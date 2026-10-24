// Spatial keys are trusted engine data. Match the field's fast Bevy maps rather
// than paying the standard library's adversarial-string hashing cost per cell.
use bevy::platform::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;

use super::naadf::{Aadf, build_aadf, linear_index, volume_len};
use super::spectrum::Radiance;
mod delta;
pub(crate) use delta::SceneStamp;

/// Transport-only sentinel outside the complete `u8` material index range.
pub const AIR: u16 = u8::MAX as u16 + 1;
pub const BRICK_EDGE: u32 = 4;
pub const REGION_EDGE: u32 = 16;
pub const PAGE_EDGE: u32 = 32;
pub const BRICKS_PER_AXIS: u32 = REGION_EDGE / BRICK_EDGE;
pub const REGION_SIZE: UVec3 = UVec3::splat(REGION_EDGE);
pub const BRICK_GRID_SIZE: UVec3 = UVec3::splat(BRICKS_PER_AXIS);
pub const PAGE_SIZE: UVec3 = UVec3::splat(PAGE_EDGE);

pub const FACE_OFFSETS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
];

/// Opaque diffuse response per band. No transmission, heat capacity or dose
/// conversion is implied; those belong to the host simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialTransport {
    /// Fraction of incident energy reflected, clamped to [0, 1] by `set`.
    pub diffuse: Vec3,
    /// Emitted energy in normalized solver units, not watts or temperature.
    pub emission: Vec3,
}

impl Default for MaterialTransport {
    fn default() -> Self {
        Self {
            diffuse: Vec3::splat(0.45),
            emission: Vec3::ZERO,
        }
    }
}

/// Band-aware optical properties for every voxel material index. This is a
/// resource so games can define emissive and reflective blocks without changing
/// the tracer. Defaults are neutral; the host assigns its own material IDs.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct VoxelMaterialTransportTable {
    entries: [MaterialTransport; 256],
}

impl Default for VoxelMaterialTransportTable {
    fn default() -> Self {
        Self {
            entries: [MaterialTransport::default(); 256],
        }
    }
}

impl VoxelMaterialTransportTable {
    pub fn get(&self, material: u8) -> MaterialTransport {
        self.entries[material as usize]
    }

    pub fn set(&mut self, material: u8, transport: MaterialTransport) {
        // A diffuse reflector cannot create energy. Bound source magnitudes
        // as well so malformed simulation input cannot poison GPU moments.
        self.entries[material as usize] = MaterialTransport {
            diffuse: transport.diffuse.bounded(1.0),
            emission: transport.emission.bounded(1.0e12),
        };
    }
}

#[derive(Clone)]
pub struct NaadfRegion {
    pub coordinate: IVec3,
    storage: RegionStorage,
    pub any_solid: bool,
    /// Geometry-owned material presence, independent of the live optical table.
    /// Source discovery can skip regions with no emitting material in O(4).
    material_mask: [u64; 4],
}

#[derive(Clone)]
enum RegionStorage {
    Empty,
    Uniform(u16),
    /// Materials only. Empty-space bounds are packed into the GPU cells and
    /// computed there from these materials (`brick_voxel_bounds`,
    /// `brick_bounds`); the CPU keeps no per-voxel bounds.
    Dense {
        materials: Box<[u16]>,
        brick_occupied: Box<[bool]>,
    },
}

impl NaadfRegion {
    fn matches_materials(&self, incoming: &[u16]) -> bool {
        match &self.storage {
            RegionStorage::Empty => incoming.iter().all(|value| *value == AIR),
            RegionStorage::Uniform(material) => incoming.iter().all(|value| value == material),
            RegionStorage::Dense { materials, .. } => materials.as_ref() == incoming,
        }
    }
    pub(crate) fn uniform_material(&self) -> Option<u16> {
        match self.storage {
            RegionStorage::Empty => Some(AIR),
            RegionStorage::Uniform(material) => Some(material),
            RegionStorage::Dense { .. } => None,
        }
    }

    pub(crate) fn contains_any_material(&self, mask: &[u64; 4]) -> bool {
        self.material_mask.iter().zip(mask).any(|(a, b)| a & b != 0)
    }

    pub fn build(coordinate: IVec3, materials: Vec<u16>) -> Self {
        assert_eq!(materials.len(), volume_len(REGION_SIZE));

        if materials.iter().all(|material| *material == AIR) {
            return Self {
                coordinate,
                storage: RegionStorage::Empty,
                any_solid: false,
                material_mask: [0; 4],
            };
        }
        if materials.iter().all(|material| *material == materials[0]) {
            let mut material_mask = [0; 4];
            material_mask[materials[0] as usize / 64] |= 1 << (materials[0] % 64);
            return Self {
                coordinate,
                storage: RegionStorage::Uniform(materials[0]),
                any_solid: true,
                material_mask,
            };
        }

        let mut brick_occupied = vec![false; volume_len(BRICK_GRID_SIZE)];
        let mut material_mask = [0; 4];
        for (index, &material) in materials.iter().enumerate() {
            if material == AIR {
                continue;
            }
            material_mask[material as usize / 64] |= 1 << (material % 64);
            let local = UVec3::new(
                index as u32 % REGION_EDGE,
                (index as u32 / REGION_EDGE) % REGION_EDGE,
                index as u32 / (REGION_EDGE * REGION_EDGE),
            );
            brick_occupied[linear_index(local / BRICK_EDGE, BRICK_GRID_SIZE)] = true;
        }
        let any_solid = brick_occupied.iter().any(|occupied| *occupied);
        Self {
            coordinate,
            storage: RegionStorage::Dense {
                materials: materials.into_boxed_slice(),
                brick_occupied: brick_occupied.into_boxed_slice(),
            },
            any_solid,
            material_mask,
        }
    }

    fn rebuild(&self, materials: Vec<u16>) -> Self {
        Self::build(self.coordinate, materials)
    }

    /// Empty-space bounds of the 64 voxels in `brick` (within the brick,
    /// voxel order x fastest); zero for solid voxels and non-dense regions.
    pub(crate) fn brick_voxel_bounds(&self, brick: UVec3) -> Vec<Aadf> {
        let brick_size = UVec3::splat(BRICK_EDGE);
        let RegionStorage::Dense { materials, .. } = &self.storage else {
            return vec![Aadf::default(); volume_len(brick_size)];
        };
        let mut blocked = [false; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize];
        for (index, cell) in blocked.iter_mut().enumerate() {
            let local = UVec3::new(index as u32 & 3, (index as u32 >> 2) & 3, index as u32 >> 4);
            *cell = materials[linear_index(brick * BRICK_EDGE + local, REGION_SIZE)] != AIR;
        }
        build_aadf(brick_size, &blocked)
    }

    /// Empty-space bounds of the region's 4^3 bricks (brick order x fastest).
    pub(crate) fn brick_bounds(&self) -> Vec<Aadf> {
        match &self.storage {
            RegionStorage::Dense { brick_occupied, .. } => build_aadf(BRICK_GRID_SIZE, brick_occupied),
            _ => vec![Aadf::default(); volume_len(BRICK_GRID_SIZE)],
        }
    }

    pub fn material(&self, local: UVec3) -> u16 {
        match &self.storage {
            RegionStorage::Empty => AIR,
            RegionStorage::Uniform(material) => *material,
            RegionStorage::Dense { materials, .. } => materials[linear_index(local, REGION_SIZE)],
        }
    }

    #[cfg(test)]
    pub fn voxel_aadf(&self, local: UVec3) -> Aadf {
        let brick = local / BRICK_EDGE;
        self.brick_voxel_bounds(brick)[linear_index(local % BRICK_EDGE, UVec3::splat(BRICK_EDGE))]
    }

    #[cfg(test)]
    pub fn brick_occupied(&self, brick: UVec3) -> bool {
        match &self.storage {
            RegionStorage::Empty => false,
            RegionStorage::Uniform(_) => true,
            RegionStorage::Dense { brick_occupied, .. } => {
                brick_occupied[linear_index(brick, BRICK_GRID_SIZE)]
            }
        }
    }

    #[cfg(test)]
    pub fn brick_aadf(&self, brick: UVec3) -> Aadf {
        self.brick_bounds()[linear_index(brick, BRICK_GRID_SIZE)]
    }
}

#[derive(Clone, Default)]
pub struct TopLevelAadf {
    pub minimum: IVec3,
    pub maximum: IVec3,
    pub size: UVec3,
    // Immutable scene snapshots can share this independently of changed voxel
    // data: only region residency and empty/nonempty topology define the field.
    fields: Arc<[Aadf]>,
}

impl TopLevelAadf {
    pub(crate) fn shares_field(&self, other: &Self) -> bool {
        self.minimum == other.minimum
            && self.size == other.size
            && Arc::ptr_eq(&self.fields, &other.fields)
    }

    fn build(regions: &HashMap<IVec3, Arc<NaadfRegion>>) -> Self {
        let Some(minimum) = regions.keys().copied().reduce(IVec3::min) else {
            return Self::default();
        };
        let maximum = regions.keys().copied().reduce(IVec3::max).unwrap();
        let size = (maximum - minimum + IVec3::ONE).as_uvec3();

        // Sparse, disconnected loaded islands must not allocate their entire
        // bounding volume (or overflow the u8 AADF extents). Traversal can safely
        // fall back to region DDA without a top-level distance field.
        if size.max_element() > 255 || size.as_u64vec3().element_product() > 262_144 {
            return Self {
                minimum,
                maximum,
                size,
                fields: Arc::default(),
            };
        }

        // Missing cells are barriers: the tracer never assumes unloaded world
        // space is empty. Loaded, empty regions are the only top-level cells
        // eligible for skipping.
        let mut blocked = vec![true; volume_len(size)];
        for (coordinate, region) in regions {
            let local = (*coordinate - minimum).as_uvec3();
            blocked[linear_index(local, size)] = region.any_solid;
        }
        let fields = build_aadf(size, &blocked);

        Self {
            minimum,
            maximum,
            size,
            fields: fields.into(),
        }
    }

    #[cfg(test)]
    pub fn contains(&self, coordinate: IVec3) -> bool {
        self.size != UVec3::ZERO
            && coordinate.cmpge(self.minimum).all()
            && coordinate.cmple(self.maximum).all()
    }

    pub fn field(&self, coordinate: IVec3) -> Aadf {
        if self.fields.is_empty() {
            return Aadf::default();
        }
        let local = (coordinate - self.minimum).as_uvec3();
        self.fields[linear_index(local, self.size)]
    }
}

#[derive(Clone, Default)]
pub struct TransportScene {
    pub regions: HashMap<IVec3, Arc<NaadfRegion>>,
    pub top_level: TopLevelAadf,
    journal: delta::SceneJournal,
}

impl TransportScene {
    pub(crate) fn stamp(&self) -> SceneStamp {
        self.journal.stamp.clone()
    }

    pub(crate) fn region_delta_since(&self, stamp: &SceneStamp) -> Option<Vec<IVec3>> {
        self.journal.since(stamp)
    }

    /// Canonical region identity delta. Lineage avoids whole-world discovery;
    /// independent scenes and consumers older than the journal compare safely.
    pub(crate) fn changed_regions(&self, previous: &Self) -> Vec<IVec3> {
        let mut keys = self
            .region_delta_since(&previous.stamp())
            .unwrap_or_else(|| {
                self.regions
                    .keys()
                    .chain(previous.regions.keys())
                    .copied()
                    .collect()
            });
        keys.retain(
            |key| match (previous.regions.get(key), self.regions.get(key)) {
                (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
                (None, None) => false,
                _ => true,
            },
        );
        keys.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        keys.dedup();
        keys
    }

    pub fn with_page_delta(
        previous: &Self,
        changed_pages: Vec<(IVec3, Vec<u16>)>,
        removed_pages: &[IVec3],
    ) -> Self {
        let mut regions = previous.regions.clone();
        let mut touched = HashSet::new();
        for page in removed_pages {
            for offset in page_region_offsets() {
                let key = page * 2 + offset;
                if regions.remove(&key).is_some() {
                    touched.insert(key);
                }
            }
        }

        // Scene construction already runs on Bevy's async compute pool. Keeping
        // this loop serial prevents a rebuild from borrowing every Rayon worker
        // and starving rendering/physics on lower-core-count machines.
        let built: Vec<_> = changed_pages
            .into_iter()
            .flat_map(|(page, materials)| build_page_regions(page, &materials, previous))
            .collect();
        for region in built {
            if previous
                .regions
                .get(&region.coordinate)
                .is_none_or(|old| !Arc::ptr_eq(old, &region))
            {
                touched.insert(region.coordinate);
            }
            regions.insert(region.coordinate, region);
        }

        touched.retain(|key| match (previous.regions.get(key), regions.get(key)) {
            (Some(a), Some(b)) => !Arc::ptr_eq(a, b),
            (None, None) => false,
            _ => true,
        });

        // A voxel/material edit inside a still-occupied region changes rays,
        // but not this level's skip bounds. Reuse only the acceleration field;
        // region identities still drive normal transport/source invalidation.
        let topology_unchanged = touched.iter().all(|key| {
            previous
                .regions
                .get(key)
                .zip(regions.get(key))
                .is_some_and(|(old, region)| old.any_solid == region.any_solid)
        });
        let top_level = if topology_unchanged {
            previous.top_level.clone()
        } else {
            TopLevelAadf::build(&regions)
        };
        let mut keys: Vec<_> = touched.into_iter().collect();
        keys.sort_unstable_by_key(|p| (p.z, p.y, p.x));
        Self {
            regions,
            top_level,
            journal: previous.journal.advance(keys),
        }
    }

    /// Returns `None` outside the loaded scene, `Some(AIR)` for loaded air, or
    /// the material index for a solid voxel.
    pub fn material_at(&self, world_voxel: IVec3) -> Option<u16> {
        let region_coordinate = world_voxel.div_euclid(IVec3::splat(REGION_EDGE as i32));
        let local = world_voxel.rem_euclid(IVec3::splat(REGION_EDGE as i32));
        self.regions
            .get(&region_coordinate)
            .map(|region| region.material(local.as_uvec3()))
    }

    #[cfg(test)]
    pub fn world_bounds(&self) -> Option<(Vec3, Vec3)> {
        (self.top_level.size != UVec3::ZERO).then(|| {
            (
                (self.top_level.minimum * REGION_EDGE as i32).as_vec3(),
                ((self.top_level.maximum + IVec3::ONE) * REGION_EDGE as i32).as_vec3(),
            )
        })
    }
}

#[derive(Clone, Debug)]
pub struct SurfaceProbe {
    pub voxel: IVec3,
    pub material: u8,
    pub order_index: usize,
    pub emission: Vec3,
    pub face_mask: u8,
    /// Test-only CPU measurements (`ProbeStatistics`); the GPU state is the
    /// authority and production probes carry none.
    #[cfg(test)]
    pub(crate) stats: Option<Box<ProbeStatistics>>,
    /// Small publication/diagnostic mirror, present without full moments.
    pub(crate) progress: ProbeProgress,
    pub(crate) display: [super::history::FaceDisplay; 6],
    /// Optional resident slot and allocation incarnation. Zero incarnation is
    /// the CPU/display-prior fallback; recycled slots cannot alias old readers.
    pub(crate) resident: [UVec2; 6],
    /// Selection in the full-probe test oracle. Production selection belongs
    /// only to the compact `ProbeWork`, never the persistent receiver.
    #[cfg(test)]
    pub(crate) trace_mask: u8,
    /// Faces refreshed in transport_revision; revision changes are lazy O(1).
    pub(crate) fresh_mask: u8,
    /// Faces whose last GPU merge the CPU rejected (selected before a revision
    /// or model change). The next selection asks the GPU to undo that merge.
    pub(crate) stale_merge: u8,
    /// Faces relit (material or exposure change) since their last accepted
    /// CPU job: that job caps their GPU history to NEAR_HISTORY. The cap is
    /// sent explicitly, never derived from `samples`: the CPU's counts are a
    /// mirror of the GPU's and can lag it (or read 0 for a face never merged).
    pub(crate) relit_mask: u8,
    pub(crate) transport_revision: u64,
    /// Independent spatial evidence consumed by statistical retention.
    /// Sunlight changes can arrive after transport refreshed in the same scene.
    /// Changes only when this probe's geometry/history is invalidated.
    pub revision: u64,
    pub source_revision: u64,
    pub updated_at: Option<Instant>,
}

/// Sequence ordering and HUD progress need no CPU radiance/moment copy.
/// Counts saturate at bootstrap, not at a claimed convergence threshold.
#[derive(Clone, Debug, Default)]
pub(crate) struct ProbeProgress {
    pub sequence: [u32; 6],
    pub bootstrap: [u8; 6],
    pub settled: u8,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FaceMeasurement {
    pub radiance: Vec3,
    pub second_moment: Vec3,
    pub direct: Vec3,
    pub custom_direct: Vec3,
    pub samples: u32,
    pub updated_at: Option<Instant>,
}

/// Test-only CPU copy of a probe's measurements, filled by the native GPU test
/// harness and fixtures. Production keeps none: the GPU state is the authority.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct ProbeStatistics {
    /// Unfiltered indirect transport measurements. Presentation history is
    /// separate; simulations never receive spatial or temporal display filters.
    pub radiance: [Vec3; 6],
    /// Direct incident energy in every band.
    pub direct: [Vec3; 6],
    /// Visible analytic contributions not already rendered by Bevy PBR.
    pub custom_direct: [Vec3; 6],
    pub samples: [u32; 6],
    pub(crate) stable_batches: [u8; 6],
    pub(crate) face_updated_at: [Option<Instant>; 6],
    /// Sampling sequence is independent of history length and invalidation.
    pub sequence: [u32; 6],
    /// Per-band second raw moment.
    pub second_moment: [Vec3; 6],
}

#[cfg(test)]
impl ProbeStatistics {
    pub(crate) const EMPTY: Self = Self {
        radiance: [Vec3::ZERO; 6],
        direct: [Vec3::ZERO; 6],
        custom_direct: [Vec3::ZERO; 6],
        samples: [0; 6],
        stable_batches: [0; 6],
        face_updated_at: [None; 6],
        sequence: [0; 6],
        second_moment: [Vec3::ZERO; 6],
    };
}

#[cfg(test)]
impl Default for ProbeStatistics {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl SurfaceProbe {
    /// A probe with test-only CPU statistics.
    #[cfg(test)]
    pub(crate) fn new(voxel: IVec3, face_mask: u8) -> Self {
        Self {
            stats: Some(Box::default()),
            ..Self::new_slim(voxel, face_mask)
        }
    }

    /// A probe without CPU statistics: production GPU transport.
    pub(crate) fn new_slim(voxel: IVec3, face_mask: u8) -> Self {
        Self {
            voxel,
            material: 0,
            order_index: 0,
            emission: Vec3::ZERO,
            face_mask,
            #[cfg(test)]
            stats: None,
            progress: default(),
            display: [super::history::FaceDisplay::default(); 6],
            resident: [UVec2::ZERO; 6],
            #[cfg(test)]
            trace_mask: 63,
            fresh_mask: 63,
            stale_merge: 0,
            relit_mask: 0,
            transport_revision: 0,
            revision: 0,
            source_revision: 0,
            updated_at: None,
        }
    }

    /// Test helper: append one raw sample to the CPU statistics.
    #[cfg(test)]
    pub(crate) fn accumulate_sample(&mut self, face: usize, sample: Vec3) {
        let sample = sample.finite_or_zero();
        let count = (self.stats().samples[face] + 1).min(super::tracer::HISTORY_LIMIT);
        let alpha = 1.0 / count as f32;
        let stats = self.stats_mut();
        stats.radiance[face] += (sample - stats.radiance[face]) * alpha;
        stats.second_moment[face] += (sample * sample - stats.second_moment[face]) * alpha;
        stats.samples[face] = count;
        stats.sequence[face] = stats.sequence[face].wrapping_add(1);
    }

    /// CPU statistics, or empty ones when this probe keeps none.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> &ProbeStatistics {
        self.stats.as_deref().unwrap_or(&ProbeStatistics::EMPTY)
    }

    /// CPU statistics, allocated on first write.
    #[cfg(test)]
    pub(crate) fn stats_mut(&mut self) -> &mut ProbeStatistics {
        self.stats.get_or_insert_with(Box::default)
    }

    /// A face's CPU measurement for a job input: radiance, second moment,
    /// direct, custom direct, samples and time. Production probes keep none
    /// (the GPU slot is the authority), so this is empty outside tests.
    pub(crate) fn measured(&self, face: usize) -> FaceMeasurement {
        #[cfg(test)]
        if let Some(stats) = &self.stats {
            return FaceMeasurement {
                radiance: stats.radiance[face],
                second_moment: stats.second_moment[face],
                direct: stats.direct[face],
                custom_direct: stats.custom_direct[face],
                samples: stats.samples[face],
                updated_at: stats.face_updated_at[face],
            };
        }
        let _ = face;
        FaceMeasurement::default()
    }

    pub(crate) fn sequence(&self, face: usize) -> u32 {
        #[cfg(test)]
        if let Some(stats) = &self.stats {
            return stats.sequence[face];
        }
        self.progress.sequence[face]
    }

    pub(crate) fn bootstrap_samples(&self, face: usize) -> u32 {
        #[cfg(test)]
        if let Some(stats) = &self.stats {
            return stats.samples[face].min(super::tracer::BOOTSTRAP_SAMPLES);
        }
        u32::from(self.progress.bootstrap[face])
    }

    pub(crate) fn update_face_mask(&mut self, face_mask: u8) {
        if self.face_mask != face_mask {
            self.request_relight();
        }
        let newly_exposed = face_mask & !self.face_mask;
        self.face_mask = face_mask;
        for face in 0..6 {
            if newly_exposed & (1 << face) == 0 {
                continue;
            }
            self.display[face] = default();
            self.resident[face] = UVec2::ZERO;
            self.progress.bootstrap[face] = 0;
            self.progress.settled &= !(1 << face);
            #[cfg(test)]
            if let Some(stats) = self.stats.as_mut() {
                stats.radiance[face] = Vec3::ZERO;
                stats.second_moment[face] = Vec3::ZERO;
                stats.direct[face] = Vec3::ZERO;
                stats.custom_direct[face] = Vec3::ZERO;
                stats.samples[face] = 0;
                stats.face_updated_at[face] = None;
            }
        }
    }

    /// Changed geometry may affect a path, but does not make every nearby
    /// receiver a new surface. Retain bounded measured history, explicitly
    /// stale, while replacing it with fresh paths. No borrowed/display energy
    /// enters these statistics. Material replacement uses the same rule;
    /// only newly exposed faces start without compatible measurements.
    pub(crate) fn request_relight(&mut self) {
        self.request_refresh();
        self.relit_mask |= self.face_mask;
        let limit = super::geometry::NEAR_HISTORY;
        self.progress.bootstrap = self.progress.bootstrap.map(|count| u32::from(count).min(limit) as u8);
        #[cfg(test)]
        if let Some(stats) = self.stats.as_mut() {
            stats.samples = stats.samples.map(|count| count.min(limit));
        }
    }

    /// A nearby edit: reject in-flight pre-edit results and stop treating the
    /// faces as settled, but keep their sample counts. How much history to
    /// keep is measured retention's decision, and it reads these counts.
    pub(crate) fn request_refresh(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.fresh_mask = 0;
        self.progress.settled = 0;
        #[cfg(test)]
        if let Some(stats) = self.stats.as_mut() {
            stats.stable_batches = [0; 6];
        }
    }

    /// Full-probe reference for test fixtures. Production model replacement uses
    /// compact `TraceWork`; keep the same measurement/display separation here.
    #[cfg(test)]
    pub(crate) fn restart_measurements(&mut self) {
        self.fresh_mask = 0;
        let stats = self.stats_mut();
        stats.samples = [0; 6];
        stats.stable_batches = [0; 6];
        stats.face_updated_at = [None; 6];
        stats.radiance = [Vec3::ZERO; 6];
        stats.second_moment = [Vec3::ZERO; 6];
        // Presentation weights belong to the retained display, not the reset
        // transport statistics. Changing only weights would visibly relight
        // unchanged shared corners on every edit.
        // Keep the previous display estimate until replacement samples arrive,
        // but never advertise it as a valid measurement to a simulation.
        self.updated_at = None;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SurfaceProbeDescriptor {
    pub voxel: IVec3,
    pub face_mask: u8,
    pub material: u8,
}

/// Pages whose probe topology can change after the supplied page delta. A
/// voxel edit affects membership in its own page and at the six touching page
/// boundaries; no other page needs to be rescanned.
pub fn affected_probe_pages(changed_pages: &[IVec3], removed_pages: &[IVec3]) -> HashSet<IVec3> {
    let mut affected = HashSet::new();
    for page in changed_pages.iter().chain(removed_pages) {
        affected.insert(*page);
        for offset in FACE_OFFSETS {
            affected.insert(*page + offset);
        }
    }
    affected
}

/// Rebuild only the probe descriptors in the requested 32-cubed pages. This
/// is the hot edit path; it avoids cloning and rescanning the complete loaded
/// world merely because one voxel changed.
pub fn scan_surface_probe_pages(
    scene: &TransportScene,
    pages: &HashSet<IVec3>,
    focus_page: IVec3,
) -> Vec<SurfaceProbeDescriptor> {
    let mut found = Vec::new();
    let mut ordered_pages: Vec<_> = pages.iter().copied().collect();
    ordered_pages.sort_unstable_by_key(|page| {
        let delta = (*page - focus_page).as_i64vec3();
        (delta.length_squared(), page.z, page.y, page.x)
    });
    for page in ordered_pages {
        let origin = page * PAGE_EDGE as i32;
        for z in 0..PAGE_EDGE {
            for y in 0..PAGE_EDGE {
                for x in 0..PAGE_EDGE {
                    let voxel = origin + UVec3::new(x, y, z).as_ivec3();
                    let Some(material) =
                        scene.material_at(voxel).filter(|material| *material != AIR)
                    else {
                        continue;
                    };

                    let mut face_mask = 0_u8;
                    for (face, offset) in FACE_OFFSETS.into_iter().enumerate() {
                        if !matches!(scene.material_at(voxel + offset), Some(material) if material != AIR)
                        {
                            face_mask |= 1 << face;
                        }
                    }
                    if face_mask != 0 {
                        found.push(SurfaceProbeDescriptor {
                            voxel,
                            face_mask,
                            material: material as u8,
                        });
                    }
                }
            }
        }
    }
    found
}

#[cfg(test)]
pub fn rebuild_surface_probes(
    scene: &TransportScene,
    previous: &HashMap<IVec3, SurfaceProbe>,
) -> (HashMap<IVec3, SurfaceProbe>, Vec<IVec3>) {
    let pages: HashSet<_> = scene
        .regions
        .keys()
        .map(|region| region.div_euclid(IVec3::splat(2)))
        .collect();
    let descriptors = scan_surface_probe_pages(scene, &pages, IVec3::ZERO);

    let mut probes = HashMap::new();
    for descriptor in descriptors {
        let voxel = descriptor.voxel;
        let face_mask = descriptor.face_mask;
        let mut probe = previous
            .get(&voxel)
            .cloned()
            .unwrap_or_else(|| SurfaceProbe::new(voxel, face_mask));
        probe.face_mask = face_mask;
        // Preserve the last stable value on screen while forcing fresh samples
        // after any geometry change.
        probe.stats_mut().samples = [0; 6];
        probes.insert(voxel, probe);
    }

    let mut order: Vec<_> = probes.keys().copied().collect();
    order.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    (probes, order)
}

fn build_page_regions(
    page: IVec3,
    page_materials: &[u16],
    previous: &TransportScene,
) -> Vec<Arc<NaadfRegion>> {
    assert_eq!(page_materials.len(), volume_len(PAGE_SIZE));
    let mut regions = Vec::with_capacity(8);
    for offset in page_region_offsets() {
        let mut materials = vec![AIR; volume_len(REGION_SIZE)];
        let page_start = offset.as_uvec3() * REGION_EDGE;
        for z in 0..REGION_EDGE {
            for y in 0..REGION_EDGE {
                for x in 0..REGION_EDGE {
                    let local = UVec3::new(x, y, z);
                    materials[linear_index(local, REGION_SIZE)] =
                        page_materials[linear_index(page_start + local, PAGE_SIZE)];
                }
            }
        }
        let coordinate = page * 2 + offset;
        if let Some(region) = previous.regions.get(&coordinate)
            && region.matches_materials(&materials)
        {
            regions.push(Arc::clone(region));
        } else {
            let region = match previous.regions.get(&coordinate) {
                Some(old) => old.rebuild(materials),
                None => NaadfRegion::build(coordinate, materials),
            };
            regions.push(Arc::new(region));
        }
    }
    regions
}

fn page_region_offsets() -> impl Iterator<Item = IVec3> {
    (0..2).flat_map(|z| (0..2).flat_map(move |y| (0..2).map(move |x| IVec3::new(x, y, z))))
}

#[cfg(test)]
mod construction_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unchanged_page_notifications_reuse_acceleration_regions() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(UVec3::new(3, 4, 5), PAGE_SIZE)] = 1;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        let next = TransportScene::with_page_delta(&scene, vec![(IVec3::ZERO, page)], &[]);
        for (key, region) in &scene.regions {
            assert!(Arc::ptr_eq(region, &next.regions[key]));
        }
    }

    #[test]
    fn material_updates_cannot_introduce_nan_or_amplifying_reflectance() {
        let mut table = VoxelMaterialTransportTable::default();
        table.set(
            7,
            MaterialTransport {
                diffuse: Vec3::splat(5.0),
                emission: Vec3::splat(f32::INFINITY),
            },
        );
        let material = table.get(7);
        assert_eq!(material.diffuse, Vec3::ONE);
        assert_eq!(
            material.emission,
            Vec3::ZERO
        );
    }

    #[test]
    fn page_delta_preserves_other_pages_and_removes_all_child_regions() {
        let solid_page = vec![1; volume_len(PAGE_SIZE)];
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, solid_page.clone()), (IVec3::X, solid_page)],
            &[],
        );
        assert_eq!(scene.regions.len(), 16);

        let scene = TransportScene::with_page_delta(&scene, vec![], &[IVec3::ZERO]);
        assert_eq!(scene.regions.len(), 8);
        assert!(scene.regions.keys().all(|coordinate| coordinate.x >= 2));
    }

    #[test]
    fn probes_exist_only_on_exposed_solid_voxels() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let voxel = UVec3::new(4, 5, 6);
        page[linear_index(voxel, PAGE_SIZE)] = 1;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );
        let (probes, _) = rebuild_surface_probes(&scene, &HashMap::new());

        let probe = probes.get(&voxel.as_ivec3()).unwrap();
        assert_eq!(probe.face_mask, 0b11_1111);
    }
}
