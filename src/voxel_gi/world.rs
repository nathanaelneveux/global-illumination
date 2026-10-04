use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use bevy::prelude::*;

use super::naadf::{Aadf, build_aadf, linear_index, volume_len};
use super::spectrum::EnergySpectrum;

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MaterialTransport {
    pub diffuse: EnergySpectrum,
    pub emission: EnergySpectrum,
}

impl Default for MaterialTransport {
    fn default() -> Self {
        Self {
            diffuse: EnergySpectrum::new(Vec3::splat(0.45), 0.35, 0.04),
            emission: EnergySpectrum::ZERO,
        }
    }
}

/// Band-aware optical properties for every voxel material index. This is a
/// resource so games can define emissive, heat-reflective, or radiation-
/// shielding blocks without changing the tracer.
#[derive(Resource, Clone, Debug, PartialEq)]
pub struct VoxelMaterialTransportTable {
    entries: [MaterialTransport; 256],
}

impl Default for VoxelMaterialTransportTable {
    fn default() -> Self {
        let mut entries = [MaterialTransport::default(); 256];
        let visible = [
            Vec3::new(0.17, 0.42, 0.08),
            Vec3::new(0.55, 0.20, 0.09),
            Vec3::new(0.46, 0.29, 0.20),
            Vec3::new(0.82, 0.88, 0.94),
        ];
        for (entry, visible) in entries.iter_mut().zip(visible) {
            entry.diffuse.visible = visible;
        }
        Self { entries }
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
}

#[derive(Clone)]
enum RegionStorage {
    Empty,
    Uniform(u16),
    Dense {
        materials: Box<[u16]>,
        voxel_aadf: Box<[Aadf]>,
        brick_occupied: Box<[bool]>,
        brick_aadf: Box<[Aadf]>,
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

    pub fn build(coordinate: IVec3, materials: Vec<u16>) -> Self {
        assert_eq!(materials.len(), volume_len(REGION_SIZE));

        if materials.iter().all(|material| *material == AIR) {
            return Self {
                coordinate,
                storage: RegionStorage::Empty,
                any_solid: false,
            };
        }
        if materials.iter().all(|material| *material == materials[0]) {
            return Self {
                coordinate,
                storage: RegionStorage::Uniform(materials[0]),
                any_solid: true,
            };
        }

        let mut voxel_aadf = vec![Aadf::default(); materials.len()];
        let mut brick_occupied = vec![false; volume_len(BRICK_GRID_SIZE)];

        for brick_z in 0..BRICKS_PER_AXIS {
            for brick_y in 0..BRICKS_PER_AXIS {
                for brick_x in 0..BRICKS_PER_AXIS {
                    let brick = UVec3::new(brick_x, brick_y, brick_z);
                    let mut blocked = vec![false; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize];
                    let brick_size = UVec3::splat(BRICK_EDGE);

                    for z in 0..BRICK_EDGE {
                        for y in 0..BRICK_EDGE {
                            for x in 0..BRICK_EDGE {
                                let local = UVec3::new(x, y, z);
                                let region_local = brick * BRICK_EDGE + local;
                                blocked[linear_index(local, brick_size)] =
                                    materials[linear_index(region_local, REGION_SIZE)] != AIR;
                            }
                        }
                    }

                    let brick_index = linear_index(brick, BRICK_GRID_SIZE);
                    brick_occupied[brick_index] = blocked.iter().any(|occupied| *occupied);
                    let local_aadf = build_aadf(brick_size, &blocked);
                    for z in 0..BRICK_EDGE {
                        for y in 0..BRICK_EDGE {
                            for x in 0..BRICK_EDGE {
                                let local = UVec3::new(x, y, z);
                                let region_local = brick * BRICK_EDGE + local;
                                voxel_aadf[linear_index(region_local, REGION_SIZE)] =
                                    local_aadf[linear_index(local, brick_size)];
                            }
                        }
                    }
                }
            }
        }

        let brick_aadf = build_aadf(BRICK_GRID_SIZE, &brick_occupied);
        let any_solid = brick_occupied.iter().any(|occupied| *occupied);
        Self {
            coordinate,
            storage: RegionStorage::Dense {
                materials: materials.into_boxed_slice(),
                voxel_aadf: voxel_aadf.into_boxed_slice(),
                brick_occupied: brick_occupied.into_boxed_slice(),
                brick_aadf: brick_aadf.into_boxed_slice(),
            },
            any_solid,
        }
    }

    pub fn material(&self, local: UVec3) -> u16 {
        match &self.storage {
            RegionStorage::Empty => AIR,
            RegionStorage::Uniform(material) => *material,
            RegionStorage::Dense { materials, .. } => materials[linear_index(local, REGION_SIZE)],
        }
    }

    pub fn voxel_aadf(&self, local: UVec3) -> Aadf {
        match &self.storage {
            RegionStorage::Dense { voxel_aadf, .. } => voxel_aadf[linear_index(local, REGION_SIZE)],
            _ => Aadf::default(),
        }
    }

    pub fn brick_occupied(&self, brick: UVec3) -> bool {
        match &self.storage {
            RegionStorage::Empty => false,
            RegionStorage::Uniform(_) => true,
            RegionStorage::Dense { brick_occupied, .. } => {
                brick_occupied[linear_index(brick, BRICK_GRID_SIZE)]
            }
        }
    }

    pub fn brick_aadf(&self, brick: UVec3) -> Aadf {
        match &self.storage {
            RegionStorage::Dense { brick_aadf, .. } => {
                brick_aadf[linear_index(brick, BRICK_GRID_SIZE)]
            }
            _ => Aadf::default(),
        }
    }
}

#[derive(Clone, Default)]
pub struct TopLevelAadf {
    pub minimum: IVec3,
    pub maximum: IVec3,
    pub size: UVec3,
    fields: Vec<Aadf>,
}

impl TopLevelAadf {
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
                fields: Vec::new(),
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
            fields,
        }
    }

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
}

impl TransportScene {
    pub fn with_page_delta(
        previous: &Self,
        changed_pages: Vec<(IVec3, Vec<u16>)>,
        removed_pages: &[IVec3],
    ) -> Self {
        let mut regions = previous.regions.clone();
        for page in removed_pages {
            for offset in page_region_offsets() {
                regions.remove(&(page * 2 + offset));
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
            regions.insert(region.coordinate, region);
        }

        let unchanged = regions.len() == previous.regions.len()
            && regions.iter().all(|(key, region)| {
                previous
                    .regions
                    .get(key)
                    .is_some_and(|old| Arc::ptr_eq(old, region))
            });
        let top_level = if unchanged {
            previous.top_level.clone()
        } else {
            TopLevelAadf::build(&regions)
        };
        Self { regions, top_level }
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
    pub emission: EnergySpectrum,
    pub face_mask: u8,
    /// Unfiltered indirect transport measurements. Presentation history is
    /// separate; simulations never receive spatial or temporal display filters.
    pub radiance: [EnergySpectrum; 6],
    /// Direct incident energy, retained for gameplay queries in every band.
    pub direct: [EnergySpectrum; 6],
    /// Visible analytic contributions not already rendered by Bevy PBR.
    pub custom_direct: [Vec3; 6],
    /// Ordinary Bevy directional-light diffuse energy, excluding custom
    /// emitter additions. Retained independently for a display-only A/B mode.
    pub direct_sky: [Vec3; 6],
    pub(crate) display: [super::history::FaceDisplay; 6],
    pub(crate) display_updated_at: Option<Instant>,
    pub samples: [u32; 6],
    /// Sampling sequence is independent of history length and invalidation.
    pub sequence: [u32; 6],
    /// Per-band second raw moment, used to expose uncertainty to simulations.
    pub second_moment: [EnergySpectrum; 6],
    /// Changes only when this probe's geometry/history is invalidated.
    pub revision: u64,
    pub source_revision: u64,
    pub updated_at: Option<Instant>,
}

impl SurfaceProbe {
    /// Worker results contain a snapshot of topology bookkeeping. Streaming
    /// may compact the order array while a trace is in flight; only the main
    /// world's current index is authoritative when a result is merged.
    pub(crate) fn preserve_topology_from(&mut self, current: &Self) {
        self.order_index = current.order_index;
    }

    pub(crate) fn new(voxel: IVec3, face_mask: u8) -> Self {
        Self {
            voxel,
            material: 0,
            order_index: 0,
            emission: EnergySpectrum::ZERO,
            face_mask,
            radiance: [EnergySpectrum::ZERO; 6],
            direct: [EnergySpectrum::ZERO; 6],
            custom_direct: [Vec3::ZERO; 6],
            direct_sky: [Vec3::ZERO; 6],
            display: [super::history::FaceDisplay::default(); 6],
            display_updated_at: None,
            samples: [0; 6],
            sequence: [0; 6],
            second_moment: [EnergySpectrum::ZERO; 6],
            revision: 0,
            source_revision: 0,
            updated_at: None,
        }
    }

    pub(crate) fn update_face_mask(&mut self, face_mask: u8) {
        if self.face_mask != face_mask {
            self.invalidate();
        }
        let newly_exposed = face_mask & !self.face_mask;
        self.face_mask = face_mask;
        for face in 0..6 {
            if newly_exposed & (1 << face) == 0 {
                continue;
            }
            self.radiance[face] = EnergySpectrum::ZERO;
            self.second_moment[face] = EnergySpectrum::ZERO;
            self.display[face] = default();
            self.direct[face] = EnergySpectrum::ZERO;
            self.custom_direct[face] = Vec3::ZERO;
            self.direct_sky[face] = Vec3::ZERO;
            self.samples[face] = 0;
        }
    }

    pub(crate) fn invalidate(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.samples = [0; 6];
        self.radiance = [EnergySpectrum::ZERO; 6];
        self.second_moment = [EnergySpectrum::ZERO; 6];
        for display in &mut self.display {
            display.confidence = display.confidence.min(0.25);
        }
        self.display_updated_at = Some(Instant::now());
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
        probe.samples = [0; 6];
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
            regions.push(Arc::new(NaadfRegion::build(coordinate, materials)));
        }
    }
    regions
}

fn page_region_offsets() -> impl Iterator<Item = IVec3> {
    (0..2).flat_map(|z| (0..2).flat_map(move |y| (0..2).map(move |x| IVec3::new(x, y, z))))
}

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
                diffuse: EnergySpectrum::new(Vec3::splat(5.0), -2.0, f32::NAN),
                emission: EnergySpectrum::new(Vec3::splat(f32::INFINITY), 1e30, -1.0),
            },
        );
        let material = table.get(7);
        assert_eq!(material.diffuse, EnergySpectrum::new(Vec3::ONE, 0.0, 0.0));
        assert_eq!(
            material.emission,
            EnergySpectrum::new(Vec3::ZERO, 1e12, 0.0)
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
