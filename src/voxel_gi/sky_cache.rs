//! Persistent world-space corner sun visibility. Visibility is traced once per
//! lattice position and quantized sun direction; the display applies each
//! light's current direction and color. Geometry changes invalidate
//! shadow-swept spatial buckets, not the world; a sun moving within one
//! quantization step changes nothing here.
use std::sync::Arc;
use std::time::{Duration, Instant};

// Trusted integer lattice keys: use the engine's fast hasher on this hot path.
use bevy::platform::collections::{HashMap, HashSet, hash_map::Entry};
use bevy::prelude::*;

use super::sky::corner_position;
use super::sources::TransportSources;
use super::tracer::MAX_PATH_DISTANCE;
use super::work::ProbeWork;
#[cfg(test)]
use super::world::SurfaceProbe;
use super::world::{AIR, FACE_OFFSETS, REGION_EDGE, TransportScene};

const BUCKET_EDGE: i32 = REGION_EDGE as i32;

/// Sky needs topology and output slots, never transport/display history.
pub(crate) trait SkyReceiver {
    fn voxel(&self) -> IVec3;
    fn face_mask(&self) -> u8;
    fn set_sky(&mut self, face: usize, value: [u32; 4]);
}

impl SkyReceiver for ProbeWork {
    fn voxel(&self) -> IVec3 {
        self.voxel
    }
    fn face_mask(&self) -> u8 {
        if self.sky_requested {
            self.face_mask
        } else {
            0
        }
    }
    fn set_sky(&mut self, face: usize, value: [u32; 4]) {
        self.sky[face] = value;
    }
}

#[cfg(test)]
impl SkyReceiver for SurfaceProbe {
    fn voxel(&self) -> IVec3 {
        self.voxel
    }
    fn face_mask(&self) -> u8 {
        self.face_mask
    }
    fn set_sky(&mut self, face: usize, value: [u32; 4]) {
        self.display[face].sky = value;
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SkyLight {
    pub direction: Vec3,
    pub color: Vec3,
}

/// The directional lights the corner cache traces: those with visible color,
/// in source order, at most `MAX_SKY_LIGHTS` (a corner's lit bit `i` is light
/// `i` of this list; the GPU kernel counts the same way).
pub(crate) fn sky_lights(sources: &TransportSources) -> Vec<SkyLight> {
    sources
        .directional
        .iter()
        .filter_map(|light| {
            let color = (light.radiance.visible - light.custom_visible).max(Vec3::ZERO);
            (color.max_element() > 0.0).then_some(SkyLight {
                direction: light.direction.normalize_or_zero(),
                color,
            })
        })
        .take(super::sky::MAX_SKY_LIGHTS)
        .collect()
}

/// Sun directions per visibility generation: a cube-projected grid with about
/// `SUN_STEP_DEGREES` cells. Cached visibility is reused until a light leaves
/// its cell; the display's lighting follows the exact direction meanwhile.
pub(crate) fn visibility_key(lights: &[SkyLight]) -> Vec<IVec3> {
    let cells = 1.0 / SUN_STEP_DEGREES.to_radians();
    lights
        .iter()
        .map(|light| {
            let direction = light.direction;
            let extent = direction.abs().max_element().max(1e-6);
            (direction / extent * cells).round().as_ivec3()
        })
        .collect()
}

/// Shadow borders move in steps of about this angle.
pub(crate) const SUN_STEP_DEGREES: f32 = 0.5;

/// While the sun moves, each visibility generation flips corners along every
/// shadow border. Publishing those changes with every batch made each resident
/// face test them all on the GPU (half the moving-sun path throughput), so
/// they publish at most this often; edits under a still sun publish at once.
const SUN_CHANGE_INTERVAL: Duration = Duration::from_millis(250);
/// The sun counts as moving for this long after its last visibility step.
const SUN_MOVING_WINDOW: Duration = Duration::from_secs(10);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SkyCacheStats {
    pub requested: usize,
    pub traced: usize,
    pub rays: usize,
    pub resident: usize,
}

#[derive(Default)]
struct Corner {
    value: u32,
    generation: u64,
    pending_batch: u64,
    pending_generation: u64,
}

#[derive(Clone, Copy)]
pub(crate) struct SkyTicket {
    generation: u64,
    batch: u64,
}

#[derive(Default)]
pub(crate) struct SkyCache {
    corners: HashMap<IVec3, Corner>,
    buckets: HashMap<IVec3, Vec<IVec3>>,
    scene: Option<Arc<TransportScene>>,
    pub lights: Vec<SkyLight>,
    key: Vec<IVec3>,
    generation: u64,
    batch: u64,
    lit_faces: u8,
    pub stats: SkyCacheStats,
    pub track_changes: bool,
    /// Resident submissions can overlap. Do not retrace shared corners that
    /// already have a still-valid pending measurement.
    pub deferred_results: bool,
    changes: HashMap<IVec3, (super::geometry::Bounds, f32)>,
    /// Last visibility step caused by sun motion, and last published changes.
    moved_at: Option<Instant>,
    published_at: Option<Instant>,
    /// Diffuse luminance per material: a flip's change is the light the
    /// corner's surfaces reflect, as edit flux measures it. Unset: 1.
    reflectance: Option<Box<[f32; 256]>>,
    /// Corners whose lit bits flipped in this visibility generation: where
    /// shadow borders were. A sun step moves borders by about a voxel, so the
    /// next generation retraces these and their axis neighbours first.
    flipped: Vec<IVec3>,
    /// Voxels around that band and around corners geometry changes
    /// invalidated, for the next sky requests.
    band: Vec<IVec3>,
}

impl SkyCache {
    pub fn ticket(&self) -> SkyTicket {
        SkyTicket {
            generation: self.generation,
            batch: self.batch,
        }
    }

    pub fn store_ticket(&mut self, ticket: SkyTicket, position: IVec3, value: u32) {
        if ticket.generation == self.generation
            && self
                .corners
                .get(&position)
                .is_some_and(|corner| corner.pending_batch == ticket.batch)
        {
            self.store(position, value);
        }
    }
    /// Build spatial bins on the tracing worker. Unknown coverage is explicit
    /// conservative evidence, not a claim that the whole sky cache is complete.
    pub fn take_changes(
        &mut self,
        scene: &TransportScene,
    ) -> Option<super::geometry::GeometryChange> {
        if self.changes.is_empty() {
            return None;
        }
        let now = Instant::now();
        let moving = self.moved_at.is_some_and(|at| now - at < SUN_MOVING_WINDOW);
        if moving && self.published_at.is_some_and(|at| now - at < SUN_CHANGE_INTERVAL) {
            // Buckets keep merging bounds and flux until the next publication.
            return None;
        }
        self.published_at = Some(now);
        let changes: Vec<_> = self.changes.drain().map(|(_, change)| change).collect();
        let bounds: Vec<_> = changes.iter().map(|(bounds, _)| *bounds).collect();
        let mut change = super::geometry::GeometryChange::from_bounds(
            scene,
            &bounds,
            super::retention::INFLUENCE_RADIUS,
        );
        change.weights = changes
            .into_iter()
            .map(|(bounds, flux)| {
                (
                    bounds,
                    super::geometry::ChangeWeight {
                        kind: super::geometry::ChangeKind::Light,
                        flux,
                    },
                )
            })
            .collect();
        Some(change)
    }

    pub fn prepare<P: SkyReceiver>(
        &mut self,
        scene: &Arc<TransportScene>,
        sources: &TransportSources,
        probes: &[P],
    ) -> Vec<IVec3> {
        self.synchronize(scene, sources);
        self.batch = self.batch.wrapping_add(1);
        let mut pending = Vec::new();
        let mut requested = 0;
        for probe in probes {
            for face in 0..6 {
                if probe.face_mask() & self.lit_faces & (1 << face) == 0 {
                    continue;
                }
                for corner in 0..4 {
                    requested += 1;
                    let position = corner_position(probe.voxel(), face, corner);
                    let entry = match self.corners.entry(position) {
                        Entry::Occupied(entry) => entry.into_mut(),
                        Entry::Vacant(entry) => {
                            self.buckets
                                .entry(position.div_euclid(IVec3::splat(BUCKET_EDGE)))
                                .or_default()
                                .push(position);
                            entry.insert(Corner::default())
                        }
                    };
                    if entry.generation != self.generation
                        && entry.pending_batch != self.batch
                        && (!self.deferred_results
                            || entry.pending_batch == 0
                            || entry.pending_generation != self.generation)
                    {
                        entry.pending_batch = self.batch;
                        entry.pending_generation = self.generation;
                        pending.push(position);
                    }
                }
            }
        }
        self.stats = SkyCacheStats {
            requested,
            traced: pending.len(),
            rays: pending.len() * self.lights.len(),
            resident: self.corners.len(),
        };
        pending
    }

    /// Forget the corners of pages whose receivers were retired. A corner
    /// shared with a remaining page is simply traced again when requested.
    pub fn retire_pages(&mut self, pages: &[IVec3]) {
        let per_page = (super::world::PAGE_EDGE as i32) / BUCKET_EDGE;
        for &page in pages {
            for z in 0..per_page {
                for y in 0..per_page {
                    for x in 0..per_page {
                        let bucket = page * per_page + IVec3::new(x, y, z);
                        if let Some(positions) = self.buckets.remove(&bucket) {
                            for position in positions {
                                self.corners.remove(&position);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Material reflectance for change flux (`store`).
    pub fn set_materials(&mut self, materials: &super::world::VoxelMaterialTransportTable) {
        let weights = Vec3::new(0.2126, 0.7152, 0.0722);
        let table = self.reflectance.get_or_insert_with(|| Box::new([1.0; 256]));
        for (material, value) in table.iter_mut().enumerate() {
            *value = materials.get(material as u8).diffuse.visible.dot(weights).clamp(0.0, 1.0);
        }
    }

    /// The most reflective solid voxel touching a corner (1 if unknown).
    fn corner_reflectance(&self, corner: IVec3) -> f32 {
        let (Some(table), Some(scene)) = (&self.reflectance, &self.scene) else {
            return 1.0;
        };
        let mut reflectance = 0.0_f32;
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    if let Some(material) = scene.material_at(corner - IVec3::new(x, y, z))
                        && material != AIR
                    {
                        let value = table.get(material as usize).copied().unwrap_or(1.0);
                        reflectance = reflectance.max(value);
                    }
                }
            }
        }
        reflectance
    }

    /// The previous generation's flipped corners and their axis neighbours,
    /// as the voxels whose faces touch them.
    fn plan_band(&mut self) {
        let mut voxels = HashSet::new();
        for corner in std::mem::take(&mut self.flipped) {
            for offset in [IVec3::ZERO, IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z] {
                let corner = corner + offset;
                if !self.corners.contains_key(&corner) {
                    continue;
                }
                for z in 0..2 {
                    for y in 0..2 {
                        for x in 0..2 {
                            voxels.insert(corner - IVec3::new(x, y, z));
                        }
                    }
                }
            }
        }
        self.band.extend(voxels);
    }

    /// Voxels to request sky for before the world sweep, nearest the focus
    /// first (pop from the end).
    pub fn take_band(&mut self, focus: IVec3) -> Vec<IVec3> {
        let mut band = std::mem::take(&mut self.band);
        band.sort_unstable_by_key(|voxel| std::cmp::Reverse((*voxel - focus).length_squared()));
        band
    }

    pub fn store(&mut self, position: IVec3, value: u32) {
        let reflectance = if self.track_changes {
            self.corner_reflectance(position)
        } else {
            1.0
        };
        let corner = self.corners.get_mut(&position).expect("planned sky corner");
        if self.track_changes && (corner.value != value || value == 0) {
            // Unknown is not black or unchanged. New/unknown corners contribute
            // conservative evidence; known equal values contribute none. Bounds
            // coalesce existing ray results, with no extra transport sampling.
            // Reflected luminance change at the corner: the sunlight change,
            // the largest over its faces at the current sun, times the most
            // reflective adjacent material (edit flux is measured the same
            // way). A first or unknown measurement belongs to new terrain,
            // whose load event already accounts for it.
            let delta = if corner.value == 0 || value == 0 {
                0.0
            } else {
                self.lights
                    .iter()
                    .enumerate()
                    .filter(|(light, _)| {
                        super::sky::lit(value, *light) != super::sky::lit(corner.value, *light)
                    })
                    .map(|(_, light)| {
                        let cosine = FACE_OFFSETS
                            .iter()
                            .map(|normal| normal.as_vec3().dot(light.direction))
                            .fold(0.0_f32, f32::max);
                        light.color.dot(Vec3::new(0.2126, 0.7152, 0.0722)) * cosine
                    })
                    .sum::<f32>()
                    * reflectance
            };
            let bucket = position.div_euclid(IVec3::splat(BUCKET_EDGE));
            self.changes
                .entry(bucket)
                .and_modify(|(bounds, flux)| {
                    bounds.min = bounds.min.min(position);
                    bounds.max = bounds.max.max(position);
                    *flux += delta;
                })
                .or_insert((super::geometry::Bounds::point(position), delta));
        }
        if corner.value != 0 && value != 0 && corner.value != value && self.flipped.len() < 1 << 20 {
            self.flipped.push(position);
        }
        corner.value = value;
        corner.pending_batch = 0;
        // Traversal-budget exhaustion is not a valid forever-cached shadow.
        corner.generation = if value == 0 {
            0
        } else {
            self.generation
        };
    }

    pub fn apply<P: SkyReceiver>(&self, probes: &mut [P]) {
        for probe in probes {
            for face in 0..6 {
                if probe.face_mask() & (1 << face) == 0 {
                    continue;
                }
                let value = if self.lit_faces & (1 << face) == 0 {
                    [super::sky::KNOWN; 4]
                } else {
                    std::array::from_fn(|corner| {
                        self.corners
                            .get(&corner_position(probe.voxel(), face, corner))
                            .map_or(0, |value| value.value)
                    })
                };
                probe.set_sky(face, value);
            }
        }
    }

    fn synchronize(&mut self, scene: &Arc<TransportScene>, sources: &TransportSources) {
        // Colors and sub-step motion only change the display's lighting.
        let lights = sky_lights(sources);
        let key = visibility_key(&lights);
        if self.generation == 0 || key != self.key {
            if self.generation != 0 {
                self.moved_at = Some(Instant::now());
                self.plan_band();
            }
            self.generation = self.generation.wrapping_add(1).max(1);
            self.key = key;
        }
        self.lit_faces = 0;
        for (face, normal) in FACE_OFFSETS.iter().enumerate() {
            if lights
                .iter()
                .any(|light| normal.as_vec3().dot(light.direction) > 0.0)
            {
                self.lit_faces |= 1 << face;
            }
        }
        self.lights = lights;
        if self
            .scene
            .as_ref()
            .is_some_and(|old| Arc::ptr_eq(old, scene))
        {
            return;
        }
        if let Some(old) = &self.scene {
            // Region identity is stable across unchanged page notifications.
            // Air-only streaming changes cannot change display visibility.
            let changed = scene.changed_regions(old);
            let mut invalidated = HashSet::new();
            let mut blockers = Vec::new();
            for &key in &changed {
                if old.regions.get(&key).is_some_and(|r| r.any_solid)
                    || scene.regions.get(&key).is_some_and(|r| r.any_solid)
                {
                    blockers.push(key);
                }
            }
            for (&bucket, positions) in &mut self.buckets {
                let minimum = (bucket * BUCKET_EDGE).as_vec3();
                let affected = blockers.iter().any(|key| {
                    let occluder_min = (*key * REGION_EDGE as i32).as_vec3();
                    self.lights.iter().any(|light| {
                        ray_intersects_box(
                            minimum,
                            light.direction,
                            occluder_min - Vec3::splat(BUCKET_EDGE as f32 + 0.01),
                            occluder_min + Vec3::splat(REGION_EDGE as f32 + 0.01),
                        )
                    })
                });
                if affected {
                    for position in positions.iter() {
                        let corner = self.corners.get_mut(position).unwrap();
                        corner.generation = 0;
                        corner.pending_batch = 0;
                        // Shadows near an edit (or under new terrain) refresh
                        // ahead of the world sweep, nearest the focus first.
                        for z in 0..2 {
                            for y in 0..2 {
                                for x in 0..2 {
                                    invalidated.insert(*position - IVec3::new(x, y, z));
                                }
                            }
                        }
                    }
                }
                // Remove retired receivers after edits/unloading. Inspect only
                // buckets touching changed geometry, including shared boundaries.
                if changed
                    .iter()
                    .any(|key| (*key - bucket).abs().max_element() <= 1)
                {
                    positions.retain(|position| {
                        let keep = has_receiver(scene, *position);
                        if !keep {
                            self.corners.remove(position);
                        }
                        keep
                    });
                }
            }
            self.buckets.retain(|_, positions| !positions.is_empty());
            self.band.extend(invalidated);
        }
        self.scene = Some(Arc::clone(scene));
    }
}

fn has_receiver(scene: &TransportScene, corner: IVec3) -> bool {
    let mut solid = false;
    let mut exterior = false;
    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                match scene.material_at(corner - IVec3::new(x, y, z)) {
                    Some(AIR) | None => exterior = true,
                    Some(_) => solid = true,
                }
            }
        }
    }
    solid && exterior
}

/// Slab intersection with the Minkowski-expanded occluder box. This tests
/// ALL potential rays from a receiving bucket, not just its center ray.
fn ray_intersects_box(origin: Vec3, direction: Vec3, minimum: Vec3, maximum: Vec3) -> bool {
    let mut near: f32 = 0.0;
    let mut far = MAX_PATH_DISTANCE + 0.01;
    for axis in 0..3 {
        if direction[axis].abs() < 1e-8 {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return false;
            }
        } else {
            let a = (minimum[axis] - origin[axis]) / direction[axis];
            let b = (maximum[axis] - origin[axis]) / direction[axis];
            near = near.max(a.min(b));
            far = far.min(a.max(b));
            if near > far {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
#[path = "sky_cache/cpu_trace.rs"]
mod cpu_trace;

#[cfg(test)]
mod tests {
    use super::cpu_trace::sample_sky_corner;
    use super::super::{
        naadf::{linear_index, volume_len},
        sky::KNOWN,
        sources::{DirectionalTransportLight, PointTransportLight},
        spectrum::EnergySpectrum,
        world::{PAGE_EDGE, PAGE_SIZE},
    };
    use super::*;

    #[test]
    fn change_evidence_distinguishes_known_black_unknown_and_equal_values() {
        let scene = floor(&[IVec3::ZERO]);
        let position = IVec3::new(4, 5, 4);
        let mut cache = SkyCache {
            track_changes: true,
            ..default()
        };
        cache.corners.insert(position, Corner::default());
        cache.store(position, KNOWN);
        assert!(
            cache.take_changes(&scene).is_some(),
            "unknown to black needs conservative evidence"
        );
        cache.store(position, KNOWN);
        assert!(
            cache.take_changes(&scene).is_none(),
            "known black is unchanged"
        );
        cache.store(position, KNOWN | 1);
        assert!(cache.take_changes(&scene).is_some(), "becoming lit counts");
        cache.store(position, KNOWN);
        assert!(
            cache.take_changes(&scene).is_some(),
            "becoming shadowed counts"
        );
        cache.store(position, 0);
        assert!(
            cache.take_changes(&scene).is_some(),
            "traversal uncertainty must not mean unchanged"
        );
    }

    #[test]
    fn moving_sun_changes_publish_a_few_times_per_second_and_edits_at_once() {
        let scene = floor(&[IVec3::ZERO]);
        let position = IVec3::new(4, 5, 4);
        let mut cache = SkyCache {
            track_changes: true,
            ..default()
        };
        cache.corners.insert(position, Corner::default());
        cache.store(position, KNOWN);
        assert!(cache.take_changes(&scene).is_some(), "a still sun publishes at once");
        cache.store(position, KNOWN | 1);
        assert!(cache.take_changes(&scene).is_some());
        // The sun just stepped: further flips merge until the interval passes.
        let now = Instant::now();
        cache.moved_at = Some(now);
        cache.store(position, KNOWN);
        assert!(cache.take_changes(&scene).is_none());
        cache.store(position, KNOWN | 1);
        cache.published_at = Some(now - SUN_CHANGE_INTERVAL);
        assert!(cache.take_changes(&scene).is_some());
        cache.store(position, KNOWN);
        cache.moved_at = Some(now - SUN_MOVING_WINDOW);
        assert!(cache.take_changes(&scene).is_some(), "the sun stopped moving");
    }

    #[test]
    fn a_sun_step_requests_the_band_around_last_steps_flips_first() {
        let scene = floor(&[IVec3::ZERO]);
        let probes = square(IVec3::new(4, 4, 4), 4);
        let mut sources = sun(Vec3::Y);
        let mut cache = SkyCache::default();
        let pending = cache.prepare(&scene, &sources, &probes);
        finish(&mut cache, &scene, pending);
        // A shadow border crossed this corner in the current generation.
        let border = IVec3::new(5, 5, 5);
        cache.store(border, KNOWN);
        sources.directional[0].direction =
            Quat::from_rotation_z(2.0_f32.to_radians()) * sources.directional[0].direction;
        cache.prepare(&scene, &sources, &probes);
        let band = cache.take_band(IVec3::new(5, 4, 5));
        assert!(band.contains(&IVec3::new(5, 4, 5)) && band.contains(&IVec3::new(4, 4, 4)));
        assert!(!band.contains(&IVec3::new(9, 4, 9)), "only the border's neighbourhood");
        assert_eq!(band.last(), Some(&IVec3::new(5, 4, 5)), "nearest the focus first");
        assert!(cache.take_band(IVec3::ZERO).is_empty());
    }

    fn sun(direction: Vec3) -> TransportSources {
        TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: direction.normalize(),
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        }
    }

    fn floor_page() -> Vec<u16> {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..PAGE_EDGE {
            for x in 0..PAGE_EDGE {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
            }
        }
        page
    }

    fn floor(pages: &[IVec3]) -> Arc<TransportScene> {
        Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            pages.iter().map(|key| (*key, floor_page())).collect(),
            &[],
        ))
    }

    fn square(origin: IVec3, edge: i32) -> Vec<SurfaceProbe> {
        (0..edge)
            .flat_map(|z| {
                (0..edge).map(move |x| SurfaceProbe::new(origin + IVec3::new(x, 0, z), 4))
            })
            .collect()
    }

    fn finish(cache: &mut SkyCache, scene: &TransportScene, pending: Vec<IVec3>) {
        for position in pending {
            cache.store(position, sample_sky_corner(scene, &cache.lights, position));
        }
    }

    #[test]
    fn hundred_square_shares_corners_across_batches_and_reuses_every_warm_sample() {
        let pages: Vec<_> = (0..4)
            .flat_map(|z| (0..4).map(move |x| IVec3::new(x, 0, z)))
            .collect();
        let scene = floor(&pages);
        let sources = sun(Vec3::Y);
        let mut probes = square(IVec3::new(0, 4, 0), 100);
        let mut cache = SkyCache::default();
        let mut total = 0;
        for batch in probes.chunks_mut(127) {
            let pending = cache.prepare(&scene, &sources, batch);
            total += pending.len();
            finish(&mut cache, &scene, pending);
            cache.apply(batch);
        }
        assert_eq!(
            total,
            101 * 101,
            "40,000 references require only 10,201 rays"
        );
        assert_eq!(cache.corners.len(), total);
        assert!(cache.prepare(&scene, &sources, &probes).is_empty());
        assert_eq!(cache.stats.requested, 40_000);
        assert_eq!(cache.stats.rays, 0);
        assert!(
            probes
                .iter()
                .all(|probe| probe.display[2].sky == [KNOWN | 1; 4])
        );
    }

    #[test]
    fn creases_share_one_visibility_word_across_receiving_normals() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(UVec3::splat(8), PAGE_SIZE)] = 2;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let sources = sun(Vec3::new(1.0, 2.0, 3.0));
        let mut probes = [SurfaceProbe::new(IVec3::splat(8), 0b01_01_01)];
        let mut cache = SkyCache::default();
        let pending = cache.prepare(&scene, &sources, &probes);
        assert_eq!(cache.stats.requested, 12);
        assert_eq!(pending.len(), 7);
        finish(&mut cache, &scene, pending);
        cache.apply(&mut probes);
        // The display projects each normal (its cosine); corners carry only
        // visibility, shared by every face that meets there.
        for face in [0, 2, 4] {
            assert_eq!(probes[0].display[face].sky, [KNOWN | 1; 4]);
        }
    }

    #[test]
    fn source_invalidation_ignores_points_environment_and_nonvisible_bands() {
        let scene = floor(&[IVec3::ZERO]);
        let probes = square(IVec3::new(4, 4, 4), 2);
        let mut sources = sun(Vec3::Y);
        let mut cache = SkyCache::default();
        let pending = cache.prepare(&scene, &sources, &probes);
        finish(&mut cache, &scene, pending);
        sources.environment = EnergySpectrum::ZERO;
        sources.points.push(PointTransportLight {
            position: Vec3::splat(10.0),
            intensity: EnergySpectrum::ONE,
            range: 30.0,
            custom_visible: Vec3::ZERO,
        });
        sources.directional[0].radiance.infrared = 10.0;
        sources.directional[0].radiance.ionizing = 2.0;
        sources.directional.push(DirectionalTransportLight {
            direction: Vec3::X,
            radiance: EnergySpectrum::new(Vec3::ONE, 50.0, 2.0),
            custom_visible: Vec3::ONE,
        });
        assert!(cache.prepare(&scene, &sources, &probes).is_empty());
        sources.directional[0].direction = Vec3::new(0.1, 1.0, 0.0).normalize();
        let pending = cache.prepare(&scene, &sources, &probes);
        assert_eq!(pending.len(), 9);
        finish(&mut cache, &scene, pending);
        // Color and sub-step motion are the display's: no retrace.
        sources.directional[0].radiance.visible *= 2.0;
        sources.directional[0].direction =
            Quat::from_rotation_z(0.1_f32.to_radians()) * sources.directional[0].direction;
        assert!(cache.prepare(&scene, &sources, &probes).is_empty());
        sources.directional.clear();
        assert!(cache.prepare(&scene, &sources, &probes).is_empty());
        let mut probes = probes;
        cache.apply(&mut probes);
        assert!(probes.iter().all(|probe| probe.display[2].sky == [KNOWN; 4]));
    }

    #[test]
    fn delayed_corners_are_deduplicated_and_cannot_revive_superseded_results() {
        let scene = floor(&[IVec3::ZERO]);
        let mut probes = square(IVec3::new(4, 4, 4), 2);
        let mut sources = sun(Vec3::Y);
        let mut cache = SkyCache {
            deferred_results: true,
            ..default()
        };
        let pending = cache.prepare(&scene, &sources, &probes);
        let first = cache.ticket();
        assert!(!pending.is_empty());
        assert!(cache.prepare(&scene, &sources, &probes).is_empty());
        let position = pending[0];
        cache.store_ticket(first, position, KNOWN | 1);
        assert_eq!(cache.corners[&position].value, KNOWN | 1);
        // A new visibility generation: the sun leaves its quantization cell.
        sources.directional[0].direction =
            Quat::from_rotation_z(2.0_f32.to_radians()) * sources.directional[0].direction;
        assert!(!cache.prepare(&scene, &sources, &probes).is_empty());
        let second = cache.ticket();
        cache.store_ticket(first, position, KNOWN | 2);
        assert_eq!(cache.corners[&position].value, KNOWN | 1);
        cache.store_ticket(second, position, KNOWN | 3);
        assert_eq!(cache.corners[&position].value, KNOWN | 3);
        let retired = Arc::new(TransportScene::default());
        cache.prepare(&retired, &sources, &[] as &[SurfaceProbe]);
        cache.store_ticket(second, position, KNOWN | 4);
        assert!(!cache.corners.contains_key(&position));
        cache.apply(&mut probes);
        assert_eq!(probes[0].display[2].sky, [0; 4]);
    }

    #[test]
    fn far_occluder_edits_refresh_only_the_shadow_sweep_and_air_streaming_keeps_cache() {
        let scene = floor(&[IVec3::ZERO, IVec3::new(3, 0, 0)]);
        let mut probes = square(IVec3::new(4, 4, 4), 2);
        probes.extend(square(IVec3::new(100, 4, 4), 2));
        let sources = sun(Vec3::Y);
        let mut cache = SkyCache::default();
        let pending = cache.prepare(&scene, &sources, &probes);
        finish(&mut cache, &scene, pending);
        let air = Arc::new(TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::new(0, 2, 0), vec![AIR; volume_len(PAGE_SIZE)])],
            &[],
        ));
        assert!(cache.prepare(&air, &sources, &probes).is_empty());
        let blocked = Arc::new(TransportScene::with_page_delta(
            &air,
            vec![(IVec3::new(0, 4, 0), vec![2; volume_len(PAGE_SIZE)])],
            &[],
        ));
        let pending = cache.prepare(&blocked, &sources, &probes);
        assert_eq!(
            pending.len(),
            9,
            "remote shadow receivers must update, unrelated terrain must not"
        );
        assert!(pending.iter().all(|position| position.x < 32));
        // The shaded receivers are requested ahead of the world sweep.
        let band = cache.take_band(IVec3::ZERO);
        assert!(band.contains(&IVec3::new(4, 4, 4)));
        assert!(band.iter().all(|voxel| voxel.x < 32), "only the shadow sweep");
        finish(&mut cache, &blocked, pending);
        cache.apply(&mut probes);
        assert!(
            probes[..4]
                .iter()
                .all(|probe| probe.display[2].sky == [KNOWN; 4])
        );
        assert!(
            probes[4..]
                .iter()
                .all(|probe| probe.display[2].sky == [KNOWN | 1; 4])
        );
        let unblocked = Arc::new(TransportScene::with_page_delta(
            &blocked,
            vec![],
            &[IVec3::new(0, 4, 0)],
        ));
        let pending = cache.prepare(&unblocked, &sources, &probes);
        assert_eq!(pending.len(), 9);
        finish(&mut cache, &unblocked, pending);
        cache.apply(&mut probes);
        assert!(
            probes
                .iter()
                .all(|probe| probe.display[2].sky == [KNOWN | 1; 4])
        );
        let distant = Arc::new(TransportScene::with_page_delta(
            &unblocked,
            vec![(IVec3::new(0, 30, 0), vec![2; volume_len(PAGE_SIZE)])],
            &[],
        ));
        assert!(
            cache.prepare(&distant, &sources, &probes).is_empty(),
            "occluders beyond the tracer's reach cannot invalidate"
        );
    }

    #[test]
    fn distant_sun_blocker_survives_skipped_scene_versions_and_expired_lineage() {
        for skipped in [3, 10] {
            let scene = floor(&[IVec3::ZERO]);
            let probes = square(IVec3::new(4, 4, 4), 2);
            let sources = sun(Vec3::Y);
            let mut cache = SkyCache::default();
            let pending = cache.prepare(&scene, &sources, &probes);
            finish(&mut cache, &scene, pending);
            let mut blocked = TransportScene::with_page_delta(
                &scene,
                vec![(IVec3::new(0, 4, 0), vec![2; volume_len(PAGE_SIZE)])],
                &[],
            );
            for _ in 0..skipped {
                blocked = TransportScene::with_page_delta(&blocked, vec![], &[]);
            }
            let blocked = Arc::new(blocked);
            assert_eq!(cache.prepare(&blocked, &sources, &probes).len(), 9);
        }
    }

    #[test]
    fn streaming_retires_corners_including_negative_shared_boundaries() {
        let mut scene = Arc::new(TransportScene::default());
        let mut cache = SkyCache::default();
        let sources = sun(Vec3::Y);
        for step in -4..4 {
            let key = IVec3::new(step, 0, -1);
            let previous = IVec3::new(step - 1, 0, -1);
            scene = Arc::new(TransportScene::with_page_delta(
                &scene,
                vec![(key, floor_page())],
                &[previous],
            ));
            let probes = square(key * PAGE_EDGE as i32 + IVec3::new(0, 4, 0), 32);
            let pending = cache.prepare(&scene, &sources, &probes);
            finish(&mut cache, &scene, pending);
            assert_eq!(
                cache.corners.len(),
                33 * 33,
                "unloading must not accumulate retired entries"
            );
        }
        scene = Arc::new(TransportScene::default());
        assert!(
            cache
                .prepare::<SurfaceProbe>(&scene, &sources, &[])
                .is_empty()
        );
        assert!(cache.corners.is_empty());
        assert!(cache.buckets.is_empty());
    }

    #[test]
    fn unknown_visibility_retries_without_discarding_known_neighbors() {
        let scene = floor(&[IVec3::ZERO]);
        let probes = square(IVec3::new(4, 4, 4), 2);
        let sources = sun(Vec3::Y);
        let mut cache = SkyCache::default();
        let pending = cache.prepare(&scene, &sources, &probes);
        let unknown = pending[0];
        finish(&mut cache, &scene, pending);
        cache.store(unknown, 0);
        assert_eq!(cache.prepare(&scene, &sources, &probes), vec![unknown]);
    }

    #[test]
    fn bucket_invalidation_conservatively_includes_every_corner_ray() {
        let box_min = Vec3::new(-17.0, 19.0, -2.0);
        let box_max = box_min + Vec3::splat(REGION_EDGE as f32);
        for direction in [
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::new(-1.0, 2.0, 0.3).normalize(),
            Vec3::new(1.0, -0.1, -2.0).normalize(),
        ] {
            for bz in -3..3 {
                for by in -3..3 {
                    for bx in -3..3 {
                        let origin = (IVec3::new(bx, by, bz) * BUCKET_EDGE).as_vec3();
                        let bucket_hit = ray_intersects_box(
                            origin,
                            direction,
                            box_min - Vec3::splat(BUCKET_EDGE as f32 + 0.01),
                            box_max + Vec3::splat(0.01),
                        );
                        for offset in [
                            Vec3::ZERO,
                            Vec3::splat(15.0),
                            Vec3::new(3.0, 14.0, 9.0),
                            Vec3::new(15.0, 0.0, 15.0),
                        ] {
                            if ray_intersects_box(
                                origin + offset + direction * 0.003,
                                direction,
                                box_min,
                                box_max,
                            ) {
                                assert!(bucket_hit, "bucket test missed a possible shadow ray");
                            }
                        }
                    }
                }
            }
        }
    }
}
