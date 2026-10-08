//! Persistent world-space corner lighting. Visibility is traced once per
//! lattice position and sun, then projected onto the six receiving normals.
//! Geometry changes invalidate shadow-swept spatial buckets, not the world.
use std::sync::Arc;

// Trusted integer lattice keys: use the engine's fast hasher on this hot path.
use bevy::platform::collections::{HashMap, hash_map::Entry};
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
        .collect()
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SkyCacheStats {
    pub requested: usize,
    pub traced: usize,
    pub rays: usize,
    pub resident: usize,
}

#[derive(Default)]
struct Corner {
    value: [u32; 6],
    generation: u64,
    pending_batch: u64,
}

#[derive(Default)]
pub(crate) struct SkyCache {
    corners: HashMap<IVec3, Corner>,
    buckets: HashMap<IVec3, Vec<IVec3>>,
    scene: Option<Arc<TransportScene>>,
    pub lights: Vec<SkyLight>,
    generation: u64,
    batch: u64,
    lit_faces: u8,
    pub stats: SkyCacheStats,
}

impl SkyCache {
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
                    if entry.generation != self.generation && entry.pending_batch != self.batch {
                        entry.pending_batch = self.batch;
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

    pub fn store(&mut self, position: IVec3, value: [u32; 6]) {
        let corner = self.corners.get_mut(&position).expect("planned sky corner");
        corner.value = value;
        // Traversal-budget exhaustion is not a valid forever-cached shadow.
        corner.generation = if value.contains(&0) {
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
                    [1; 4]
                } else {
                    std::array::from_fn(|corner| {
                        self.corners[&corner_position(probe.voxel(), face, corner)].value[face]
                    })
                };
                probe.set_sky(face, value);
            }
        }
    }

    fn synchronize(&mut self, scene: &Arc<TransportScene>, sources: &TransportSources) {
        let lights = sky_lights(sources);
        if self.generation == 0 || lights != self.lights {
            self.generation = self.generation.wrapping_add(1).max(1);
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
        }
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
            let mut changed = Vec::new();
            let mut blockers = Vec::new();
            for (&key, region) in &old.regions {
                let next = scene.regions.get(&key);
                if next.is_none_or(|next| !Arc::ptr_eq(region, next)) {
                    changed.push(key);
                    if region.any_solid || next.is_some_and(|region| region.any_solid) {
                        blockers.push(key);
                    }
                }
            }
            for (&key, region) in &scene.regions {
                if !old.regions.contains_key(&key) {
                    changed.push(key);
                    if region.any_solid {
                        blockers.push(key);
                    }
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
                        self.corners.get_mut(position).unwrap().generation = 0;
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
mod tests {
    use super::super::{
        naadf::{linear_index, volume_len},
        sky::{pack, unpack},
        sources::{DirectionalTransportLight, PointTransportLight},
        spectrum::EnergySpectrum,
        tracer::sample_sky_corner,
        world::{PAGE_EDGE, PAGE_SIZE},
    };
    use super::*;

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
                .all(|probe| probe.display[2].sky == [pack(Vec3::ONE); 4])
        );
    }

    #[test]
    fn creases_share_visibility_but_project_each_receiving_normal_separately() {
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
        for face in [0, 2, 4] {
            let expected = pack(Vec3::splat(
                FACE_OFFSETS[face]
                    .as_vec3()
                    .dot(sources.directional[0].direction),
            ));
            assert_eq!(probes[0].display[face].sky, [expected; 4]);
        }
        assert!(unpack(probes[0].display[0].sky[0]).x < unpack(probes[0].display[4].sky[0]).x);
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
        sources.directional[0].radiance.visible *= 2.0;
        assert_eq!(cache.prepare(&scene, &sources, &probes).len(), 9);
        sources.directional.clear();
        assert!(cache.prepare(&scene, &sources, &probes).is_empty());
        let mut probes = probes;
        cache.apply(&mut probes);
        assert!(probes.iter().all(|probe| probe.display[2].sky == [1; 4]));
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
        finish(&mut cache, &blocked, pending);
        cache.apply(&mut probes);
        assert!(
            probes[..4]
                .iter()
                .all(|probe| probe.display[2].sky == [1; 4])
        );
        assert!(
            probes[4..]
                .iter()
                .all(|probe| probe.display[2].sky == [pack(Vec3::ONE); 4])
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
                .all(|probe| probe.display[2].sky == [pack(Vec3::ONE); 4])
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
        cache.store(unknown, [0; 6]);
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
