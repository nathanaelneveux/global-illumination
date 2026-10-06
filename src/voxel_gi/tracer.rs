use std::sync::Arc;

use bevy::prelude::*;
use rayon::prelude::*;

use super::emission::{EmissionDistribution, face_point};
use super::sampling::pcg;
use super::sampling::{hemisphere, sample_1d, stream_id};
use super::spectrum::EnergySpectrum;
use super::work::{FaceWork, TraceWork};
#[cfg(test)]
use super::world::SurfaceProbe;
use super::world::{
    AIR, BRICK_EDGE, BRICKS_PER_AXIS, FACE_OFFSETS, REGION_EDGE, TransportScene,
    VoxelMaterialTransportTable,
};

const RAY_EPSILON: f32 = 0.001;
const MAX_TRACE_STEPS: usize = 2_048;
pub(crate) const MAX_PATH_DISTANCE: f32 = 768.0;
const MAX_BOUNCES: u32 = 3;
/// Coverage milestone, not a convergence or quality guarantee. Sampling never
/// stops here; the scheduler continually improves every loaded surface.
pub const BOOTSTRAP_SAMPLES: u32 = 8;
pub const HISTORY_LIMIT: u32 = 256;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DirectionalTransportLight {
    /// Direction from a surface toward the light.
    pub direction: Vec3,
    pub radiance: EnergySpectrum,
    pub custom_visible: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PointTransportLight {
    pub position: Vec3,
    pub intensity: EnergySpectrum,
    pub range: f32,
    pub custom_visible: Vec3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TransportSources {
    pub environment: EnergySpectrum,
    pub directional: Vec<DirectionalTransportLight>,
    pub points: Vec<PointTransportLight>,
}

impl Default for TransportSources {
    fn default() -> Self {
        Self {
            environment: EnergySpectrum::new(Vec3::new(0.16, 0.24, 0.38), 0.08, 0.0),
            directional: Vec::new(),
            points: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub voxel: IVec3,
    pub position: Vec3,
    pub normal: Vec3,
    pub material: u8,
    pub distance: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum TraceResult {
    Hit(RayHit),
    Escaped,
    Unknown,
}

#[derive(Clone, Copy, PartialEq)]
enum UnloadedPolicy {
    /// Unknown space cannot certify energy arriving at a simulation surface.
    Conservative,
    /// Match the visible scene: only loaded geometry can cast a display shadow.
    VisibleGeometry,
}

impl TransportScene {
    #[cfg(test)]
    pub fn trace_ray(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> Option<RayHit> {
        match self.trace_ray_result(origin, direction, max_distance) {
            TraceResult::Hit(hit) => Some(hit),
            TraceResult::Escaped | TraceResult::Unknown => None,
        }
    }

    fn trace_ray_result(&self, origin: Vec3, direction: Vec3, max_distance: f32) -> TraceResult {
        self.trace_ray_with_policy(
            origin,
            direction,
            max_distance,
            UnloadedPolicy::Conservative,
        )
    }

    fn trace_ray_with_policy(
        &self,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
        policy: UnloadedPolicy,
    ) -> TraceResult {
        let direction = direction.normalize_or_zero();
        if direction == Vec3::ZERO {
            return TraceResult::Escaped;
        }

        let Some((world_minimum, world_maximum)) = self.world_bounds() else {
            return TraceResult::Escaped;
        };
        let Some((entry_distance, _, entry_normal)) =
            ray_box_interval(origin, direction, world_minimum, world_maximum)
        else {
            return TraceResult::Escaped;
        };
        let mut distance = entry_distance.max(0.0);
        if distance > max_distance {
            return TraceResult::Escaped;
        }

        let mut position = origin + direction * (distance + RAY_EPSILON);
        distance += RAY_EPSILON;
        let mut last_normal = entry_normal;

        for _ in 0..MAX_TRACE_STEPS {
            if distance > max_distance {
                return TraceResult::Escaped;
            }

            let voxel = position.floor().as_ivec3();
            let region_coordinate = voxel.div_euclid(IVec3::splat(REGION_EDGE as i32));
            if !self.top_level.contains(region_coordinate) {
                return TraceResult::Escaped;
            }
            let Some(region) = self.regions.get(&region_coordinate) else {
                if policy == UnloadedPolicy::VisibleGeometry {
                    let (travel, normal) = advance_empty_box(
                        position,
                        direction,
                        region_coordinate,
                        IVec3::ZERO,
                        IVec3::ZERO,
                        REGION_EDGE as f32,
                    );
                    position += direction * travel;
                    distance += travel;
                    last_normal = normal;
                    continue;
                }
                return if self.has_loaded_region_ahead(position, direction) {
                    TraceResult::Unknown
                } else {
                    TraceResult::Escaped
                };
            };

            if !region.any_solid {
                let field = self.top_level.field(region_coordinate);
                let (travel, normal) = advance_empty_box(
                    position,
                    direction,
                    region_coordinate,
                    field.negative().as_ivec3(),
                    field.positive().as_ivec3(),
                    REGION_EDGE as f32,
                );
                position += direction * travel;
                distance += travel;
                last_normal = normal;
                continue;
            }

            let region_origin = region_coordinate * REGION_EDGE as i32;
            let region_local = (voxel - region_origin).as_uvec3();
            let brick = region_local / BRICK_EDGE;
            if !region.brick_occupied(brick) {
                let field = region.brick_aadf(brick);
                let global_brick = region_coordinate * BRICKS_PER_AXIS as i32 + brick.as_ivec3();
                let (travel, normal) = advance_empty_box(
                    position,
                    direction,
                    global_brick,
                    field.negative().as_ivec3(),
                    field.positive().as_ivec3(),
                    BRICK_EDGE as f32,
                );
                position += direction * travel;
                distance += travel;
                last_normal = normal;
                continue;
            }

            let material = region.material(region_local);
            if material != AIR {
                return TraceResult::Hit(RayHit {
                    voxel,
                    position,
                    normal: last_normal,
                    material: material as u8,
                    distance,
                });
            }

            let field = region.voxel_aadf(region_local);
            let (travel, normal) = advance_empty_box(
                position,
                direction,
                voxel,
                field.negative().as_ivec3(),
                field.positive().as_ivec3(),
                1.0,
            );
            position += direction * travel;
            distance += travel;
            last_normal = normal;
        }

        TraceResult::Unknown
    }

    /// Sparse streaming domains are rarely rectangular. A missing region is
    /// an unknown occluder only when the ray would re-enter loaded terrain; if
    /// no loaded cell remains ahead, this is the actual world frontier.
    fn has_loaded_region_ahead(&self, mut position: Vec3, direction: Vec3) -> bool {
        for _ in 0..MAX_TRACE_STEPS {
            let region = position
                .floor()
                .as_ivec3()
                .div_euclid(IVec3::splat(REGION_EDGE as i32));
            if !self.top_level.contains(region) {
                return false;
            }
            if self.regions.contains_key(&region) {
                return true;
            }

            let (travel, _) = advance_empty_box(
                position,
                direction,
                region,
                IVec3::ZERO,
                IVec3::ZERO,
                REGION_EDGE as f32,
            );
            position += direction * travel;
        }
        true
    }
}

#[cfg(test)]
pub fn trace_probe_batch(
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
    probes: Vec<SurfaceProbe>,
    samples_per_face: u32,
) -> Vec<SurfaceProbe> {
    let emitters = EmissionDistribution::build(&scene, &materials);
    trace_probe_batch_with_emitters(
        scene,
        sources,
        materials,
        probes,
        samples_per_face,
        &emitters,
        &mut super::sky_cache::SkyCache::default(),
    )
}

#[cfg(test)]
pub(crate) fn trace_probe_batch_with_emitters(
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
    mut probes: Vec<SurfaceProbe>,
    samples_per_face: u32,
    emitters: &EmissionDistribution,
    sky_cache: &mut super::sky_cache::SkyCache,
) -> Vec<SurfaceProbe> {
    let mut work = TraceWork::from_probes(&probes, samples_per_face == 0);
    trace_work_batch(
        &scene,
        &sources,
        &materials,
        &mut work,
        samples_per_face,
        emitters,
        sky_cache,
    );
    work.copy_measurements(&mut probes);
    if samples_per_face != 0 {
        for probe in &mut probes {
            probe.emission = materials.get(probe.material).emission;
        }
    }
    probes
}

pub(crate) fn trace_work_batch(
    scene: &Arc<TransportScene>,
    sources: &TransportSources,
    materials: &VoxelMaterialTransportTable,
    work: &mut TraceWork,
    samples_per_face: u32,
    emitters: &EmissionDistribution,
    sky_cache: &mut super::sky_cache::SkyCache,
) {
    trace_sky_batch(scene, sources, &mut work.probes, sky_cache);
    if samples_per_face != 0 {
        work.faces.par_iter_mut().for_each(|face| {
            trace_face(scene, sources, materials, face, samples_per_face, emitters);
        });
    }
}

pub(crate) fn trace_sky_batch<P: super::sky_cache::SkyReceiver>(
    scene: &Arc<TransportScene>,
    sources: &TransportSources,
    probes: &mut [P],
    sky_cache: &mut super::sky_cache::SkyCache,
) {
    let pending = sky_cache.prepare(scene, sources, probes);
    let sky: Vec<_> = pending
        .par_iter()
        .map(|position| {
            (
                *position,
                sample_sky_corner(scene, &sky_cache.lights, *position),
            )
        })
        .collect();
    for (position, value) in sky {
        sky_cache.store(position, value);
    }
    sky_cache.apply(probes);
}

fn trace_face(
    scene: &TransportScene,
    sources: &TransportSources,
    materials: &VoxelMaterialTransportTable,
    work: &mut FaceWork,
    samples_per_face: u32,
    emitters: &EmissionDistribution,
) {
    // The caller installs this work in the GI-owned, bounded Rayon pool. That
    // retains useful path-level parallelism without occupying Bevy's render and
    // gameplay workers.
    let face = work.face;
    let normal = FACE_OFFSETS[face].as_vec3();
    let voxel_center = work.voxel.as_vec3() + Vec3::splat(0.5);
    let surface_position = voxel_center + normal * 0.5;
    (work.direct, work.custom_direct) = direct_at_surface(scene, sources, surface_position, normal);
    let stream = stream_id(work.voxel, face as u32);
    for _ in 0..samples_per_face {
        let sequence = work.sequence;
        let direction = hemisphere(
            normal,
            sample_1d(stream, sequence, 0),
            sample_1d(stream, sequence, 1),
        );
        let origin = voxel_center + normal * 0.501;
        let sample = trace_path(
            scene, sources, materials, origin, direction, stream, sequence, normal, emitters,
        ) + sample_emitter(
            scene,
            materials,
            emitters,
            surface_position,
            normal,
            stream,
            sequence,
            0,
            true,
        );
        let sample = sample.finite_or_zero();
        let count = (work.samples + 1).min(HISTORY_LIMIT);
        let alpha = 1.0 / count as f32;
        work.radiance += (sample - work.radiance) * alpha;
        work.second_moment += (sample * sample - work.second_moment) * alpha;
        work.samples = count;
        work.sequence = work.sequence.wrapping_add(1);
    }
    work.updated_at = Some(std::time::Instant::now());
}

#[cfg(test)]
pub(crate) fn accumulate_sample(probe: &mut SurfaceProbe, face: usize, sample: EnergySpectrum) {
    let sample = sample.finite_or_zero();
    let count = (probe.samples[face] + 1).min(HISTORY_LIMIT);
    let alpha = 1.0 / count as f32;
    probe.radiance[face] += (sample - probe.radiance[face]) * alpha;
    probe.second_moment[face] += (sample * sample - probe.second_moment[face]) * alpha;
    probe.samples[face] = count;
    probe.sequence[face] = probe.sequence[face].wrapping_add(1);
}

fn trace_path(
    scene: &TransportScene,
    sources: &TransportSources,
    materials: &VoxelMaterialTransportTable,
    mut origin: Vec3,
    mut direction: Vec3,
    stream: u32,
    sequence: u32,
    mut previous_normal: Vec3,
    emitters: &EmissionDistribution,
) -> EnergySpectrum {
    let mut result = EnergySpectrum::ZERO;
    let mut throughput = EnergySpectrum::ONE;

    for bounce in 0..MAX_BOUNCES {
        let hit = match scene.trace_ray_result(origin, direction, MAX_PATH_DISTANCE) {
            TraceResult::Hit(hit) => hit,
            TraceResult::Escaped => {
                let horizon = direction.y.mul_add(0.35, 0.65).clamp(0.15, 1.0);
                result += throughput * sources.environment * horizon;
                break;
            }
            TraceResult::Unknown => break,
        };

        let material = materials.get(hit.material);
        let cosine = previous_normal.dot(direction).max(0.0) / std::f32::consts::PI;
        let emission_cosine = hit.normal.dot(-direction).max(1e-8);
        let light_pdf = if emitters.total_weight > 0.0 {
            material.emission.max_component() / emitters.total_weight * hit.distance * hit.distance
                / emission_cosine
        } else {
            0.0
        };
        let emission_weight = if light_pdf > 0.0 {
            cosine / (cosine + light_pdf)
        } else {
            1.0
        };
        result += throughput * material.emission * emission_weight;
        let direct = direct_radiance(scene, sources, hit).0;
        let emissive_direct = sample_emitter(
            scene,
            materials,
            emitters,
            hit.position,
            hit.normal,
            stream,
            sequence,
            bounce + 1,
            bounce + 1 < MAX_BOUNCES,
        );
        result += throughput * material.diffuse * (direct + emissive_direct);
        throughput = throughput * material.diffuse;

        if throughput.max_component() < 0.002 {
            break;
        }
        if bounce >= 1 {
            let survival = throughput.max_component().clamp(0.1, 0.95);
            if sample_1d(stream, sequence, 8 + bounce) > survival {
                break;
            }
            throughput *= survival.recip();
        }

        origin = hit.position + hit.normal * (RAY_EPSILON * 2.0);
        previous_normal = hit.normal;
        direction = hemisphere(
            hit.normal,
            sample_1d(stream, sequence, 2 + bounce * 2),
            sample_1d(stream, sequence, 3 + bounce * 2),
        );
    }

    result.finite_or_zero()
}

fn sample_emitter(
    scene: &TransportScene,
    materials: &VoxelMaterialTransportTable,
    emitters: &EmissionDistribution,
    position: Vec3,
    normal: Vec3,
    stream: u32,
    sequence: u32,
    bounce: u32,
    has_cosine_path: bool,
) -> EnergySpectrum {
    let stream = stream ^ pcg(bounce.wrapping_add(0xa341_316c));
    let Some(emitter) = emitters.select(sample_1d(stream, sequence, 12)) else {
        return EnergySpectrum::ZERO;
    };
    let target = face_point(
        emitter,
        sample_1d(stream, sequence, 13),
        sample_1d(stream, sequence, 14),
    );
    let delta = target - position;
    let distance_squared = delta.length_squared();
    if !(0.0001..=MAX_PATH_DISTANCE * MAX_PATH_DISTANCE).contains(&distance_squared) {
        return EnergySpectrum::ZERO;
    }
    let distance = distance_squared.sqrt();
    let direction = delta / distance;
    let receiver_cosine = normal.dot(direction).max(0.0);
    let emitter_cosine = FACE_OFFSETS[emitter.face as usize]
        .as_vec3()
        .dot(-direction)
        .max(0.0);
    if receiver_cosine <= 0.0 || emitter_cosine <= 0.0 || emitter.probability <= 0.0 {
        return EnergySpectrum::ZERO;
    }
    let light_pdf = emitter.probability * distance_squared / emitter_cosine;
    let cosine_pdf = receiver_cosine / std::f32::consts::PI;
    let weight = if has_cosine_path {
        light_pdf / (light_pdf + cosine_pdf)
    } else {
        1.0
    };
    if scene.trace_ray_result(position + normal * 0.002, direction, distance - 0.008)
        != TraceResult::Escaped
    {
        return EnergySpectrum::ZERO;
    }
    materials.get(emitter.material).emission * (cosine_pdf * weight / light_pdf)
}

fn direct_radiance(
    scene: &TransportScene,
    sources: &TransportSources,
    hit: RayHit,
) -> (EnergySpectrum, Vec3) {
    let mut radiance = EnergySpectrum::ZERO;
    let mut custom = Vec3::ZERO;
    let shadow_origin = hit.position + hit.normal * (RAY_EPSILON * 2.0);

    for light in &sources.directional {
        let direction = light.direction.normalize_or_zero();
        let cosine = hit.normal.dot(direction).max(0.0);
        if cosine > 0.0
            && scene.trace_ray_result(shadow_origin, direction, MAX_PATH_DISTANCE)
                == TraceResult::Escaped
        {
            radiance += light.radiance * cosine;
            custom += light.custom_visible * cosine;
        }
    }

    for light in &sources.points {
        let to_light = light.position - hit.position;
        let distance_squared = to_light.length_squared();
        if distance_squared <= f32::EPSILON || distance_squared > light.range * light.range {
            continue;
        }
        let distance = distance_squared.sqrt();
        let direction = to_light / distance;
        let cosine = hit.normal.dot(direction).max(0.0);
        if cosine > 0.0
            && scene.trace_ray_result(shadow_origin, direction, distance - RAY_EPSILON * 4.0)
                == TraceResult::Escaped
        {
            radiance += light.intensity * (cosine / distance_squared.max(1.0));
            custom += light.custom_visible * (cosine / distance_squared.max(1.0));
        }
    }

    (radiance, custom)
}

fn direct_at_surface(
    scene: &TransportScene,
    sources: &TransportSources,
    position: Vec3,
    normal: Vec3,
) -> (EnergySpectrum, Vec3) {
    direct_radiance(
        scene,
        sources,
        RayHit {
            voxel: position.floor().as_ivec3(),
            position,
            normal,
            material: 0,
            distance: 0.0,
        },
    )
}

#[cfg(test)]
fn sky_at_corner(
    scene: &TransportScene,
    sources: &TransportSources,
    voxel: IVec3,
    face: usize,
    corner: usize,
) -> u32 {
    sample_sky_corner(
        scene,
        &super::sky_cache::sky_lights(sources),
        super::sky::corner_position(voxel, face, corner),
    )[face]
}

pub(crate) fn sample_sky_corner(
    scene: &TransportScene,
    lights: &[super::sky_cache::SkyLight],
    position: IVec3,
) -> [u32; 6] {
    let mut value = [Vec3::ZERO; 6];
    for light in lights {
        let direction = light.direction;
        // One canonical ray origin for every receiver sharing this vertex.
        // Offset toward the light, not along a receiving face's normal. Exact
        // axis ties consistently use the positive grid cell in both backends.
        let origin = position.as_vec3() + direction * (RAY_EPSILON * 2.0);
        match scene.trace_ray_with_policy(
            origin,
            direction,
            MAX_PATH_DISTANCE,
            UnloadedPolicy::VisibleGeometry,
        ) {
            TraceResult::Escaped => {
                for (face, normal) in FACE_OFFSETS.iter().enumerate() {
                    value[face] += light.color * normal.as_vec3().dot(direction).max(0.0);
                }
            }
            TraceResult::Hit(_) => (),
            TraceResult::Unknown => return [0; 6],
        }
    }
    value.map(super::sky::pack)
}

fn advance_empty_box(
    position: Vec3,
    direction: Vec3,
    cell: IVec3,
    negative: IVec3,
    positive: IVec3,
    cell_size: f32,
) -> (f32, Vec3) {
    let minimum = (cell - negative).as_vec3() * cell_size;
    let maximum = (cell + positive + IVec3::ONE).as_vec3() * cell_size;
    let mut best = f32::INFINITY;
    let mut normal = Vec3::ZERO;

    for axis in 0..3 {
        let component = direction[axis];
        if component.abs() <= 1e-8 {
            continue;
        }
        let plane = if component > 0.0 {
            maximum[axis]
        } else {
            minimum[axis]
        };
        let distance = (plane - position[axis]) / component;
        if distance >= -RAY_EPSILON && distance < best {
            best = distance.max(0.0);
            normal = Vec3::ZERO;
            normal[axis] = if component > 0.0 { -1.0 } else { 1.0 };
        }
    }

    ((best + RAY_EPSILON).max(RAY_EPSILON), normal)
}

fn ray_box_interval(
    origin: Vec3,
    direction: Vec3,
    minimum: Vec3,
    maximum: Vec3,
) -> Option<(f32, f32, Vec3)> {
    let mut entry = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    let mut entry_normal = Vec3::ZERO;

    for axis in 0..3 {
        if direction[axis].abs() <= 1e-8 {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return None;
            }
            continue;
        }

        let inverse = direction[axis].recip();
        let first = (minimum[axis] - origin[axis]) * inverse;
        let second = (maximum[axis] - origin[axis]) * inverse;
        let near = first.min(second);
        let far = first.max(second);
        if near > entry {
            entry = near;
            entry_normal = Vec3::ZERO;
            entry_normal[axis] = if first < second { -1.0 } else { 1.0 };
        }
        exit = exit.min(far);
    }

    (exit >= entry.max(0.0)).then_some((entry, exit, entry_normal))
}

#[cfg(test)]
fn random_f32(seed: &mut u32) -> f32 {
    *seed = pcg(*seed);
    (*seed >> 9) as f32 * (1.0 / (1_u32 << 23) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::naadf::{linear_index, volume_len};
    use crate::voxel_gi::world::{PAGE_SIZE, TransportScene};

    #[test]
    fn sky_corners_preserve_wall_contact_shadow_missed_by_face_center() {
        let (scene, sources, probes) = super::super::sky::contact_shadow_fixture();
        let values = trace_probe_batch(
            scene,
            sources,
            Arc::new(VoxelMaterialTransportTable::default()),
            probes,
            1,
        );
        for probe in &values {
            assert!(
                probe.direct[2].visible.x > 0.9,
                "face center misses the small shadow"
            );
            let corners = probe.display[2].sky;
            assert_eq!(corners[0], 1, "near-left corner must be measured shadow");
            assert_eq!(corners[2], 1, "near-right corner must be measured shadow");
            assert!(super::super::sky::unpack(corners[1]).x > 0.9);
            assert!(super::super::sky::unpack(corners[3]).x > 0.9);
        }
        assert_eq!(values[0].display[2].sky[2..], values[1].display[2].sky[..2]);
    }

    #[test]
    fn sky_display_crosses_streaming_gaps_but_simulation_stays_conservative() {
        let empty = vec![AIR; volume_len(PAGE_SIZE)];
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![
                (IVec3::ZERO, empty.clone()),
                (IVec3::new(2, 0, 0), empty.clone()),
            ],
            &[],
        );
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::X,
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        assert_eq!(
            scene.trace_ray_result(Vec3::new(32.002, 8.0, 8.0), Vec3::X, 768.0),
            TraceResult::Unknown
        );
        let before = sky_at_corner(&scene, &sources, IVec3::new(31, 8, 8), 0, 0);
        assert!(super::super::sky::unpack(before).x > 0.99);
        let filled = TransportScene::with_page_delta(&scene, vec![(IVec3::X, empty)], &[]);
        assert!(
            super::super::sky::unpack(sky_at_corner(&filled, &sources, IVec3::new(31, 8, 8), 0, 0))
                .x
                > 0.99
        );
        let blocked = TransportScene::with_page_delta(
            &filled,
            vec![(IVec3::new(2, 0, 0), vec![2; volume_len(PAGE_SIZE)])],
            &[IVec3::X],
        );
        assert_eq!(
            sky_at_corner(&blocked, &sources, IVec3::new(31, 8, 8), 0, 0),
            1,
            "loaded occluders beyond a gap must still cast shadows"
        );
    }

    #[test]
    fn sealed_room_has_no_sky_leak_and_small_emitter_lights_it() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 4..21 {
            for y in 4..17 {
                for x in 4..21 {
                    if x == 4 || x == 20 || y == 4 || y == 16 || z == 4 || z == 20 {
                        page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 3;
                    }
                }
            }
        }
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        ));
        let probe = SurfaceProbe::new(IVec3::new(12, 4, 12), 4);
        let dark = trace_probe_batch(
            Arc::clone(&scene),
            TransportSources::default(),
            Arc::new(VoxelMaterialTransportTable::default()),
            vec![probe.clone()],
            64,
        );
        assert_eq!(
            dark[0].radiance[2],
            EnergySpectrum::ZERO,
            "sky must not leak through a one-voxel wall"
        );
        page[linear_index(UVec3::new(12, 12, 12), PAGE_SIZE)] = 9;
        let scene = Arc::new(TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            9,
            super::super::world::MaterialTransport {
                emission: EnergySpectrum::new(Vec3::splat(8.0), 4.0, 2.0),
                ..default()
            },
        );
        let lit = trace_probe_batch(
            scene,
            TransportSources::default(),
            Arc::new(materials),
            vec![probe],
            64,
        );
        assert!(lit[0].radiance[2].visible.x > 0.01);
        assert!(lit[0].radiance[2].infrared > 0.005);
        assert!(lit[0].radiance[2].ionizing > 0.001);
    }

    #[test]
    fn open_sky_matches_cosine_integral() {
        let probe = SurfaceProbe::new(IVec3::ZERO, 4);
        let sources = TransportSources::default();
        let expected = sources.environment * (0.65 + 0.35 * 2.0 / 3.0);
        let result = trace_probe_batch(
            Arc::new(TransportScene::default()),
            sources,
            Arc::new(VoxelMaterialTransportTable::default()),
            vec![probe],
            256,
        );
        assert!((result[0].radiance[2] - expected).visible.length() < 0.001);
    }

    #[test]
    fn naadf_tracer_hits_wall_and_reports_entry_normal() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..PAGE_SIZE.z {
            for y in 0..PAGE_SIZE.y {
                page[linear_index(UVec3::new(12, y, z), PAGE_SIZE)] = 1;
            }
        }
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );

        let hit = scene
            .trace_ray(Vec3::new(2.5, 8.5, 8.5), Vec3::X, 100.0)
            .unwrap();
        assert_eq!(hit.voxel.x, 12);
        assert_eq!(hit.normal, Vec3::NEG_X);
        assert!((hit.distance - 9.5).abs() < 0.01);
    }

    #[test]
    fn hierarchical_traversal_matches_reference_dda() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let mut state = 0x5eed_1234_u32;
        for z in 0..PAGE_SIZE.z {
            for y in 0..PAGE_SIZE.y {
                for x in 0..PAGE_SIZE.x {
                    state = pcg(state);
                    if state.is_multiple_of(19) {
                        page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 1;
                    }
                }
            }
        }
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );

        for _ in 0..600 {
            let mut random = || {
                state = pcg(state);
                random_f32(&mut state)
            };
            let origin = Vec3::new(
                random() * 44.0 - 6.0,
                random() * 44.0 - 6.0,
                random() * 44.0 - 6.0,
            );
            let direction = Vec3::new(
                random() * 2.0 - 1.0,
                random() * 2.0 - 1.0,
                random() * 2.0 - 1.0,
            )
            .normalize_or_zero();
            if direction == Vec3::ZERO {
                continue;
            }

            let accelerated = scene
                .trace_ray(origin, direction, 100.0)
                .map(|hit| hit.voxel);
            let reference = reference_dda(&scene, origin, direction, 100.0);
            assert_eq!(
                accelerated, reference,
                "origin={origin:?}, direction={direction:?}"
            );
        }
    }

    #[test]
    fn direct_non_visible_energy_is_cached_at_the_surface() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let voxel = IVec3::new(8, 8, 8);
        page[linear_index(voxel.as_uvec3(), PAGE_SIZE)] = 1;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let probe = SurfaceProbe {
            voxel,
            face_mask: 1 << 2,
            radiance: [EnergySpectrum::ZERO; 6],
            direct: [EnergySpectrum::ZERO; 6],
            samples: [0; 6],
            ..SurfaceProbe::new(voxel, 1 << 2)
        };
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::Y,
                radiance: EnergySpectrum::new(Vec3::ZERO, 3.0, 5.0),
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };

        let result = trace_probe_batch(
            scene,
            sources,
            Arc::new(VoxelMaterialTransportTable::default()),
            vec![probe],
            1,
        );
        assert_eq!(result[0].direct[2].infrared, 3.0);
        assert_eq!(result[0].direct[2].ionizing, 5.0);
    }

    #[test]
    fn direct_measurement_includes_sun_points_and_custom_emission_with_occlusion() {
        let voxel = IVec3::new(8, 8, 8);
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(voxel.as_uvec3(), PAGE_SIZE)] = 1;
        let mut scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        let sources = TransportSources {
            environment: EnergySpectrum::ZERO,
            directional: vec![DirectionalTransportLight {
                direction: Vec3::Y,
                radiance: EnergySpectrum::new(Vec3::new(3.0, 4.0, 5.0), 6.0, 7.0),
                custom_visible: Vec3::ONE,
            }],
            points: vec![PointTransportLight {
                position: Vec3::new(8.5, 10.0, 8.5),
                intensity: EnergySpectrum::new(Vec3::splat(10.0), 0.0, 0.0),
                range: 4.0,
                custom_visible: Vec3::splat(2.0),
            }],
        };
        let position = Vec3::new(8.5, 9.0, 8.5);
        let (total, custom) = direct_at_surface(&scene, &sources, position, Vec3::Y);
        assert_eq!(custom, Vec3::splat(3.0));
        assert_eq!(total.visible, Vec3::new(13.0, 14.0, 15.0));
        assert_eq!(total.infrared, 6.0);
        assert_eq!(total.ionizing, 7.0);
        // The roof blocks the sun, but the point source below it remains lit.
        page[linear_index(UVec3::new(8, 12, 8), PAGE_SIZE)] = 1;
        scene = TransportScene::with_page_delta(&scene, vec![(IVec3::ZERO, page)], &[]);
        let (total, custom) = direct_at_surface(&scene, &sources, position, Vec3::Y);
        assert_eq!(custom, Vec3::splat(2.0));
        assert_eq!(total.visible, Vec3::splat(10.0));
    }

    #[test]
    fn unloaded_holes_are_unknown_not_open_sky() {
        let mut near_page = vec![AIR; volume_len(PAGE_SIZE)];
        near_page[linear_index(UVec3::new(31, 8, 8), PAGE_SIZE)] = 1;
        let far_page = vec![AIR; volume_len(PAGE_SIZE)];
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, near_page), (IVec3::new(2, 0, 0), far_page)],
            &[],
        );

        assert_eq!(
            scene.trace_ray_result(Vec3::new(32.01, 8.5, 8.5), Vec3::X, 100.0),
            TraceResult::Unknown
        );
    }

    #[test]
    fn material_index_255_does_not_collide_with_air() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(UVec3::new(12, 8, 8), PAGE_SIZE)] = 255;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );

        let hit = scene
            .trace_ray(Vec3::new(2.5, 8.5, 8.5), Vec3::X, 100.0)
            .unwrap();
        assert_eq!(hit.material, 255);
    }

    fn reference_dda(
        scene: &TransportScene,
        origin: Vec3,
        direction: Vec3,
        max_distance: f32,
    ) -> Option<IVec3> {
        let (minimum, maximum) = scene.world_bounds()?;
        let (entry, exit, _) = ray_box_interval(origin, direction, minimum, maximum)?;
        let mut distance = entry.max(0.0) + RAY_EPSILON;
        let end = exit.min(max_distance);

        for _ in 0..MAX_TRACE_STEPS {
            if distance > end {
                return None;
            }
            let position = origin + direction * distance;
            let voxel = position.floor().as_ivec3();
            if matches!(scene.material_at(voxel), Some(material) if material != AIR) {
                return Some(voxel);
            }

            let mut travel = f32::INFINITY;
            for axis in 0..3 {
                if direction[axis].abs() <= 1e-8 {
                    continue;
                }
                let boundary = if direction[axis] > 0.0 {
                    voxel[axis] as f32 + 1.0
                } else {
                    voxel[axis] as f32
                };
                let axis_travel = (boundary - position[axis]) / direction[axis];
                if axis_travel >= 0.0 {
                    travel = travel.min(axis_travel);
                }
            }
            distance += travel + RAY_EPSILON;
        }
        None
    }
}
