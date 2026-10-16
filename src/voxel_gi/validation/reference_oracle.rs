//! Independent finite-depth cosine estimator for offline diagnosis. It never
//! selects an emitter, evaluates an emitter PDF, uses MIS, or reads GI history.
//! Traversal and source/material snapshots are shared with transport; traversal
//! has its separate scalar-DDA parity tests. This is an estimator cross-check,
//! not a second implementation of voxel storage or raster presentation.
use super::super::{
    EnergySpectrum, TransportSettings, VoxelMaterialTransportTable,
    sources::TransportSources,
    tracer::{MAX_PATH_DISTANCE, TraceResult, UnloadedPolicy},
    world::{FACE_OFFSETS, TransportScene},
};
use bevy::prelude::*;

#[derive(Clone, Debug, serde::Serialize)]
pub(in crate::voxel_gi) struct Estimate {
    pub paths: u32,
    pub mean: [f64; 5],
    pub standard_error: [f64; 5],
}

/// Finite-depth target matching the current solver: analytic sources at each
/// diffuse hit and one final emission connection, with no terminal environment
/// connection. Every connection is cosine sampled here. Independent uniform
/// random numbers replace the production Halton/Sobol dimension allocation.
pub(super) fn estimate(
    scene: &TransportScene,
    materials: &VoxelMaterialTransportTable,
    sources: &TransportSources,
    transport: TransportSettings,
    voxel: IVec3,
    face: usize,
    paths: u32,
) -> Estimate {
    estimate_seeded(scene, materials, sources, transport, voxel, face, paths, 0)
}

/// Independent random replicas for estimator audits. Seed zero preserves the
/// original offline reference stream. No production sampler code is used.
pub(in crate::voxel_gi) fn estimate_seeded(
    scene: &TransportScene,
    materials: &VoxelMaterialTransportTable,
    sources: &TransportSources,
    transport: TransportSettings,
    voxel: IVec3,
    face: usize,
    paths: u32,
    seed: u64,
) -> Estimate {
    assert!(paths > 1);
    let transport = transport.normalized();
    let normal = FACE_OFFSETS[face].as_vec3();
    let surface = voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5;
    let mut random = Random::new(voxel, face);
    random.0 ^= seed;
    let mut sum = [0.0; 5];
    let mut square = [0.0; 5];
    for _ in 0..paths {
        let mut energy = EnergySpectrum::ZERO;
        let mut throughput = EnergySpectrum::ONE;
        let mut position = surface;
        let mut n = normal;
        for bounce in 0..transport.bounces {
            let direction = random.cosine(n);
            let policy = if transport.reach < MAX_PATH_DISTANCE {
                UnloadedPolicy::BoundedTransport
            } else {
                UnloadedPolicy::Conservative
            };
            let hit = match scene.trace_ray_with_policy(
                position + n * 0.002,
                direction,
                transport.reach,
                policy,
            ) {
                TraceResult::Escaped => {
                    energy += throughput
                        * sources.environment
                        * (0.65 + 0.35 * direction.y).clamp(0.15, 1.0);
                    break;
                }
                TraceResult::Unknown => break,
                TraceResult::Hit(hit) => hit,
            };
            let material = materials.get(hit.material);
            energy += throughput * material.emission;
            throughput = throughput * material.diffuse;
            energy += throughput * direct(scene, sources, hit.position, hit.normal);
            position = hit.position;
            n = hit.normal;
            if bounce + 1 == transport.bounces {
                let direction = random.cosine(n);
                if let TraceResult::Hit(last) =
                    scene.trace_ray_result(position + n * 0.002, direction, transport.reach)
                {
                    energy += throughput * materials.get(last.material).emission;
                }
            }
        }
        for (band, value) in channels(energy).into_iter().enumerate() {
            sum[band] += f64::from(value);
            square[band] += f64::from(value).powi(2);
        }
    }
    let mean = sum.map(|s| s / f64::from(paths));
    Estimate {
        paths,
        mean,
        standard_error: std::array::from_fn(|b| {
            ((square[b] / f64::from(paths) - mean[b].powi(2)).max(0.0) / f64::from(paths - 1))
                .sqrt()
        }),
    }
}

fn direct(scene: &TransportScene, sources: &TransportSources, p: Vec3, n: Vec3) -> EnergySpectrum {
    let mut result = EnergySpectrum::ZERO;
    for light in &sources.directional {
        let d = light.direction.normalize_or_zero();
        let cosine = n.dot(d).max(0.0);
        if cosine > 0.0
            && scene.trace_ray_result(p + n * 0.002, d, MAX_PATH_DISTANCE) == TraceResult::Escaped
        {
            result += light.radiance * cosine;
        }
    }
    for light in &sources.points {
        let delta = light.position - p;
        let distance_squared = delta.length_squared();
        if distance_squared <= f32::EPSILON || distance_squared > light.range.powi(2) {
            continue;
        }
        let distance = distance_squared.sqrt();
        let d = delta / distance;
        let cosine = n.dot(d).max(0.0);
        if cosine > 0.0
            && scene.trace_ray_result(p + n * 0.002, d, distance - 0.004) == TraceResult::Escaped
        {
            result += light.intensity * (cosine / distance_squared.max(1.0));
        }
    }
    result
}

pub(super) fn channels(value: EnergySpectrum) -> [f32; 5] {
    [
        value.visible.x,
        value.visible.y,
        value.visible.z,
        value.infrared,
        value.ionizing,
    ]
}

/// SplitMix64, local to the oracle. The high 24 bits map exactly into [0,1).
/// This intentionally shares no production stream hash or sampling dimensions.
struct Random(u64);
impl Random {
    fn new(voxel: IVec3, face: usize) -> Self {
        let mut state = 0x6a09e667f3bcc909;
        for value in [voxel.x as u32, voxel.y as u32, voxel.z as u32, face as u32] {
            state = (state ^ u64::from(value)).wrapping_mul(0x9e3779b97f4a7c15);
        }
        Self(state)
    }
    fn uniform(&mut self) -> f32 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^= z >> 31;
        (z >> 40) as f32 * (1.0 / 16_777_216.0)
    }
    fn cosine(&mut self, normal: Vec3) -> Vec3 {
        let r = self.uniform().sqrt();
        let angle = std::f32::consts::TAU * self.uniform();
        let z = (1.0 - r * r).max(0.0).sqrt();
        let tangent = if normal.x.abs() < 0.5 {
            Vec3::X
        } else {
            Vec3::Y
        };
        let bitangent = normal.cross(tangent);
        tangent * (r * angle.cos()) + bitangent * (r * angle.sin()) + normal * z
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::{emission::EmissionDistribution, work::TraceWork, world::SurfaceProbe};
    use std::sync::Arc;

    #[test]
    fn independent_cosine_oracle_matches_open_sky_integral() {
        let scene = TransportScene::default();
        let sources = TransportSources::default();
        for (face, normal) in FACE_OFFSETS.into_iter().enumerate() {
            let estimate = estimate(
                &scene,
                &default(),
                &sources,
                default(),
                IVec3::ZERO,
                face,
                32768,
            );
            let expected = sources.environment * (0.65 + 0.35 * normal.y as f32 * 2.0 / 3.0);
            for b in 0..5 {
                assert!(
                    (estimate.mean[b] - f64::from(channels(expected)[b])).abs()
                        <= 5.0 * estimate.standard_error[b] + 1e-6
                );
            }
        }
    }

    #[test]
    #[ignore = "offline native cave estimator cross-check"]
    fn cave_independent_cosine_oracle_checks_production_mis() {
        let cave = crate::voxel_gi::research::cave();
        let mut gpu = crate::voxel_gi::gpu::tests::native_tracer();
        let emitters = Arc::new(EmissionDistribution::build(&cave.scene, &cave.materials));
        let mut failures = 0;
        for index in [0, 5, 16, 35, 57, 73, 88, 100] {
            let probe: SurfaceProbe = cave.receivers[index].clone();
            let face = probe.face_mask.trailing_zeros() as usize;
            let oracle = estimate(
                &cave.scene,
                &cave.materials,
                &cave.sources,
                default(),
                probe.voxel,
                face,
                262144,
            );
            let mut work = TraceWork::default();
            work.push(&probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
            work.probes[0].sky_requested = false;
            let mut mean = [0.0; 5];
            for block in 0..8 {
                let f = &mut work.faces[0];
                f.samples = 0;
                f.radiance = EnergySpectrum::ZERO;
                f.second_moment = EnergySpectrum::ZERO;
                f.sequence = 4194304 + block * 32768;
                for _ in 0..32 {
                    gpu.trace_compact(
                        cave.scene.clone(),
                        &cave.sources,
                        &cave.materials,
                        &mut work,
                        1024,
                        &emitters,
                        false,
                    )
                    .unwrap();
                }
                for (sum, value) in mean.iter_mut().zip(channels(work.faces[0].radiance)) {
                    *sum += f64::from(value) / 8.0;
                }
            }
            let rejected: Vec<_> = (0..5)
                .filter(|&b| {
                    (mean[b] - oracle.mean[b]).abs()
                        > 8.0 * oracle.standard_error[b] + 0.005 * oracle.mean[b].abs() + 0.001
                })
                .collect();
            failures += rejected.len();
            eprintln!(
                "COSINE_ORACLE site={index} voxel={:?} face={face} gpu={mean:?} cpu={:?} se={:?} rejected={rejected:?}",
                probe.voxel, oracle.mean, oracle.standard_error
            );
        }
        assert_eq!(
            failures, 0,
            "production estimator differs from independent cosine integration"
        );
    }
}
