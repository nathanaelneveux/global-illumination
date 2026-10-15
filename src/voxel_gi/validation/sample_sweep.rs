//! Frozen-scene prefixes of the production estimator. No adaptive scheduling,
//! history resets or temporal/spatial filters beyond ordinary corner display.
//! Disjoint sequence windows expose sensitivity to where accumulation starts;
//! these are not independent random scrambles or statistical confidence bounds.
use super::super::{
    emission::EmissionDistribution,
    gpu::GpuTracer,
    settings::TransportSettings,
    sources::TransportSources,
    work::TraceWork,
    world::{SurfaceProbe, TransportScene, VoxelMaterialTransportTable},
};
use bevy::prelude::*;
use serde::Serialize;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

pub(super) const COUNTS: &[u32] = &[
    16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
    24576, 32768,
];
// Include the real startup prefix and unaligned windows. All are disjoint from
// one another and the reference's 4,194,304 / 6,291,456 starts, below 2^24.
const STARTS: [u32; 4] = [0, 574_117, 1_853_903, 3_102_719];
pub(super) type Means = Vec<[f64; 5]>;

pub(super) fn requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_SAMPLE_SWEEP").is_ok_and(|v| v == "1") || spatial_requested()
}

pub(super) fn spatial_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_SPATIAL_SWEEP").is_ok_and(|v| v == "1")
}

pub(super) fn reuse_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_REUSE_SWEEP").is_ok_and(|v| v == "1")
}

fn counts(control: bool) -> &'static [u32] {
    if spatial_requested() {
        &[16, 64, 256, 512, 1024, 2048, 4096]
    } else if reuse_requested() && control {
        &[1024, 1536, 2048]
    } else if reuse_requested() {
        &[64, 256, 512, 768, 1024, 1280, 1536, 2048, 4096]
    } else {
        COUNTS
    }
}

pub(super) fn path_count(faces: usize) -> u64 {
    let paths = *counts(false).last().unwrap()
        + if reuse_requested() {
            *counts(true).last().unwrap()
        } else {
            0
        };
    faces as u64 * u64::from(paths) * STARTS.len() as u64
}

#[derive(Serialize)]
pub(super) struct Prefix {
    pub paths: u32,
    pub means: Means,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_moments: Option<Vec<[f32; 3]>>,
    /// Sum of native compute-pass timestamps through this whole-scene prefix.
    /// None when timestamps are unavailable; never substitute readback latency.
    pub gpu_ms: Option<f64>,
}

#[derive(Serialize)]
pub(super) struct Window {
    pub sequence_start: u32,
    pub prefixes: Vec<Prefix>,
}

/// Keep complete face values, not only a few selected receivers. The caller
/// bakes its disjoint reference on this same immutable scene, then renders each
/// prefix through the regular material/texture/tonemapping path.
#[expect(
    clippy::too_many_arguments,
    reason = "explicit immutable diagnostic inputs"
)]
pub(super) fn run(
    gpu: &mut GpuTracer,
    scene: &Arc<TransportScene>,
    sources: &TransportSources,
    materials: &VoxelMaterialTransportTable,
    transport: TransportSettings,
    emitters: &Arc<EmissionDistribution>,
    sites: &[(IVec3, usize)],
    progress: &AtomicU64,
    control: bool,
) -> Result<Vec<Window>, String> {
    let mut windows = Vec::new();
    for sequence_start in STARTS {
        let mut prefixes: Vec<_> = counts(control)
            .iter()
            .map(|&paths| Prefix {
                paths,
                means: vec![[0.0; 5]; sites.len()],
                second_moments: spatial_requested().then(|| vec![[0.0; 3]; sites.len()]),
                gpu_ms: Some(0.0),
            })
            .collect();
        // Reuse needs a representative neighborhood, not a 512-face reference
        // serialization tile. Both samplers get identical 2048-face batches.
        let chunk_size = if reuse_requested() || spatial_requested() {
            2048
        } else {
            512
        };
        for (chunk_index, chunk) in sites.chunks(chunk_size).enumerate() {
            let mut work = TraceWork {
                transport,
                ..default()
            };
            for &(voxel, face) in chunk {
                let mut probe = SurfaceProbe::new(voxel, 1 << face);
                probe.sequence[face] = sequence_start;
                work.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                work.probes.last_mut().unwrap().sky_requested = false;
                work.probes.last_mut().unwrap().revision = u64::from(sequence_start) + 1;
            }
            let mut completed = 0;
            let mut gpu_ms = Some(0.0_f64);
            for prefix in &mut prefixes {
                while completed < prefix.paths {
                    let paths = (prefix.paths - completed).min(
                        if reuse_requested() || spatial_requested() {
                            16
                        } else {
                            1024
                        },
                    );
                    gpu.trace_compact(
                        scene.clone(),
                        sources,
                        materials,
                        &mut work,
                        paths,
                        emitters,
                        false,
                    )?;
                    completed += paths;
                    gpu_ms = gpu_ms
                        .zip(gpu.last_gpu_ms)
                        .map(|(sum, ms)| sum + f64::from(ms));
                    progress.fetch_add(chunk.len() as u64 * u64::from(paths), Ordering::Relaxed);
                }
                for (i, face) in work.faces.iter().enumerate() {
                    if face.samples != prefix.paths
                        || face.sequence != sequence_start + prefix.paths
                    {
                        return Err("Sweep lost its cumulative sample count or sequence".into());
                    }
                    prefix.means[chunk_index * chunk_size + i] =
                        super::reference_oracle::channels(face.radiance).map(f64::from);
                    if let Some(moments) = &mut prefix.second_moments {
                        moments[chunk_index * chunk_size + i] =
                            face.second_moment.visible.to_array();
                    }
                }
                prefix.gpu_ms = prefix.gpu_ms.zip(gpu_ms).map(|(sum, ms)| sum + ms);
            }
        }
        windows.push(Window {
            sequence_start,
            prefixes,
        });
    }
    Ok(windows)
}

/// Identical nonrecursive filter to the native resident pass, applied to frozen
/// raw prefixes so temporal response and scheduling do not confound the image
/// comparison. This oracle is checked against native GPU presentation in tests.
pub(super) fn spatial_values(sites: &[(IVec3, usize)], prefix: &Prefix) -> Means {
    use super::super::spatial::{Sample, neighbors, reconstruct};
    let indices: bevy::platform::collections::HashMap<_, _> = sites
        .iter()
        .copied()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    let moments = prefix
        .second_moments
        .as_ref()
        .expect("spatial sweep moments");
    let samples: Vec<_> = prefix
        .means
        .iter()
        .zip(moments)
        .map(|(m, s)| {
            let mean = Vec3::new(m[0] as f32, m[1] as f32, m[2] as f32);
            Sample {
                mean,
                second: Vec3::from_array(*s),
                count: prefix.paths,
                display: mean,
            }
        })
        .collect();
    sites
        .iter()
        .enumerate()
        .map(|(i, &(voxel, face))| {
            let rgb = reconstruct(
                samples[i],
                neighbors(voxel, face, |p, f| {
                    indices.get(&(p, f)).map(|&i| samples[i])
                }),
                1.0,
            );
            let mut result = prefix.means[i];
            result[..3].copy_from_slice(&rgb.to_array().map(f64::from));
            result
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_windows_are_disjoint_below_history_ceiling_and_sampler_period() {
        assert!(COUNTS.windows(2).all(|v| v[0] < v[1]));
        let max = *COUNTS.last().unwrap();
        assert!(max < super::super::super::tracer::HISTORY_LIMIT);
        let starts = [STARTS.as_slice(), &[4_194_304, 6_291_456]].concat();
        for (i, &start) in starts.iter().enumerate() {
            let length = if i < STARTS.len() { max } else { 262_144 };
            assert!(start + length < 1 << 24);
            for &next in &starts[i + 1..] {
                assert!(start + length <= next);
            }
        }
        assert_eq!(path_count(15163), 1_987_444_736);
    }
}
