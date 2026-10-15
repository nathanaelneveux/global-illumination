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
    std::env::var("VOXEL_GI_VALIDATE_SAMPLE_SWEEP").is_ok_and(|v| v == "1")
}

pub(super) fn reuse_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_REUSE_SWEEP").is_ok_and(|v| v == "1")
}

fn counts() -> &'static [u32] {
    if reuse_requested() {
        &[64, 256, 512, 768, 1024, 1280, 1536, 2048, 4096]
    } else {
        COUNTS
    }
}

pub(super) fn path_count(faces: usize) -> u64 {
    faces as u64 * u64::from(*counts().last().unwrap()) * STARTS.len() as u64
}

#[derive(Serialize)]
pub(super) struct Prefix {
    pub paths: u32,
    pub means: Means,
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
) -> Result<Vec<Window>, String> {
    let mut windows = Vec::new();
    for sequence_start in STARTS {
        let mut prefixes: Vec<_> = counts()
            .iter()
            .map(|&paths| Prefix {
                paths,
                means: vec![[0.0; 5]; sites.len()],
            })
            .collect();
        for (chunk_index, chunk) in sites.chunks(512).enumerate() {
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
            for prefix in &mut prefixes {
                while completed < prefix.paths {
                    let paths =
                        (prefix.paths - completed).min(if reuse_requested() { 16 } else { 1024 });
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
                    progress.fetch_add(chunk.len() as u64 * u64::from(paths), Ordering::Relaxed);
                }
                for (i, face) in work.faces.iter().enumerate() {
                    if face.samples != prefix.paths
                        || face.sequence != sequence_start + prefix.paths
                    {
                        return Err("Sweep lost its cumulative sample count or sequence".into());
                    }
                    prefix.means[chunk_index * 512 + i] =
                        super::reference_oracle::channels(face.radiance).map(f64::from);
                }
            }
        }
        windows.push(Window {
            sequence_start,
            prefixes,
        });
    }
    Ok(windows)
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
