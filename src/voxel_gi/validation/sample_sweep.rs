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
use serde::{Deserialize, Serialize};
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
        || spatial_requested()
}

/// Converge the resident surface cache on the frozen scene, then measure the
/// reference and raw prefixes against that frozen cache.
pub(super) fn cache_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_CACHE_SWEEP").is_ok_and(|v| v == "1")
}

/// Short-history batches reach the cache's feedback fixed point; long-history
/// batches then average down the cached values' own noise.
pub(super) const CACHE_CONVERGE_BATCHES: [(u32, u32); 2] = [(256, 640), (65_536, 512)];

pub(super) fn spatial_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_SPATIAL_SWEEP").is_ok_and(|v| v == "1") || evidence_requested()
}

pub(super) fn evidence_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_EVIDENCE_SWEEP").is_ok_and(|v| v == "1")
        || evidence_targets_requested()
}

fn evidence_targets_requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_EVIDENCE_TARGETS").is_ok_and(|v| !v.is_empty() && v != "0")
}

/// Matched presentation variants; all consume exactly the same raw prefixes.
pub(super) fn evidence_targets() -> Vec<u32> {
    if evidence_targets_requested() {
        parse_evidence_targets(&std::env::var("VOXEL_GI_VALIDATE_EVIDENCE_TARGETS").unwrap())
            .expect("EVIDENCE_TARGETS must be 1 or distinct comma-separated targets in [2,4096]")
    } else {
        vec![512]
    }
}

fn parse_evidence_targets(value: &str) -> Option<Vec<u32>> {
    if value == "1" {
        return Some(vec![512, 1024, 2048, 4096]);
    }
    let targets: Vec<u32> = value
        .split(',')
        .map(|v| v.trim().parse().ok())
        .collect::<Option<_>>()?;
    targets
        .iter()
        .enumerate()
        .all(|(i, t)| (2..=4096).contains(t) && !targets[..i].contains(t))
        .then_some(targets)
}

fn counts() -> &'static [u32] {
    if evidence_targets_requested() {
        &[4, 16, 64, 128, 256, 512, 1024, 1280, 1536, 2048]
    } else if evidence_requested() {
        &[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    } else if spatial_requested() {
        &[16, 64, 256, 512, 1024, 2048, 4096]
    } else {
        COUNTS
    }
}

pub(super) fn path_count(faces: usize) -> u64 {
    faces as u64 * u64::from(*counts().last().unwrap()) * STARTS.len() as u64
}

#[derive(Serialize, Deserialize)]
pub(super) struct Prefix {
    pub paths: u32,
    pub means: Means,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_moments: Option<Vec<[f32; 3]>>,
    /// Sum of native compute-pass timestamps through this whole-scene prefix.
    /// None when timestamps are unavailable; never substitute readback latency.
    pub gpu_ms: Option<f64>,
}

#[derive(Serialize, Deserialize)]
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
                second_moments: spatial_requested().then(|| vec![[0.0; 3]; sites.len()]),
                gpu_ms: Some(0.0),
            })
            .collect();
        // Spatial variants need representative neighbourhoods in one batch.
        let chunk_size = if spatial_requested() {
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
                        if spatial_requested() {
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
                    // Samples reaching unloaded space are excluded, not black.
                    if face.samples > prefix.paths
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

/// Presentation-only target-evidence sweep. Same sites, paths and ordinary
/// material renderer as the control; no additional transport rays. The support
/// report names its heuristic explicitly rather than calling it convergence.
pub(super) fn evidence_values(
    sites: &[(IVec3, usize)],
    prefix: &Prefix,
    target: u32,
    max_donors: u32,
) -> (Means, serde_json::Value) {
    use super::super::spatial::{Sample, evidence};
    let indices: bevy::platform::collections::HashMap<_, _> = sites
        .iter()
        .copied()
        .enumerate()
        .map(|(i, k)| (k, i))
        .collect();
    let samples: Vec<_> = prefix
        .means
        .iter()
        .zip(prefix.second_moments.as_ref().unwrap())
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
    let mut gather = evidence::Gather::default();
    let mut borrowed = 0;
    let mut candidates = 0;
    let mut reached = 0;
    let mut effective = 0.0f64;
    let mut radius2 = 0;
    let mut values = prefix.means.clone();
    for (i, &(voxel, face)) in sites.iter().enumerate() {
        gather.collect(
            voxel,
            face,
            samples[i],
            evidence::candidate_limit(target, prefix.paths, max_donors),
            |p, f| indices.get(&(p, f)).map(|&j| samples[j]),
        );
        let result = evidence::resolve(&gather.sites, target, 1.0);
        values[i][..3].copy_from_slice(&result.rgb.to_array().map(f64::from));
        borrowed += result.borrowed;
        candidates += gather.sites.len() - 1;
        reached += usize::from(result.effective >= target as f32 - 0.1);
        effective += f64::from(result.effective);
        radius2 = radius2.max(
            gather
                .sites
                .iter()
                .map(|(p, _)| p.length_squared())
                .max()
                .unwrap_or(0),
        );
    }
    (
        values,
        serde_json::json!({"paths":prefix.paths,"target":target,"max_donors":max_donors,"faces":sites.len(),
        "mean_effective_support":effective/sites.len() as f64,"reached_fraction":reached as f64/sites.len() as f64,
        "mean_borrowed_faces":borrowed as f64/sites.len() as f64,"mean_candidates":candidates as f64/sites.len() as f64,
        "max_candidate_radius":(radius2 as f64).sqrt(),
        "note":"Weighted support heuristic, not independent samples or certified accuracy."}),
    )
}

#[cfg(test)]
mod footprint_replay;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_evidence_targets_preserve_legacy_and_reject_invalid_lists() {
        assert_eq!(
            parse_evidence_targets("1"),
            Some(vec![512, 1024, 2048, 4096])
        );
        assert_eq!(parse_evidence_targets("1536, 2048"), Some(vec![1536, 2048]));
        for value in ["", "0", "4097", "1536,1536", "1536,", "nan"] {
            assert_eq!(parse_evidence_targets(value), None, "{value}");
        }
    }

    #[test]
    fn raw_prefix_work_does_not_inherit_gameplay_presentation() {
        // Frozen prefixes use TraceWork directly, not the live dispatch path
        // that copies gameplay settings. Keep the physical control unfiltered.
        let work = TraceWork::default();
        assert_eq!(work.spatial_reconstruction, 0.0);
        assert_eq!(work.spatial_evidence_target, 0);
        assert_eq!(work.response_seconds, 0.0);
    }

    #[test]
    fn prefix_windows_are_disjoint_below_history_ceiling_and_sampler_period() {
        for counts in [COUNTS] {
            assert!(counts.windows(2).all(|v| v[0] < v[1]));
            assert!(*counts.last().unwrap() < super::super::super::tracer::HISTORY_LIMIT);
        }
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
