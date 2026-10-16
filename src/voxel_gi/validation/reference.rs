//! Offline accuracy oracle for the exact end-of-workload scene. This runs only
//! after the timed report is saved, on an exiting headless validation run.
//! Two disjoint sequence blocks expose reference uncertainty; neither is truth.
use super::super::{
    emission::EmissionDistribution,
    gpu::{GpuTracer, GpuTransportDevice},
    work::TraceWork,
    world::SurfaceProbe,
};
use super::*;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};

// Dense corner neighborhoods reduce reconstruction noise spatially. Two
// disjoint 65k-path replicas also expose the remaining oracle uncertainty.
const BLOCKS: u32 = 2;
const PATHS_PER_BLOCK: u32 = 32_768;
// Path dimensions wrap at 2^24. Stay below that period and away from the live
// sequence prefix; 2^24 itself would alias its primary/bounce directions.
const REFERENCE_SEQUENCE_START: u32 = 4_194_304;

fn channels(energy: EnergySpectrum) -> [f32; 5] {
    [
        energy.visible.x,
        energy.visible.y,
        energy.visible.z,
        energy.infrared,
        energy.ionizing,
    ]
}

/// Converged, equal-confidence counterpart of the renderer's connected
/// coplanar reconstruction. Face-center references alone cannot measure the
/// preferred corner appearance; a blurred target would conceal gradient bias.
pub(super) struct ReferencePatch {
    pub sites: Vec<(IVec3, usize)>,
    pub corners: Vec<[Vec<usize>; 4]>,
}

impl ReferencePatch {
    pub fn new(centers: &[(IVec3, usize)], exposed: impl Fn(IVec3, usize) -> bool) -> Self {
        let mut sites = centers.to_vec();
        let mut indices: std::collections::HashMap<_, _> = centers
            .iter()
            .copied()
            .enumerate()
            .map(|(i, p)| (p, i))
            .collect();
        let mut corners = Vec::new();
        for &(voxel, face) in centers {
            let (u, v) = match face / 2 {
                0 => (IVec3::Y, IVec3::Z),
                1 => (IVec3::X, IVec3::Z),
                _ => (IVec3::X, IVec3::Y),
            };
            let mut neighbor = |offset| {
                let p = voxel + offset;
                exposed(p, face).then(|| {
                    *indices.entry((p, face)).or_insert_with(|| {
                        sites.push((p, face));
                        sites.len() - 1
                    })
                })
            };
            let patch = [(-1, -1), (1, -1), (-1, 1), (1, 1)].map(|(x, y)| {
                let mut members = vec![centers.iter().position(|p| *p == (voxel, face)).unwrap()];
                let a = neighbor(u * x);
                let b = neighbor(v * y);
                members.extend(a);
                members.extend(b);
                if a.is_some() || b.is_some() {
                    members.extend(neighbor(u * x + v * y));
                }
                members
            });
            corners.push(patch);
        }
        Self { sites, corners }
    }

    pub fn reconstruct(&self, values: &[[f64; 5]]) -> Vec<[[f64; 5]; 4]> {
        self.corners
            .iter()
            .map(|patch| {
                patch.each_ref().map(|members| {
                    std::array::from_fn(|band| {
                        members.iter().map(|&i| values[i][band]).sum::<f64>() / members.len() as f64
                    })
                })
            })
            .collect()
    }
}

#[derive(Default)]
pub(super) struct ReferenceRun {
    task: Option<Task<Result<(), String>>>,
}

impl ReferenceRun {
    pub fn start_retention(
        &mut self,
        snapshots: super::retention_audit::SceneSnapshots,
        device: Option<&GpuTransportDevice>,
        settings: &VoxelGiSettings,
        output: PathBuf,
    ) -> Result<(), String> {
        self.task = Some(super::retention_audit::start_reference(
            snapshots, device, settings, output,
        )?);
        Ok(())
    }

    pub fn requested() -> bool {
        std::env::var_os("VOXEL_GI_VALIDATE_REFERENCE").is_some()
    }

    pub fn start(
        &mut self,
        runtime: &super::super::VoxelGiRuntime,
        device: Option<&GpuTransportDevice>,
        settings: &VoxelGiSettings,
        profile: Profile,
        output: PathBuf,
    ) -> Result<(), String> {
        if !crate::headless_validation() || std::env::var_os("VOXEL_GI_VALIDATE_EXIT").is_none() {
            return Err("Offline reference requires headless validation with EXIT enabled".into());
        }
        let device = device
            .ok_or("Offline reference requires the native GPU device")?
            .clone();
        let scene = runtime.scene.clone();
        let materials = runtime.materials.clone();
        let sources = runtime.sources.clone();
        let transport = settings.transport;
        let revision = runtime.generation;
        let sites = super::temporal::sites(profile);
        let blocks = match std::env::var("VOXEL_GI_VALIDATE_REFERENCE_BLOCKS") {
            Ok(value) => value.parse::<u32>().ok().filter(|v| (2..=128).contains(v))
                .ok_or("REFERENCE_BLOCKS must be an integer in 2..=128")?,
            Err(std::env::VarError::NotPresent) => BLOCKS,
            Err(error) => return Err(error.to_string()),
        };
        let stride = (blocks * PATHS_PER_BLOCK).max(1_048_576);
        // Both replicas must remain disjoint and below the direction period.
        assert!(REFERENCE_SEQUENCE_START + stride + blocks * PATHS_PER_BLOCK < 1 << 24);
        let cpu_sites: Vec<usize> = match std::env::var("VOXEL_GI_VALIDATE_REFERENCE_CPU_SITES") {
            Ok(value) => value.split(',').map(|value| value.parse::<usize>()
                .ok().filter(|&index| index < sites.len())
                .ok_or("REFERENCE_CPU_SITES must list valid receiver indices"))
                .collect::<Result<_, _>>()?,
            Err(std::env::VarError::NotPresent) => Vec::new(),
            Err(error) => return Err(error.to_string()),
        };
        if cpu_sites.len() > 8 { return Err("REFERENCE_CPU_SITES is limited to eight receivers".into()); }
        self.task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let mut gpu = GpuTracer::new(device)?;
            let emitters = std::sync::Arc::new(EmissionDistribution::build(&scene, &materials));
            let patch = ReferencePatch::new(&sites, |voxel, face| {
                scene.material_at(voxel).is_some_and(|m| m != super::super::world::AIR)
                    && scene.material_at(voxel + FACE_OFFSETS[face]) == Some(super::super::world::AIR)
            });
            if let Ok(site) = std::env::var("VOXEL_GI_VALIDATE_PATH_DIAGNOSTIC") {
                let site: usize = site.parse().map_err(|_| "PATH_DIAGNOSTIC must name a receiver index")?;
                let &(voxel, face) = sites.get(site).ok_or("PATH_DIAGNOSTIC receiver index out of range")?;
                let mut work = TraceWork { transport, ..default() };
                let mut cpu = TraceWork { transport, ..default() };
                for sequence in 0..2048 {
                    let mut probe = SurfaceProbe::new(voxel, 1 << face);
                    probe.sequence[face] = sequence;
                    work.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                    work.probes.last_mut().unwrap().sky_requested = false;
                    cpu.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                    cpu.probes.last_mut().unwrap().sky_requested = false;
                }
                super::super::tracer::trace_work_batch(&scene, &sources, &materials, &mut cpu, 1, &emitters, &mut default());
                gpu.trace_compact(scene.clone(), &sources, &materials, &mut work, 1, &emitters, false)?;
                let paths: Vec<_> = cpu.faces.iter().zip(&work.faces).enumerate().map(|(sequence,(a,b))| {
                    json!({"sequence":sequence,"cpu":channels(a.radiance),"gpu":channels(b.radiance)})
                }).collect();
                let data = json!({"voxel":voxel.to_array(),"face":face,"paths":paths,
                    "note":"Individual fresh paths on the frozen final scene, all five bands. CPU/GPU comparison, not independent accuracy evidence."});
                std::fs::write(output.join("reference-paths.json"), serde_json::to_vec(&data).unwrap()).map_err(|e|e.to_string())?;
            }
            let mut references = Vec::new();
            for replicate in 0..2 {
                let mut work = TraceWork {
                    transport,
                    ..default()
                };
                for &(voxel, face) in &patch.sites {
                    if scene.material_at(voxel).is_none_or(|m| m == super::super::world::AIR)
                        || scene.material_at(voxel + FACE_OFFSETS[face])
                            != Some(super::super::world::AIR)
                    {
                        return Err(format!("Reference receiver is not exposed: {voxel:?}/{face}"));
                    }
                    let mut probe = SurfaceProbe::new(voxel, 1 << face);
                    probe.sequence[face] = REFERENCE_SEQUENCE_START + replicate * stride;
                    work.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                    work.probes.last_mut().unwrap().sky_requested = false;
                }
                let mut estimate = vec![[0.0_f64; 5]; patch.sites.len()];
                // Each block stays below the production accumulator ceiling.
                // Average blocks independently; running one long history would
                // turn into an EMA and not provide this many effective paths.
                for _ in 0..blocks {
                    for face in &mut work.faces {
                        face.samples = 0;
                        face.radiance = EnergySpectrum::ZERO;
                        face.second_moment = EnergySpectrum::ZERO;
                    }
                    for _ in 0..PATHS_PER_BLOCK / 1024 {
                        gpu.trace_compact(
                            scene.clone(), &sources, &materials, &mut work,
                            1024, &emitters, false,
                        )?;
                    }
                    for (mean, face) in estimate.iter_mut().zip(&work.faces) {
                        for (mean, sample) in mean.iter_mut().zip(channels(face.radiance)) {
                            *mean += f64::from(sample) / f64::from(blocks);
                        }
                    }
                }
                references.push(estimate);
            }
            let corners: Vec<_> = references.iter().map(|r| patch.reconstruct(r)).collect();
            let faces: Vec<_> = references.iter().map(|r| &r[..sites.len()]).collect();
            let cpu: Vec<_> = cpu_sites.into_iter().map(|index| {
                let (voxel, face) = sites[index];
                let estimate = super::reference_oracle::estimate(&scene, &materials, &sources,
                    transport, voxel, face, 1_048_576);
                json!({"site":index,"voxel":voxel.to_array(),"face":face,"estimate":estimate})
            }).collect();
            let data = json!({"schema_version":2,"scene_generation":revision,
                "regions":scene.regions.len(),"paths_per_replicate":PATHS_PER_BLOCK * blocks,
                "sequence_start":REFERENCE_SEQUENCE_START,"replicate_stride":stride,
                "transport":transport,"sites":sites.iter().map(|(v,f)|json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
                "references":faces,"corner_references":corners,
                "independent_cosine_checks":cpu,
                "patch_sites":patch.sites.iter().map(|(v,f)|json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
                "corner_members":patch.corners,"patch_references":references,
                "note":"Exact frozen end-of-workload scene; two disjoint high-sample sequence blocks. Corners use the renderer's connected coplanar reconstruction at converged equal confidence. Computed after timing/captures. Early error is relative to the final scene, not time-varying ground truth or tone-mapped pixel error."});
            std::fs::write(output.join("reference.json"), serde_json::to_vec(&data).unwrap())
                .map_err(|error| error.to_string())
        }));
        Ok(())
    }

    pub fn poll(&mut self, commands: &mut Commands, evidence: &mut CaptureEvidence) {
        let Some(task) = self.task.as_mut() else {
            return;
        };
        let Some(result) = block_on(poll_once(task)) else {
            return;
        };
        self.task = None;
        match result {
            Ok(()) => {
                commands.write_message(AppExit::Success);
            }
            Err(error) => {
                evidence.fail(&format!("Offline reference failed: {error}"));
                commands.write_message(AppExit::error());
            }
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_patch_matches_shared_corners_and_excludes_disconnected_diagonals() {
        let centers = [
            (IVec3::ZERO, 2),
            (IVec3::X, 2),
            (IVec3::Z, 2),
            (IVec3::X + IVec3::Z, 2),
        ];
        let patch = ReferencePatch::new(&centers, |p, f| centers.contains(&(p, f)));
        let values = [[1.0; 5], [3.0; 5], [5.0; 5], [7.0; 5]];
        let corners = patch.reconstruct(&values);
        assert_eq!(patch.sites.len(), 4);
        for (site, corner) in [(0, 3), (1, 2), (2, 1), (3, 0)] {
            assert_eq!(corners[site][corner], [4.0; 5]);
        }
        let diagonal = ReferencePatch::new(&centers[..1], |p, _| {
            p == IVec3::ZERO || p == IVec3::X + IVec3::Z
        });
        assert_eq!(diagonal.sites.len(), 1);
        assert_eq!(diagonal.reconstruct(&values[..1]), vec![[[1.0; 5]; 4]]);
    }
}
