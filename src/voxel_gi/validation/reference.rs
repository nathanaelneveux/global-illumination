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

const BLOCKS: u32 = 8;
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

#[derive(Default)]
pub(super) struct ReferenceRun {
    task: Option<Task<Result<(), String>>>,
}

impl ReferenceRun {
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
        self.task = Some(AsyncComputeTaskPool::get().spawn(async move {
            let mut gpu = GpuTracer::new(device)?;
            let emitters = std::sync::Arc::new(EmissionDistribution::build(&scene, &materials));
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
                for &(voxel, face) in &sites {
                    if scene.material_at(voxel).is_none_or(|m| m == super::super::world::AIR)
                        || scene.material_at(voxel + FACE_OFFSETS[face])
                            != Some(super::super::world::AIR)
                    {
                        return Err(format!("Reference receiver is not exposed: {voxel:?}/{face}"));
                    }
                    let mut probe = SurfaceProbe::new(voxel, 1 << face);
                    probe.sequence[face] = REFERENCE_SEQUENCE_START + replicate * 1_048_576;
                    work.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                    work.probes.last_mut().unwrap().sky_requested = false;
                }
                let mut estimate = vec![[0.0_f64; 5]; sites.len()];
                // Each block stays below the production accumulator ceiling.
                // Average blocks independently; running one long history would
                // turn into an EMA and not provide this many effective paths.
                for _ in 0..BLOCKS {
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
                            *mean += f64::from(sample) / f64::from(BLOCKS);
                        }
                    }
                }
                references.push(estimate);
            }
            let data = json!({"schema_version":1,"scene_generation":revision,
                "regions":scene.regions.len(),"paths_per_replicate":PATHS_PER_BLOCK * BLOCKS,
                "sequence_start":REFERENCE_SEQUENCE_START,"replicate_stride":1_048_576,
                "transport":transport,"sites":sites.iter().map(|(v,f)|json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
                "references":references,
                "note":"Exact frozen end-of-workload scene; two disjoint high-sample sequence blocks. Computed after timing/captures. Early error is relative to the final scene, not a time-varying ground truth."});
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
