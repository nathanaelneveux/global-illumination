//! Paired transport-model images on the actual frozen, audited cave snapshot.
//! This reuses the reference renderer without rebaking an unchanged high-sample
//! oracle. Completion establishes a valid comparison, not visual equivalence.
use super::*;
use crate::voxel_gi::TransportSettings;

const STARTS: [u32; 2] = [574_117, 1_853_903];

pub(super) fn requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_MODEL_COMPARISON").is_ok_and(|v| v == "1")
}

pub(super) fn start(
    runtime: &crate::voxel_gi::VoxelGiRuntime,
    device: GpuTransportDevice,
    output: PathBuf,
    origin: IVec3,
    sites: Vec<(IVec3, usize)>,
    base: ProbeMap,
) -> Result<ReferenceImage, String> {
    let paths: u32 = std::env::var("VOXEL_GI_MODEL_PATHS")
        .unwrap_or_else(|_| "8192".into())
        .parse()
        .map_err(|_| "MODEL_PATHS must be an integer")?;
    if !(1024..=BLOCK_PATHS).contains(&paths) || !paths.is_multiple_of(DISPATCH_PATHS) {
        return Err("MODEL_PATHS must be a multiple of 1024 in [1024,32768]".into());
    }
    let models = [
        TransportSettings::default(),
        TransportSettings {
            reach: 64.,
            bounces: 3,
        },
        TransportSettings {
            reach: 96.,
            bounces: 3,
        },
        TransportSettings {
            reach: 128.,
            bounces: 3,
        },
        TransportSettings {
            reach: 256.,
            bounces: 3,
        },
        TransportSettings {
            reach: 384.,
            bounces: 3,
        },
        TransportSettings {
            reach: 768.,
            bounces: 2,
        },
    ];
    let progress = Arc::new(AtomicU64::new(0));
    let total_paths =
        sites.len() as u64 * u64::from(paths) * STARTS.len() as u64 * models.len() as u64;
    let worker_progress = progress.clone();
    let worker_output = output.clone();
    let scene = runtime.scene.clone();
    let sources = runtime.sources.clone();
    let materials = runtime.materials.clone();
    let generation = runtime.generation;
    let task = AsyncComputeTaskPool::get().spawn(async move {
        let mut gpu = GpuTracer::new(device)?;
        let emitters = Arc::new(EmissionDistribution::build(&scene,&materials));
        let mut frames = VecDeque::new();
        let mut rows = Vec::new();
        let mut baseline: Vec<Means> = Vec::new();
        for (index, model) in models.into_iter().enumerate() {
            let mut windows = Vec::new();
            for sequence in STARTS {
                let mut means = vec![[0.;5];sites.len()];
                for (chunk_index, chunk) in sites.chunks(CHUNK_FACES).enumerate() {
                    let mut work = TraceWork { transport:model, ..default() };
                    for &(voxel, face) in chunk {
                        let mut p = SurfaceProbe::new(voxel,1<<face);
                        p.sequence[face]=sequence;
                        work.push(&p,1<<face,0,0,false,u32::MAX,u32::MAX);
                        work.probes.last_mut().unwrap().sky_requested=false;
                    }
                    for _ in 0..paths/DISPATCH_PATHS {
                        gpu.trace_compact(scene.clone(),&sources,&materials,&mut work,
                            DISPATCH_PATHS,&emitters,false)?;
                        worker_progress.fetch_add(chunk.len() as u64*u64::from(DISPATCH_PATHS),
                            Ordering::Relaxed);
                    }
                    for (i, face) in work.faces.iter().enumerate() {
                        if face.samples!=paths || face.sequence!=sequence+paths {
                            return Err("Model comparison lost its sample count/sequence".into());
                        }
                        means[chunk_index*CHUNK_FACES+i] =
                            super::super::reference_oracle::channels(face.radiance).map(f64::from);
                    }
                }
                windows.push(means);
            }
            if index==0 { baseline=windows.clone(); }
            let mean: Means = windows[0].iter().zip(&windows[1])
                .map(|(a,b)| std::array::from_fn(|i| (a[i]+b[i])*0.5)).collect();
            let errors: Vec<_> = windows.iter().zip(&baseline).map(|(a,b)| {
                let errors:Vec<_> = a.iter().zip(b).map(|(a,b)| rgb_error(a,b)).collect();
                percentiles(&errors)
            }).collect();
            let name = format!("model-reach-{}-depth-{}",model.reach as u32,model.bounces);
            info!(?model,?errors,"Audited cave model comparison complete");
            rows.push(json!({"model":model,"name":name,"mean":mean,
                "windows":windows,"paired_rgb_error":errors}));
            frames.push_back(ReferenceFrame { name,values:mean,pending:default(),confidence:None });
        }
        let data = json!({"schema_version":1,"kind":"paired_model_comparison",
            "scene_generation":generation,"origin":origin.to_array(),
            "paths_per_window":paths,"sequences":STARTS,
            "sites":sites.iter().map(|(v,f)| json!({"voxel":v.to_array(),"face":f})).collect::<Vec<_>>(),
            "models":rows,
            "note":"Exact frozen composed cave, audited authored geometry, all chamber/halo faces. Same two sample windows at every model; images use their mean, equal reconstruction confidence, regular textures and tonemapping. Completion is not a visual-equivalence or perfect-reference pass. No runtime history/scheduling or interactive FPS claim."});
        std::fs::write(worker_output.join("cave-model-comparison.json"),
            serde_json::to_vec(&data).unwrap()).map_err(|e| e.to_string())?;
        Ok(Bake { frames,base,sites,accepted:true,
            qualification:json!({"kind":"paired_model_comparison",
                "measurement_complete":true,"quality_pass_claimed":false,
                "paths_per_window":paths,"windows":STARTS.len()}) })
    });
    Ok(ReferenceImage {
        frozen: true,
        task: Some(task),
        progress,
        total_paths,
        output,
        live_wait: Some(12),
        comparison: true,
        ..default()
    })
}
