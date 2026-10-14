//! Offline near-history calibration on audited, composed-world fixtures. Record
//! common native paths once; replay policy candidates without retracing them.
//! Off-camera edits isolate retained receivers from mesh replacement.
//! This is a cache-response experiment, not a gameplay/frame-pacing benchmark.
use super::near_fixture::Fixture;
use super::*;
use crate::voxel_gi::{naadf::linear_index, world::PAGE_SIZE};
use std::io::{BufWriter, Write};

const WARM: u32 = 4096;
const BATCH: u32 = 16;
const STEPS: usize = 256;
const STARTS: [u32; 2] = [17, 131_119];
const REFERENCE_PATHS: u32 = 131_072;

pub(super) fn requested() -> bool {
    std::env::var_os("VOXEL_GI_VALIDATE_NEAR_HISTORY").is_some()
}

fn write_moments(writer: &mut impl Write, work: &TraceWork) -> Result<(), String> {
    for face in &work.faces {
        for value in [face.radiance.visible, face.second_moment.visible] {
            for v in value.to_array() {
                writer
                    .write_all(&v.to_le_bytes())
                    .map_err(|e| e.to_string())?;
            }
        }
    }
    Ok(())
}

fn jobs(sites: &[(IVec3, usize)], starts: impl IntoIterator<Item = u32>) -> TraceWork {
    let mut work = TraceWork::default();
    for start in starts {
        for &(voxel, face) in sites {
            let mut p = SurfaceProbe::new(voxel, 1 << face);
            p.sequence[face] = start;
            work.push(&p, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
            work.probes.last_mut().unwrap().sky_requested = false;
        }
    }
    work
}

/// Preserve the actual runtime snapshot except for one existing, off-camera
/// voxel. In particular, do not use a material-wide emission toggle or pretend
/// that adding a lamp is a global analytic-light change.
fn remove_torch(
    scene: &Arc<crate::voxel_gi::world::TransportScene>,
    origin: IVec3,
) -> Result<(Arc<crate::voxel_gi::world::TransportScene>, IVec3), String> {
    let voxel = origin + IVec3::new(5, 10, 56);
    let expected = if crate::cave_scene::enabled() {
        7
    } else {
        u16::from(crate::terrain::TerrainMaterial::CaveTorch.index())
    };
    if scene.material_at(voxel) != Some(expected) {
        return Err("Near-history fixture requires its authored torch".into());
    }
    Ok((replace_voxel(scene, voxel, AIR)?, voxel))
}

fn replace_voxel(
    scene: &Arc<crate::voxel_gi::world::TransportScene>,
    voxel: IVec3,
    material: u16,
) -> Result<Arc<crate::voxel_gi::world::TransportScene>, String> {
    let page = crate::voxel_gi::page_of(voxel);
    let page_origin = page * 32;
    let mut cells = Vec::with_capacity(32 * 32 * 32);
    for z in 0..32 {
        for y in 0..32 {
            for x in 0..32 {
                cells.push(
                    scene
                        .material_at(page_origin + IVec3::new(x, y, z))
                        .ok_or("Near-history edit page is not fully resident")?,
                );
            }
        }
    }
    cells[linear_index((voxel - page_origin).as_uvec3(), PAGE_SIZE)] = material;
    Ok(Arc::new(
        crate::voxel_gi::world::TransportScene::with_page_delta(scene, vec![(page, cells)], &[]),
    ))
}

fn render_plan(path: &str, sites: &[(IVec3, usize)]) -> Result<VecDeque<ReferenceFrame>, String> {
    let path = PathBuf::from(path);
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let expected = json!(
        sites
            .iter()
            .map(|(v, f)| json!([v.to_array(), f]))
            .collect::<Vec<_>>()
    );
    if manifest["schema"] != 1 || manifest["sites"] != expected {
        return Err("Replay receiver identity/order differs from the audited fixture".into());
    }
    if manifest["fixture"].as_str().unwrap_or("cave-torch") != Fixture::selected()?.name() {
        return Err("Replay fixture differs from the active scene".into());
    }
    let mut frames = VecDeque::new();
    for frame in manifest["frames"]
        .as_array()
        .ok_or("Replay has no frames")?
    {
        let name = frame["name"].as_str().ok_or("Frame has no name")?;
        if Fixture::selected()?.is_house() && frame["sky_state"].as_str().is_none() {
            return Err("House replay must specify its target-state sunlight".into());
        }
        if name.is_empty() || !name.bytes().all(|v| v.is_ascii_alphanumeric() || v == b'-') {
            return Err("Unsafe replay frame name".into());
        }
        let file = path.parent().unwrap().join(format!("{name}.bin"));
        let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
        if bytes.len() != sites.len() * 16 {
            return Err("Replay RGB/confidence buffer length differs".into());
        }
        let mut values = Vec::with_capacity(sites.len());
        let mut confidence = Vec::with_capacity(sites.len());
        for row in bytes.chunks_exact(16) {
            let v: [f32; 4] = std::array::from_fn(|i| {
                f32::from_le_bytes(row[i * 4..i * 4 + 4].try_into().unwrap())
            });
            if v.iter().any(|x| !x.is_finite() || *x < 0.) || v[3] > 1. {
                return Err("Replay contains invalid energy/confidence".into());
            }
            values.push([f64::from(v[0]), f64::from(v[1]), f64::from(v[2]), 0., 0.]);
            confidence.push(v[3]);
        }
        frames.push_back(ReferenceFrame {
            name: name.into(),
            values,
            confidence: Some(confidence),
            sky: if let Some(state) = frame["sky_state"].as_str() {
                let sky: Vec<[u32; 4]> = serde_json::from_value(manifest["skies"][state].clone())
                    .map_err(|e| e.to_string())?;
                if sky.len() != sites.len() {
                    return Err("Replay sky receiver count differs".into());
                }
                Some(sky)
            } else {
                None
            },
            pending: default(),
        });
    }
    if frames.is_empty() {
        return Err("Replay is empty".into());
    }
    Ok(frames)
}

pub(super) fn start(
    runtime: &crate::voxel_gi::VoxelGiRuntime,
    device: GpuTransportDevice,
    output: PathBuf,
    origin: IVec3,
    mut sites: Vec<(IVec3, usize)>,
    base: ProbeMap,
) -> Result<ReferenceImage, String> {
    let mode = std::env::var("VOXEL_GI_VALIDATE_NEAR_HISTORY").unwrap();
    let fixture = Fixture::selected()?;
    let edits = fixture.edits(origin);
    if fixture.is_house() {
        // Keep only persistent receivers. Preserve historical cave tape identity;
        // its edited receiver is behind the camera and cannot affect the image.
        sites.retain(|(p, _)| !edits.contains(p));
    }
    let progress = Arc::new(AtomicU64::new(0));
    let total_paths = sites.len() as u64
        * 2
        * (u64::from(REFERENCE_PATHS) * 2
            + STARTS.len() as u64 * (u64::from(WARM) + STEPS as u64 * u64::from(BATCH)));
    let worker_progress = progress.clone();
    let worker_output = output.clone();
    let mut scene = runtime.scene.clone();
    let sources = runtime.sources.clone();
    let mut materials = runtime.materials.clone();
    let generation = runtime.generation;
    let edit = edits[0];
    let mut changed = scene.clone();
    if fixture == Fixture::CaveTorch {
        changed = remove_torch(&scene, origin)?.0;
    } else if fixture == Fixture::HouseEmitter {
        if scene.material_at(edit) != Some(AIR) {
            return Err("House test emitter location is not empty".into());
        }
        if scene
            .regions
            .values()
            .any(|r| r.contains_any_material(&[0, 0, 0, 1 << 58]))
        {
            return Err("Diagnostic material 250 is already in use".into());
        }
        Arc::make_mut(&mut materials).set(
            250,
            MaterialTransport {
                diffuse: EnergySpectrum::new(Vec3::splat(0.2), 0.1, 0.01),
                emission: EnergySpectrum::new(Vec3::new(12., 3., 0.3), 4., 1.),
            },
        );
        scene = replace_voxel(&scene, edit, 250)?;
    } else {
        for &voxel in &edits {
            if scene.material_at(voxel).is_none_or(|v| v == AIR) {
                return Err("House opening edit has no authored solid voxel".into());
            }
            changed = replace_voxel(&changed, voxel, AIR)?;
        }
    }
    let scale = std::env::var("VOXEL_GI_NEAR_EMITTER_SCALE")
        .unwrap_or_else(|_| "1".into())
        .parse::<f32>()
        .map_err(|_| "NEAR_EMITTER_SCALE must be a number")?;
    if !scale.is_finite() || !(1.0..=1024.0).contains(&scale) {
        return Err("NEAR_EMITTER_SCALE must be finite and in [1,1024]".into());
    }
    if scale != 1.0 {
        if fixture != Fixture::CaveTorch {
            return Err("Emitter scaling is only defined for cave-torch".into());
        }
        // One diagnostic host material, never a change to the global torch
        // palette. A bright-source control prevents tiny edits below the FLIP
        // gate from making a slow response policy look universally superior.
        if scene
            .regions
            .values()
            .any(|r| r.contains_any_material(&[0, 0, 0, 1 << 58]))
        {
            return Err("Diagnostic material 250 is already in use".into());
        }
        let mut optical = materials.get(scene.material_at(edit).unwrap() as u8);
        optical.emission *= scale;
        Arc::make_mut(&mut materials).set(250, optical);
        scene = replace_voxel(&scene, edit, 250)?;
    }
    let (eye, target, fov) = fixture.view();
    for &voxel in &edits {
        if !fixture.edit_is_offscreen(voxel) {
            return Err("Calibration edit intersects the captured view".into());
        }
    }
    let task = AsyncComputeTaskPool::get().spawn(async move {
        if mode != "record" {
            let frames = render_plan(&mode, &sites)?;
            return Ok(Bake { frames, base, sites, accepted:true,
                qualification:json!({"kind":"near_history_replay","quality_pass_claimed":false,
                    "plan":mode,"note":"Fixed receiving geometry; offline cache images, not gameplay timing."}) });
        }
        let mut gpu = GpuTracer::new(device)?;
        let mut states = Vec::new();
        let mut frames = VecDeque::new();
        for (name, scene) in [("before",scene),("after",changed)] {
            let sky = if fixture.is_house() {
                let mut work = jobs(&sites, [0]);
                for probe in &mut work.probes { probe.sky_requested = true; }
                crate::voxel_gi::tracer::trace_sky_batch(&scene, &sources, &mut work.probes, &mut default());
                Some(work.probes.iter().zip(&sites).map(|(p,(_,face))| p.sky[*face]).collect::<Vec<_>>())
            } else { None };
            let emitters = Arc::new(EmissionDistribution::build(&scene,&materials));
            let mut replicas = Vec::new();
            for replica in 0..2 {
                let mut means = vec![[0.;5];sites.len()];
                for (chunk_index, chunk) in sites.chunks(CHUNK_FACES).enumerate() {
                    let mut work = jobs(chunk, [4_194_304 + replica * 2_097_152]);
                    for _ in 0..REFERENCE_PATHS/BLOCK_PATHS {
                        for face in &mut work.faces {
                            face.samples=0;
                            face.radiance=EnergySpectrum::ZERO;
                            face.second_moment=EnergySpectrum::ZERO;
                        }
                        for _ in 0..BLOCK_PATHS/DISPATCH_PATHS {
                            gpu.trace_compact(scene.clone(),&sources,&materials,&mut work,DISPATCH_PATHS,&emitters,false)?;
                            worker_progress.fetch_add(chunk.len() as u64*u64::from(DISPATCH_PATHS),Ordering::Relaxed);
                        }
                        for (i,face) in work.faces.iter().enumerate() {
                            for (sum,value) in means[chunk_index*CHUNK_FACES+i].iter_mut()
                                .zip(crate::voxel_gi::validation::reference_oracle::channels(face.radiance)) {
                                *sum+=f64::from(value)/f64::from(REFERENCE_PATHS/BLOCK_PATHS);
                            }
                        }
                    }
                }
                frames.push_back(ReferenceFrame { name:format!("near-{name}-replica-{replica}"),
                    values:means.clone(),pending:default(),confidence:None,sky:sky.clone() });
                replicas.push(means);
            }
            let mean:Means=replicas[0].iter().zip(&replicas[1])
                .map(|(a,b)| std::array::from_fn(|i|(a[i]+b[i])*0.5)).collect();
            frames.push_back(ReferenceFrame { name:format!("near-{name}-reference"),
                values:mean.clone(),pending:default(),confidence:None,sky:sky.clone() });
            let file_name=format!("near-{name}-moments.bin");
            let mut writer=BufWriter::new(std::fs::File::create(worker_output.join(&file_name)).map_err(|e|e.to_string())?);
            // Binary layout: window, face chunk (<=512), warm then groups of
            // sixteen fresh batches, batch, site, [RGB mean, RGB raw second].
            // Parallel independent jobs preserve the exact path sequence while
            // avoiding one GPU round trip for every tiny batch.
            for sequence in STARTS {
                for chunk in sites.chunks(CHUNK_FACES) {
                    let mut work=jobs(chunk,[sequence]);
                    for _ in 0..WARM/DISPATCH_PATHS {
                        gpu.trace_compact(scene.clone(),&sources,&materials,&mut work,DISPATCH_PATHS,&emitters,false)?;
                    }
                    write_moments(&mut writer,&work)?;
                    for first in (0..STEPS).step_by(16) {
                        let starts=(first..first+16).map(|step| sequence+WARM+step as u32*BATCH);
                        let mut work=jobs(chunk,starts);
                        gpu.trace_compact(scene.clone(),&sources,&materials,&mut work,BATCH,&emitters,false)?;
                        if work.faces.iter().any(|f|f.samples!=BATCH) {
                            return Err("Native tape lost its batch sample count".into());
                        }
                        write_moments(&mut writer,&work)?;
                    }
                    worker_progress.fetch_add(chunk.len() as u64*(u64::from(WARM)+STEPS as u64*u64::from(BATCH)),Ordering::Relaxed);
                }
            }
            writer.flush().map_err(|e|e.to_string())?;
            states.push(json!({"name":name,"replicas":replicas,"mean":mean,"moments":file_name,"sky":sky}));
        }
        let metadata=json!({"schema":1,"kind":"near_history_tape","compiled_manifest_dir":env!("CARGO_MANIFEST_DIR"),
            "scene_generation":generation,"fixture":fixture.name(),"origin":origin.to_array(),"view":{"eye":eye.to_array(),"target":target.to_array(),"fov_degrees":fov},"transport":crate::voxel_gi::TransportSettings::default(),
            "warm_paths":WARM,"batch_paths":BATCH,"steps":STEPS,"starts":STARTS,"chunk_faces":CHUNK_FACES,
            "reference_paths_per_replica":REFERENCE_PATHS,"edit":edit.to_array(),"edits":edits.iter().map(|p|p.to_array()).collect::<Vec<_>>(),"emitter_scale":scale,
            "sites":sites.iter().map(|(v,f)|json!([v.to_array(),f])).collect::<Vec<_>>(),"states":states,
            "note":"Off-camera authored edit removed/restored, plus exact unchanged-lighting control. Frozen composed-world geometry and sources with common native path windows. House direct-sun corners are solved separately for each state. Only retained receivers are calibrated, not newly exposed topology. No scheduling, display smoothing or gameplay seconds in this tape. Binary little-endian f32 moments layout is documented in near_history.rs."});
        std::fs::write(worker_output.join("near-history.json"),serde_json::to_vec(&metadata).unwrap()).map_err(|e|e.to_string())?;
        Ok(Bake {frames,base,sites,accepted:true,qualification:json!({"kind":"near_history_tape",
            "measurement_complete":true,"quality_pass_claimed":false})})
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_replay_matches_the_production_display_resolve() {
        let now = Instant::now();
        for (mean, display, samples, expected, confidence) in [
            (
                Vec3::new(0.01, 0.02, 0.03),
                Vec3::ONE,
                32,
                Vec3::new(0.98887787, 0.98899021, 0.98910256),
                0.99313831,
            ),
            (
                Vec3::new(4., 2., 1.),
                Vec3::ZERO,
                256,
                Vec3::new(0.03296449, 0.01648225, 0.00824112),
                0.95036899,
            ),
        ] {
            let mut probe = SurfaceProbe::new(IVec3::ZERO, 1);
            probe.radiance[0] = EnergySpectrum::new(mean, 0., 0.);
            probe.second_moment[0] = EnergySpectrum::new(mean * mean * 100., 0., 0.);
            probe.samples[0] = samples;
            probe.display[0].indirect = display;
            probe.display[0].confidence = 1.;
            probe.resolve_face(0, Some(now - Duration::from_millis(100)), now, 0.075);
            assert!((probe.display[0].indirect - expected).abs().max_element() < 1e-6);
            assert!((probe.display[0].confidence - confidence).abs() < 1e-6);
        }
    }

    #[test]
    fn native_history_windows_are_disjoint_and_below_history_ceiling() {
        assert!(WARM < crate::voxel_gi::tracer::HISTORY_LIMIT);
        for pair in STARTS.windows(2) {
            assert!(pair[0] + WARM + STEPS as u32 * BATCH < pair[1]);
        }
        assert!(STARTS.last().unwrap() + WARM + STEPS as u32 * BATCH < 4_194_304);
        assert!(REFERENCE_PATHS.is_multiple_of(BLOCK_PATHS));
        assert!(STEPS.is_multiple_of(16));
    }
}
