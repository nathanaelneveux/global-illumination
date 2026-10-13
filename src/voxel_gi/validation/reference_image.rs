//! Opt-in whole-chamber reference. Freeze real-time publication, bake disjoint
//! sample blocks on the exact scene, then upload through the ordinary display
//! path. This diagnostic never contributes measurements to gameplay history.
use super::super::{
    emission::EmissionDistribution,
    gpu::{GpuTracer, GpuTransportDevice},
    render::{GpuProbePatchQueue, GpuTransportCache, try_patch_probe_values},
    work::TraceWork,
    world::{AIR, SurfaceProbe},
};
use super::*;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

const BLOCK_PATHS: u32 = 32_768;
const DISPATCH_PATHS: u32 = 1024;
const CHUNK_FACES: usize = 512;

type ProbeMap = HashMap<IVec3, SurfaceProbe>;
type Means = Vec<[f64; 5]>;

struct ReferenceLookup<'a>(&'a ProbeMap);
impl super::super::field::ProbeLookup for ReferenceLookup<'_> {
    fn lookup(&self, voxel: &IVec3) -> Option<&SurfaceProbe> {
        self.0.get(voxel)
    }
}

#[derive(Resource, Default)]
pub(super) struct ReferenceImage {
    frozen: bool,
    task: Option<Task<Result<Bake, String>>>,
    progress: Arc<AtomicU64>,
    total_paths: u64,
    next_log: f64,
    output: PathBuf,
    frames: VecDeque<ReferenceFrame>,
    current: Option<ReferenceFrame>,
    settle_frames: usize,
    capture_count: Option<usize>,
    complete: bool,
    live_wait: Option<usize>,
    qualification: Option<serde_json::Value>,
}

struct ReferenceFrame {
    name: &'static str,
    probes: ProbeMap,
    pending: VecDeque<IVec3>,
}

struct Bake {
    frames: VecDeque<ReferenceFrame>,
    qualification: serde_json::Value,
}

pub(super) fn requested() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_REFERENCE_IMAGE").is_ok_and(|v| v == "1")
}

pub(super) fn install(app: &mut App) {
    if requested() {
        app.init_resource::<ReferenceImage>()
            .add_systems(
                PostStartup,
                setup_residency.after(super::setup_capture_target),
            )
            .configure_sets(
                Update,
                super::super::VoxelGiSystems.run_if(transport_running),
            )
            .add_systems(Update, tick.after(run_validation));
    }
}

/// A complete-reference fixture needs the chamber resident independently of
/// the measured view. Use the same non-rendering host camera as the background
/// workload, placed near the floor; the geometry audit still verifies every
/// chamber voxel. This changes only the opt-in offline fixture's residency.
fn setup_residency(mut commands: Commands, evidence: Res<CaptureEvidence>) {
    let origin = if crate::cave_scene::enabled() {
        Vec3::ZERO
    } else {
        crate::structures::showcase::CAVE_ORIGIN.as_vec3()
    };
    let eye = origin + Vec3::new(16.0, 6.0, 32.0);
    super::background::spawn_loader(
        &mut commands,
        &evidence,
        "Offline reference chamber residency (not rendered)",
        Transform::from_translation(eye).looking_at(origin + Vec3::new(40.0, 22.0, 32.0), Vec3::Y),
    );
}

fn transport_running(reference: Res<ReferenceImage>) -> bool {
    !reference.frozen
}

fn exposed(scene: &super::super::world::TransportScene, voxel: IVec3, face: usize) -> bool {
    scene.material_at(voxel).is_some_and(|m| m != AIR)
        && scene.material_at(voxel + FACE_OFFSETS[face]) == Some(AIR)
}

/// All chamber-facing surfaces, plus every exposed coplanar neighbor needed
/// by corner reconstruction. Includes floor, ceiling, spire and emitters.
fn chamber_sites(
    scene: &super::super::world::TransportScene,
    origin: IVec3,
) -> Vec<(IVec3, usize)> {
    let mut sites = HashSet::new();
    for z in 4..=59 {
        for y in 4..=43 {
            for x in 4..=59 {
                let voxel = origin + IVec3::new(x, y, z);
                if scene.material_at(voxel).is_none_or(|m| m == AIR) {
                    continue;
                }
                for (face, normal) in FACE_OFFSETS.into_iter().enumerate() {
                    let air = voxel + normal - origin;
                    if air.cmpge(IVec3::splat(5)).all()
                        && air.cmple(IVec3::new(58, 42, 58)).all()
                        && exposed(scene, voxel, face)
                    {
                        let (u, v) = match face / 2 {
                            0 => (IVec3::Y, IVec3::Z),
                            1 => (IVec3::X, IVec3::Z),
                            _ => (IVec3::X, IVec3::Y),
                        };
                        for a in -1..=1 {
                            for b in -1..=1 {
                                let p = voxel + u * a + v * b;
                                if exposed(scene, p, face) {
                                    sites.insert((p, face));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut sites: Vec<_> = sites.into_iter().collect();
    sites.sort_unstable_by_key(|(p, f)| (p.x, p.y, p.z, *f));
    sites
}

impl ReferenceImage {
    pub(super) fn start(
        runtime: &super::super::VoxelGiRuntime,
        field: &VoxelEnergyField,
        cache: &GpuTransportCache,
        queue: &GpuProbePatchQueue,
        device: Option<&GpuTransportDevice>,
        settings: &VoxelGiSettings,
        world: &VoxelWorld<PrototypeWorld>,
        generator: &PrototypeWorld,
        profile: Profile,
        output: PathBuf,
    ) -> Result<Self, String> {
        if profile != Profile::Cave
            || !crate::headless_validation()
            || std::env::var_os("VOXEL_GI_VALIDATE_EXIT").is_none()
        {
            return Err(
                "Reference image requires headless cave validation with EXIT enabled".into(),
            );
        }
        let paths: u32 = std::env::var("VOXEL_GI_REFERENCE_PATHS")
            .unwrap_or_else(|_| "262144".into())
            .parse()
            .map_err(|_| "REFERENCE_PATHS must be an integer")?;
        if !(BLOCK_PATHS..=1_048_576).contains(&paths) || !paths.is_multiple_of(BLOCK_PATHS) {
            return Err("REFERENCE_PATHS must be a multiple of 32768 in [32768,1048576]".into());
        }
        let device = device
            .ok_or("Reference image requires native GPU transport")?
            .clone();
        let origin = if crate::cave_scene::enabled() {
            IVec3::ZERO
        } else {
            crate::structures::showcase::CAVE_ORIGIN
        };
        let lookup = world.get_voxel_fn();
        let chunk_lookup = world.get_chunk_data_fn();
        let mut unknown = 0;
        let mut mismatched = 0;
        let mut samples = Vec::new();
        let mut pages: HashMap<IVec3, (usize, usize)> = default();
        for z in 4..=59 {
            for y in 4..=43 {
                for x in 4..=59 {
                    let voxel = origin + IVec3::new(x, y, z);
                    let expected = match generator.generated_voxel(voxel) {
                        WorldVoxel::Solid(id) => u16::from(id),
                        _ => AIR,
                    };
                    let actual = runtime.scene.material_at(voxel);
                    if actual == Some(expected) {
                        continue;
                    }
                    let page = super::super::page_of(voxel);
                    let counts = pages.entry(page).or_default();
                    if actual.is_none() {
                        unknown += 1;
                        counts.0 += 1;
                    } else {
                        mismatched += 1;
                        counts.1 += 1;
                    }
                    if samples.len() < 24 {
                        samples.push(json!({"voxel":voxel.to_array(),"expected":expected,
                            "scene":actual,"bvw":format!("{:?}",lookup(voxel))}));
                    }
                }
            }
        }
        let pages: Vec<_> = pages.into_iter().map(|(page,(unknown,mismatched))| {
            let chunk = chunk_lookup(page);
            json!({"page":page.to_array(),"unknown":unknown,"mismatched":mismatched,
                "bvw_chunk_present":chunk.is_some(),"bvw_chunk_empty":chunk.as_ref().map(|c| c.is_empty())})
        }).collect();
        let audit = json!({"unknown_voxels":unknown,"mismatched_voxels":mismatched,
            "pages":pages,"samples":samples,"note":"Unedited cave profile: compare every authored chamber voxel against the host generator and sample BVW values at mismatches. Unknown and wrong-material cells invalidate a complete-chamber reference."});
        std::fs::write(
            output.join("cave-geometry-audit.json"),
            serde_json::to_vec_pretty(&audit).unwrap(),
        )
        .map_err(|e| e.to_string())?;
        if unknown > 0 || mismatched > 0 {
            return Err(format!(
                "Cave GI snapshot differs from generated geometry: {unknown} unknown and {mismatched} wrong-material voxels; see cave-geometry-audit.json"
            ));
        }
        let sites = chamber_sites(&runtime.scene, origin);
        if sites.is_empty() || !cache.ready {
            return Err("Reference image has no chamber receivers or live display cache".into());
        }
        let live = cache.validation_mirror(queue, &sites);
        let mut base = ProbeMap::default();
        for (&(voxel, face), displayed) in sites.iter().zip(&live) {
            let probe = field
                .probes
                .get(&voxel)
                .filter(|p| p.face_mask & (1 << face) != 0 && displayed[3] > 0.0)
                .ok_or_else(|| {
                    format!("Reference face missing from field/display: {voxel:?}/{face}")
                })?;
            base.entry(voxel).or_insert_with(|| probe.clone());
        }
        let progress = Arc::new(AtomicU64::new(0));
        let total_paths = sites.len() as u64 * u64::from(paths) * 2;
        info!(
            faces = sites.len(),
            paths_per_replica = paths,
            total_paths,
            "Freezing GI publication for offline full-cave reference"
        );
        let scene = runtime.scene.clone();
        let sources = runtime.sources.clone();
        let materials = runtime.materials.clone();
        let transport = settings.transport;
        let generation = runtime.generation;
        let worker_progress = progress.clone();
        let worker_output = output.clone();
        let task = AsyncComputeTaskPool::get().spawn(async move {
            let mut gpu = GpuTracer::new(device)?;
            let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
            let mut replicas = Vec::new();
            for replica in 0..2 {
                let mut means = vec![[0.0_f64; 5]; sites.len()];
                for (chunk_index, chunk) in sites.chunks(CHUNK_FACES).enumerate() {
                    let mut work = TraceWork { transport, ..default() };
                    for &(voxel, face) in chunk {
                        let mut probe = SurfaceProbe::new(voxel, 1 << face);
                        probe.sequence[face] = 4_194_304 + replica * 2_097_152;
                        work.push(&probe, 1 << face, 0, 0, false, u32::MAX, u32::MAX);
                    }
                    for _ in 0..paths / BLOCK_PATHS {
                        for face in &mut work.faces {
                            face.samples = 0;
                            face.radiance = EnergySpectrum::ZERO;
                            face.second_moment = EnergySpectrum::ZERO;
                        }
                        for _ in 0..BLOCK_PATHS / DISPATCH_PATHS {
                            gpu.trace_compact(scene.clone(), &sources, &materials,
                                &mut work, DISPATCH_PATHS, &emitters, false)?;
                            worker_progress.fetch_add(chunk.len() as u64 * u64::from(DISPATCH_PATHS), Ordering::Relaxed);
                        }
                        for (i, face) in work.faces.iter().enumerate() {
                            for (sum, value) in means[chunk_index * CHUNK_FACES + i].iter_mut()
                                .zip(super::reference_oracle::channels(face.radiance))
                            {
                                *sum += f64::from(value) / f64::from(paths / BLOCK_PATHS);
                            }
                        }
                    }
                    // Deterministic direct/sky terms come from the frozen scene
                    // through the same path as normal display publication.
                    for (job, face) in work.probes.iter().zip(&work.faces) {
                        let probe = base.get_mut(&face.voxel).unwrap();
                        probe.direct[face.face] = face.direct;
                        probe.custom_direct[face.face] = face.custom_direct;
                        probe.display[face.face].sky = job.sky[face.face];
                    }
                }
                replicas.push(means);
            }
            let mean: Means = replicas[0].iter().zip(&replicas[1])
                .map(|(a,b)| std::array::from_fn(|i| (a[i] + b[i]) * 0.5)).collect();
            let discrepancy: Vec<_> = replicas[0].iter().zip(&replicas[1])
                .map(|(a,b)| rgb_error(a, b)).collect();
            let mut oracle_checks = Vec::new();
            for (local, face) in [
                (IVec3::new(4,22,15),0), (IVec3::new(59,9,50),1),
                (IVec3::new(22,37,59),5), (IVec3::new(8,43,56),3),
                (IVec3::new(8,4,8),2), (IVec3::new(30,22,32),1),
                (IVec3::new(7,40,8),3), (IVec3::new(59,37,36),1),
            ] {
                let voxel = origin + local;
                if let Some(index) = sites.iter().position(|&s| s == (voxel, face)) {
                    let oracle = super::reference_oracle::estimate(&scene, &materials, &sources,
                        transport, voxel, face, 262144);
                    let rejected: Vec<_> = (0..5).filter(|&b| {
                        (mean[index][b] - oracle.mean[b]).abs()
                            > 8.0 * oracle.standard_error[b] + 0.005 * oracle.mean[b].abs() + 0.001
                    }).collect();
                    oracle_checks.push(json!({"voxel":voxel.to_array(),"face":face,
                        "production":mean[index],"cosine_oracle":oracle,"rejected_bands":rejected}));
                }
            }
            let discrepancy_summary = percentiles(&discrepancy);
            let rejected_bands: usize = oracle_checks.iter().map(|c| c["rejected_bands"].as_array().unwrap().len()).sum();
            let qualification = json!({"qualified":rejected_bands==0 && oracle_checks.len()==8
                && discrepancy_summary["p99"].as_f64().unwrap()<=0.01
                && discrepancy_summary["max"].as_f64().unwrap()<=0.05,
                "oracle_rejected_bands":rejected_bands,"oracle_sites":oracle_checks.len(),
                "replica_relative_rgb_difference":discrepancy_summary,
                "criteria":"Eight independent cosine checks pass; replica RGB difference p99 <=1%, maximum <=5%. These are empirical agreement checks, not certified confidence bounds."});
            let rows: Vec<_> = sites.iter().enumerate().map(|(i, &(voxel,face))| {
                let p = &base[&voxel];
                json!({"voxel":voxel.to_array(),"face":face,
                    "material":p.material,"source_revision":p.source_revision,
                    "transport_revision":p.transport_revision,
                    "live_display":live[i],"live_history":p.samples[face],
                    "live_measured":super::reference_oracle::channels(p.radiance[face]),
                    "replicas":[replicas[0][i],replicas[1][i]],"mean":mean[i],
                    "replica_relative_rgb_difference":discrepancy[i]})
            }).collect();
            let data = json!({"schema_version":1,"compiled_manifest_dir":env!("CARGO_MANIFEST_DIR"),
                "scene_generation":generation,"regions":scene.regions.len(),"origin":origin.to_array(),
                "transport":transport,"paths_per_replica":paths,"replicas":2,
                "block_paths":BLOCK_PATHS,"face_count":sites.len(),
                "qualification":qualification,"oracle_checks":oracle_checks,
                "faces":rows,"note":"Frozen exact scene. Full chamber surfaces and coplanar reconstruction halo; outside chamber retains frozen live values. f64 average of independent blocks below history ceiling. Reference bypasses scheduling, resets and display smoothing, but uses the production finite-depth estimator. Two replica images expose residual noise; independent cosine CPU checks audit estimator bias. No infinite-bounce or mathematically perfect-reference claim."});
            std::fs::write(worker_output.join("cave-reference.json"), serde_json::to_vec(&data).unwrap())
                .map_err(|e| e.to_string())?;
            let frames = [
                ("reference-replica-a", &replicas[0]),
                ("reference-replica-b", &replicas[1]),
                ("cave-reference", &mean),
                ("cave-reference-held", &mean),
            ].into_iter().map(|(name, values)| {
                let mut probes = base.clone();
                for (&(voxel, face), value) in sites.iter().zip(values) {
                    let p = probes.get_mut(&voxel).unwrap();
                    p.display[face].indirect = Vec3::new(value[0] as f32,value[1] as f32,value[2] as f32);
                    p.display[face].confidence = 1.0;
                }
                let pending = probes.keys().copied().collect();
                ReferenceFrame { name, probes, pending }
            }).collect();
            Ok(Bake { frames, qualification })
        });
        Ok(Self {
            frozen: true,
            task: Some(task),
            progress,
            total_paths,
            output,
            live_wait: Some(12),
            ..default()
        })
    }
}

fn rgb_error(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    let norm = |v: &[f64; 5]| (v[..3].iter().map(|x| x * x).sum::<f64>()).sqrt();
    let difference: [f64; 5] = std::array::from_fn(|i| a[i] - b[i]);
    norm(&difference) / ((norm(a) + norm(b)) * 0.5).max(0.02)
}

fn percentiles(values: &[f64]) -> serde_json::Value {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"p50":sorted[sorted.len()/2],"p95":sorted[(sorted.len()*95/100).min(sorted.len()-1)],
        "p99":sorted[(sorted.len()*99/100).min(sorted.len()-1)],"max":sorted.last()})
}

fn tick(
    mut reference: ResMut<ReferenceImage>,
    mut evidence: ResMut<CaptureEvidence>,
    cache: Res<GpuTransportCache>,
    queue: Res<GpuProbePatchQueue>,
    time: Res<Time<Real>>,
    mut commands: Commands,
) {
    if !reference.frozen || reference.complete || evidence.failed {
        return;
    }
    if time.elapsed_secs_f64() >= reference.next_log {
        info!(
            percent = 100.0 * reference.progress.load(Ordering::Relaxed) as f64
                / reference.total_paths as f64,
            "Offline cave reference progress"
        );
        reference.next_log = time.elapsed_secs_f64() + 10.0;
    }
    if let Some(wait) = reference.live_wait.as_mut() {
        if *wait > 0 {
            *wait -= 1;
            return;
        }
        let target = evidence.render_target.clone().unwrap();
        commands
            .spawn(Screenshot::image(target))
            .observe(save_to_disk(reference.output.join("live-frozen.png")))
            .observe(check_capture);
        reference.capture_count = Some(evidence.completed);
        reference.live_wait = None;
    }
    if let Some(task) = reference.task.as_mut() {
        match block_on(poll_once(task)) {
            None => return,
            Some(Ok(bake)) => {
                reference.frames = bake.frames;
                reference.qualification = Some(bake.qualification);
                reference.task = None;
            }
            Some(Err(error)) => {
                evidence.fail(&format!("Full-cave reference failed: {error}"));
                commands.write_message(AppExit::error());
                return;
            }
        }
    }
    if let Some(count) = reference.capture_count {
        if evidence.completed <= count {
            return;
        }
        reference.capture_count = None;
        reference.current = None;
    }
    if reference.current.is_none() {
        reference.current = reference.frames.pop_front();
        reference.settle_frames = 12;
        if reference.current.is_none() {
            reference.complete = true;
            let saved = std::fs::write(
                reference.output.join("cave-reference-status.json"),
                json!({"status":"completed","captures":5,"frozen":true,"geometry_audit_passed":true,
                    "qualification":reference.qualification})
                .to_string(),
            );
            if let Err(error) = saved {
                evidence.fail(&format!("Reference status publication failed: {error}"));
                commands.write_message(AppExit::error());
            } else {
                let qualified = reference
                    .qualification
                    .as_ref()
                    .and_then(|q| q["qualified"].as_bool())
                    .unwrap_or(false);
                commands.write_message(if qualified {
                    AppExit::Success
                } else {
                    AppExit::error()
                });
            }
            return;
        }
    }
    let frame = reference.current.as_mut().unwrap();
    if !frame.pending.is_empty() {
        try_patch_probe_values(
            &cache,
            &queue,
            &ReferenceLookup(&frame.probes),
            std::iter::from_fn(|| frame.pending.pop_front()),
            Duration::from_millis(2),
        );
        return;
    }
    if reference.settle_frames > 0 {
        reference.settle_frames -= 1;
        return;
    }
    let frame = reference.current.as_ref().unwrap();
    let target = evidence.render_target.clone().unwrap();
    let path = reference.output.join(format!("{}.png", frame.name));
    commands
        .spawn(Screenshot::image(target))
        .observe(save_to_disk(path))
        .observe(check_capture);
    reference.capture_count = Some(evidence.completed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_cave_reference_includes_floor_ceiling_spire_and_sources() {
        let cave = crate::voxel_gi::research::cave();
        let sites = chamber_sites(&cave.scene, IVec3::ZERO);
        assert!(sites.len() > 10_000);
        for site in [
            (IVec3::new(4, 22, 15), 0),
            (IVec3::new(59, 9, 50), 1),
            (IVec3::new(22, 37, 59), 5),
            (IVec3::new(8, 43, 56), 3),
            (IVec3::new(8, 4, 8), 2),
            (IVec3::new(30, 22, 32), 1),
            (IVec3::new(7, 40, 8), 3),
        ] {
            assert!(sites.contains(&site), "missing reference site {site:?}");
        }
        assert!(sites.iter().all(|&(v, f)| exposed(&cave.scene, v, f)));
        let distinct: HashSet<_> = sites.iter().copied().collect();
        assert_eq!(distinct.len(), sites.len());
    }

    #[test]
    fn reference_freeze_prevents_transport_systems_from_mutating_the_image() {
        #[derive(Resource, Default)]
        struct Publications(usize);
        let mut app = App::new();
        app.init_resource::<ReferenceImage>()
            .init_resource::<Publications>()
            .configure_sets(
                Update,
                super::super::super::VoxelGiSystems.run_if(transport_running),
            )
            .add_systems(
                Update,
                (|mut p: ResMut<Publications>| p.0 += 1)
                    .in_set(super::super::super::VoxelGiSystems),
            );
        app.update();
        assert_eq!(app.world().resource::<Publications>().0, 1);
        app.world_mut().resource_mut::<ReferenceImage>().frozen = true;
        app.update();
        app.update();
        assert_eq!(app.world().resource::<Publications>().0, 1);
    }
}
