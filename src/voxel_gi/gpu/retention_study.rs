//! Re-derive the whole-range retention scale for the surface cache.
//!
//! The distance curve keeps `H(d) = (d / s)^4` samples after an edit at
//! distance `d`. Its analytic basis is the MSE-optimal retained weight
//! `H = sigma^2 / D^2` with an inverse-square change bound `D ~ 1/d^2`; the
//! scale `s` absorbs per-sample variance, so `s ~ sigma^(-1/2)`. This study
//! replays real edits on every exposed face of a sealed scene: converge, edit,
//! cap each face's history with a candidate curve by its distance to the change,
//! then follow its error against that estimator's own converged post-edit
//! answer. It runs the cache and the explicit (cache-disabled) estimator so
//! the old fitted scale doubles as a check of the method.
use super::*;
use crate::voxel_gi::research::{ResearchFixture, channels};
use crate::voxel_gi::world::{AIR, FACE_OFFSETS, PAGE_SIZE, SurfaceProbe};
use crate::voxel_gi::naadf::{linear_index, volume_len};
use serde_json::json;

const CHECKPOINTS: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];

fn exposed_probes(scene: &TransportScene, min: IVec3, max: IVec3) -> Vec<SurfaceProbe> {
    let mut probes = Vec::new();
    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let voxel = IVec3::new(x, y, z);
                if scene.material_at(voxel).is_none_or(|m| m == AIR) {
                    continue;
                }
                let mut mask = 0;
                for (face, offset) in FACE_OFFSETS.iter().enumerate() {
                    let n = voxel + *offset;
                    if n.cmpge(min).all() && n.cmple(max).all() && scene.material_at(n) == Some(AIR) {
                        mask |= 1 << face;
                    }
                }
                if mask != 0 {
                    probes.push(SurfaceProbe::new(voxel, mask));
                }
            }
        }
    }
    probes
}

fn with_voxel(scene: &Arc<TransportScene>, voxel: IVec3, material: u16) -> Arc<TransportScene> {
    let edge = PAGE_SIZE.x as i32;
    let page = voxel.div_euclid(IVec3::splat(edge));
    let mut materials = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..PAGE_SIZE.z {
        for y in 0..PAGE_SIZE.y {
            for x in 0..PAGE_SIZE.x {
                let local = UVec3::new(x, y, z);
                materials[linear_index(local, PAGE_SIZE)] = scene
                    .material_at(page * edge + local.as_ivec3())
                    .unwrap_or(AIR);
            }
        }
    }
    let local = (voxel - page * edge).as_uvec3();
    materials[linear_index(local, PAGE_SIZE)] = material;
    Arc::new(TransportScene::with_page_delta(scene, vec![(page, materials)], &[]))
}

fn luminance(energy: EnergySpectrum) -> f64 {
    let c = channels(energy);
    0.2126 * f64::from(c[0]) + 0.7152 * f64::from(c[1]) + 0.0722 * f64::from(c[2])
}

struct Case {
    name: &'static str,
    fixture: ResearchFixture,
    before: Arc<TransportScene>,
    after: Arc<TransportScene>,
    changed: Vec3,
    /// Luminance flux of the changed emitter: emission luminance times its
    /// exposed faces (unit area). Bounds the change seen at distance d by
    /// flux / (pi d^2), without visibility or cosines.
    flux: f64,
    /// Positive when the edit adds the emitter, negative when it removes it.
    signed: f64,
    probes: Vec<SurfaceProbe>,
}

fn emitter_flux(scene: &TransportScene, materials: &VoxelMaterialTransportTable, voxel: IVec3, material: u16) -> f64 {
    let exposed = FACE_OFFSETS
        .iter()
        .filter(|o| scene.material_at(voxel + **o) == Some(AIR))
        .count();
    luminance(materials.get(material as u8).emission) * exposed as f64
}

#[derive(Clone, Copy)]
enum Arm {
    Retain,
    Reset,
    Scale(f32),
    /// MSE-optimal H = c * sigma^2 / D^2 with D = flux / (pi d^2).
    Physical(f64),
    /// The same plus a floor of k samples: new light brings variance that
    /// scales with its own size, sigma_new^2 ~ sigma^2 + k D^2.
    PhysicalFloor(f64, f64),
    /// Predict the new state's per-sample noise from the face's own relative
    /// noise times its predicted new mean max(M + signed D, 0).
    PhysicalSigned(f64),
}

impl Arm {
    fn label(self) -> String {
        match self {
            Arm::Retain => "retain".into(),
            Arm::Reset => "reset".into(),
            Arm::Scale(s) => format!("{s}"),
            Arm::Physical(c) => format!("phys{c}"),
            Arm::PhysicalFloor(c, k) => format!("phys{c}+{k}"),
            Arm::PhysicalSigned(c) => format!("signed{c}"),
        }
    }
}

fn work_for(case: &Case) -> TraceWork {
    let mut work = TraceWork::from_probes(&case.probes, false);
    for probe in &mut work.probes {
        probe.sky_requested = false;
    }
    work
}

fn trace(gpu: &mut GpuTracer, case: &Case, scene: &Arc<TransportScene>, emitters: &Arc<EmissionDistribution>, work: &mut TraceWork) {
    gpu.trace_compact(scene.clone(), &case.fixture.sources, &case.fixture.materials, work, 16, emitters, false)
        .unwrap();
}

fn converge(gpu: &mut GpuTracer, case: &Case, scene: &Arc<TransportScene>, emitters: &Arc<EmissionDistribution>, work: &mut TraceWork) {
    for (limit, batches) in [(256, 320), (65_536, 256)] {
        work.history_limit = limit;
        for _ in 0..batches {
            trace(gpu, case, scene, emitters, work);
        }
    }
    work.history_limit = 65_536;
}

fn emitters_for(case: &Case, scene: &TransportScene) -> Arc<EmissionDistribution> {
    Arc::new(EmissionDistribution::with_importance(
        scene,
        &case.fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ))
}

fn hallway_cases() -> Vec<Case> {
    let fixture = crate::voxel_gi::research::hallway();
    let emissive = u16::from(crate::terrain::TerrainMaterial::Emissive.index());
    let rock = u16::from(crate::terrain::TerrainMaterial::CaveRock.index());
    let (min, max) = (IVec3::new(-128, -32, -32), IVec3::new(-1, -1, -1));
    let mut lamps = Vec::new();
    for z in min.z..=max.z {
        for y in min.y..=max.y {
            for x in min.x..=max.x {
                let v = IVec3::new(x, y, z);
                if fixture.scene.material_at(v) == Some(emissive) {
                    lamps.push(v);
                }
            }
        }
    }
    assert!(!lamps.is_empty(), "hallway lamp");
    let lamp = lamps[0];
    let lit = fixture.scene.clone();
    let dark = with_voxel(&lit, lamp, rock);
    let probes = exposed_probes(&lit, min, max);
    let changed = lamp.as_vec3() + Vec3::splat(0.5);
    let flux = emitter_flux(&lit, &fixture.materials, lamp, emissive);
    let copy = |scene: &Arc<TransportScene>| ResearchFixture {
        scene: scene.clone(),
        materials: fixture.materials.clone(),
        sources: fixture.sources.clone(),
        receivers: Vec::new(),
    };
    vec![
        Case { name: "hallway-lamp-off", fixture: copy(&lit), before: lit.clone(), after: dark.clone(), changed, flux, signed: -1.0, probes: probes.clone() },
        Case { name: "hallway-lamp-on", fixture: copy(&lit), before: dark, after: lit, changed, flux, signed: 1.0, probes },
    ]
}

fn cave_cases() -> Vec<Case> {
    let fixture = crate::voxel_gi::research::cave();
    // Middle torch of the west wall; the lava, crystals, spire, other torches
    // and skylight keep both states lit and noisy.
    let torch = IVec3::new(5, 10, 32);
    assert_eq!(fixture.scene.material_at(torch), Some(7), "cave torch");
    let lit = fixture.scene.clone();
    let dark = with_voxel(&lit, torch, 2);
    let probes = exposed_probes(&lit, IVec3::splat(4), IVec3::new(59, 43, 59));
    let changed = torch.as_vec3() + Vec3::splat(0.5);
    let flux = emitter_flux(&lit, &fixture.materials, torch, 7);
    let copy = || ResearchFixture {
        scene: lit.clone(),
        materials: fixture.materials.clone(),
        sources: fixture.sources.clone(),
        receivers: Vec::new(),
    };
    vec![
        Case { name: "cave-torch-off", fixture: copy(), before: lit.clone(), after: dark.clone(), changed, flux, signed: -1.0, probes: probes.clone() },
        Case { name: "cave-torch-on", fixture: copy(), before: dark, after: lit.clone(), changed, flux, signed: 1.0, probes },
    ]
}

#[test]
#[ignore = "native retention-scale study for cache and explicit estimators (hallway lamp)"]
fn retention_curve_study() {
    let suffix = std::env::var("RETENTION_STUDY_ARMS").map_or(String::new(), |a| format!("-{a}"));
    run_study(hallway_cases(), &format!("target/retention-curve-study{suffix}.json"));
}

#[test]
#[ignore = "native retention-scale study for cache and explicit estimators (cave torch)"]
fn retention_curve_study_cave() {
    let suffix = std::env::var("RETENTION_STUDY_ARMS").map_or(String::new(), |a| format!("-{a}"));
    run_study(cave_cases(), &format!("target/retention-curve-study-cave{suffix}.json"));
}

fn run_study(cases: Vec<Case>, output: &str) {
    let native = tests::native_tracer();
    let explicit = super::SHADER.replacen("if cached.found {", "if false {", 1);
    assert_ne!(explicit, super::SHADER, "cache termination marker");
    let arms: Vec<Arm> = match std::env::var("RETENTION_STUDY_ARMS").as_deref() {
        Ok("physical") => vec![Arm::Physical(0.25), Arm::Physical(1.0), Arm::Physical(4.0), Arm::Physical(16.0)],
        Ok("signed") => vec![Arm::PhysicalSigned(0.25), Arm::PhysicalSigned(1.0), Arm::PhysicalSigned(4.0), Arm::PhysicalSigned(16.0)],
        Ok("floor") => vec![Arm::PhysicalFloor(1.0, 4.0), Arm::PhysicalFloor(1.0, 16.0), Arm::PhysicalFloor(1.0, 64.0),
            Arm::PhysicalFloor(4.0, 4.0), Arm::PhysicalFloor(4.0, 16.0), Arm::PhysicalFloor(4.0, 64.0)],
        _ => vec![Arm::Retain, Arm::Reset, Arm::Scale(6.0), Arm::Scale(9.0), Arm::Scale(12.0), Arm::Scale(15.0),
            Arm::Scale(18.0), Arm::Scale(24.0), Arm::Scale(32.0)],
    };
    let mut rows = Vec::new();
    for case in cases {
        let before_emitters = emitters_for(&case, &case.before);
        let after_emitters = emitters_for(&case, &case.after);
        for (mode, shader) in [("cache", super::SHADER.to_string()), ("explicit", explicit.clone())] {
            // Converged post-edit answer of this estimator.
            let mut reference_gpu = GpuTracer::with_shader(native.context.clone(), true, &shader).unwrap();
            reference_gpu.enable_resident().unwrap();
            let mut reference_work = work_for(&case);
            converge(&mut reference_gpu, &case, &case.after, &after_emitters, &mut reference_work);
            let reference: Vec<f64> = reference_work.faces.iter().map(|f| luminance(f.radiance)).collect();
            let peak = reference.iter().cloned().fold(0.0, f64::max);
            for arm in arms.iter().copied() {
                let mut gpu = GpuTracer::with_shader(native.context.clone(), true, &shader).unwrap();
                gpu.enable_resident().unwrap();
                let mut work = work_for(&case);
                converge(&mut gpu, &case, &case.before, &before_emitters, &mut work);
                let before: Vec<f64> = work.faces.iter().map(|f| luminance(f.radiance)).collect();
                // A sealed scene can go exactly dark: normalize by the brighter
                // scene's peak too, so 0/0 never happens for unreached faces.
                let floor = 1e-3 * peak.max(before.iter().cloned().fold(0.0, f64::max));
                let distances: Vec<f32> = work
                    .faces
                    .iter()
                    .map(|f| {
                        let center = f.voxel.as_vec3() + Vec3::splat(0.5) + FACE_OFFSETS[f.face].as_vec3() * 0.5;
                        center.distance(case.changed)
                    })
                    .collect();
                // Median relative per-sample noise: fallback for faces that were dark.
                let relative_noise = {
                    let mut r: Vec<f64> = work.faces.iter().filter_map(|f| {
                        let mean = luminance(f.radiance);
                        let second = channels(f.second_moment);
                        let m = channels(f.radiance);
                        let w = [0.2126, 0.7152, 0.0722];
                        let var: f64 = (0..3).map(|i| w[i] * f64::from((second[i] - m[i] * m[i]).max(0.0))).sum();
                        (mean > 1e-6).then(|| var.sqrt() / mean)
                    }).collect();
                    r.sort_by(f64::total_cmp);
                    let dark: f64 = std::env::var("RETENTION_DARK_NOISE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                    r.get(r.len() / 2).copied().unwrap_or(dark)
                };
                for (face, &d) in work.faces.iter_mut().zip(&distances) {
                    face.history_cap = match arm {
                        Arm::PhysicalSigned(c) => {
                            let m = channels(face.radiance);
                            let second = channels(face.second_moment);
                            let w = [0.2126, 0.7152, 0.0722];
                            let var: f64 = (0..3).map(|i| w[i] * f64::from((second[i] - m[i] * m[i]).max(0.0))).sum();
                            let mean = luminance(face.radiance);
                            let change = case.flux / (std::f64::consts::PI * f64::from(d.max(1.0)).powi(2));
                            let relative = if mean > 1e-6 { var.sqrt() / mean } else { relative_noise };
                            let predicted = (mean + case.signed * change).max(0.0);
                            let new_variance = var.max((relative * predicted).powi(2));
                            (c * new_variance / (change * change).max(1e-30)).min(65_536.0) as u32
                        }
                        Arm::Retain => u32::MAX,
                        Arm::Reset => 0,
                        Arm::Scale(s) => ((d / s).powi(4).round() as f64).min(65_536.0) as u32,
                        Arm::Physical(c) | Arm::PhysicalFloor(c, _) => {
                            // Per-sample luminance variance from this face's own moments.
                            let mean = channels(face.radiance);
                            let second = channels(face.second_moment);
                            let weights = [0.2126, 0.7152, 0.0722];
                            let variance: f64 = (0..3)
                                .map(|i| weights[i] * f64::from((second[i] - mean[i] * mean[i]).max(0.0)))
                                .sum();
                            let change = case.flux / (std::f64::consts::PI * f64::from(d.max(1.0)).powi(2));
                            let floor = if let Arm::PhysicalFloor(_, k) = arm { k } else { 0.0 };
                            (c * variance / (change * change).max(1e-30) + floor).min(65_536.0) as u32
                        }
                    };
                }
                let mut errors = Vec::new();
                let mut completed = 0;
                for checkpoint in CHECKPOINTS {
                    while completed < checkpoint {
                        trace(&mut gpu, &case, &case.after, &after_emitters, &mut work);
                        for face in &mut work.faces {
                            face.history_cap = u32::MAX;
                        }
                        completed += 1;
                    }
                    let per_face: Vec<f64> = work
                        .faces
                        .iter()
                        .enumerate()
                        .map(|(i, f)| {
                            let scale = before[i].max(reference[i]).max(floor);
                            (luminance(f.radiance) - reference[i]) / scale
                        })
                        .collect();
                    errors.push(per_face);
                }
                let label = arm.label();
                eprintln!("RETENTION {} {mode} arm={label} done", case.name);
                rows.push(json!({"case": case.name, "mode": mode, "scale": label,
                    "checkpoints_batches": CHECKPOINTS, "distances": distances,
                    "errors": errors}));
            }
        }
    }
    std::fs::write(output, serde_json::to_vec(&rows).unwrap()).unwrap();
}

