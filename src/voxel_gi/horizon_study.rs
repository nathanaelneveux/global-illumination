//! Fixed-work model-bias study. The production kernel, source PDFs, residency,
//! receiver set and sample windows stay fixed; only diffuse reach/depth vary.
//! No history truncation, temporal filtering or wall-clock scheduling is used.
use super::*;
use crate::voxel_gi::TransportSettings;

const WINDOWS: [u32; 2] = [574_117, 1_853_903];
const BLOCK: u32 = 32_768;
const DISPATCH: u32 = 512;

fn canyon_fixture() -> Fixture {
    let f = crate::voxel_gi::research::canyon();
    Fixture {
        name: "daylight_canyon",
        scene: f.scene,
        sources: f.sources,
        materials: f.materials,
        probes: f.receivers,
        analytic: None,
    }
}

fn ceiling_fixture() -> Fixture {
    let mut f = fixture(2);
    f.name = "skylight_ceiling";
    f.probes = (6..26)
        .step_by(2)
        .flat_map(|z| {
            (6..26)
                .step_by(2)
                .filter(move |x| !(13..=17).contains(x) || !(13..=17).contains(&z))
                .map(move |x| SurfaceProbe::new(IVec3::new(x, 20, z), 8))
        })
        .collect();
    f
}

fn models() -> Vec<TransportSettings> {
    if let Ok(value) = std::env::var("VOXEL_GI_HORIZON_MODELS") {
        return value
            .split(',')
            .map(|pair| {
                let (reach, bounces) = pair.split_once(':').expect("models: reach:depth,...");
                let m = TransportSettings {
                    reach: reach.parse().unwrap(),
                    bounces: bounces.parse().unwrap(),
                };
                assert_eq!(m, m.normalized(), "out-of-range model");
                m
            })
            .collect();
    }
    [16., 32., 64., 128., 192., 256., 384., 512., 768.]
        .into_iter()
        .flat_map(|reach| (1..=3).map(move |bounces| TransportSettings { reach, bounces }))
        .collect()
}

/// Return f64 block averages, never the capped runtime EMA. Splitting both
/// candidate and baseline into the same batches preserves common random paths.
fn integrate(
    gpu: &mut GpuTracer,
    f: &Fixture,
    emitters: &Arc<EmissionDistribution>,
    model: TransportSettings,
    sequence: u32,
    paths: u32,
) -> (Vec<[f64; 5]>, Option<f64>) {
    let mut means = Vec::new();
    let mut gpu_ms = Some(0.0);
    for probes in f.probes.chunks(256) {
        let mut work = TraceWork::from_probes(probes, false);
        work.transport = model;
        for p in &mut work.probes {
            p.sky_requested = false;
        }
        for face in &mut work.faces {
            face.sequence = sequence;
        }
        let mut sums = vec![[0.0; 5]; work.faces.len()];
        let mut remaining = paths;
        while remaining > 0 {
            let block = remaining.min(BLOCK);
            for face in &mut work.faces {
                face.samples = 0;
                face.radiance = EnergySpectrum::ZERO;
                face.second_moment = EnergySpectrum::ZERO;
            }
            let mut pending = block;
            while pending > 0 {
                let n = pending.min(DISPATCH);
                gpu.trace_compact(
                    f.scene.clone(),
                    &f.sources,
                    &f.materials,
                    &mut work,
                    n,
                    emitters,
                    false,
                )
                .unwrap();
                // Some Metal dispatches report zero/identical endpoints. Keep
                // accuracy evidence, but never report those as free GPU work.
                gpu_ms = gpu_ms
                    .zip(gpu.last_gpu_ms)
                    .map(|(sum, ms)| sum + f64::from(ms));
                pending -= n;
            }
            for (sum, face) in sums.iter_mut().zip(&work.faces) {
                for (sum, value) in sum.iter_mut().zip(channels(face.radiance)) {
                    *sum += f64::from(value) * f64::from(block) / f64::from(paths);
                }
            }
            remaining -= block;
        }
        means.extend(sums);
    }
    (means, gpu_ms)
}

fn summary(values: &[f64]) -> Value {
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    json!({"mean":values.iter().sum::<f64>()/values.len() as f64,
        "p50":sorted[sorted.len()/2],"p95":sorted[sorted.len()*95/100],
        "p99":sorted[sorted.len()*99/100],"max":sorted.last()})
}

/// A symmetric 0.02-floor normalization, shared with the cave reference. This
/// is a linear transport error, not a perceptual score or convergence proof.
fn difference(a: &[[f64; 5]], b: &[[f64; 5]]) -> Value {
    assert_eq!(a.len(), b.len());
    let rgb: Vec<_> = a
        .iter()
        .zip(b)
        .map(|(a, b)| {
            let norm = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let delta = std::array::from_fn::<_, 3, _>(|i| a[i] - b[i]);
            norm(&delta) / ((norm(&a[..3]) + norm(&b[..3])) * 0.5).max(0.02)
        })
        .collect();
    let bands: Vec<_> = (0..5)
        .map(|i| {
            let errors: Vec<_> = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a[i] - b[i]).abs() / ((a[i].abs() + b[i].abs()) * 0.5).max(0.02))
                .collect();
            summary(&errors)
        })
        .collect();
    json!({"rgb":summary(&rgb),"bands":bands})
}

#[test]
fn horizon_difference_is_symmetric_and_reports_small_dark_values() {
    let a = [[0.0; 5], [1.0; 5]];
    let b = [[0.001; 5], [0.9; 5]];
    assert_eq!(difference(&a, &b), difference(&b, &a));
    assert_eq!(difference(&a, &a)["rgb"]["max"], 0.0);
    assert!(
        (difference(&a, &b)["bands"][0]["mean"].as_f64().unwrap() - (0.05 + 0.1 / 0.95) * 0.5)
            .abs()
            < 1e-12
    );
}

#[test]
#[ignore = "native reach/depth model matrix; no runtime changes"]
fn horizon_reach_depth_study() {
    let paths: u32 = std::env::var("VOXEL_GI_HORIZON_PATHS")
        .unwrap_or_else(|_| "4096".into())
        .parse()
        .unwrap();
    assert!((512..=1_048_576).contains(&paths));
    let models = models();
    let selected = std::env::var("VOXEL_GI_HORIZON_FIXTURES").unwrap_or_default();
    let output = std::env::var("VOXEL_GI_HORIZON_OUTPUT")
        .unwrap_or_else(|_| "target/horizon-study.json".into());
    let mut gpu = super::super::tests::native_tracer();
    let mut fixtures = vec![fixture(0), fixture(1), fixture(2), ceiling_fixture()];
    let cave = crate::voxel_gi::research::cave();
    fixtures.push(Fixture {
        name: "many_emitter_cave",
        scene: cave.scene,
        sources: cave.sources,
        materials: cave.materials,
        probes: cave.receivers,
        analytic: None,
    });
    fixtures.push(canyon_fixture());
    let mut reports = Vec::new();
    for f in fixtures {
        if !selected.is_empty() && !selected.split(',').any(|n| n == f.name) {
            continue;
        }
        let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
        let mut baseline = Vec::new();
        for sequence in WINDOWS {
            baseline.push(integrate(
                &mut gpu,
                &f,
                &emitters,
                TransportSettings::default(),
                sequence,
                paths,
            ));
        }
        eprintln!(
            "HORIZON_BASELINE {} receivers={} windows={} paths={paths}",
            f.name,
            baseline[0].0.len(),
            WINDOWS.len()
        );
        let mut runs = Vec::new();
        for model in &models {
            let mut windows = Vec::new();
            for (window, sequence) in WINDOWS.into_iter().enumerate() {
                let (means, ms) = if *model == TransportSettings::default() {
                    baseline[window].clone()
                } else {
                    integrate(&mut gpu, &f, &emitters, *model, sequence, paths)
                };
                let diff = difference(&means, &baseline[window].0);
                eprintln!(
                    "HORIZON {} reach={} depth={} window={window} {}",
                    f.name, model.reach, model.bounces, diff["rgb"]
                );
                windows.push(json!({"sequence_start":sequence,"means":means,
                    "paired_difference":diff,"gpu_ms":ms}));
            }
            runs.push(json!({"model":model,"windows":windows}));
        }
        let sites: Vec<_> = f
            .probes
            .iter()
            .flat_map(|p| {
                (0..6).filter(|face| p.face_mask & (1 << face) != 0).map(
                    |face| json!({"voxel":p.voxel.to_array(),"face":face,"material":p.material}),
                )
            })
            .collect();
        reports.push(json!({"fixture":f.name,"sites":sites,
            "baseline_windows":baseline.iter().map(|b| &b.0).collect::<Vec<_>>(),
            "baseline_replica_difference":difference(&baseline[0].0,&baseline[1].0),
            "analytic":f.analytic.map(|v| v.into_iter().map(channels).collect::<Vec<_>>()),
            "runs":runs}));
        let data = json!({"schema_version":1,"paths_per_window":paths,"windows":WINDOWS,
            "baseline_model":TransportSettings::default(),"fixtures":reports,
            "note":"Fully resident fixed geometry; production kernel, paired sample windows. Sparse receiver sites, no display reconstruction/filtering, no interactive FPS claim. Reach is per leg, directional visibility and conservative sky frontier remain independent. Matched differences screen model bias but do not certify visual equivalence; compare replica noise and confirm boundary candidates with more samples and images."});
        std::fs::write(&output, serde_json::to_vec(&data).unwrap()).unwrap();
    }
    assert!(!reports.is_empty(), "no fixtures selected");
}
