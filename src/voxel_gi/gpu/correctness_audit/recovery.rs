//! Diagnose persistent outliers without hiding them in a long-run mean. Raw
//! single-path outputs separate sequence/weight tails from accumulation; a
//! resident replay checks that the same tail receives exactly its batch weight.
use super::*;
use serde_json::json;

const SITES: [usize; 2] = [193, 208];
const PREFIXES: [usize; 16] = [
    16, 64, 256, 512, 1024, 2048, 2240, 2256, 4096, 8192, 16384, 32768, 49152, 65536, 98304, 131072,
];

fn raw_paths(
    gpu: &mut GpuTracer,
    fixture: &crate::voxel_gi::research::ResearchFixture,
    emitters: &Arc<EmissionDistribution>,
    start: u32,
    paths: u32,
) -> [Vec<[f32; 5]>; 2] {
    let mut outputs = std::array::from_fn(|_| Vec::with_capacity(paths as usize));
    for first in (0..paths).step_by(8192) {
        let mut work = TraceWork::default();
        for sequence in start + first..start + (first + 8192).min(paths) {
            for site in SITES {
                let probe = &fixture.receivers[site];
                work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
                work.probes.last_mut().unwrap().sky_requested = false;
                work.faces.last_mut().unwrap().sequence = sequence;
            }
        }
        gpu.trace_compact(
            fixture.scene.clone(),
            &fixture.sources,
            &fixture.materials,
            &mut work,
            1,
            emitters,
            false,
        )
        .unwrap();
        for (i, face) in work.faces.iter().enumerate() {
            assert_eq!(face.samples, 1);
            assert_eq!(face.sequence, start + first + i as u32 / 2 + 1);
            let value = channels(face.radiance);
            assert!(value.into_iter().all(|v| v.is_finite() && v >= 0.0));
            outputs[i % 2].push(value);
        }
    }
    outputs
}

fn summarize(samples: &[[f32; 5]], start: u32) -> serde_json::Value {
    let mut sum = [0.0; 5];
    let mut square = [0.0; 5];
    let mut max_red = 0.0_f64;
    let mut rows = Vec::new();
    for (i, sample) in samples.iter().enumerate() {
        for b in 0..5 {
            sum[b] += f64::from(sample[b]);
            square[b] += f64::from(sample[b]).powi(2);
        }
        max_red = max_red.max(f64::from(sample[0]));
        if PREFIXES.contains(&(i + 1)) {
            let n = (i + 1) as f64;
            let mean = sum.map(|v| v / n);
            let standard_error: [f64; 5] =
                std::array::from_fn(|b| ((square[b] / n - mean[b] * mean[b]).max(0.0) / n).sqrt());
            rows.push(
                json!({"paths":i+1, "mean":mean, "standard_error_diagnostic":standard_error,
                "max_red":max_red, "max_red_share":max_red/sum[0].max(1e-30)}),
            );
        }
    }
    let mut sorted: Vec<_> = samples.iter().enumerate().collect();
    sorted.sort_unstable_by(|a, b| b.1[0].total_cmp(&a.1[0]));
    let top: Vec<_> = sorted
        .iter()
        .take(16)
        .map(|(i, v)| json!({"sequence":start + *i as u32, "value":v}))
        .collect();
    let tail_counts: Vec<_> = [0.001, 0.01, 0.1, 0.5, 1.0].map(|threshold| {
        json!({"red_above":threshold, "count":samples.iter().filter(|v|v[0]>threshold).count()})
    }).into();
    json!({"prefixes":rows,"top_paths":top,"tail_counts":tail_counts})
}

#[test]
#[ignore = "native raw hallway prefix/tail audit, independent RNG and roulette ablations"]
fn hallway_raw_prefix_recovery_audit() {
    let fixture = crate::voxel_gi::research::hallway_loaded();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let native = tests::native_tracer();
    let mut rows = Vec::new();
    for (name, iid, roulette) in [
        ("legacy_qmc_roulette", false, true),
        ("iid", true, true),
        ("production", false, false),
        ("iid_no_roulette", true, false),
    ] {
        let mut gpu =
            GpuTracer::with_shader(native.context.clone(), true, &shader(iid, roulette, false))
                .unwrap();
        for replica in 0..4 {
            let start = replica * 1_048_576;
            let samples = raw_paths(&mut gpu, &fixture, &emitters, start, 131072);
            for (site, values) in SITES.into_iter().zip(samples) {
                let report = summarize(&values, start);
                eprintln!(
                    "RECOVERY {name} start={start} site={site} final={} top={}",
                    report["prefixes"].as_array().unwrap().last().unwrap(),
                    report["top_paths"][0]
                );
                rows.push(json!({"variant":name,"start":start,"site":site,"statistics":report}));
            }
        }
    }
    std::fs::write("target/hallway-prefix-recovery.json", serde_json::to_vec_pretty(&json!({
        "notes":"Raw paths; no guiding, reuse, history, adaptive scheduling or reconstruction. QMC windows are deterministic, not independent IID confidence intervals. Statistics describe tails, not a quality pass.",
        "rows":rows
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "native resident hallway outlier replay against individually measured paths"]
fn hallway_resident_outlier_weight_matches_raw_sum() {
    let fixture = crate::voxel_gi::research::hallway_loaded();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let mut raw = tests::native_tracer();
    let samples = raw_paths(&mut raw, &fixture, &emitters, 0, 8192);
    let mut resident = GpuTracer::new(raw.context.clone()).unwrap();
    resident.enable_resident().unwrap();
    let mut probes: Vec<_> = SITES.map(|site| fixture.receivers[site].clone()).into();
    let mut sums = [[0.0_f64; 5]; 2];
    let mut squares = sums;
    let mut rows = Vec::new();
    for first in (0..8192).step_by(16) {
        let mut work = TraceWork::from_probes(&probes, false);
        work.response_seconds = 0.075;
        for job in &mut work.probes {
            job.sky_requested = false;
        }
        resident
            .trace_compact(
                fixture.scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut work,
                16,
                &emitters,
                false,
            )
            .unwrap();
        for (i, face) in work.faces.iter().enumerate() {
            assert_eq!(face.samples, (first + 16) as u32);
            assert_eq!(face.sequence, (first + 16) as u32);
            for sample in &samples[i][first..first + 16] {
                for b in 0..5 {
                    sums[i][b] += f64::from(sample[b]);
                    squares[i][b] += f64::from(sample[b]).powi(2);
                }
            }
            for (name, measured, expected) in [
                ("mean", channels(face.radiance), sums[i]),
                ("second", channels(face.second_moment), squares[i]),
            ] {
                for (a, e) in measured.into_iter().zip(expected) {
                    let e = e / (first + 16) as f64;
                    assert!(
                        (f64::from(a) - e).abs() <= e.abs() * 0.0001 + 1e-10,
                        "{name} site={} count={}: {a} != {e}",
                        SITES[i],
                        first + 16
                    );
                }
            }
            if PREFIXES.contains(&(first + 16)) {
                rows.push(
                    json!({"site":SITES[i],"paths":first+16,"mean":channels(face.radiance),
                    "second_moment":channels(face.second_moment),"sequence":face.sequence}),
                );
            }
        }
        work.copy_measurements(&mut probes);
    }
    std::fs::write(
        "target/hallway-resident-recovery.json",
        serde_json::to_vec_pretty(&rows).unwrap(),
    )
    .unwrap();
}
