//! Fixed-work emission-estimator comparison, separate from history policies.
use super::*;
use crate::voxel_gi::research::{cave, channels};

fn values(work: &TraceWork) -> Vec<EnergySpectrum> {
    work.faces.iter().map(|f| f.radiance).collect()
}

fn percentile(mut values: Vec<f32>, fraction: f32) -> f32 {
    values.sort_by(f32::total_cmp);
    values[((values.len() - 1) as f32 * fraction).round() as usize]
}

fn error(values: &[EnergySpectrum], reference: &[EnergySpectrum]) -> f32 {
    percentile(
        values
            .iter()
            .zip(reference)
            .flat_map(|(a, b)| {
                channels(*a)
                    .into_iter()
                    .zip(channels(*b))
                    .map(|(a, b)| (a - b).abs() / (0.02 + b.abs()))
            })
            .collect(),
        0.95,
    )
}

#[test]
#[ignore = "native terminal-mixture equal-path/reference/throughput study"]
fn terminal_mixture_native_study() {
    let cave = cave();
    let mut candidate = super::tests::native_tracer();
    let source =
        std::fs::read_to_string(std::env::var("VOXEL_GI_BASELINE_SHADER").unwrap()).unwrap();
    let mut baseline = GpuTracer::with_shader(candidate.context.clone(), true, &source).unwrap();
    let emitters = Arc::new(EmissionDistribution::build(&cave.scene, &cave.materials));
    let mut cpu = TraceWork::from_probes(&cave.receivers[..8], false);
    let mut native = TraceWork::from_probes(&cave.receivers[..8], false);
    crate::voxel_gi::tracer::trace_work_batch(
        &cave.scene,
        &cave.sources,
        &cave.materials,
        &mut cpu,
        8192,
        &emitters,
        &mut default(),
    );
    candidate
        .trace_compact(
            cave.scene.clone(),
            &cave.sources,
            &cave.materials,
            &mut native,
            8192,
            &emitters,
            false,
        )
        .unwrap();
    for (a, b) in values(&cpu).into_iter().zip(values(&native)) {
        for (a, b) in channels(a).into_iter().zip(channels(b)) {
            assert!(
                (a - b).abs() < 0.005 * (0.02 + a.abs()),
                "CPU/GPU {a} vs {b}"
            );
        }
    }
    let mut references = Vec::new();
    for (mode, tracer) in [&mut baseline, &mut candidate].into_iter().enumerate() {
        let mut work = TraceWork::from_probes(&cave.receivers, false);
        for face in &mut work.faces {
            face.sequence = 4_194_304 + mode as u32 * 1_048_576;
        }
        for _ in 0..32 {
            tracer
                .trace_compact(
                    cave.scene.clone(),
                    &cave.sources,
                    &cave.materials,
                    &mut work,
                    1024,
                    &emitters,
                    false,
                )
                .unwrap();
        }
        references.push(values(&work));
    }
    let reference: Vec<_> = references[0]
        .iter()
        .zip(&references[1])
        .map(|(a, b)| (*a + *b) * 0.5)
        .collect();
    let disagreement = error(&references[0], &references[1]);
    assert!(
        disagreement < 0.06,
        "high-sample estimators disagree: {disagreement}"
    );
    eprintln!("TERMINAL_REFERENCE all_band_disagreement_p95={disagreement}");
    for (mode, tracer) in [&mut baseline, &mut candidate].into_iter().enumerate() {
        let mut work = TraceWork::from_probes(&cave.receivers, false);
        for batch in 1..=64 {
            tracer
                .trace_compact(
                    cave.scene.clone(),
                    &cave.sources,
                    &cave.materials,
                    &mut work,
                    16,
                    &emitters,
                    false,
                )
                .unwrap();
            if [4, 16, 64].contains(&batch) {
                eprintln!(
                    "TERMINAL_ERROR mode={mode} paths={} all_band_p95={}",
                    batch * 16,
                    error(&values(&work), &reference)
                );
            }
        }
    }
    let probes: Vec<_> = cave.receivers.iter().cycle().take(1024).cloned().collect();
    let mut work = TraceWork::from_probes(&probes, false);
    let mut times: [Vec<f32>; 2] = default();
    for round in 0..40 {
        for mode in [round % 2, 1 - round % 2] {
            for face in &mut work.faces {
                face.samples = 0;
                face.sequence = round as u32 * 16;
            }
            let tracer = if mode == 0 {
                &mut baseline
            } else {
                &mut candidate
            };
            tracer
                .trace_compact(
                    cave.scene.clone(),
                    &cave.sources,
                    &cave.materials,
                    &mut work,
                    16,
                    &emitters,
                    false,
                )
                .unwrap();
            if round >= 8 {
                times[mode].push(tracer.last_gpu_ms.expect("native GPU timestamp"));
            }
        }
    }
    eprintln!(
        "TERMINAL_GPU_MS baseline={} mixture={}",
        percentile(times[0].clone(), 0.5),
        percentile(times[1].clone(), 0.5)
    );
}
