//! Raw matched-prefix qualification of conditional emitter-neighborhood guiding.
//! No spatial/temporal display and no history resets inside a measured prefix.
use super::*;

fn work(fixture: &super::super::research::ResearchFixture, start: u32) -> TraceWork {
    let mut work = TraceWork::default();
    for probe in &fixture.receivers {
        work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
        work.probes.last_mut().unwrap().sky_requested = false;
        work.faces.last_mut().unwrap().sequence = start;
    }
    work
}

#[test]
#[ignore = "native raw hallway/cave sampler qualification"]
fn conditional_guiding_prefix_study() {
    let reference_path = "target/conditional-guiding-prefix-study.json";
    let prior: Vec<serde_json::Value> = std::fs::read(reference_path).ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default();
    let control = tests::native_tracer();
    let mut tracers = vec![(0, control)];
    for radius in [2, 4, 8, 16] {
        let shader = SHADER.replace(
            "override GUIDE_RADIUS: f32 = 0.0;",
            &format!("override GUIDE_RADIUS: f32 = {radius}.0;"),
        );
        let tracer = GpuTracer::with_shader(tracers[0].1.context.clone(), true, &shader).unwrap();
        tracers.push((radius, tracer));
    }
    let mut rows = Vec::new();
    for (name, fixture) in [
        ("hallway", super::super::research::hallway()),
        ("cave", super::super::research::cave()),
    ] {
        let importance = super::super::VoxelGiSettings::default().sampling_importance();
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            importance,
        ));
        for (radius, gpu) in &mut tracers {
            for start in [0, 574117, 1853903, 3102719] {
                let mut work = work(&fixture, start);
                let mut completed = 0;
                let mut gpu_ms = Some(0.0);
                let started = Instant::now();
                for count in [128, 512, 2048, 4096, 8192] {
                    while completed < count {
                        let batch = (count - completed).min(128);
                        gpu.trace_compact(
                            fixture.scene.clone(),
                            &fixture.sources,
                            &fixture.materials,
                            &mut work,
                            batch,
                            &emitters,
                            false,
                        )
                        .unwrap();
                        completed += batch;
                        gpu_ms = gpu_ms.zip(gpu.last_gpu_ms).map(|(sum, ms)| sum + ms);
                    }
                    rows.push(serde_json::json!({"fixture":name, "radius":radius, "start":start, "paths":count, "gpu_ms":gpu_ms, "worker_ms":started.elapsed().as_secs_f64()*1000.,
                        "means":work.faces.iter().map(|f| super::super::research::channels(f.radiance)).collect::<Vec<_>>()}));
                }
            }
            eprintln!("completed {name} radius={radius}");
        }
        // Reference uses the unchanged control and disjoint windows. Block
        // averaging avoids the finite history cap turning this into an EMA.
        for start in [4194304, 5242880] {
            if let Some(reference) = prior.iter().find(|r| r["fixture"] == name && r["start"] == start && r["paths"] == 1048576) {
                rows.push(reference.clone());
                continue;
            }
            let gpu = &mut tracers[0].1;
            let mut mean = vec![[0.0_f64; 5]; fixture.receivers.len()];
            for block in 0..32 {
                let mut work = work(&fixture, start + block * 32768);
                for _ in 0..32 {
                    gpu.trace_compact(
                        fixture.scene.clone(),
                        &fixture.sources,
                        &fixture.materials,
                        &mut work,
                        1024,
                        &emitters,
                        false,
                    )
                    .unwrap();
                }
                for (sum, f) in mean.iter_mut().zip(&work.faces) {
                    for (sum, v) in sum
                        .iter_mut()
                        .zip(super::super::research::channels(f.radiance))
                    {
                        *sum += f64::from(v) / 32.;
                    }
                }
            }
            rows.push(serde_json::json!({"fixture":name, "radius":0,"start":start,"paths":1048576,"means":mean}));
            eprintln!("reference {name} start={start}");
        }
    }
    let path = if prior.is_empty() { reference_path } else { "target/conditional-guiding-primary-study.json" };
    std::fs::write(path, serde_json::to_vec(&rows).unwrap()).unwrap();
    println!("raw study written to {path}");
}
