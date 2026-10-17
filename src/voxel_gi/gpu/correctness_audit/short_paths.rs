//! Measure early roulette's variance/cost tradeoff before changing gameplay.
//! Raw matched prefixes only; sparse field errors are not image qualification.
use super::*;
use serde_json::json;

fn work(fixture: &crate::voxel_gi::research::ResearchFixture, start: u32) -> TraceWork {
    let mut work = TraceWork::default();
    for probe in &fixture.receivers {
        work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
        work.probes.last_mut().unwrap().sky_requested = false;
        work.faces.last_mut().unwrap().sequence = start;
    }
    work
}

#[test]
#[ignore = "native paired short-path roulette variance and dispatch cost study"]
fn short_path_roulette_study() {
    let native = tests::native_tracer();
    let mut tracers = [
        (
            "roulette",
            GpuTracer::with_shader(native.context.clone(), true, &shader(false, true, false))
                .unwrap(),
        ),
        (
            "no_roulette",
            GpuTracer::with_shader(native.context.clone(), true, &shader(false, false, false))
                .unwrap(),
        ),
    ];
    let mut rows = Vec::new();
    for (name, fixture) in [
        (
            "hallway_loaded_full",
            crate::voxel_gi::research::hallway_loaded(),
        ),
        (
            "cave_materials",
            super::field::cave_with_material_receivers(),
        ),
    ] {
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
        ));
        let sites: Vec<_> = (0..fixture.receivers.len()).collect();
        // Disjoint block-averaged references. Both arms use the same reference;
        // these deterministic QMC windows are not independent confidence bounds.
        let references: Vec<_> = [4_194_304, 6_291_456]
            .map(|start| {
                estimates(
                    &mut tracers[0].1,
                    &fixture,
                    &emitters,
                    default(),
                    false,
                    &sites,
                    262144,
                    start,
                )
            })
            .into();
        rows.push(json!({"fixture":name, "references":references,
            "sites":fixture.receivers.iter().map(|p| json!({"voxel":p.voxel.to_array(), "mask":p.face_mask})).collect::<Vec<_>>()}));
        for start in [0, 574_117, 1_853_903, 3_102_719] {
            for (variant, gpu) in &mut tracers {
                let mut work = work(&fixture, start);
                let mut completed = 0;
                for paths in [16, 64, 128, 256, 512, 1024, 2048, 2256, 4096, 8192] {
                    while completed < paths {
                        let batch = (paths - completed).min(128);
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
                    }
                    rows.push(json!({"fixture":name, "variant":variant, "start":start, "paths":paths,
                        "means":work.faces.iter().map(|f| channels(f.radiance)).collect::<Vec<_>>(),
                        "seconds":work.faces.iter().map(|f| channels(f.second_moment)).collect::<Vec<_>>()}));
                }
            }
        }
        // Warm both pipelines, then alternate order. Same 2048 faces x16 paths,
        // with every prefix reset; no app/compiler should overlap this test.
        let mut dispatch = TraceWork::default();
        for probe in fixture.receivers.iter().cycle().take(2048) {
            dispatch.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
            dispatch.probes.last_mut().unwrap().sky_requested = false;
        }
        for round in 0..26 {
            for i in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let (variant, gpu) = &mut tracers[i];
                for face in &mut dispatch.faces {
                    face.samples = 0;
                    face.sequence = round * 16;
                    face.radiance = EnergySpectrum::ZERO;
                    face.second_moment = EnergySpectrum::ZERO;
                }
                gpu.trace_compact(
                    fixture.scene.clone(),
                    &fixture.sources,
                    &fixture.materials,
                    &mut dispatch,
                    16,
                    &emitters,
                    false,
                )
                .unwrap();
                if round >= 2 {
                    rows.push(json!({"fixture":name,"variant":variant,"round":round,"gpu_ms":gpu.last_gpu_ms}));
                }
            }
        }
        eprintln!("SHORT_PATHS {name} complete");
    }
    std::fs::write(
        "target/short-path-roulette-study.json",
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
}
