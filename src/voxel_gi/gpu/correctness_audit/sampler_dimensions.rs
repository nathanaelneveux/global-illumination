//! Compare path-dimension samplers on matched raw prefixes. Production rotated
//! Halton correlates high prime bases (roulette vs terminal choice) at small
//! counts; padded Owen-scrambled Sobol pairs and an IID counter control share
//! the same estimator, fixtures and references. Sparse-field metrics only.
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

fn all_sobol_shader() -> String {
    replace_once(
        &shader(false, true, false),
        "if dimension < 12u {",
        "if false && dimension < 12u {",
    )
}

#[test]
fn sampler_dimension_study_shaders_validate() {
    let source = all_sobol_shader();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "native matched raw-prefix comparison of path-dimension samplers"]
fn sampler_dimension_study() {
    let native = tests::native_tracer();
    let context = native.context.clone();
    let mut tracers = [
        (
            "halton",
            GpuTracer::with_shader(context.clone(), true, &shader(false, true, false)).unwrap(),
        ),
        (
            "sobol",
            GpuTracer::with_shader(context.clone(), true, &all_sobol_shader()).unwrap(),
        ),
        (
            "iid",
            GpuTracer::with_shader(context.clone(), true, &shader(true, true, false)).unwrap(),
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
        // The IID control supplies references independent of both QMC arms.
        let references: Vec<_> = [8_388_608, 12_582_912]
            .map(|start| {
                estimates(
                    &mut tracers[2].1,
                    &fixture,
                    &emitters,
                    default(),
                    &sites,
                    262144,
                    start,
                )
            })
            .into();
        rows.push(json!({"fixture":name, "references":references}));
        for start in [0, 574_117, 1_853_903, 3_102_719, 4_000_037, 5_123_457] {
            for (variant, gpu) in &mut tracers {
                let mut work = work(&fixture, start);
                let mut completed = 0;
                for paths in [16, 32, 64, 128, 256, 512, 1024, 2048] {
                    while completed < paths {
                        let batch = (paths - completed).min(16);
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
                        "means":work.faces.iter().map(|f| channels(f.radiance)).collect::<Vec<_>>()}));
                }
            }
        }
        eprintln!("SAMPLER_DIMENSIONS {name} complete");
    }
    std::fs::write(
        "target/sampler-dimension-study.json",
        serde_json::to_vec(&rows).unwrap(),
    )
    .unwrap();
}
