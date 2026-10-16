//! Estimator correctness, separate from display quality and throughput. The
//! independent CPU oracle uses cosine-only SplitMix64 paths and no roulette.
//! GPU ablations modify only test shaders; none is a gameplay fallback.
use super::*;
use crate::voxel_gi::research::channels;
use crate::voxel_gi::{TransportSettings, validation::reference_oracle};

fn replace_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "audit shader marker: {from}"
    );
    source.replacen(from, to, 1)
}

fn shader(iid: bool, roulette: bool, cutoff: bool) -> String {
    let mut shader = replace_once(
        SHADER,
        "fn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {",
        "fn production_sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {",
    );
    shader.push_str(if iid {
        // Counter-keyed pseudorandom control: no Halton/Sobol mapping or shared
        // sample dimensions. Distinct sequence windows supply distinct counters.
        "\nfn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {
            let word = pcg(pcg(sequence ^ 0x93d765ddu) ^ pcg(stream ^ 0x6c8e9cf5u) ^ pcg(dimension ^ 0xb5297a4du));
            return f32(word >> 8u) / 16777216.0;
        }\n"
    } else {
        "\nfn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {
            return production_sample_1d(stream, sequence, dimension);
        }\n"
    });
    if !roulette {
        shader = replace_once(&shader, "if bounce >= 1u {", "if false {");
    }
    if !cutoff {
        shader = replace_once(
            &shader,
            "if maximum < 0.002 { break; }",
            "// Audit: no small-throughput truncation.",
        );
    }
    shader
}

#[test]
fn correctness_audit_shader_variants_validate() {
    for iid in [false, true] {
        for roulette in [false, true] {
            let source = shader(iid, roulette, roulette);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
}

#[test]
#[ignore = "native analytic furnace checks all bands and finite path depths"]
fn native_furnace_matches_geometric_series() {
    let fixture = crate::voxel_gi::research::furnace();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let mut gpu = tests::native_tracer();
    let material = fixture.materials.get(4);
    for bounces in 1..=3 {
        let mut expected = material.emission;
        let mut term = material.emission;
        for _ in 0..bounces {
            term = term * material.diffuse;
            expected += term;
        }
        for guided in [false, true] {
            let mut work = TraceWork {
                transport: TransportSettings {
                    bounces,
                    ..default()
                },
                emitter_guiding: guided,
                ..default()
            };
            for probe in &fixture.receivers {
                work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
                work.probes.last_mut().unwrap().sky_requested = false;
            }
            for _ in 0..64 {
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
            for face in work.faces {
                for (a, e) in channels(face.radiance).into_iter().zip(channels(expected)) {
                    eprintln!(
                        "FURNACE depth={bounces} guided={guided} voxel={:?} measured={a} expected={e}",
                        face.voxel
                    );
                    assert!((a / e - 1.0).abs() < 0.015, "analytic furnace: {a} != {e}");
                }
            }
        }
    }
}

#[test]
#[ignore = "native raw estimator audit against independent CPU cosine paths"]
fn hallway_estimator_correctness_audit() {
    let paths: u32 = std::env::var("VOXEL_GI_AUDIT_PATHS")
        .map(|v| v.parse().unwrap())
        .unwrap_or(262144);
    assert!(paths >= 16384 && paths.is_multiple_of(16384));
    let oracle_paths: u32 = std::env::var("VOXEL_GI_AUDIT_ORACLE_PATHS")
        .map(|v| v.parse().unwrap())
        .unwrap_or(paths);
    let fixture = crate::voxel_gi::research::hallway_loaded();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let native = tests::native_tracer();
    let mut variants: Vec<_> = [
        ("production", false, true, true, false),
        ("iid", true, true, true, false),
        ("iid_no_roulette", true, false, false, false),
        ("guided_iid", true, true, true, true),
        ("guided_production", false, true, true, true),
        ("guided_iid_no_cutoff", true, true, false, true),
        ("guided_iid_no_roulette", true, false, false, true),
    ]
    .into_iter()
    .map(|(name, iid, roulette, cutoff, guided)| {
        let tracer =
            GpuTracer::with_shader(native.context.clone(), true, &shader(iid, roulette, cutoff))
                .unwrap();
        (name, guided, tracer)
    })
    .collect();
    let mut rows = Vec::new();
    for bounces in 1..=3 {
        let transport = TransportSettings {
            bounces,
            ..default()
        };
        for site in [193, 208] {
            if oracle_paths == 0 {
                continue;
            }
            let probe = &fixture.receivers[site];
            for replica in 0..2 {
                let started = Instant::now();
                let oracle = reference_oracle::estimate_seeded(
                    &fixture.scene,
                    &fixture.materials,
                    &fixture.sources,
                    transport,
                    probe.voxel,
                    2,
                    oracle_paths,
                    0x9e3779b97f4a7c15u64.wrapping_mul(replica + 1),
                );
                eprintln!(
                    "ORACLE depth={bounces} site={site} replica={replica} red={} se={} seconds={:.2}",
                    oracle.mean[0],
                    oracle.standard_error[0],
                    started.elapsed().as_secs_f64()
                );
                rows.push(
                    serde_json::json!({"variant":"oracle", "bounces":bounces, "site":site,
                    "replica":replica, "estimate":oracle}),
                );
            }
        }
        for (name, guided, gpu) in &mut variants {
            for replica in 0..2 {
                let mut sums = [[0.0; 5]; 2];
                let mut squares = [[0.0; 5]; 2];
                // Exact block averages: never feed a >65536-path run through
                // the runtime's capped-history exponential moving average.
                for first_block in (0..paths / 16384).step_by(64) {
                    let mut work = TraceWork {
                        transport,
                        emitter_guiding: *guided,
                        ..default()
                    };
                    for block in first_block..(first_block + 64).min(paths / 16384) {
                        for site in [193, 208] {
                            let probe = &fixture.receivers[site];
                            work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
                            work.probes.last_mut().unwrap().sky_requested = false;
                            work.faces.last_mut().unwrap().sequence =
                                4194304 + replica * paths + block * 16384;
                        }
                    }
                    for _ in 0..16 {
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
                    for (index, face) in work.faces.iter().enumerate() {
                        let i = index % 2;
                        for b in 0..5 {
                            sums[i][b] +=
                                f64::from(channels(face.radiance)[b]) * 16384.0 / f64::from(paths);
                            squares[i][b] += f64::from(channels(face.second_moment)[b]) * 16384.0
                                / f64::from(paths);
                        }
                    }
                }
                for (i, site) in [193, 208].into_iter().enumerate() {
                    let se: [f64; 5] = std::array::from_fn(|b| {
                        ((squares[i][b] - sums[i][b].powi(2)).max(0.0) / f64::from(paths - 1))
                            .sqrt()
                    });
                    eprintln!(
                        "GPU {name} depth={bounces} site={site} replica={replica} red={} sample_se={}",
                        sums[i][0], se[0]
                    );
                    rows.push(serde_json::json!({"variant":name,"bounces":bounces,"site":site,"replica":replica,
                        "estimate":{"paths":paths,"mean":sums[i],"standard_error":se},
                        "note":"QMC sample standard error is diagnostic, not an IID confidence interval"}));
                }
            }
        }
        std::fs::write(
            "target/hallway-correctness-audit.json",
            serde_json::to_vec_pretty(&rows).unwrap(),
        )
        .unwrap();
    }
    // Evidence collection, not a pass/fail accuracy certificate: heavy tails
    // require checking replica consistency and oracle uncertainty explicitly.
}
