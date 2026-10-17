//! Estimator correctness, separate from display quality and throughput. The
//! independent CPU oracle uses cosine-only SplitMix64 paths and no roulette.
//! GPU ablations modify only test shaders; none is a gameplay fallback.
use super::*;
use crate::voxel_gi::research::channels;
use crate::voxel_gi::{TransportSettings, validation::reference_oracle};

mod field;
mod recovery;
mod short_paths;

/// Raw block-averaged estimates. A duplicated receiver represents a disjoint
/// sequence block, not borrowed evidence. Runtime history never exceeds 16384.
fn estimates(
    gpu: &mut GpuTracer,
    fixture: &crate::voxel_gi::research::ResearchFixture,
    emitters: &Arc<EmissionDistribution>,
    transport: TransportSettings,
    guided: bool,
    sites: &[usize],
    paths: u32,
    start: u32,
) -> Vec<reference_oracle::Estimate> {
    assert!(!sites.is_empty() && paths > 0 && paths.is_multiple_of(16384));
    let mut sums = vec![[0.0; 5]; sites.len()];
    let mut squares = sums.clone();
    let blocks_per_batch = (4096 / sites.len()).clamp(1, 64);
    for first_block in (0..paths / 16384).step_by(blocks_per_batch) {
        let mut work = TraceWork {
            transport,
            emitter_guiding: guided,
            ..default()
        };
        for block in first_block..(first_block + blocks_per_batch as u32).min(paths / 16384) {
            for &site in sites {
                let probe = &fixture.receivers[site];
                work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
                work.probes.last_mut().unwrap().sky_requested = false;
                work.faces.last_mut().unwrap().sequence = start + block * 16384;
            }
        }
        for _ in 0..16 {
            gpu.trace_compact(
                fixture.scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut work,
                1024,
                emitters,
                false,
            )
            .unwrap();
        }
        for (index, face) in work.faces.iter().enumerate() {
            assert_eq!(face.samples, 16384);
            let i = index % sites.len();
            for b in 0..5 {
                sums[i][b] += f64::from(channels(face.radiance)[b]) * 16384.0 / f64::from(paths);
                squares[i][b] +=
                    f64::from(channels(face.second_moment)[b]) * 16384.0 / f64::from(paths);
            }
        }
    }
    sums.into_iter()
        .zip(squares)
        .map(|(mean, square)| reference_oracle::Estimate {
            paths,
            mean,
            standard_error: std::array::from_fn(|b| {
                ((square[b] - mean[b].powi(2)).max(0.0) / f64::from(paths - 1)).sqrt()
            }),
        })
        .collect()
}

fn replace_once(source: &str, from: &str, to: &str) -> String {
    assert_eq!(
        source.matches(from).count(),
        1,
        "audit shader marker: {from}"
    );
    source.replacen(from, to, 1)
}

fn shader(iid: bool, roulette: bool, legacy_cutoff: bool) -> String {
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
    if legacy_cutoff {
        shader = replace_once(
            &shader,
            "if maximum == 0.0 { break; }",
            "if maximum < 0.002 { break; }",
        );
    }
    shader
}

fn legacy_visibility_shader(source: &str) -> String {
    replace_once(
        source,
        "if trace(shadow_origin, shadow_delta / shadow_distance, shadow_distance).status != 0u { return result; }",
        "if trace(position + normal * 0.002, direction, distance - 0.008).status != 0u { return result; }",
    )
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
#[ignore = "native exact-path regression for nonzero throughput termination"]
fn native_low_throughput_path_keeps_bright_suffix() {
    let fixture = crate::voxel_gi::research::absorbing_corner();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let native = tests::native_tracer();
    let mut source = replace_once(
        SHADER,
        "fn trace_receiver(surface: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32, anchor: GuideAnchor) -> Energy {",
        "fn production_trace_receiver(surface: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32, anchor: GuideAnchor) -> Energy {",
    );
    // Replay the analytically checked CPU witness, including its roulette coin.
    // Only its initial direction is forced; all suffix sampling is production.
    source.push_str("\nfn trace_receiver(surface: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32, anchor: GuideAnchor) -> Energy {
        let direction = normalize(vec3(-1.0, 1.0, 0.0));
        return trace_path_record(surface + normal * 0.001, direction, normal, 123u, 19u,
            direction.y / 3.14159265359, false, anchor).suffix;
    }\n");
    for cutoff in [false, true] {
        let shader = if cutoff {
            replace_once(
                &source,
                "if maximum == 0.0 { break; }",
                "if maximum < 0.002 { break; }",
            )
        } else {
            source.clone()
        };
        let mut gpu = GpuTracer::with_shader(native.context.clone(), true, &shader).unwrap();
        let mut work = TraceWork::default();
        work.push(&fixture.receivers[0], 4, 0, 0, false, u32::MAX, u32::MAX);
        work.probes[0].sky_requested = false;
        gpu.trace_compact(
            fixture.scene.clone(),
            &fixture.sources,
            &fixture.materials,
            &mut work,
            1,
            &emitters,
            false,
        )
        .unwrap();
        let expected = if cutoff { 0.0 } else { 9.490358 };
        for value in channels(work.faces[0].radiance) {
            assert!(
                (value - expected).abs() < 0.001,
                "legacy_cutoff={cutoff}: {value} != {expected}"
            );
        }
    }
}

#[test]
#[ignore = "paired native dispatch cost; run without another GI app or compiler"]
fn cutoff_removal_dispatch_cost() {
    let native = tests::native_tracer();
    let variants: Vec<_> = [
        ("unguided", false, false),
        ("legacy_guided", true, true),
        ("corrected_guided", true, false),
    ]
    .into_iter()
    .map(|(name, guided, cutoff)| {
        (
            name,
            guided,
            GpuTracer::with_shader(native.context.clone(), true, &shader(false, true, cutoff))
                .unwrap(),
        )
    })
    .collect();
    dispatch_cost(variants, "target/cutoff-removal-dispatch-cost.json");
}

#[test]
#[ignore = "paired endpoint-visibility GPU cost; run without other GPU work or compilation"]
fn endpoint_visibility_dispatch_cost() {
    let native = tests::native_tracer();
    let variants = [
        ("legacy-endpoints", legacy_visibility_shader(SHADER)),
        ("corrected-endpoints", SHADER.to_owned()),
    ]
    .into_iter()
    .map(|(name, source)| {
        (
            name,
            true,
            GpuTracer::with_shader(native.context.clone(), true, &source).unwrap(),
        )
    })
    .collect();
    dispatch_cost(variants, "target/endpoint-visibility-dispatch-cost.json");
}

fn dispatch_cost(mut variants: Vec<(&str, bool, GpuTracer)>, output: &str) {
    let mut rows = Vec::new();
    for (name, fixture) in [
        ("hallway", crate::voxel_gi::research::hallway_loaded()),
        ("cave", crate::voxel_gi::research::cave()),
    ] {
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
        ));
        let mut work = TraceWork::default();
        for probe in fixture.receivers.iter().cycle().take(2048) {
            work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
            work.probes.last_mut().unwrap().sky_requested = false;
        }
        for round in 0..26 {
            for offset in 0..variants.len() {
                let index = (offset + round as usize) % variants.len();
                let (variant, guided, gpu) = &mut variants[index];
                work.emitter_guiding = *guided;
                for face in &mut work.faces {
                    face.sequence = round * 16;
                }
                let started = Instant::now();
                gpu.trace_compact(
                    fixture.scene.clone(),
                    &fixture.sources,
                    &fixture.materials,
                    &mut work,
                    16,
                    &emitters,
                    false,
                )
                .unwrap();
                if round >= 2 {
                    rows.push(
                        serde_json::json!({"fixture":name,"variant":variant,"round":round,
                        "gpu_ms":gpu.last_gpu_ms.expect("GPU timestamps for cost audit"),
                        "worker_ms":started.elapsed().as_secs_f64()*1000.0}),
                    );
                }
            }
        }
    }
    std::fs::write(output, serde_json::to_vec_pretty(&rows).unwrap()).unwrap();
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
        ("production", false, true, false, false),
        ("iid", true, true, false, false),
        ("iid_no_roulette", true, false, false, false),
        ("guided_iid", true, true, false, true),
        ("guided_production", false, true, false, true),
        ("legacy_guided_iid_cutoff", true, true, true, true),
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
