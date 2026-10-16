//! Wider frozen-field correctness gates. No display reconstruction, history
//! policy, or scheduling. IID comparisons use a conservative six-standard-error
//! diagnostic gate; QMC windows are recorded without pretending they are IID.
use super::*;
use crate::voxel_gi::{
    MaterialTransport,
    research::{self, ResearchFixture},
};
use rayon::prelude::*;

struct Case {
    name: String,
    fixture: ResearchFixture,
    sites: Vec<usize>,
    transport: TransportSettings,
    analytic: Option<[f64; 5]>,
}

fn cave_with_material_receivers() -> ResearchFixture {
    use crate::voxel_gi::{
        sampling::stream_id,
        world::{AIR, FACE_OFFSETS, SurfaceProbe},
    };
    let mut fixture = research::cave();
    let mut groups = std::collections::BTreeMap::<(u8, usize), Vec<(u32, IVec3)>>::new();
    for z in 4..60 {
        for y in 4..44 {
            for x in 4..60 {
                let voxel = IVec3::new(x, y, z);
                let Some(material) = fixture.scene.material_at(voxel).filter(|&m| m != AIR) else {
                    continue;
                };
                for (face, offset) in FACE_OFFSETS.iter().enumerate() {
                    let neighbor = voxel + *offset;
                    if (5..59).contains(&neighbor.x)
                        && (5..43).contains(&neighbor.y)
                        && (5..59).contains(&neighbor.z)
                        && fixture.scene.material_at(neighbor) == Some(AIR)
                    {
                        groups
                            .entry((material as u8, face))
                            .or_default()
                            .push((stream_id(voxel, face as u32), voxel));
                    }
                }
            }
        }
    }
    // Deterministic stratification across material/normal, adding floor, spire,
    // torch and crystal receivers missing from the original wall/ceiling grid.
    for ((material, face), mut candidates) in groups {
        candidates.sort_unstable_by_key(|p| p.0);
        for (_, voxel) in candidates.into_iter().take(2) {
            if fixture
                .receivers
                .iter()
                .any(|p| p.voxel == voxel && p.face_mask == 1 << face)
            {
                continue;
            }
            let mut p = SurfaceProbe::new(voxel, 1 << face);
            p.material = material;
            fixture.receivers.push(p);
        }
    }
    fixture
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    cases.push(Case {
        name: "grazing-emitter-plane".into(),
        fixture: research::grazing_emitter_plane(false),
        sites: vec![0],
        transport: default(),
        analytic: Some([research::grazing_emitter_view_factor(); 5]),
    });
    for (label, bounces, reach, stride) in [
        ("cave-field", 3, 768.0, 1),
        ("cave-depth1", 1, 768.0, 11),
        ("cave-reach16", 3, 16.0, 11),
        ("cave-reach64", 3, 64.0, 11),
    ] {
        let fixture = cave_with_material_receivers();
        let sites = (0..fixture.receivers.len()).step_by(stride).collect();
        cases.push(Case {
            name: label.into(),
            fixture,
            sites,
            transport: TransportSettings { bounces, reach },
            analytic: None,
        });
    }
    // Geometry does not change across these known-answer tests. Different
    // spectral albedos exercise roulette's shared survival probability without
    // letting RGB energy conceal lost nonvisual transport (or vice versa).
    for (label, diffuse, emission) in [
        (
            "black",
            EnergySpectrum::ZERO,
            EnergySpectrum::new(Vec3::new(0.0, 0.01, 10000.0), 100.0, 0.0),
        ),
        (
            "white",
            EnergySpectrum::ONE,
            EnergySpectrum::new(Vec3::new(0.001, 1.0, 1000.0), 0.01, 100.0),
        ),
        (
            "absorbing",
            EnergySpectrum::ONE * 0.01,
            EnergySpectrum::ONE * 10000.0,
        ),
        (
            "hidden-only",
            EnergySpectrum::new(Vec3::ZERO, 0.95, 0.05),
            EnergySpectrum::new(Vec3::ZERO, 100.0, 1000.0),
        ),
        (
            "spectral",
            EnergySpectrum::new(Vec3::new(0.0, 0.01, 0.99), 0.7, 0.3),
            EnergySpectrum::new(Vec3::new(10000.0, 1.0, 0.001), 0.01, 100.0),
        ),
    ] {
        let mut fixture = research::furnace();
        fixture
            .materials
            .set(4, MaterialTransport { diffuse, emission });
        let mut expected = emission;
        let mut term = emission;
        for _ in 0..3 {
            term = term * diffuse;
            expected += term;
        }
        let sites = (0..fixture.receivers.len()).collect();
        cases.push(Case {
            name: format!("furnace-{label}"),
            fixture,
            sites,
            transport: default(),
            analytic: Some(channels(expected).map(f64::from)),
        });
    }
    let mut point = research::absorbing_corner();
    point.materials.set(
        4,
        MaterialTransport {
            diffuse: EnergySpectrum::new(Vec3::new(0.55, 0.2, 0.99), 0.8, 0.04),
            emission: EnergySpectrum::ZERO,
        },
    );
    point.sources.environment = EnergySpectrum::ZERO;
    point
        .sources
        .points
        .push(crate::voxel_gi::sources::PointTransportLight {
            position: Vec3::new(6.0, 8.0, 6.0),
            intensity: EnergySpectrum::new(Vec3::new(2.0, 5.0, 20.0), 50.0, 100.0),
            range: 20.0,
            custom_visible: Vec3::ZERO,
        });
    cases.push(Case {
        name: "point-corner".into(),
        sites: vec![0],
        fixture: point,
        transport: default(),
        analytic: None,
    });
    cases
}

/// Pool independent equally sized estimates, retaining within-path variance.
/// This does not manufacture independent evidence from the QMC variant.
fn pooled(replicas: &[Vec<reference_oracle::Estimate>], site: usize) -> ([f64; 5], [f64; 5]) {
    let n = replicas.len() as f64;
    let mean = std::array::from_fn(|b| replicas.iter().map(|r| r[site].mean[b]).sum::<f64>() / n);
    let se = std::array::from_fn(|b| {
        replicas
            .iter()
            .map(|r| r[site].standard_error[b].powi(2))
            .sum::<f64>()
            .sqrt()
            / n
    });
    (mean, se)
}

#[test]
#[ignore = "native full cave receiver and material-boundary estimator qualification"]
fn transport_field_correctness_gate() {
    let paths: u32 = std::env::var("VOXEL_GI_FIELD_PATHS")
        .map(|v| v.parse().unwrap())
        .unwrap_or(262144);
    assert!(paths >= 16384 && paths.is_multiple_of(16384));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let native = tests::native_tracer();
    let mut variants: Vec<_> = [
        ("iid", true, false),
        ("guided-iid", true, true),
        ("guided-qmc", false, true),
    ]
    .into_iter()
    .map(|(name, iid, guided)| {
        (
            name,
            guided,
            GpuTracer::with_shader(native.context.clone(), true, &shader(iid, true, false))
                .unwrap(),
        )
    })
    .collect();
    let run = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis();
    let output = format!("target/transport-field-{run}.json");
    let mut rows = Vec::new();
    let mut failures = Vec::new();
    for case in cases() {
        let started = Instant::now();
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &case.fixture.scene,
            &case.fixture.materials,
            crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
        ));
        let mut oracle = Vec::new();
        for replica in 0..2u64 {
            let estimates: Vec<_> = pool.install(|| {
                case.sites
                    .par_iter()
                    .map(|&site| {
                        let probe = &case.fixture.receivers[site];
                        reference_oracle::estimate_seeded(
                            &case.fixture.scene,
                            &case.fixture.materials,
                            &case.fixture.sources,
                            case.transport,
                            probe.voxel,
                            probe.face_mask.trailing_zeros() as usize,
                            paths,
                            (replica + 17).wrapping_mul(0x9e3779b97f4a7c15),
                        )
                    })
                    .collect()
            });
            oracle.push(estimates);
        }
        eprintln!(
            "FIELD {}: {} receivers, oracle {:.1}s",
            case.name,
            case.sites.len(),
            started.elapsed().as_secs_f64()
        );
        rows.push(serde_json::json!({"case":case.name,"variant":"oracle","transport":case.transport,
            "sites":case.sites,"receivers":case.sites.iter().map(|&s| {
                let p=&case.fixture.receivers[s]; serde_json::json!({"voxel":p.voxel.to_array(),"face":p.face_mask.trailing_zeros()})
            }).collect::<Vec<_>>(), "replicas":oracle,"analytic":case.analytic}));
        for (name, guided, gpu) in &mut variants {
            let replicas: Vec<_> = (0..2)
                .map(|r| {
                    estimates(
                        gpu,
                        &case.fixture,
                        &emitters,
                        case.transport,
                        *guided,
                        &case.sites,
                        paths,
                        4194304 + r * paths,
                    )
                })
                .collect();
            let mut max_z = 0.0f64;
            let mut max_relative = 0.0f64;
            for (i, &site) in case.sites.iter().enumerate() {
                let (reference, reference_se) = pooled(&oracle, i);
                let (actual, actual_se) = pooled(&replicas, i);
                for b in 0..5 {
                    let expected = case.analytic.map_or(reference[b], |a| a[b]);
                    let se = (actual_se[b].powi(2)
                        + if case.analytic.is_some() {
                            0.0
                        } else {
                            reference_se[b].powi(2)
                        })
                    .sqrt();
                    let delta = (actual[b] - expected).abs();
                    let roundoff = 0.0002 * expected.abs().max(actual[b].abs()) + 1e-10;
                    let z = (delta - roundoff).max(0.0) / se.max(1e-30);
                    max_z = max_z.max(z);
                    if expected != 0.0 {
                        max_relative = max_relative.max(delta / expected.abs());
                    }
                    // Sample SE is not an IID confidence interval for QMC.
                    // Analytic tests also require a fixed 1.5% error gate there.
                    let rejected = if *name == "guided-qmc" {
                        case.analytic.is_some() && delta > 0.015 * expected.abs() + 1e-10
                    } else {
                        z > 6.0
                    };
                    if rejected {
                        failures.push(
                            serde_json::json!({"case":case.name,"variant":name,"site":site,"band":b,
                        "expected":expected,"actual":actual[b],"combined_se":se,"z":z}),
                        );
                    }
                }
            }
            eprintln!(
                "FIELD {} {name}: max diagnostic z={max_z:.3}, relative={max_relative:.5}",
                case.name
            );
            rows.push(
                serde_json::json!({"case":case.name,"variant":name,"replicas":replicas,
                "max_diagnostic_z":max_z,"max_relative_error":max_relative}),
            );
        }
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(&serde_json::json!({"schema":1,"paths_per_replica":paths,
            "replicas":2,"rows":rows,"failures":failures,"passed":false}))
            .unwrap(),
        )
        .unwrap();
    }
    std::fs::write(
        &output,
        serde_json::to_vec_pretty(&serde_json::json!({"schema":1,"paths_per_replica":paths,
        "replicas":2,"rows":rows,"failures":failures,"passed":failures.is_empty()}))
        .unwrap(),
    )
    .unwrap();
    eprintln!(
        "FIELD evidence: {output}; {} failed band checks",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "independent estimator disagreements: {failures:?}"
    );
}

/// Independent cosine integration on the GPU isolates estimator logic from
/// CPU/GPU traversal differences. It deliberately uses no light distribution,
/// MIS, guide, roulette, cache, or history policy.
fn cosine_shader() -> String {
    let source = shader(true, false, false);
    let start = source.find("fn trace_receiver(").unwrap();
    let end = source[start..].find("fn evaluate_record(").unwrap() + start;
    let replacement = r#"
fn trace_receiver(surface: vec3<f32>, normal: vec3<f32>, stream: u32, sequence: u32, anchor: GuideAnchor) -> Energy {
    var result: Energy;
    var weight = Energy(vec4(1.0), 1.0, vec3(0.0));
    var position = surface;
    var n = normal;
    for (var bounce = 0u; bounce < DIFFUSE_HITS; bounce += 1u) {
        let d = hemisphere(n, sample_1d(stream, sequence, bounce * 2u), sample_1d(stream, sequence, bounce * 2u + 1u));
        let hit = trace_diffuse(position + n * 0.002, d);
        if hit.status == 0u {
            let horizon = clamp(0.65 + 0.35 * d.y, 0.15, 1.0);
            result.value += weight.value * params.environment * horizon;
            result.radiation += weight.radiation * params.extra.x * horizon;
            break;
        }
        if hit.status != 1u { break; }
        let material = materials[hit.material];
        result.value += weight.value * material.emission;
        result.radiation += weight.radiation * material.emission_extra.x;
        weight.value *= material.diffuse;
        weight.radiation *= material.diffuse_extra.x;
        let direct = direct_at(hit.position, hit.normal);
        result.value += weight.value * direct.value;
        result.radiation += weight.radiation * direct.radiation;
        position = hit.position;
        n = hit.normal;
        if bounce + 1u == DIFFUSE_HITS {
            let last_d = hemisphere(n, sample_1d(stream, sequence, 6u), sample_1d(stream, sequence, 7u));
            let last = trace(position + n * 0.002, last_d, connection_reach());
            if last.status == 1u {
                result.value += weight.value * materials[last.material].emission;
                result.radiation += weight.radiation * materials[last.material].emission_extra.x;
            }
        }
    }
    return result;
}
"#;
    format!("{}{}{}", &source[..start], replacement, &source[end..])
}

#[test]
#[ignore = "native estimator/traversal isolation at cave disagreement witnesses"]
fn cave_witness_estimator_isolation() {
    let fixture = cave_with_material_receivers();
    let sites = [159, 160];
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let native = tests::native_tracer();
    let mut variants: Vec<_> = [
        ("production", shader(true, true, false)),
        ("no-roulette", shader(true, false, false)),
        ("cosine-only", cosine_shader()),
    ]
    .into_iter()
    .map(|(name, source)| {
        (
            name,
            GpuTracer::with_shader(native.context.clone(), true, &source).unwrap(),
        )
    })
    .collect();
    for bounces in 1..=3 {
        let transport = TransportSettings {
            bounces,
            ..default()
        };
        let mut references = Vec::new();
        for &site in &sites {
            let p = &fixture.receivers[site];
            let reference = reference_oracle::estimate_seeded(
                &fixture.scene,
                &fixture.materials,
                &fixture.sources,
                transport,
                p.voxel,
                p.face_mask.trailing_zeros() as usize,
                1048576,
                457,
            );
            eprintln!("WITNESS depth={bounces} site={site} cpu={reference:?}");
            references.push(reference);
        }
        for (name, gpu) in &mut variants {
            let actual = estimates(
                gpu, &fixture, &emitters, transport, false, &sites, 1048576, 4194304,
            );
            eprintln!("WITNESS depth={bounces} {name}: {actual:?}");
            for (reference, actual) in references.iter().zip(&actual) {
                for b in 0..5 {
                    let se = reference.standard_error[b].hypot(actual.standard_error[b]);
                    assert!(
                        (reference.mean[b] - actual.mean[b]).abs()
                            <= 6.0 * se + 0.0002 * reference.mean[b].abs() + 1e-10,
                        "depth={bounces} variant={name} band={b}: {reference:?} vs {actual:?}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "native analytic grazing-angle visibility regression with legacy ablation"]
fn grazing_emitter_plane_matches_view_factor() {
    let fixture = research::grazing_emitter_plane(false);
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let expected = research::grazing_emitter_view_factor();
    let native = tests::native_tracer();
    let legacy = legacy_visibility_shader(&shader(true, true, false));
    for (name, source) in [
        ("corrected", shader(true, true, false)),
        ("cosine-only", cosine_shader()),
        ("legacy", legacy),
    ] {
        let mut gpu = GpuTracer::with_shader(native.context.clone(), true, &source).unwrap();
        let actual = estimates(
            &mut gpu,
            &fixture,
            &emitters,
            default(),
            false,
            &[0],
            1048576,
            4194304,
        )
        .remove(0);
        eprintln!("GRAZING {name}: analytic={expected} actual={actual:?}");
        for b in 0..5 {
            if name == "legacy" {
                assert!(
                    actual.mean[b] < expected - 6.0 * actual.standard_error[b],
                    "legacy ablation must expose the false shadow"
                );
            } else {
                assert!(
                    (actual.mean[b] - expected).abs()
                        < 6.0 * actual.standard_error[b] + expected * 0.0002
                );
            }
        }
    }
}
