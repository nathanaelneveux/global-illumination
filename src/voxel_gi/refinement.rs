//! Opt-in fixed-work refinement diagnostics. No asynchronous streaming,
//! adaptive scheduling, display uploads or wall-clock-dependent invalidation.
//! Controlled scene/source changes and history policies have separate studies.
//! Reports transport and presentation separately against independent analytic
//! integration where possible, otherwise a disjoint high-sample reference.
use super::*;
use crate::voxel_gi::{
    naadf::{linear_index, volume_len},
    sources::DirectionalTransportLight,
    world::{AIR, MaterialTransport, PAGE_SIZE},
};
use serde_json::{Value, json};

#[path = "history_audit.rs"]
mod history_audit;

#[path = "history_qualification.rs"]
mod history_qualification;

#[path = "settling_study.rs"]
mod settling_study;

#[path = "horizon_study.rs"]
mod horizon_study;

#[path = "distance_history_study.rs"]
mod distance_history_study;

struct Fixture {
    name: &'static str,
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: VoxelMaterialTransportTable,
    probes: Vec<SurfaceProbe>,
    analytic: Option<Vec<EnergySpectrum>>,
}

fn fixture(kind: u32) -> Fixture {
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 4..=27 {
        for y in 4..=20 {
            for x in 4..=27 {
                let shell = x == 4 || x == 27 || y == 4 || y == 20 || z == 4 || z == 27;
                let skylight =
                    kind == 2 && y == 20 && (13..=17).contains(&x) && (13..=17).contains(&z);
                if (kind == 0 && y == 4) || (kind != 0 && shell && !skylight) {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] =
                        if kind == 2 && x == 4 { 3 } else { 2 };
                }
            }
        }
    }
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(
        3,
        MaterialTransport {
            diffuse: EnergySpectrum::new(Vec3::new(0.8, 0.06, 0.03), 0.5, 0.1),
            ..default()
        },
    );
    let lamps = [IVec3::new(8, 20, 10), IVec3::new(22, 20, 21)];
    let energies = [
        EnergySpectrum::new(Vec3::new(24.0, 2.0, 0.5), 12.0, 3.0),
        EnergySpectrum::new(Vec3::new(0.5, 2.0, 24.0), 3.0, 12.0),
    ];
    if kind == 1 {
        materials.set(
            2,
            MaterialTransport {
                diffuse: EnergySpectrum::ZERO,
                ..default()
            },
        );
        for (index, lamp) in lamps.iter().enumerate() {
            page[linear_index(lamp.as_uvec3(), PAGE_SIZE)] = 4 + index as u16;
            materials.set(
                4 + index as u8,
                MaterialTransport {
                    diffuse: EnergySpectrum::ZERO,
                    emission: energies[index],
                },
            );
        }
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let probes: Vec<_> = (6..26)
        .step_by(2)
        .flat_map(|z| {
            (6..26)
                .step_by(2)
                .map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4))
        })
        .collect();
    let sources = TransportSources {
        directional: if kind == 2 {
            vec![DirectionalTransportLight {
                direction: Vec3::new(0.3, 1.0, 0.2).normalize(),
                radiance: EnergySpectrum::new(Vec3::splat(6.0), 3.0, 1.0),
                custom_visible: Vec3::ZERO,
            }]
        } else {
            vec![]
        },
        ..default()
    };
    let analytic = if kind == 0 {
        Some(vec![
            sources.environment * (0.65 + 0.35 * 2.0 / 3.0);
            probes.len()
        ])
    } else if kind == 1 {
        // Independent midpoint area quadrature, not the path sampler or MIS.
        // The two downward ceiling squares are the only visible emission;
        // all other surfaces have zero reflectance and the room is sealed.
        Some(
            probes
                .iter()
                .map(|probe| {
                    let receiver = probe.voxel.as_vec3() + Vec3::new(0.5, 1.0, 0.5);
                    let mut sum = EnergySpectrum::ZERO;
                    const EDGE: u32 = 128;
                    for (lamp, emission) in lamps.iter().zip(energies) {
                        let mut factor = 0.0;
                        for z in 0..EDGE {
                            for x in 0..EDGE {
                                let target = lamp.as_vec3()
                                    + Vec3::new(
                                        (x as f32 + 0.5) / EDGE as f32,
                                        0.0,
                                        (z as f32 + 0.5) / EDGE as f32,
                                    );
                                let delta = target - receiver;
                                let d2 = delta.length_squared();
                                factor += delta.y * delta.y / (std::f32::consts::PI * d2 * d2);
                            }
                        }
                        sum += emission * (factor / (EDGE * EDGE) as f32);
                    }
                    sum
                })
                .collect(),
        )
    } else {
        None
    };
    Fixture {
        name: [
            "analytic_open_sky",
            "analytic_colored_emitters",
            "colored_skylight_room",
        ][kind as usize],
        scene,
        sources,
        materials,
        probes,
        analytic,
    }
}

/// Reuse the same controlled geometry in native sampler/ownership studies.
pub(super) fn research_fixture(kind: u32) -> crate::voxel_gi::research::ResearchFixture {
    let fixture = fixture(kind);
    crate::voxel_gi::research::ResearchFixture {
        scene: fixture.scene,
        sources: fixture.sources,
        materials: fixture.materials,
        receivers: fixture.probes,
    }
}

fn channels(energy: EnergySpectrum) -> [f32; 5] {
    [
        energy.visible.x,
        energy.visible.y,
        energy.visible.z,
        energy.infrared,
        energy.ionizing,
    ]
}

/// Compare spatial caps, a forced restart and full compatible-history retention. Every
/// post-change path has an independent exact-zero reference in all five bands.
/// Removal, replacement and occlusion must not need different energy policies.
#[test]
#[ignore = "native five-band history response diagnostic"]
fn scene_history_source_extinction_response() {
    use crate::voxel_gi::geometry::{GeometryHistory, SceneChanges};
    let mut gpu = super::tests::native_tracer();
    let mut fixture = fixture(1);
    fixture.sources.environment = EnergySpectrum::ZERO;
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    gpu.trace(
        fixture.scene.clone(),
        &fixture.sources,
        &fixture.materials,
        &mut fixture.probes,
        2048,
        &emitters,
    )
    .unwrap();
    let lamps = [IVec3::new(8, 20, 10), IVec3::new(22, 20, 21)];
    for kind in ["remove", "replace", "occlude"] {
        let mut page: Vec<_> = (0..32)
            .flat_map(|z| (0..32).flat_map(move |y| (0..32).map(move |x| IVec3::new(x, y, z))))
            .map(|v| fixture.scene.material_at(v).unwrap())
            .collect();
        for lamp in lamps {
            if kind == "occlude" {
                for normal in crate::voxel_gi::world::FACE_OFFSETS {
                    page[linear_index((lamp + normal).as_uvec3(), PAGE_SIZE)] = 2;
                }
            } else {
                page[linear_index(lamp.as_uvec3(), PAGE_SIZE)] =
                    if kind == "remove" { AIR } else { 2 };
            }
        }
        let dark = Arc::new(TransportScene::with_page_delta(
            &fixture.scene,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let mut journal = GeometryHistory::default();
        journal.record(
            0,
            1,
            SceneChanges::between(&fixture.scene, &dark)
                .transport
                .unwrap(),
        );
        let emitters = Arc::new(EmissionDistribution::build(&dark, &fixture.materials));
        for (policy, cap, paths, batches) in [
            ("reset", 0, 4, 256),
            ("spatial", 16, 4, 256),
            ("retain-all", u32::MAX, 128, 160),
        ] {
            let start = Instant::now();
            let mut probes = fixture.probes.clone();
            for p in &mut probes {
                assert_eq!(journal.history_limit(p.voxel, 0, 1), 16);
                p.samples[2] = p.samples[2].min(cap);
                p.display[2].indirect = p.radiance[2].visible;
                p.face_updated_at[2] = Some(start);
            }
            let mut last_raw_bad = 0;
            let mut last_display_bad = 0;
            for batch in 1..=batches {
                let before = probes.clone();
                gpu.trace(
                    dark.clone(),
                    &fixture.sources,
                    &fixture.materials,
                    &mut probes,
                    paths,
                    &emitters,
                )
                .unwrap();
                if batch == 1 {
                    let cpu = crate::voxel_gi::tracer::trace_probe_batch(
                        dark.clone(),
                        fixture.sources.clone(),
                        Arc::new(fixture.materials.clone()),
                        before.clone(),
                        paths,
                    );
                    for (a, b) in probes.iter().zip(cpu) {
                        assert!((a.radiance[2] - b.radiance[2]).visible.abs().max_element() < 1e-6);
                        assert!((a.radiance[2].infrared - b.radiance[2].infrared).abs() < 1e-6);
                        assert!((a.radiance[2].ionizing - b.radiance[2].ionizing).abs() < 1e-6);
                    }
                }
                let now = start + Duration::from_millis(batch * 50);
                for ((p, previous), initial) in probes.iter_mut().zip(&before).zip(&fixture.probes)
                {
                    let retained = initial.samples[2].min(cap) as f32;
                    let expected = retained / (retained + (batch * u64::from(paths)) as f32);
                    for (mean, lit) in channels(p.radiance[2])
                        .into_iter()
                        .zip(channels(initial.radiance[2]))
                    {
                        assert!(lit > 0.0);
                        assert!(
                            (mean / lit - expected).abs() < 1e-5,
                            "{kind}: obsolete energy follows only the generic retained weight"
                        );
                        if mean / lit > 0.1 {
                            last_raw_bad = batch;
                        }
                    }
                    p.resolve_display(previous, now, 0.075);
                    assert!(
                        p.display[2]
                            .indirect
                            .cmple(previous.display[2].indirect + Vec3::splat(1e-7))
                            .all()
                    );
                    if ((p.display[2].indirect / initial.radiance[2].visible).max_element()) > 0.1 {
                        last_display_bad = batch;
                    }
                    p.face_updated_at[2] = Some(now);
                }
            }
            assert!(
                last_display_bad < batches,
                "response must settle inside the observed interval"
            );
            eprintln!(
                "extinction {kind} policy={policy}: all-band raw <10% at {} fresh paths; worst RGB display <10% at {} fresh paths (virtual 50ms/{paths} paths, not gameplay latency)",
                (last_raw_bad + 1) * u64::from(paths),
                (last_display_bad + 1) * u64::from(paths)
            );
        }
    }
}

fn errors(values: &[EnergySpectrum], reference: &[EnergySpectrum]) -> Value {
    let mut squares = [0.0; 5];
    let mut normalized = Vec::new();
    for (value, expected) in values.iter().zip(reference) {
        for (band, (value, expected)) in channels(*value)
            .into_iter()
            .zip(channels(*expected))
            .enumerate()
        {
            let error = value - expected;
            squares[band] += error * error;
            normalized.push(error.abs() / (0.02 + expected.abs()));
        }
    }
    normalized.sort_by(f32::total_cmp);
    json!({"band_rmse": squares.map(|s| (s / values.len() as f32).sqrt()),
        "relative_p95": normalized[normalized.len() * 95 / 100],
        "relative_max": normalized.last()})
}

fn percentile(mut values: Vec<f32>, percentile: usize) -> f32 {
    values.sort_by(f32::total_cmp);
    values[(values.len() - 1) * percentile / 100]
}

/// Sampler experiments deliberately change paths, so compare equal work to one
/// independent reference, not path-by-path equality or each sampler's own mean.
#[test]
#[ignore = "native paired sampler study; requires VOXEL_GI_BASELINE_SHADER"]
fn sampler_candidate_benchmark() {
    let source =
        std::fs::read_to_string(std::env::var("VOXEL_GI_BASELINE_SHADER").unwrap()).unwrap();
    let mut candidate = super::tests::native_tracer();
    let mut baseline = GpuTracer::with_shader(candidate.context.clone(), true, &source).unwrap();
    for kind in 0..3 {
        let fixture = fixture(kind);
        let emitters = Arc::new(EmissionDistribution::build(
            &fixture.scene,
            &fixture.materials,
        ));
        let mut high = fixture.probes.clone();
        for p in &mut high {
            p.sequence[2] = 1_048_576;
        }
        baseline
            .trace(
                fixture.scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut high,
                65_536,
                &emitters,
            )
            .unwrap();
        let reference: Vec<_> = fixture
            .analytic
            .clone()
            .unwrap_or_else(|| high.iter().map(|p| p.radiance[2]).collect());
        let mut expected = Vec::new();
        let mut estimates: [[Vec<EnergySpectrum>; 6]; 2] = default();
        let mut times: [Vec<f32>; 2] = default();
        for replicate in 0..8 {
            expected.extend_from_slice(&reference);
            let mut probes = [fixture.probes.clone(), fixture.probes.clone()];
            for group in &mut probes {
                for p in group {
                    p.sequence[2] = replicate * 4096;
                }
            }
            for batch in 1..=256 {
                for mode in (0..2).map(|offset| (offset + batch + replicate as usize) % 2) {
                    let gpu = if mode == 0 {
                        &mut baseline
                    } else {
                        &mut candidate
                    };
                    gpu.trace(
                        fixture.scene.clone(),
                        &fixture.sources,
                        &fixture.materials,
                        &mut probes[mode],
                        4,
                        &emitters,
                    )
                    .unwrap();
                    if batch > 4 {
                        times[mode].push(gpu.last_gpu_ms.unwrap());
                    }
                    if let Some(index) = [2, 8, 16, 32, 128, 256].iter().position(|b| *b == batch) {
                        estimates[mode][index].extend(probes[mode].iter().map(|p| p.radiance[2]));
                    }
                }
            }
        }
        for mode in 0..2 {
            let checkpoints: Vec<_> = estimates[mode]
                .iter()
                .zip([8, 32, 64, 128, 512, 1024])
                .map(|(values, paths)| json!({"paths":paths,"error":errors(values, &expected)}))
                .collect();
            eprintln!(
                "Sampler study: {}",
                json!({"fixture":fixture.name,"mode":(["baseline","candidate"][mode]),"replicates":8,"gpu_p50_ms":percentile(times[mode].clone(),50),"checkpoints":checkpoints})
            );
        }
    }
}

#[test]
#[ignore = "fixed-work refinement diagnostic; requires a native compute adapter"]
fn fixed_scene_refinement_diagnostic() {
    let mut gpu = super::tests::native_tracer();
    let mut reports = Vec::new();
    for kind in 0..3 {
        let fixture = fixture(kind);
        let emitters = Arc::new(EmissionDistribution::build(
            &fixture.scene,
            &fixture.materials,
        ));
        let mut reference_probes = fixture.probes.clone();
        for probe in &mut reference_probes {
            probe.sequence[2] = 65_536;
        }
        gpu.trace(
            fixture.scene.clone(),
            &fixture.sources,
            &fixture.materials,
            &mut reference_probes,
            16_384,
            &emitters,
        )
        .unwrap();
        let integrated: Vec<_> = reference_probes.iter().map(|p| p.radiance[2]).collect();
        let reference = fixture.analytic.as_ref().unwrap_or(&integrated);
        let reference_check = errors(&integrated, reference);
        if kind < 2 {
            assert!(
                reference_check["relative_p95"].as_f64().unwrap() < 0.015,
                "independent integration disagrees: {reference_check}"
            );
        }
        let start = Instant::now();
        let mut probes = fixture.probes.clone();
        let mut checkpoints = Vec::new();
        let mut raw_steps = Vec::new();
        let mut display_steps = Vec::new();
        let mut gpu_ms = Vec::new();
        let mut timeline = Vec::new();
        for batch in 1..=512 {
            let previous = probes.clone();
            gpu.trace(
                fixture.scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut probes,
                4,
                &emitters,
            )
            .unwrap();
            if let Some(ms) = gpu.last_gpu_ms {
                gpu_ms.push(ms);
            }
            let now = start + Duration::from_millis(batch * 50);
            for (index, (probe, old)) in probes.iter_mut().zip(&previous).enumerate() {
                probe.resolve_display(old, now, 0.075);
                probe.face_updated_at[2] = Some(now);
                if batch > 64 {
                    let scale = 0.02 + reference[index].visible.max_element();
                    raw_steps.push(
                        (probe.radiance[2].visible - old.radiance[2].visible)
                            .abs()
                            .max_element()
                            / scale,
                    );
                    display_steps.push(
                        (probe.display[2].indirect - old.display[2].indirect)
                            .abs()
                            .max_element()
                            / scale,
                    );
                }
            }
            if [2, 8, 16, 32, 64, 128, 256, 512].contains(&batch) {
                let raw: Vec<_> = probes.iter().map(|p| p.radiance[2]).collect();
                let display: Vec<_> = probes
                    .iter()
                    .map(|p| EnergySpectrum::new(p.display[2].indirect, 0.0, 0.0))
                    .collect();
                let visible_reference: Vec<_> = reference
                    .iter()
                    .map(|e| EnergySpectrum::new(e.visible, 0.0, 0.0))
                    .collect();
                checkpoints.push(json!({"paths_per_face": batch * 4,
                    "raw": errors(&raw, reference), "display": errors(&display, &visible_reference)}));
            }
            timeline.push(json!({"paths_per_face": batch * 4,
                "raw": probes.iter().map(|p| channels(p.radiance[2])).collect::<Vec<_>>(),
                "display": probes.iter().map(|p| p.display[2].indirect.to_array()).collect::<Vec<_>>()}));
        }
        let report = json!({"fixture": fixture.name, "receivers": probes.len(),
            "reference_check": reference_check, "checkpoints": checkpoints,
            "late_raw_step_p95": percentile(raw_steps.clone(), 95),
            "late_raw_step_max": percentile(raw_steps, 100),
            "late_display_step_p95": percentile(display_steps.clone(), 95),
            "late_display_step_max": percentile(display_steps, 100),
            "gpu_batch_ms_p50": (!gpu_ms.is_empty()).then(|| percentile(gpu_ms, 50))});
        // Broad fixture-specific regression bounds, not universal convergence
        // guarantees. They reject the original rolling-history plateau and the
        // high-base emitter prefixes at the same path count and display policy.
        if kind == 1 {
            assert!(
                report["checkpoints"][2]["raw"]["relative_p95"]
                    .as_f64()
                    .unwrap()
                    < 0.09
            );
            assert!(
                report["checkpoints"][7]["raw"]["relative_p95"]
                    .as_f64()
                    .unwrap()
                    < 0.01
            );
        } else if kind == 2 {
            assert!(
                report["checkpoints"][7]["raw"]["relative_p95"]
                    .as_f64()
                    .unwrap()
                    < 0.20
            );
            assert!(report["late_display_step_p95"].as_f64().unwrap() < 0.02);
        }
        eprintln!("Refinement: {report}");
        reports.push(json!({"summary": report, "reference": reference.iter().map(|v| channels(*v)).collect::<Vec<_>>(),
            "timeline": timeline}));
    }
    if let Ok(path) = std::env::var("VOXEL_GI_REFINEMENT_REPORT") {
        std::fs::write(path, serde_json::to_vec(&json!({"fixtures": reports,
            "notes": "Fixed scene, 100 faces, 4 paths per batch, virtual 50ms display cadence. Not gameplay FPS or wall-clock response. References use disjoint 16384-path estimates or independent analytic/quadrature values."})).unwrap()).unwrap();
    }
}

/// Matched-sample history tradeoff, independent of scheduling and streaming.
/// Closure has an exact zero-energy reference; the open roof uses two disjoint
/// high-sample estimates, whose disagreement is reported rather than hidden.
#[test]
#[ignore = "matched-ray geometry response diagnostic; requires a native compute adapter"]
fn geometry_history_response_sweep() {
    let mut gpu = super::tests::native_tracer();
    let mut fixture = fixture(2);
    let mut closed_page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for y in 0..32 {
            for x in 0..32 {
                let v = UVec3::new(x, y, z);
                closed_page[linear_index(v, PAGE_SIZE)] =
                    if y == 20 && (13..=17).contains(&x) && (13..=17).contains(&z) {
                        2
                    } else {
                        fixture.scene.material_at(v.as_ivec3()).unwrap()
                    };
            }
        }
    }
    let closed = Arc::new(TransportScene::with_page_delta(
        &fixture.scene,
        vec![(IVec3::ZERO, closed_page.clone())],
        &[],
    ));
    closed_page[linear_index(UVec3::new(16, 20, 16), PAGE_SIZE)] = AIR;
    let open = Arc::new(TransportScene::with_page_delta(
        &closed,
        vec![(IVec3::ZERO, closed_page)],
        &[],
    ));
    fixture.probes = [6, 10, 16, 21, 25]
        .into_iter()
        .flat_map(|z| {
            [6, 10, 16, 21, 25]
                .into_iter()
                .filter(move |&x| x != 16 || z != 16)
                .map(move |x| SurfaceProbe::new(IVec3::new(x, 20, z), 8))
        })
        .collect();
    let face = 3;
    let emitters = Arc::new(EmissionDistribution::build(&open, &fixture.materials));
    let mut references = Vec::new();
    for sequence in [65_536, 1_048_576] {
        let mut probes = fixture.probes.clone();
        let mut integrated = fixture.probes.clone();
        for probe in &mut probes {
            probe.sequence[face] = sequence;
        }
        for _ in 0..8 {
            for probe in &mut probes {
                probe.samples[face] = 0;
                probe.radiance[face] = EnergySpectrum::ZERO;
                probe.second_moment[face] = EnergySpectrum::ZERO;
            }
            gpu.trace(
                open.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut probes,
                65_536,
                &emitters,
            )
            .unwrap();
            for (total, batch) in integrated.iter_mut().zip(&probes) {
                total.radiance[face] += batch.radiance[face] / 8.0;
                total.second_moment[face] += batch.second_moment[face] / 8.0;
            }
        }
        references.push(integrated);
    }
    let reference: Vec<_> = references[0]
        .iter()
        .zip(&references[1])
        .map(|(a, b)| {
            (
                (a.radiance[face] + b.radiance[face]) * 0.5,
                (a.second_moment[face] + b.second_moment[face]) * 0.5,
            )
        })
        .collect();
    let disagreement: Vec<_> = references[0]
        .iter()
        .zip(&references[1])
        .zip(&reference)
        .map(|((a, b), (mean, _))| {
            (a.radiance[face].visible - b.radiance[face].visible)
                .abs()
                .max_element()
                / mean.visible.max_element().max(1e-6)
        })
        .collect();
    let mut reports = Vec::new();
    const CAPS: [u32; 6] = [16, 24, 32, 64, 128, 256];
    for closing in [false, true] {
        let scene = if closing {
            closed.clone()
        } else {
            open.clone()
        };
        let emitters = Arc::new(EmissionDistribution::build(&scene, &fixture.materials));
        let start = Instant::now();
        let mut fresh = fixture.probes.clone();
        let mut candidates: Vec<_> = CAPS
            .iter()
            .map(|&cap| {
                fixture
                    .probes
                    .iter()
                    .zip(&reference)
                    .map(|(probe, &(mean, second))| {
                        let mut p = probe.clone();
                        p.samples[face] = cap;
                        p.radiance[face] = if closing { mean } else { EnergySpectrum::ZERO };
                        p.second_moment[face] = if closing {
                            second
                        } else {
                            EnergySpectrum::ZERO
                        };
                        p.display[face].indirect = p.radiance[face].visible;
                        p.face_updated_at[face] = Some(start);
                        p
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut steps = vec![Vec::new(); CAPS.len()];
        let mut curves = vec![Vec::new(); CAPS.len()];
        for batch in 1..=640 {
            // Trace each fresh stream once, then feed identical moments into
            // every candidate. No different random sequence or ray budget.
            for p in &mut fresh {
                p.samples[face] = 0;
                p.radiance[face] = EnergySpectrum::ZERO;
                p.second_moment[face] = EnergySpectrum::ZERO;
            }
            gpu.trace(
                scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut fresh,
                4,
                &emitters,
            )
            .unwrap();
            if closing {
                assert!(
                    fresh
                        .iter()
                        .all(|p| p.radiance[face] == EnergySpectrum::ZERO)
                );
            }
            let now = start + Duration::from_millis(batch * 50);
            for (index, probes) in candidates.iter_mut().enumerate() {
                let mut raw_error = Vec::new();
                let mut display_error = Vec::new();
                for ((p, new), &(mean, _)) in probes.iter_mut().zip(&fresh).zip(&reference) {
                    let old_display = p.display[face].indirect;
                    p.samples[face] += 4;
                    let alpha = 4.0 / p.samples[face] as f32;
                    p.radiance[face] += (new.radiance[face] - p.radiance[face]) * alpha;
                    p.second_moment[face] +=
                        (new.second_moment[face] - p.second_moment[face]) * alpha;
                    p.resolve_face(face, Some(now - Duration::from_millis(50)), now, 0.075);
                    let target = if closing { Vec3::ZERO } else { mean.visible };
                    let scale = mean.visible.max_element().max(1e-6);
                    raw_error.push((p.radiance[face].visible - target).abs().max_element() / scale);
                    display_error
                        .push((p.display[face].indirect - target).abs().max_element() / scale);
                    if batch <= 100 {
                        steps[index].push(
                            (p.display[face].indirect - old_display).abs().max_element() / scale,
                        );
                    }
                }
                curves[index].push(json!({"fresh_paths":batch*4, "raw_error_p95":percentile(raw_error,95), "display_error_p95":percentile(display_error,95)}));
            }
        }
        for (index, cap) in CAPS.into_iter().enumerate() {
            // First point after the LAST threshold violation, not the first
            // lucky crossing. None means not reached during this workload.
            let sustained = |threshold: f64| {
                curves[index]
                    .iter()
                    .rposition(|p| p["display_error_p95"].as_f64().unwrap() > threshold)
                    .map_or(Some(4), |i| {
                        curves[index]
                            .get(i + 1)
                            .map(|p| p["fresh_paths"].as_u64().unwrap())
                    })
            };
            let summary = json!({"closing":closing, "history_cap":cap,
                "stay_within_10_percent_paths":sustained(0.1), "stay_within_20_percent_paths":sustained(0.2),
                "first_100_batches_display_step_p95":percentile(steps[index].clone(),95),
                "display_step_max":percentile(steps[index].clone(),100),
                "checkpoints":curves[index].iter().filter(|p|[32,64,128,256,512,1024,2048,2560].contains(&p["fresh_paths"].as_u64().unwrap())).collect::<Vec<_>>()});
            eprintln!("History response: {summary}");
            reports.push(json!({"summary":summary,"curve":curves[index]}));
        }
    }
    let report = json!({"reference_disagreement_p95":percentile(disagreement,95),"cases":reports,
        "notes":"24 fixed roof receivers, one-block skylight, identical 4-path batches across caps. Closure is exactly zero. Opening reference averages two disjoint 524288-path estimates; reported disagreement limits accuracy claims. Virtual 50ms display cadence, NOT gameplay latency or FPS. No adaptive scheduler or streaming."});
    if let Ok(path) = std::env::var("VOXEL_GI_RESPONSE_REPORT") {
        std::fs::write(path, serde_json::to_vec(&report).unwrap()).unwrap();
    }
}
