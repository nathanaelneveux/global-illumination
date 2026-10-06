//! Opt-in fixed-work refinement diagnostic. No streaming, edits, adaptive
//! scheduling, display uploads, or wall-clock-dependent history invalidation.
//! Reports transport and presentation separately against independent analytic
//! integration where possible, otherwise a disjoint high-sample reference.
use super::*;
use crate::voxel_gi::{
    naadf::{linear_index, volume_len},
    tracer::DirectionalTransportLight,
    world::{AIR, MaterialTransport, PAGE_SIZE},
};
use serde_json::{Value, json};

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

fn channels(energy: EnergySpectrum) -> [f32; 5] {
    [
        energy.visible.x,
        energy.visible.y,
        energy.visible.z,
        energy.infrared,
        energy.ionizing,
    ]
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
