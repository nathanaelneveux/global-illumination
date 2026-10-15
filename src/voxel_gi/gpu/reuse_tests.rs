//! Native fixed-work study. Each trial starts from an empty measurement with
//! disjoint sequences; references are independent, not the candidate's tail.
use super::*;
use crate::voxel_gi::{
    research::{cave, channels},
    world::{AIR, FACE_OFFSETS},
};

fn trace(
    tracer: &mut GpuTracer,
    fixture: &crate::voxel_gi::research::ResearchFixture,
    emitters: &Arc<EmissionDistribution>,
    work: &mut TraceWork,
    samples: u32,
) {
    tracer
        .trace_compact(
            fixture.scene.clone(),
            &fixture.sources,
            &fixture.materials,
            work,
            samples,
            emitters,
            false,
        )
        .unwrap();
}

fn fresh(probes: &[SurfaceProbe], revision: u64, sequence: u32) -> TraceWork {
    let mut work = TraceWork::from_probes(probes, false);
    for job in &mut work.probes {
        job.revision = revision;
        job.sky_requested = false;
    }
    for face in &mut work.faces {
        face.sequence = sequence;
    }
    work
}

#[test]
#[ignore = "native adaptive wide sharing: mixed groups, tails and mature independent path parity"]
fn wide_reuse_mixed_groups_and_mature_fallback() {
    let mut control = super::tests::native_tracer();
    control.enable_resident().unwrap();
    let mut candidate = GpuTracer::new(control.context.clone()).unwrap();
    candidate.enable_wide_reuse().unwrap();
    let fixture = super::refinement::research_fixture(2);
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let probes: Vec<_> = (6..13)
        .flat_map(|z| (6..15).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
        .collect();
    for spp in [1, 3, 7, 16] {
        let mut a = fresh(&probes, u64::from(spp), 99);
        let mut b = fresh(&probes, u64::from(spp), 99);
        for work in [&mut a, &mut b] {
            for (i, f) in work.faces.iter_mut().enumerate() {
                f.samples = [0, 256, 1024][i % 3];
            }
        }
        trace(&mut control, &fixture, &emitters, &mut a, spp);
        trace(&mut candidate, &fixture, &emitters, &mut b, spp);
        let links: Vec<_> = b.faces.iter().map(|f| f.neighbor).collect();
        let mut shared = 0;
        for (i, (a, b)) in a.faces.iter().zip(&b.faces).enumerate() {
            assert_eq!((a.samples, a.sequence), (b.samples, b.sequence));
            assert!(
                channels(b.radiance)
                    .iter()
                    .all(|v| v.is_finite() && *v >= 0.0)
            );
            if i % 3 == 2 {
                assert_eq!(b.neighbor, 0);
                for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                    assert!((a - b).abs() < 1e-5 * (0.02 + a.abs()), "mature {a} vs {b}");
                }
            }
            if b.neighbor > 0 {
                shared += 1;
                let mut next = b.neighbor as usize - 1;
                let mut size = 1;
                while next != i {
                    size += 1;
                    assert!(size <= 4);
                    assert_ne!(links[next], 0);
                    next = links[next] as usize - 1;
                }
            }
        }
        assert!(shared > 0);
    }
}

fn percentile(mut xs: Vec<f32>, p: f32) -> f32 {
    xs.sort_by(f32::total_cmp);
    xs[((xs.len() - 1) as f32 * p).round() as usize]
}

#[test]
#[ignore = "native adaptive sharing study: identical RGB priority, disjoint prefixes and cumulative GPU work"]
fn wide_reuse_quality_and_cost() {
    let mut independent = super::tests::native_tracer();
    independent.enable_resident().unwrap();
    let mut paired = GpuTracer::new(independent.context.clone()).unwrap();
    paired.enable_reuse().unwrap();
    let mut wide = GpuTracer::new(independent.context.clone()).unwrap();
    wide.enable_wide_reuse().unwrap();
    for kind in 0..4 {
        let mut fixture = if kind == 3 {
            cave()
        } else {
            super::refinement::research_fixture(kind)
        };
        if kind == 3 {
            let mut keys = bevy::platform::collections::HashSet::new();
            let mut dense = Vec::new();
            for probe in &fixture.receivers {
                let face = probe.face_mask.trailing_zeros() as usize;
                let normal = FACE_OFFSETS[face];
                let axes: Vec<_> = [IVec3::X, IVec3::Y, IVec3::Z]
                    .into_iter()
                    .filter(|a| a.dot(normal) == 0)
                    .collect();
                for x in -2..=2 {
                    for y in -2..=2 {
                        let p = probe.voxel + axes[0] * x + axes[1] * y;
                        if fixture.scene.material_at(p).is_some_and(|m| m != AIR)
                            && fixture.scene.material_at(p + normal) == Some(AIR)
                            && keys.insert((p, face))
                        {
                            dense.push(SurfaceProbe::new(p, 1 << face));
                        }
                    }
                }
            }
            fixture.receivers = dense;
        } else {
            fixture.receivers = (6..26)
                .flat_map(|z| (6..26).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
                .collect();
        }
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            super::super::EnergySpectrum::new(Vec3::ONE, 0.125, 0.125),
        ));
        let mut reference = fresh(&fixture.receivers, 1, 8_388_608);
        for _ in 0..8 {
            trace(&mut independent, &fixture, &emitters, &mut reference, 8192);
        }
        let refs: Vec<_> = reference
            .faces
            .iter()
            .map(|f| channels(f.radiance))
            .collect();
        // Compile/warm all kernels separately from the measured prefixes.
        for gpu in [&mut independent, &mut paired, &mut wide] {
            let mut work = fresh(&fixture.receivers, 2, 7_000_000);
            trace(gpu, &fixture, &emitters, &mut work, 16);
        }
        for window in 0..4 {
            let mut works: Vec<_> = (0..3)
                .map(|_| fresh(&fixture.receivers, 100 + window, window as u32 * 574_117))
                .collect();
            let mut cost = [0.0; 3];
            for round in 0..128 {
                for k in 0..3 {
                    let mode = (round + k) % 3;
                    let gpu = match mode {
                        0 => &mut independent,
                        1 => &mut paired,
                        _ => &mut wide,
                    };
                    trace(gpu, &fixture, &emitters, &mut works[mode], 16);
                    cost[mode] += gpu.last_gpu_ms.unwrap();
                }
                let paths = (round + 1) * 16;
                if [16, 64, 256, 512, 1024, 2048].contains(&paths) {
                    for mode in 0..3 {
                        let work = &works[mode];
                        let errors = corner_errors(work, &refs);
                        let mut bias = [0.0f64; 5];
                        for (face, reference) in work.faces.iter().zip(&refs) {
                            for (i, (a, b)) in channels(face.radiance)
                                .into_iter()
                                .zip(reference)
                                .enumerate()
                            {
                                bias[i] += f64::from(a - b) / refs.len() as f64;
                            }
                        }
                        eprintln!(
                            "WIDE_RESULT kind={kind} mode={} window={window} paths={paths} faces={} shared={} gpu_ms={} corner_p95={} corner_p99={} bias={bias:?}",
                            ["independent", "paired", "wide"][mode],
                            refs.len(),
                            work.faces.iter().filter(|f| f.neighbor != 0).count(),
                            cost[mode],
                            percentile(errors.clone(), 0.95),
                            percentile(errors, 0.99)
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "native five-band CPU/GPU parity with unequal spectral sampling importance"]
fn spectral_importance_preserves_cpu_gpu_energy_parity() {
    let mut gpu = super::tests::native_tracer();
    // A black room isolates the changed light-selection/MIS path from chaotic
    // CPU/Metal multi-bounce floating-point divergence in the full cave.
    let mut fixture = super::refinement::research_fixture(1);
    fixture.materials.set(
        4,
        super::super::MaterialTransport {
            diffuse: super::super::EnergySpectrum::ZERO,
            emission: super::super::EnergySpectrum::new(Vec3::ZERO, 24., 12.),
        },
    );
    for importance in [1.0, 0.125, 0.01] {
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            super::super::EnergySpectrum::new(Vec3::ONE, importance, importance),
        ));
        let mut a = fresh(&fixture.receivers[..17], 1, 19);
        let mut b = fresh(&fixture.receivers[..17], 1, 19);
        super::super::tracer::trace_work_batch(
            &fixture.scene,
            &fixture.sources,
            &fixture.materials,
            &mut a,
            256,
            &emitters,
            &mut default(),
        );
        trace(&mut gpu, &fixture, &emitters, &mut b, 256);
        for (a, b) in a.faces.iter().zip(&b.faces) {
            for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                assert!(
                    (a - b).abs() < 1e-4 * (0.02 + a.abs()),
                    "importance={importance}: {a} vs {b}"
                );
            }
        }
    }
}

#[test]
#[ignore = "native cross-scene early-corner accuracy and matched-work GPU cost"]
fn reuse_cross_scene_quality() {
    let mut baseline = super::tests::native_tracer();
    baseline.enable_resident().unwrap();
    let mut candidate = GpuTracer::new(baseline.context.clone()).unwrap();
    candidate.enable_reuse().unwrap();
    for kind in 0..3 {
        let mut fixture = super::refinement::research_fixture(kind);
        fixture.receivers = (6..26)
            .flat_map(|z| (6..26).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            super::super::EnergySpectrum::new(Vec3::ONE, 0.125, 0.125),
        ));
        let mut reference = fresh(&fixture.receivers, 1, 8_388_608);
        for _ in 0..4 {
            trace(&mut baseline, &fixture, &emitters, &mut reference, 8192);
        }
        let refs: Vec<_> = reference
            .faces
            .iter()
            .map(|f| channels(f.radiance))
            .collect();
        for paths in [64, 256, 1024, 2048] {
            for (mode, gpu) in [("independent", &mut baseline), ("reuse", &mut candidate)] {
                let mut errors = Vec::new();
                for window in 0..4 {
                    let mut work = fresh(
                        &fixture.receivers,
                        100 + paths as u64 * 4 + window,
                        window as u32 * 574_117,
                    );
                    for _ in 0..paths / 16 {
                        trace(gpu, &fixture, &emitters, &mut work, 16);
                    }
                    errors.extend(corner_errors(&work, &refs));
                }
                eprintln!(
                    "REUSE_SCENE kind={kind} mode={mode} paths={paths} corner_p95={} corner_p99={}",
                    percentile(errors.clone(), 0.95),
                    percentile(errors, 0.99)
                );
            }
        }
        let mut a = fresh(&fixture.receivers, 999_999, 0);
        let mut b = fresh(&fixture.receivers, 999_999, 0);
        let mut cost = [Vec::new(), Vec::new()];
        for round in 0..40 {
            for mode in [round % 2, 1 - round % 2] {
                let (gpu, work) = if mode == 0 {
                    (&mut baseline, &mut a)
                } else {
                    (&mut candidate, &mut b)
                };
                trace(gpu, &fixture, &emitters, work, 16);
                if round >= 8 {
                    cost[mode].push(gpu.last_gpu_ms.unwrap());
                }
            }
        }
        eprintln!(
            "REUSE_SCENE_COST kind={kind} baseline_p50={} reuse_p50={} baseline_p95={} reuse_p95={}",
            percentile(cost[0].clone(), 0.5),
            percentile(cost[1].clone(), 0.5),
            percentile(cost[0].clone(), 0.95),
            percentile(cost[1].clone(), 0.95)
        );
    }
}

fn corner_errors(work: &TraceWork, refs: &[[f32; 5]]) -> Vec<f32> {
    let indices: bevy::platform::collections::HashMap<_, _> = work
        .faces
        .iter()
        .enumerate()
        .map(|(i, f)| ((f.voxel, f.face), i))
        .collect();
    let mut errors = Vec::new();
    for (i, face) in work.faces.iter().enumerate() {
        let (u, v) = match face.face / 2 {
            0 => (IVec3::Y, IVec3::Z),
            1 => (IVec3::X, IVec3::Z),
            _ => (IVec3::X, IVec3::Y),
        };
        for x in [-1, 1] {
            for y in [-1, 1] {
                let neighbors = [u * x, v * y];
                let mut members = vec![i];
                for delta in neighbors {
                    if let Some(&n) = indices.get(&(face.voxel + delta, face.face)) {
                        members.push(n);
                    }
                }
                if members.len() > 1
                    && let Some(&n) = indices.get(&(face.voxel + u * x + v * y, face.face))
                {
                    members.push(n);
                }
                let value = members
                    .iter()
                    .map(|&i| work.faces[i].radiance.visible)
                    .sum::<Vec3>()
                    / members.len() as f32;
                let reference = members
                    .iter()
                    .map(|&i| Vec3::from_slice(&refs[i]))
                    .sum::<Vec3>()
                    / members.len() as f32;
                errors.push((value - reference).length() / reference.length().max(0.02));
            }
        }
    }
    errors
}

#[test]
#[ignore = "native paired workgroup partial receiver and sample tails"]
fn paired_reuse_handles_partial_workgroups_and_sample_lanes() {
    let mut baseline = super::tests::native_tracer();
    let mut candidate = GpuTracer::new(baseline.context.clone()).unwrap();
    candidate.enable_reuse().unwrap();
    let fixture = cave();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    for spp in [1, 3, 4, 7, 16] {
        let mut a = fresh(&fixture.receivers[..17], spp as u64, 99);
        let mut b = fresh(&fixture.receivers[..17], spp as u64, 99);
        trace(&mut baseline, &fixture, &emitters, &mut a, spp);
        trace(&mut candidate, &fixture, &emitters, &mut b, spp);
        for (a, b) in a.faces.iter().zip(&b.faces) {
            assert_eq!((a.samples, a.sequence), (b.samples, b.sequence));
            for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                assert!(
                    (a - b).abs() < 1e-4 * (0.02 + a.abs()),
                    "spp={spp} {a} vs {b}"
                );
            }
        }
    }
}

#[test]
#[ignore = "native cooperative lane-layout cost sweep for fresh path sharing"]
fn reuse_lane_cost_sweep() {
    let mut baseline = super::tests::native_tracer();
    let fixture = cave();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let mut probes = Vec::new();
    for y in 7..40 {
        for z in 7..57 {
            probes.push(SurfaceProbe::new(IVec3::new(4, y, z), 1));
        }
    }
    let mut candidates: Vec<_> = [1, 2, 4, 8, 16, 32]
        .into_iter()
        .map(|lanes| {
            let mut tracer = GpuTracer::new(baseline.context.clone()).unwrap();
            tracer.enable_reuse().unwrap();
            tracer.fixed_path_lanes = Some(lanes);
            tracer
        })
        .collect();
    let mut works: Vec<_> = (0..7).map(|_| fresh(&probes, 1, 0)).collect();
    let mut times: Vec<Vec<f32>> = (0..7).map(|_| Vec::new()).collect();
    for round in 0..24 {
        for k in 0..7 {
            let mode = (k + round) % 7;
            let tracer = if mode == 0 {
                &mut baseline
            } else {
                &mut candidates[mode - 1]
            };
            trace(tracer, &fixture, &emitters, &mut works[mode], 16);
            if round >= 4 {
                times[mode].push(tracer.last_gpu_ms.unwrap());
            }
        }
    }
    for (mode, ms) in times.into_iter().enumerate() {
        eprintln!(
            "REUSE_LANES lanes={} p50={} p95={}",
            [0, 1, 2, 4, 8, 16, 32][mode],
            percentile(ms.clone(), 0.5),
            percentile(ms, 0.95)
        );
    }
    for candidate in &works[2..] {
        for (a, b) in works[1].faces.iter().zip(&candidate.faces) {
            for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                assert!((a - b).abs() < 1e-4 * (0.02 + a.abs()));
            }
        }
    }
}

#[test]
#[ignore = "native exact zero-energy reconnection pruning and alternating cost"]
fn zero_energy_reconnection_pruning_is_exact() {
    let device = super::tests::native_tracer().context;
    let original = SHADER.replace("if paired && record_has_energy(record)", "if paired");
    assert_ne!(original, SHADER);
    let mut a = GpuTracer::with_shader(device.clone(), true, &original).unwrap();
    let mut b = GpuTracer::new(device).unwrap();
    a.enable_reuse().unwrap();
    b.enable_reuse().unwrap();
    let fixture = cave();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let probes: Vec<_> = (7..40)
        .flat_map(|y| (7..57).map(move |z| SurfaceProbe::new(IVec3::new(4, y, z), 1)))
        .collect();
    let mut left = fresh(&probes, 1, 319);
    let mut right = fresh(&probes, 1, 319);
    let mut costs = [Vec::new(), Vec::new()];
    for round in 0..48 {
        for mode in [round % 2, 1 - round % 2] {
            let (gpu, work) = if mode == 0 {
                (&mut a, &mut left)
            } else {
                (&mut b, &mut right)
            };
            trace(gpu, &fixture, &emitters, work, 16);
            if round >= 8 {
                costs[mode].push(gpu.last_gpu_ms.unwrap());
            }
        }
        for (a, b) in left.faces.iter().zip(&right.faces) {
            assert_eq!((a.samples, a.sequence), (b.samples, b.sequence));
            for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                assert!((a - b).abs() <= 1e-6 * (0.02 + a.abs()), "mean {a} vs {b}");
            }
            for (a, b) in channels(a.second_moment)
                .into_iter()
                .zip(channels(b.second_moment))
            {
                assert!(
                    (a - b).abs() <= 1e-6 * (0.02 + a.abs()),
                    "moment {a} vs {b}"
                );
            }
        }
    }
    eprintln!(
        "REUSE_ZERO original_p50={} pruned_p50={} original_p95={} pruned_p95={}",
        percentile(costs[0].clone(), 0.5),
        percentile(costs[1].clone(), 0.5),
        percentile(costs[0].clone(), 0.95),
        percentile(costs[1].clone(), 0.95)
    );
}

#[test]
#[ignore = "native five-band recorded-path and compatible reuse study"]
fn compatible_reuse_native_study() {
    let mut baseline = super::tests::native_tracer();
    let mut prioritized_control = GpuTracer::new(baseline.context.clone()).unwrap();
    prioritized_control.enable_resident().unwrap();
    let mut candidate = GpuTracer::new(baseline.context.clone()).unwrap();
    candidate.enable_reuse().unwrap();
    let mut fixture = cave();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let importance = std::env::var("VOXEL_GI_STUDY_SIM_IMPORTANCE")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(1.0);
    let prioritized = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        super::super::EnergySpectrum::new(Vec3::ONE, importance, importance),
    ));
    // Sparse, unpaired receivers verify recording/reconnection does not change
    // the original estimator, including terminal emission and all five bands.
    let sparse = fixture.receivers.clone();
    let mut a = fresh(&sparse, 1, 0);
    let mut b = fresh(&sparse, 1, 0);
    trace(&mut baseline, &fixture, &emitters, &mut a, 16);
    trace(&mut candidate, &fixture, &emitters, &mut b, 16);
    assert!(b.faces.iter().all(|f| f.neighbor == 0));
    for (a, b) in a.faces.iter().zip(&b.faces) {
        for (a, b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
            assert!(
                (a - b).abs() < 0.0001 * (0.02 + a.abs()),
                "unpaired {a} vs {b}"
            );
        }
    }
    // Dense 3x3 patches cover walls, ceiling, lava, blue spire and green sources.
    let mut keys = bevy::platform::collections::HashSet::new();
    let mut dense = Vec::new();
    for probe in &sparse {
        let face = probe.face_mask.trailing_zeros() as usize;
        let normal = FACE_OFFSETS[face];
        let axes: Vec<_> = [IVec3::X, IVec3::Y, IVec3::Z]
            .into_iter()
            .filter(|a| a.dot(normal) == 0)
            .collect();
        for x in -1..=1 {
            for y in -1..=1 {
                let p = probe.voxel + axes[0] * x + axes[1] * y;
                if fixture.scene.material_at(p).is_some_and(|m| m != AIR)
                    && fixture.scene.material_at(p + normal) == Some(AIR)
                    && keys.insert((p, face))
                {
                    dense.push(SurfaceProbe::new(p, 1 << face));
                }
            }
        }
    }
    fixture.receivers = dense;
    let mut paired = fresh(&fixture.receivers, 888, 0);
    trace(&mut candidate, &fixture, &emitters, &mut paired, 16);
    for (i, face) in paired.faces.iter().enumerate() {
        if face.neighbor > 0 {
            assert_eq!(
                paired.faces[face.neighbor as usize - 1].neighbor,
                i as u32 + 1
            );
        }
    }
    // The expanded proposal admits formerly clipped density ratios. Its mean
    // is qualified below against the independent high-sample path estimator.
    assert!(paired.faces[2].radiance.visible.x.is_finite());
    // Wider patches include donor sites outside each corner's 2x2 footprint.
    for probe in &sparse {
        let face = probe.face_mask.trailing_zeros() as usize;
        let normal = FACE_OFFSETS[face];
        let axes: Vec<_> = [IVec3::X, IVec3::Y, IVec3::Z]
            .into_iter()
            .filter(|a| a.dot(normal) == 0)
            .collect();
        for x in -2..=2 {
            for y in -2..=2 {
                let p = probe.voxel + axes[0] * x + axes[1] * y;
                if fixture.scene.material_at(p).is_some_and(|m| m != AIR)
                    && fixture.scene.material_at(p + normal) == Some(AIR)
                    && keys.insert((p, face))
                {
                    fixture.receivers.push(SurfaceProbe::new(p, 1 << face));
                }
            }
        }
    }
    let mut reference = fresh(&fixture.receivers, 2, 8_388_608);
    for _ in 0..8 {
        trace(&mut baseline, &fixture, &emitters, &mut reference, 8192);
    }
    let refs: Vec<_> = reference
        .faces
        .iter()
        .map(|f| channels(f.radiance))
        .collect();
    eprintln!("REUSE_REFERENCE faces={} samples=65536", refs.len());
    for spp in [16u32, 64, 256, 1024, 2048, 8192] {
        for (mode, tracer) in [
            ("baseline", &mut baseline),
            ("priority", &mut prioritized_control),
            ("reuse", &mut candidate),
        ] {
            let mut errors = Vec::new();
            let mut corners = Vec::new();
            let mut gpu_ms = Vec::new();
            let mut bias = [0.0f64; 5];
            let mut paired = 0;
            for trial in 0..4 {
                let mut work = fresh(
                    &fixture.receivers,
                    u64::from(spp) * 16 + trial + 100,
                    trial as u32 * 1_048_576,
                );
                let mut cost = 0.0;
                for _ in 0..spp / 16 {
                    trace(
                        tracer,
                        &fixture,
                        if mode != "baseline" {
                            &prioritized
                        } else {
                            &emitters
                        },
                        &mut work,
                        16,
                    );
                    cost += tracer.last_gpu_ms.unwrap();
                }
                paired = work.faces.iter().filter(|f| f.neighbor != 0).count();
                gpu_ms.push(cost);
                corners.extend(corner_errors(&work, &refs));
                for (face, reference) in work.faces.iter().zip(&refs) {
                    let values = channels(face.radiance);
                    for band in 0..5 {
                        bias[band] += (values[band] - reference[band]) as f64;
                    }
                    let a = face.radiance.visible;
                    let b = Vec3::from_slice(reference);
                    errors.push((a - b).length() / b.length().max(0.02));
                }
            }
            bias.iter_mut().for_each(|v| *v /= (refs.len() * 4) as f64);
            eprintln!(
                "REUSE_RESULT mode={mode} spp={spp} paired={paired} gpu_ms={} error_p50={} error_p95={} corner_p95={} bias={bias:?}",
                percentile(gpu_ms, 0.5),
                percentile(errors.clone(), 0.5),
                percentile(errors, 0.95),
                percentile(corners, 0.95)
            );
        }
    }
    let mut a = fresh(&fixture.receivers, 999_000, 0);
    let mut b = fresh(&fixture.receivers, 999_000, 0);
    let mut costs = [Vec::new(), Vec::new()];
    for round in 0..50 {
        if round % 2 == 0 {
            trace(&mut baseline, &fixture, &emitters, &mut a, 16);
            if round >= 10 {
                costs[0].push(baseline.last_gpu_ms.unwrap());
            }
            trace(&mut candidate, &fixture, &emitters, &mut b, 16);
            if round >= 10 {
                costs[1].push(candidate.last_gpu_ms.unwrap());
            }
        } else {
            trace(&mut candidate, &fixture, &emitters, &mut b, 16);
            if round >= 10 {
                costs[1].push(candidate.last_gpu_ms.unwrap());
            }
            trace(&mut baseline, &fixture, &emitters, &mut a, 16);
            if round >= 10 {
                costs[0].push(baseline.last_gpu_ms.unwrap());
            }
        }
    }
    eprintln!(
        "REUSE_PAIRED_COST baseline_p50={} reuse_p50={} baseline_p95={} reuse_p95={}",
        percentile(costs[0].clone(), 0.5),
        percentile(costs[1].clone(), 0.5),
        percentile(costs[0].clone(), 0.95),
        percentile(costs[1].clone(), 0.95)
    );
}
