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

fn percentile(mut xs: Vec<f32>, p: f32) -> f32 {
    xs.sort_by(f32::total_cmp);
    xs[((xs.len() - 1) as f32 * p).round() as usize]
}

#[test]
#[ignore = "native paired workgroup partial receiver and sample tails"]
fn paired_reuse_handles_partial_workgroups_and_sample_lanes() {
    let mut baseline = super::tests::native_tracer();
    let mut candidate = GpuTracer::new(baseline.context.clone()).unwrap();
    candidate.enable_reuse().unwrap();
    let fixture = cave();
    let emitters = Arc::new(EmissionDistribution::build(&fixture.scene, &fixture.materials));
    for spp in [1, 3, 4, 7, 16] {
        let mut a = fresh(&fixture.receivers[..17], spp as u64, 99);
        let mut b = fresh(&fixture.receivers[..17], spp as u64, 99);
        trace(&mut baseline, &fixture, &emitters, &mut a, spp);
        trace(&mut candidate, &fixture, &emitters, &mut b, spp);
        for (a,b) in a.faces.iter().zip(&b.faces) {
            assert_eq!((a.samples,a.sequence),(b.samples,b.sequence));
            for (a,b) in channels(a.radiance).into_iter().zip(channels(b.radiance)) {
                assert!((a-b).abs() < 1e-4 * (0.02+a.abs()), "spp={spp} {a} vs {b}");
            }
        }
    }
}

#[test]
#[ignore = "native five-band recorded-path and compatible reuse study"]
fn compatible_reuse_native_study() {
    let mut baseline = super::tests::native_tracer();
    let mut candidate = GpuTracer::new(baseline.context.clone()).unwrap();
    candidate.enable_reuse().unwrap();
    let mut fixture = cave();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
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
    // This lane's first sample reconnects, its second is rejected. Metal must
    // write zero, not carry the first iteration's donated energy into sample 8.
    // Golden checked against the independent recorded-path gather kernel.
    assert!((paired.faces[2].radiance.visible.x - 1.8486694).abs() < 2e-5);
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
    for (mode, tracer) in [("baseline", &mut baseline), ("reuse", &mut candidate)] {
        for spp in [16u32, 64, 256, 1024, 2048, 8192] {
            let mut errors = Vec::new();
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
                    trace(tracer, &fixture, &emitters, &mut work, 16);
                    cost += tracer.last_gpu_ms.unwrap();
                }
                paired = work.faces.iter().filter(|f| f.neighbor != 0).count();
                gpu_ms.push(cost);
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
                "REUSE_RESULT mode={mode} spp={spp} paired={paired} gpu_ms={} error_p50={} error_p95={} bias={bias:?}",
                percentile(gpu_ms, 0.5),
                percentile(errors.clone(), 0.5),
                percentile(errors, 0.95)
            );
        }
    }
}
