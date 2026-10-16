//! Opt-in exact-path diagnosis. No scheduling, history retention or display.
use super::*;

#[test]
fn hallway_guide_anchors() {
    let fixture = super::super::research::hallway_loaded();
    let emitters = EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        super::super::VoxelGiSettings::default().sampling_importance(),
    );
    for index in [10, 70, 98, 140, 172, 193, 208, 209, 235] {
        let p = fixture.receivers[index].voxel;
        let surface = p.as_vec3() + Vec3::new(0.5, 1., 0.5);
        let visible = |voxel: IVec3| {
            let delta = voxel.as_vec3() + Vec3::splat(0.5) - surface;
            matches!(fixture.scene.trace_ray_result(surface+Vec3::Y*0.002,delta.normalize(),delta.length()+0.01), TraceResult::Hit(hit) if hit.voxel == voxel)
        };
        println!(
            "GUIDE receiver={p:?} anchor={:?}",
            emitters.guide_at(surface, Vec3::Y, visible)
        );
        assert_eq!(
            emitters.guide_at(surface, Vec3::Y, visible).unwrap().voxel,
            IVec3::new(-124, -11, -20)
        );
    }
}

#[test]
#[ignore = "native guided CPU/GPU matched-path parity"]
fn guided_native_prefix_parity() {
    let mut gpu = super::super::gpu::tests::native_tracer();
    for fixture in [
        super::super::research::hallway(),
        super::super::research::hallway_loaded(),
        super::super::research::cave(),
    ] {
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            super::super::VoxelGiSettings::default().sampling_importance(),
        ));
        for start in [0, 2240, 574117] {
            let mut work = TraceWork {
                emitter_guiding: true,
                ..default()
            };
            for probe in fixture
                .receivers
                .iter()
                .step_by(7)
                .chain(fixture.receivers.get(208))
            {
                work.push(probe, probe.face_mask, 0, 0, false, u32::MAX, u32::MAX);
                work.probes.last_mut().unwrap().sky_requested = false;
                work.faces.last_mut().unwrap().sequence = start;
            }
            let mut cpu = work.faces.clone();
            for face in &mut cpu {
                trace_face(
                    &fixture.scene,
                    &fixture.sources,
                    &fixture.materials,
                    face,
                    16,
                    &emitters,
                    work.transport,
                    true,
                );
            }
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
            for (cpu, gpu) in cpu.iter().zip(&work.faces) {
                for (a, b) in super::super::research::channels(cpu.radiance)
                    .into_iter()
                    .zip(super::super::research::channels(gpu.radiance))
                {
                    assert!(
                        (a - b).abs() < 0.00005 + a.abs() * 0.003,
                        "receiver={:?}/{} start={start}: cpu={a} gpu={b}",
                        cpu.voxel,
                        cpu.face
                    );
                }
            }
        }
    }
}

#[test]
fn guide_hemisphere_rejection_never_tests_visibility() {
    for fixture in [
        super::super::research::hallway(),
        super::super::research::hallway_loaded(),
    ] {
        let emitters = EmissionDistribution::build(&fixture.scene, &fixture.materials);
        for normal in FACE_OFFSETS.map(|n| n.as_vec3()) {
            assert!(
                emitters
                    .guide_at(normal * 4096.0, normal, |_| panic!(
                        "behind-plane source visibility query"
                    ))
                    .is_none()
            );
        }
    }
}

#[test]
fn guided_hallway_batch_avoids_the_reproduced_bright_return_path() {
    let fixture = super::super::research::hallway_loaded();
    let emitters = EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        super::super::VoxelGiSettings::default().sampling_importance(),
    );
    let mut work = TraceWork::default();
    work.push(&fixture.receivers[208], 4, 0, 0, false, u32::MAX, u32::MAX);
    work.faces[0].sequence = 2240;
    let mut control = work.faces[0].clone();
    let mut candidate = control.clone();
    for (face, guided) in [(&mut control, false), (&mut candidate, true)] {
        trace_face(
            &fixture.scene,
            &fixture.sources,
            &fixture.materials,
            face,
            16,
            &emitters,
            work.transport,
            guided,
        );
    }
    println!(
        "known batch: control={:?} candidate={:?}",
        control.radiance, candidate.radiance
    );
    assert!(candidate.radiance.visible.x < control.radiance.visible.x * 0.1);
}

#[test]
#[ignore = "native exact-path replay of the live hallway outlier"]
fn hallway_outlier_paths() {
    let fixture = super::super::research::hallway();
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let mut gpu = super::super::gpu::tests::native_tracer();
    let probe = &fixture.receivers[208];
    assert_eq!(probe.voxel, IVec3::new(-54, -12, -20));
    let mut maximum = (0., 0);
    for sequence in 2240..2256 {
        let mut work = TraceWork::default();
        work.push(probe, 4, 0, 0, false, u32::MAX, u32::MAX);
        work.probes[0].sky_requested = false;
        work.faces[0].sequence = sequence;
        let mut cpu = work.faces[0].clone();
        trace_face(
            &fixture.scene,
            &fixture.sources,
            &fixture.materials,
            &mut cpu,
            1,
            &emitters,
            work.transport,
            false,
        );
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
        println!(
            "path {sequence} cpu={:?} gpu={:?}",
            cpu.radiance, work.faces[0].radiance
        );
        assert!(
            (cpu.radiance.visible - work.faces[0].radiance.visible)
                .abs()
                .max_element()
                < 0.001
        );
        if cpu.radiance.visible.x > maximum.0 {
            maximum = (cpu.radiance.visible.x, sequence);
        }
    }
    let sequence = maximum.1;
    println!("maximum {maximum:?}");
    let mut normal = Vec3::Y;
    let mut surface = probe.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5;
    let mut origin = surface + normal * RAY_EPSILON;
    let stream = stream_id(probe.voxel, 2);
    let mut direction = hemisphere(
        normal,
        sample_1d(stream, sequence, 0),
        sample_1d(stream, sequence, 1),
    );
    let mut throughput = EnergySpectrum::ONE;
    for bounce in 0..3 {
        let hit = fixture
            .scene
            .trace_ray_result(origin, direction, MAX_PATH_DISTANCE);
        println!(
            "bounce {bounce} origin={origin:?} direction={direction:?} throughput={throughput:?} hit={hit:?}"
        );
        let TraceResult::Hit(hit) = hit else {
            break;
        };
        let material = fixture.materials.get(hit.material);
        let probability = emitters.probability_at(
            surface,
            normal,
            hit.voxel,
            FACE_OFFSETS
                .iter()
                .position(|n| n.as_vec3() == hit.normal)
                .unwrap() as u32,
            hit.material,
            &fixture.materials,
        );
        let cosine = normal.dot(direction).max(0.) / std::f32::consts::PI;
        let light_pdf = probability * hit.distance.powi(2) / hit.normal.dot(-direction).max(1e-8);
        let nee = if bounce == 2 {
            sample_terminal_emission(
                &fixture.scene,
                &fixture.materials,
                &emitters,
                hit.position,
                hit.normal,
                stream,
                sequence,
                bounce,
                MAX_PATH_DISTANCE,
                DirectionGuide::default(),
            )
        } else {
            sample_emitter(
                &fixture.scene,
                &fixture.materials,
                &emitters,
                hit.position,
                hit.normal,
                stream,
                sequence,
                bounce + 1,
                MAX_PATH_DISTANCE,
                DirectionGuide::default(),
            )
        };
        println!(
            "  emission={:?} pdf_light={light_pdf} pdf_cosine={cosine} nee={nee:?}",
            material.emission
        );
        if bounce == 2 {
            break;
        }
        throughput = throughput * material.diffuse;
        if bounce >= 1 {
            let survival = throughput.max_component().clamp(0.1, 0.95);
            let coin = sample_1d(stream, sequence, 8 + bounce);
            println!("  roulette={coin} survival={survival}");
            if coin > survival {
                break;
            }
            throughput *= survival.recip();
        }
        normal = hit.normal;
        surface = hit.position;
        origin = surface + normal * 0.002;
        direction = hemisphere(
            normal,
            sample_1d(stream, sequence, 2 + bounce * 2),
            sample_1d(stream, sequence, 3 + bounce * 2),
        );
    }
}
