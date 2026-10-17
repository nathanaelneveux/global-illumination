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

/// A contribution ledger for the reproducible loaded-world witness. Rebuilds
/// the sum from production connection routines, checking each term against the
/// ordinary CPU estimator. This is not an independent correctness oracle.
#[test]
#[ignore = "CPU exact-path weight ledger for the loaded hallway witness"]
fn hallway_known_path_weight_ledger() {
    use super::super::research::channels;
    use serde_json::json;
    let fixture = super::super::research::hallway_loaded();
    let emitters = EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        super::super::VoxelGiSettings::default().sampling_importance(),
    );
    let probe = &fixture.receivers[208];
    let sequence = 2243;
    let stream = stream_id(probe.voxel, 2);
    let mut normal = Vec3::Y;
    let mut surface = probe.voxel.as_vec3() + Vec3::new(0.5, 1.0, 0.5);
    let mut origin = surface + normal * RAY_EPSILON;
    let mut direction = hemisphere(
        normal,
        sample_1d(stream, sequence, 0),
        sample_1d(stream, sequence, 1),
    );
    let mut throughput = EnergySpectrum::ONE;
    let primary = sample_emitter(
        &fixture.scene,
        &fixture.materials,
        &emitters,
        surface,
        normal,
        stream,
        sequence,
        0,
        MAX_PATH_DISTANCE,
        DirectionGuide::default(),
    );
    let mut sum = primary;
    let mut rows = Vec::new();
    for bounce in 0..3 {
        let TraceResult::Hit(hit) =
            fixture
                .scene
                .trace_ray_result(origin, direction, MAX_PATH_DISTANCE)
        else {
            panic!("witness must hit at every bounce");
        };
        let material = fixture.materials.get(hit.material);
        assert_eq!(
            material.emission.max_component(),
            0.0,
            "this witness only receives connection emission"
        );
        let direct = direct_radiance(&fixture.scene, &fixture.sources, hit).0;
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
        let contribution = throughput * material.diffuse * (direct + nee);
        sum += contribution;
        let mut row = json!({"bounce":bounce,"position":hit.position.to_array(),
            "normal":hit.normal.to_array(),"direction":direction.to_array(),"distance":hit.distance,
            "throughput_before":channels(throughput),"reflectance":channels(material.diffuse),
            "connection_estimate":channels(nee),"contribution":channels(contribution)});
        if bounce == 2 {
            let coin = sample_1d(stream, sequence, 8 + bounce);
            assert!(coin < 0.5, "known witness uses the light branch");
            let light_stream = stream ^ pcg((bounce + 1).wrapping_add(0xa341_316c));
            let selected = emitters
                .select_at(
                    hit.position,
                    hit.normal,
                    sample_1d(light_stream, sequence, 12),
                )
                .unwrap();
            let target = face_point(
                selected,
                sample_1d(light_stream, sequence, 13),
                sample_1d(light_stream, sequence, 14),
            );
            let delta = target - hit.position;
            let d2 = delta.length_squared();
            let wi = delta.normalize();
            let emitter_cosine = FACE_OFFSETS[selected.face as usize].as_vec3().dot(-wi);
            let cosine_pdf = hit.normal.dot(wi) / std::f32::consts::PI;
            let light_pdf = selected.probability * d2 / emitter_cosine;
            let queried = emitters.probability_at(
                hit.position,
                hit.normal,
                selected.voxel,
                selected.face,
                selected.material,
                &fixture.materials,
            );
            assert!((queried - selected.probability).abs() < selected.probability * 1e-5);
            let emission = fixture.materials.get(selected.material).emission;
            let factor = 2.0 * cosine_pdf / (light_pdf + cosine_pdf);
            for (a, b) in channels(nee).into_iter().zip(channels(emission * factor)) {
                assert!((a - b).abs() < 1e-5);
            }
            row["terminal_light"] = json!({"branch_coin":coin,"branch_probability":0.5,
                "emitter_voxel":selected.voxel.to_array(),"emitter_face":selected.face,
                "target":target.to_array(),"area":1.0,"distance":d2.sqrt(),
                "selection_probability":selected.probability,"queried_probability":queried,
                "cosine_pdf":cosine_pdf,"light_pdf":light_pdf,"mixture_factor":factor,
                "emission":channels(emission)});
        } else {
            throughput = throughput * material.diffuse;
            if bounce >= 1 {
                let survival = throughput.max_component().clamp(0.1, 0.95);
                let coin = sample_1d(stream, sequence, 8 + bounce);
                assert!(coin <= survival);
                row["roulette"] =
                    json!({"coin":coin,"survival":survival,"multiplier":survival.recip()});
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
        rows.push(row);
    }
    let mut work = TraceWork::default();
    work.push(probe, 4, 0, 0, false, u32::MAX, u32::MAX);
    work.faces[0].sequence = sequence;
    trace_face(
        &fixture.scene,
        &fixture.sources,
        &fixture.materials,
        &mut work.faces[0],
        1,
        &emitters,
        work.transport,
        false,
    );
    for (a, b) in channels(sum)
        .into_iter()
        .zip(channels(work.faces[0].radiance))
    {
        assert!((a - b).abs() < 1e-5);
    }
    let ledger = json!({"site":208,"sequence":sequence,"primary":channels(primary),
        "total":channels(sum),"bounces":rows});
    eprintln!("WEIGHT LEDGER {ledger}");
    std::fs::write(
        "target/hallway-path-2243-ledger.json",
        serde_json::to_vec_pretty(&ledger).unwrap(),
    )
    .unwrap();
}
