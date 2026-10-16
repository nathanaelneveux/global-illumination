//! Opt-in exact-path diagnosis. No scheduling, history retention or display.
use super::*;

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
