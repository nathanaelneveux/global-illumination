//! Variable-model CPU/Metal parity and range semantics, not a timing benchmark.
use super::*;
use crate::voxel_gi::{
    TransportSettings,
    naadf::{linear_index, volume_len},
    sources::DirectionalTransportLight,
    tracer::trace_work_batch,
    world::{AIR, MaterialTransport, PAGE_SIZE},
};

#[test]
#[ignore = "requires a native compute adapter"]
fn tuning_short_reach_preserves_known_empty_streaming_frontier() {
    let mut gpu = super::tests::native_tracer();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    let voxel = IVec3::new(31, 8, 8);
    page[linear_index(voxel.as_uvec3(), PAGE_SIZE)] = 1;
    let base = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let extended = Arc::new(TransportScene::with_page_delta(
        &base,
        vec![(IVec3::X, vec![AIR; volume_len(PAGE_SIZE)])],
        &[],
    ));
    let sources = TransportSources::default();
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::default());
    let mut results = Vec::new();
    for scene in [base, extended] {
        let receivers = [SurfaceProbe::new(voxel, 1)];
        let mut cpu = TraceWork::from_probes(&receivers, false);
        let mut native = TraceWork::from_probes(&receivers, false);
        cpu.transport.reach = 16.0;
        native.transport = cpu.transport;
        trace_work_batch(
            &scene,
            &sources,
            &materials,
            &mut cpu,
            256,
            &emitters,
            &mut default(),
        );
        gpu.trace_compact(
            scene,
            &sources,
            &materials,
            &mut native,
            256,
            &emitters,
            false,
        )
        .unwrap();
        assert!(
            (cpu.faces[0].radiance - native.faces[0].radiance)
                .visible
                .abs()
                .max_element()
                < 0.001
        );
        results.push(native.faces[0].radiance);
    }
    assert_eq!(results[0], results[1]);
    assert!(results[0].visible.max_element() > 0.1);
}

#[test]
#[ignore = "requires a native compute adapter"]
fn tuning_reach_depth_and_batch_size_match_cpu_without_sky_leaks() {
    let mut gpu = super::tests::native_tracer();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    // Roof is 27 voxels from the receiver: a 16-voxel path must not turn a
    // sealed room into sky. Emitting ceiling also exercises terminal-depth MIS.
    for z in 2..=30 {
        for y in 2..=30 {
            for x in 2..=30 {
                if [x, y, z].into_iter().any(|v| v == 2 || v == 30) {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] =
                        if y == 30 { 2 } else { 1 };
                }
            }
        }
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(
        1,
        MaterialTransport {
            diffuse: EnergySpectrum::ONE * 0.6,
            emission: EnergySpectrum::ZERO,
        },
    );
    materials.set(
        2,
        MaterialTransport {
            diffuse: EnergySpectrum::ONE * 0.4,
            emission: EnergySpectrum::new(Vec3::new(1.0, 2.0, 4.0), 3.0, 2.0),
        },
    );
    let sources = TransportSources {
        environment: EnergySpectrum::ONE,
        directional: vec![DirectionalTransportLight {
            direction: Vec3::Y,
            radiance: EnergySpectrum::ONE,
            custom_visible: Vec3::ZERO,
        }],
        points: vec![],
    };
    let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
    let receivers = [
        SurfaceProbe::new(IVec3::new(16, 2, 16), 4),
        SurfaceProbe::new(IVec3::new(2, 12, 16), 1),
    ];
    let mut sky = super::super::sky_cache::SkyCache::default();
    for reach in [1.0, 16.0, 64.0, 768.0] {
        for bounces in [1, 2, 3] {
            let mut cpu = TraceWork::from_probes(&receivers, false);
            let mut native = TraceWork::from_probes(&receivers, false);
            cpu.transport = TransportSettings { reach, bounces };
            native.transport = cpu.transport;
            // Nonmultiples of the four cooperative lanes and several batches.
            for samples in [1, 3, 4, 16, 256] {
                trace_work_batch(
                    &scene, &sources, &materials, &mut cpu, samples, &emitters, &mut sky,
                );
                gpu.trace_compact(
                    scene.clone(),
                    &sources,
                    &materials,
                    &mut native,
                    samples,
                    &emitters,
                    false,
                )
                .unwrap();
                for (a, b) in cpu.faces.iter().zip(&native.faces) {
                    assert_eq!(a.samples, b.samples);
                    assert_eq!(a.sequence, b.sequence);
                    assert!(
                        (a.radiance - b.radiance).visible.abs().max_element() < 0.003,
                        "reach={reach} bounces={bounces} samples={samples}: {:?} != {:?}",
                        a.radiance,
                        b.radiance
                    );
                    assert!((a.radiance.infrared - b.radiance.infrared).abs() < 0.003);
                    assert!((a.radiance.ionizing - b.radiance.ionizing).abs() < 0.003);
                    assert!(
                        (a.second_moment - b.second_moment)
                            .visible
                            .abs()
                            .max_element()
                            < 0.02
                    );
                    assert_eq!(
                        a.direct,
                        EnergySpectrum::ZERO,
                        "roof must still shadow direct sun"
                    );
                    assert_eq!(a.direct, b.direct);
                    if reach == 1.0 {
                        assert_eq!(a.radiance, EnergySpectrum::ZERO);
                    }
                    if reach >= 64.0 {
                        assert!(a.radiance.visible.max_element() > 0.01);
                    }
                }
                for (a, b) in cpu.probes.iter().zip(&native.probes) {
                    assert_eq!(a.sky, b.sky);
                }
            }
        }
    }
    // No emitters: finite diffuse rays must never invent environment through
    // the roof even though the environment itself is bright.
    materials.set(2, materials.get(1));
    assert_eq!(
        gpu.quality_pipelines.len(),
        12,
        "reach values reuse three depths × two horizon modes × two lane layouts"
    );
    let emitters = Arc::new(EmissionDistribution::default());
    for reach in [1.0, 16.0, 768.0] {
        let mut work = TraceWork::from_probes(&receivers, false);
        work.transport.reach = reach;
        gpu.trace_compact(
            scene.clone(),
            &sources,
            &materials,
            &mut work,
            512,
            &emitters,
            false,
        )
        .unwrap();
        assert!(
            work.faces
                .iter()
                .all(|face| face.radiance == EnergySpectrum::ZERO)
        );
    }
}
