//! Raw-prefix qualification: grouping sequential visits must not change paths,
//! moments, unknown accounting or unfinished tails. No reconstruction runs.
use super::*;
use crate::voxel_gi::{emission::EmissionDistribution, research, work::TraceWork};

#[derive(Clone, Copy, ShaderType)]
struct Receiver {
    surface: Vec3,
    face: u32,
    voxel: IVec3,
    sequence: u32,
}

#[test]
#[ignore = "native raw visible-receiver batching parity"]
fn native_view_batches_preserve_raw_prefixes_and_pending_tails() {
    let mut gpu = crate::voxel_gi::gpu::tests::native_tracer();
    let fixture = research::furnace();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene, &fixture.materials, VoxelGiSettings::default().sampling_importance(),
    ));
    let mut work = TraceWork::default();
    gpu.submit_compact(fixture.scene.clone(), &fixture.sources, &fixture.materials,
        &mut work, 0, &emitters, true).unwrap();
    let mut scene = gpu.view_scene().unwrap();
    let mut inputs = Vec::new();
    for probe in &fixture.receivers {
        for face in 0..6 {
            if probe.face_mask & (1 << face) != 0 {
                let normal = crate::voxel_gi::world::FACE_OFFSETS[face].as_vec3();
                inputs.push(Receiver { surface: probe.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5,
                    voxel: probe.voxel, face: face as u32, sequence: 0 });
            }
        }
    }
    assert!(!inputs.is_empty());
    let count = inputs.len() as u32;
    scene.params.counts.x = count;
    let device = &gpu.context.device;
    let queue = &gpu.context.queue;
    let mut world = World::new();
    world.insert_resource(device.clone());
    world.insert_resource(queue.clone());
    let renderer = ViewRenderer::from_world(&mut world);
    let kernel = &renderer.trace[0];
    let run = |visits: u32| {
        let mut camera = CameraState::new(device, count);
        let history = &mut camera.history[0];
        history.receivers.upload(device, queue, &ShaderBuffer::from(inputs.clone()).data.unwrap()).unwrap();
        camera.params.upload(device, queue, &ShaderBuffer::from(vec![scene.params]).data.unwrap()).unwrap();
        camera.control.upload(device, queue, &ShaderBuffer::from(vec![Control {
            history: UVec4::new(0, 0, 0, 1), ..default()
        }]).data.unwrap()).unwrap();
        camera.wave.upload(device, queue, &ShaderBuffer::from(vec![Wave {
            range: UVec4::new(0, count * visits, 3, 0), ticket: UVec4::ZERO,
        }]).data.unwrap()).unwrap();
        let mut words = vec![0u32; 12 + count as usize];
        words[0] = count;
        for i in 0..count { words[12 + i as usize] = i; }
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        camera.queue.upload(device, queue, &bytes).unwrap();
        let buffers = [&camera.params.buffer, &scene.buffers[0], &scene.buffers[1],
            &scene.buffers[2], &scene.buffers[3], &scene.buffers[4], &history.receivers.buffer,
            &history.paths.buffer, &scene.buffers[5], &scene.buffers[6], &scene.buffers[7]];
        let groups: Vec<_> = kernel.bindings.iter().enumerate().map(|(group, bindings)| {
            let entries: Vec<_> = bindings.iter().map(|&binding| {
                let buffer = match (group, binding) {
                    (0, _) => buffers[binding as usize],
                    (1, 0) => &history.tails.buffer,
                    (1, 1) => &camera.wave.buffer,
                    (2, 0) => &camera.control.buffer,
                    (2, 5) => &history.estimates.buffer,
                    (2, 10) => &camera.queue.buffer,
                    _ => panic!("unexpected trace binding"),
                };
                BindGroupEntry { binding, resource: buffer.as_entire_binding() }
            }).collect();
            device.create_bind_group("Raw batching parity", &kernel.pipeline.get_bind_group_layout(group as u32).into(), &entries)
        }).collect();
        let mut encoder = device.create_command_encoder(&default());
        for _ in 0..128 / visits {
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&kernel.pipeline);
            for (i, group) in groups.iter().enumerate() { pass.set_bind_group(i as u32, &**group, &[]); }
            pass.dispatch_workgroups(count.div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        [&history.receivers, &history.paths, &history.tails, &history.estimates].map(|buffer| {
            crate::voxel_gi::gpu::maintenance_tests::read_words(&gpu.context, &buffer.buffer, buffer.capacity)
        })
    };
    let serial = run(1);
    assert_eq!(serial, run(8));
    assert_eq!(serial, run(32));
    assert!(serial[3].chunks_exact(16).any(|e| f32::from_bits(e[3]) > 16.0));
}
