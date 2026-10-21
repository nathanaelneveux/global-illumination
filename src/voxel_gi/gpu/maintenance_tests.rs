//! Worker lifecycle must not depend on receiver count or ray allowance.
use super::*;
use crate::voxel_gi as gi;
use bevy::{
    ecs::system::RunSystemOnce,
    tasks::{AsyncComputeTaskPool, TaskPoolBuilder, block_on},
};

fn world(
    runtime: gi::VoxelGiRuntime,
    field: gi::VoxelEnergyField,
    device: GpuTransportDevice,
) -> World {
    AsyncComputeTaskPool::get_or_init(|| TaskPoolBuilder::new().num_threads(1).build());
    let mut world = World::new();
    world.insert_resource(runtime);
    world.insert_resource(field);
    world.insert_resource(device);
    world.init_resource::<gi::VoxelGiSettings>();
    world.init_resource::<gi::VoxelGiStats>();
    world.init_resource::<gi::telemetry::GiWorkTimings>();
    world
}

fn run_worker(world: &mut World) -> gi::TraceBatchResult {
    world
        .run_system_once(gi::dispatch::start_trace_task)
        .unwrap();
    let mut runtime = world.resource_mut::<gi::VoxelGiRuntime>();
    let result = block_on(runtime.trace_task.take().expect("maintenance worker"));
    assert!(result.gpu_error.is_none(), "{:?}", result.gpu_error);
    assert!(result.pending.is_empty(), "maintenance needs no readback");
    assert_eq!(
        runtime.batch_counter, 0,
        "maintenance does not advance ray batches"
    );
    result
}

#[test]
#[ignore = "native far-only scene publication regression"]
fn far_only_model_changes_publish_without_near_receivers() {
    let device = super::tests::native_tracer().context;
    let (scene, sources, _) = super::tests::open_floor_fixture();
    let runtime = gi::VoxelGiRuntime { scene, sources, ..default() };
    let mut world = world(runtime, default(), device);
    let first = run_worker(&mut world);
    let held = first.gpu_tracer.as_ref().unwrap().view_scene().unwrap();
    assert_eq!(held.params.transport.y, 0.0);
    {
        let mut runtime = world.resource_mut::<gi::VoxelGiRuntime>();
        runtime.gpu_tracer = first.gpu_tracer;
        runtime.emission_cache = first.emission_cache;
        runtime.sources.environment.visible *= 0.5;
        runtime.sources.directional.clear();
        runtime.sources.points.push(gi::sources::PointTransportLight {
            position: Vec3::new(8.0, 8.0, 8.0), intensity: gi::EnergySpectrum::ZERO,
            range: 32.0, custom_visible: Vec3::ONE,
        });
        let mut materials = (*runtime.materials).clone();
        let mut material = materials.get(2);
        material.emission.visible = Vec3::splat(4.0);
        materials.set(2, material);
        runtime.materials = Arc::new(materials);
    }
    let next = run_worker(&mut world);
    let current = next.gpu_tracer.as_ref().unwrap().view_scene().unwrap();
    assert_eq!(current.params.environment, held.params.environment * Vec4::new(0.5, 0.5, 0.5, 1.0));
    assert_eq!(current.params.maximum.w, 0);
    assert_eq!(current.params.transport.y, 1.0, "custom direct RGB reaches far presentation");
    for index in [3, 4, 5] {
        assert_ne!(held.buffers[index].id(), current.buffers[index].id(),
            "changed model allocation must not mutate a retained descriptor");
    }
    assert_eq!(held.buffers[0].id(), current.buffers[0].id(), "unchanged geometry shares storage");
    assert!(current.params.extra.w > 0.0, "far-only emitters are built");
}

#[test]
#[ignore = "native worker lifecycle regression"]
fn retirements_survive_poll_only_and_empty_receiver_dispatch() {
    let device = super::tests::native_tracer().context;
    // Poll-only backpressure, no receivers, and exhausted budget without any
    // pending readback must all deliver the same retirement transaction.
    for (near_receiver, pending_poll) in [(true, true), (false, false), (true, false)] {
        let mut gpu = GpuTracer::new(device.clone()).unwrap();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = super::tests::open_floor_fixture();
        let mut work = TraceWork::from_probes(&probes[..1], false);
        gpu.trace_compact(
            scene.clone(),
            &sources,
            &default(),
            &mut work,
            1,
            &Arc::new(EmissionDistribution::default()),
            false,
        )
        .unwrap();
        let address = work.faces[0].resident;
        let retired = gi::page_of(work.faces[0].voxel);
        let mut runtime = gi::VoxelGiRuntime::default();
        runtime.scene = scene;
        runtime.gpu_tracer = Some(gpu);
        runtime.gpu_allowance_ms = -100.0;
        runtime.slot_retirements.push(retired);
        if pending_poll {
            runtime.gpu_pending.push_back(gi::PendingTransport {
                snapshot: None,
                batch: default(),
            });
        }
        let mut field = gi::VoxelEnergyField::default();
        if near_receiver {
            let other = IVec3::new(64, 4, 8);
            field.order.push(other);
            field.probes.insert(other, SurfaceProbe::new_slim(other, 4));
        }
        let mut world = world(runtime, field, device.clone());
        let result = run_worker(&mut world);
        assert!(
            world
                .resource::<gi::VoxelGiRuntime>()
                .slot_retirements
                .is_empty()
        );
        let tracer = result.gpu_tracer.unwrap();
        assert!(
            tracer
                .resident
                .as_ref()
                .unwrap()
                .picked_face(address.x, address.y)
                .is_none()
        );
        // Inspect the GPU index, not just CPU ownership: paths must no longer
        // terminate on the retired face, even without a subsequent ray batch.
        let index = tracer.resident.as_ref().unwrap().cache_buffers().1;
        let words = read_words(&tracer.context, &index.buffer, index.capacity);
        let entries = words[0] as usize + 1;
        assert!(
            words[4..4 + entries * 4]
                .chunks_exact(4)
                .all(|entry| entry[3] == u32::MAX)
        );
    }
}

pub(super) fn read_words(context: &GpuTransportDevice, source: &Buffer, size: u64) -> Vec<u32> {
    let mut readback = UploadBuffer::new(
        &context.device,
        "maintenance assertion",
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    );
    readback.ensure(&context.device, size as usize).unwrap();
    let mut encoder = context.device.create_command_encoder(&default());
    encoder.copy_buffer_to_buffer(source, 0, &readback.buffer, 0, size);
    let submission = context.queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback
        .buffer
        .slice(..size)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    context
        .device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let view = readback.buffer.slice(..size).get_mapped_range();
    let words = view
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect();
    drop(view);
    readback.buffer.unmap();
    words
}

#[test]
#[ignore = "native transport cache allocation-generation regression"]
fn retained_cache_pairs_stay_coherent_when_either_member_grows() {
    let mut gpu = super::tests::native_tracer();
    gpu.enable_resident().unwrap();
    let (scene, sources, probes) = super::tests::open_floor_fixture();
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::default());
    let mut work = TraceWork::from_probes(&probes[..1], false);
    gpu.trace_compact(scene.clone(), &sources, &materials, &mut work, 16, &emitters, false).unwrap();
    for (count, grow_index) in [(2, false), (600, true)] {
        let resident = gpu.resident.as_ref().unwrap();
        let (state, index) = resident.cache_buffers();
        let held = (state.shared(), index.shared());
        let state_words = read_words(&gpu.context, &held.0.buffer, held.0.capacity);
        let index_words = read_words(&gpu.context, &held.1.buffer, held.1.capacity);
        let mut work = TraceWork::from_probes(&probes[..count], false);
        gpu.trace_compact(scene.clone(), &sources, &materials, &mut work, 16, &emitters, false).unwrap();
        let (state, index) = gpu.resident.as_ref().unwrap().cache_buffers();
        assert_ne!(state.buffer.id(), held.0.buffer.id());
        assert_ne!(index.buffer.id(), held.1.buffer.id());
        assert_eq!(index.capacity > held.1.capacity, grow_index);
        assert_eq!(read_words(&gpu.context, &held.0.buffer, held.0.capacity), state_words);
        assert_eq!(read_words(&gpu.context, &held.1.buffer, held.1.capacity), index_words);
        // Seeded histories survive the copy; later merges do not mutate the
        // retained generation. Check native words rather than CPU ownership.
        let current = read_words(&gpu.context, &state.buffer, state.capacity);
        assert!(current[5] >= 32, "first face's history survived growth");
    }
}

#[test]
#[ignore = "native far-only geometry maintenance regression"]
fn far_only_scene_initializes_updates_and_then_idles_without_near_receivers() {
    let device = super::tests::native_tracer().context;
    let (scene, _, _) = super::tests::open_floor_fixture();
    let mut runtime = gi::VoxelGiRuntime::default();
    runtime.scene = scene.clone();
    runtime.gpu_allowance_ms = -100.0;
    let mut world = world(runtime, default(), device);
    let result = run_worker(&mut world);
    assert!(result.geometry.is_some());
    assert!(
        result
            .gpu_tracer
            .as_ref()
            .unwrap()
            .scene
            .as_ref()
            .is_some_and(|s| Arc::ptr_eq(s, &scene))
    );
    {
        let mut runtime = world.resource_mut::<gi::VoxelGiRuntime>();
        runtime.gpu_tracer = result.gpu_tracer;
        runtime.scene = Arc::new(TransportScene::default());
    }
    let result = run_worker(&mut world);
    assert_eq!(
        result.geometry.as_ref().unwrap().1.counts.z,
        0,
        "empty directory mask"
    );
    let uploaded = result.gpu_tracer.as_ref().unwrap().uploaded_geometry_bytes;
    world.resource_mut::<gi::VoxelGiRuntime>().gpu_tracer = result.gpu_tracer;
    world
        .run_system_once(gi::dispatch::start_trace_task)
        .unwrap();
    let runtime = world.resource::<gi::VoxelGiRuntime>();
    assert!(
        runtime.trace_task.is_none(),
        "unchanged far-only world stays idle"
    );
    assert_eq!(
        runtime.gpu_tracer.as_ref().unwrap().uploaded_geometry_bytes,
        uploaded
    );
}
