//! Far tier: faces under the pixel threshold own no receiver, and their
//! cells trace, accumulate and reproject per pixel.
use super::coverage_tests::Rig;
use super::*;
use crate::{
    emission::EmissionDistribution,
    naadf::{linear_index, volume_len},
    work::TraceWork,
    world::{AIR, PAGE_SIZE},
};

/// IEEE half to f32 (`pack2x16float` output; no infinities expected).
fn f16(bits: u16) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 31);
    let mantissa = f32::from(bits & 1023);
    sign * if exponent == 0 { mantissa * 2f32.powi(-24) } else { (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15) }
}

#[test]
#[ignore = "native per-pixel tier"]
fn native_small_faces_are_lit_per_pixel_without_receivers() {
    // A floor seen from 96 voxels up: a face is a fraction of a pixel.
    let mut gpu = crate::gpu::tests::native_tracer();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for x in 0..32 {
        for y in 0..4 {
            for z in 0..32 {
                page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 2;
            }
        }
    }
    let scene = Arc::new(TransportScene::with_page_delta(&TransportScene::default(), vec![(IVec3::ZERO, page)], &[]));
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
    gpu.submit_compact(scene, &TransportSources::default(), &materials,
        &mut TraceWork::default(), 0, &emitters).unwrap();
    let scene = gpu.view_scene().unwrap();
    let device = &gpu.context.device;
    let queue = &gpu.context.queue;
    let rig = Rig::new(device);
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("Per-pixel tier"), source: ShaderSource::Wgsl(rig.source.clone().into()),
    });
    let kernels: Vec<_> = [("classify_pixels", PIXEL_CLASSIFY_BINDINGS), ("size_pixel_queue", [&[][..], &[], &[23, 24]]),
        ("trace_pixel_queue", PIXEL_TRACE_BINDINGS)].into_iter().map(|(name, bindings)| {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name), layout: None, module: &module, entry_point: Some(name),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("PATH_STEPS", 128.0), ("BUDGETED_RAYS", 1.0)],
                ..default()
            }, cache: None,
        });
        (pipeline, bindings)
    }).collect();
    let grid = 4u32;
    let mut camera = CameraState::new(device, UVec2::splat(grid));
    let mask = camera.history[0].hash.capacity as u32 / 4 - 1;
    camera.params.upload(device, queue, &ShaderBuffer::from(vec![scene.params]).data.unwrap()).unwrap();
    let cells = UVec2::splat(8);
    let pixels = PixelState::new(device, cells);
    let output = storage(device, "pixel output", PIXEL_HEADER_BYTES as usize + 64 * 16);
    let depths: Vec<u8> = (0..256).flat_map(|_| 0.5f32.to_le_bytes()).collect();
    let normals: Vec<u8> = (0..256).flat_map(|_| [0.5f32, 1., 0.5, 1.]).flat_map(f32::to_le_bytes).collect();
    rig.upload(queue, &depths, &normals);
    // Pixel x -> world x, pixel y -> world z (two voxels per pixel); depth
    // 0.5 -> the floor top at y = 4.
    let view_at = |shift: f32| Mat4::from_cols(Vec4::new(16., 0., 0., 0.), Vec4::new(0., 0., -16., 0.),
        Vec4::new(0., 8., 0., 0.), Vec4::new(16. + shift, 0., 16., 1.));
    let control = |frame: u32, world: Mat4, previous: Option<Mat4>, enabled: bool| Control {
        world_from_clip: world, clip_from_world: world.inverse(),
        camera: Vec4::new(16., 100., 16., 0.),
        viewport: UVec4::new(0, 0, 16, 16),
        grid: UVec4::new(grid, grid, grid * grid, mask),
        history: UVec4::new(0, mask, frame, 1),
        pixel_axis: Vec4::new(0., -1., 0., 4.),
        pixel_grid: UVec4::new(cells.x, cells.y, 2, u32::from(enabled)),
        pixel_tune: Vec4::new(4., 32., 1., f32::from_bits(if previous.is_some() { frame - 1 } else { u32::MAX })),
        previous_clip_from_world: previous.unwrap_or(world.inverse()),
        ..default()
    };
    // Discovery: no receiver for owned faces; the same view without the
    // tier does allocate them.
    for enabled in [false, true] {
        camera.control.upload(device, queue, &ShaderBuffer::from(vec![control(0, view_at(0.), None, enabled)]).data.unwrap()).unwrap();
        rig.run(device, queue, &scene, &camera, grid, 0);
        let words = crate::gpu::maintenance_tests::read_words(
            &gpu.context, &camera.history[0].receivers.buffer, u64::from(grid * grid) * 32);
        let owned = words.chunks_exact(8).filter(|w| w[3] < 6).count();
        if enabled { assert_eq!(owned, 0, "per-pixel faces own no receiver"); }
        else { assert!(owned > 0, "control view allocates receivers"); }
    }
    let mut run = |frame: u32, world: Mat4, previous: Option<Mat4>| -> Vec<u32> {
        let (cur, prev) = (frame as usize % 2, 1 - frame as usize % 2);
        camera.control.upload(device, queue, &ShaderBuffer::from(vec![control(frame, world, previous, true)]).data.unwrap()).unwrap();
        let group0: Vec<&Buffer> = vec![&camera.params.buffer, &scene.buffers[0], &scene.buffers[1], &scene.buffers[2],
            &scene.buffers[3], &scene.buffers[4], &camera.history[0].receivers.buffer, &camera.history[0].paths.buffer,
            &scene.buffers[5], &scene.buffers[6], &scene.buffers[7]];
        let make = |pipeline: &ComputePipeline, sets: [&[u32]; 3]| -> Vec<BindGroup> { sets.iter().enumerate().map(|(group, bindings)| {
            let entries: Vec<_> = bindings.iter().map(|&binding| {
                let resource = match (group, binding) {
                    (0, b) => group0[b as usize].as_entire_binding(),
                    (2, 0) => camera.control.buffer.as_entire_binding(),
                    (2, 1) => camera.history[1].receivers.buffer.as_entire_binding(),
                    (2, 4) => camera.history[1].estimates.buffer.as_entire_binding(),
                    (2, 6) => camera.history[1].hash.buffer.as_entire_binding(),
                    (2, 8) => wgpu::BindingResource::TextureView(&rig.depth_view),
                    (2, 9) => wgpu::BindingResource::TextureView(&rig.normal_view),
                    (2, 14..=16) => scene.buffers[6].as_entire_binding(),
                    (2, 19) => pixels.history[prev].buffer.as_entire_binding(),
                    (2, 20) => pixels.history[cur].buffer.as_entire_binding(),
                    (2, 21) => pixels.stats.buffer.as_entire_binding(),
                    (2, 22) => output.buffer.as_entire_binding(),
                    (2, 23) => pixels.queue.buffer.as_entire_binding(),
                    (2, 24) => pixels.args.as_entire_binding(),
                    _ => unreachable!("binding {group}/{binding}"),
                };
                BindGroupEntry { binding, resource }
            }).collect();
            device.create_bind_group("Per-pixel tier", &pipeline.get_bind_group_layout(group as u32).into(), &entries)
        }).collect() };
        let mut encoder = device.create_command_encoder(&default());
        encoder.clear_buffer(&pixels.stats.buffer, 0, None);
        encoder.clear_buffer(&pixels.queue.buffer, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&default());
            for (stage, (pipeline, sets)) in kernels.iter().enumerate() {
                pass.set_pipeline(pipeline);
                for (i, group) in make(pipeline, *sets).iter().enumerate() { pass.set_bind_group(i as u32, &**group, &[]); }
                if stage == 2 { pass.dispatch_workgroups_indirect(&pixels.args, 0); } else { pass.dispatch_workgroups(1, 1, 1); }
            }
        }
        queue.submit([encoder.finish()]);
        crate::gpu::maintenance_tests::read_words(&gpu.context, &output.buffer, PIXEL_HEADER_BYTES + 64 * 16)
    };
    let counts = |words: &[u32]| -> Vec<f32> {
        words[16..].chunks_exact(4).map(|c| {
            assert_eq!(c[2] & 7, 2, "owned top faces");
            assert_eq!(c[2] >> 3, words[15], "stamped with this frame");
            f16((c[1] >> 16) as u16)
        }).collect()
    };
    let mut previous = None;
    for frame in 0..6 {
        let words = run(frame, view_at(0.), previous);
        assert_eq!(&words[8..12], &[8, 8, 2, 1], "header grid");
        assert_eq!(words[15], frame, "header stamp");
        let counts = counts(&words);
        assert!(counts.iter().all(|&n| n == frame as f32 + 1.), "frame {frame}: {counts:?}");
        let light = f16((words[16] & 0xffff) as u16);
        assert!(light > 0.0, "the floor sees the sky");
        previous = Some(view_at(0.).inverse());
    }
    // A one-cell pan (four voxels): history follows the world point.
    let words = run(6, view_at(4.), previous);
    let counts = counts(&words);
    let kept = counts.iter().filter(|&&n| n > 1.).count();
    eprintln!("pan: {kept} of 64 cells kept history: {counts:?}");
    assert!(kept >= 48, "reprojected history: {kept} of 64");
}
