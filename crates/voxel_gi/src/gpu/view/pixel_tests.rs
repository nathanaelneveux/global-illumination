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

/// Pixel x -> world x, pixel y -> world z (two voxels per pixel); depth 0.5
/// -> the floor top at y = 4, seen from 96 voxels up: a face is a fraction
/// of a pixel. `shift` pans the view along +x.
fn view_at(shift: f32) -> Mat4 {
    Mat4::from_cols(Vec4::new(16., 0., 0., 0.), Vec4::new(0., 0., -16., 0.),
        Vec4::new(0., 8., 0., 0.), Vec4::new(16. + shift, 0., 16., 1.))
}

/// A 16x16 view of a 32x32 floor, the discovery rig and the three far-tier
/// kernels, at a given cell size and trace rate.
struct Far {
    gpu: GpuTracer,
    scene: SceneBindings,
    rig: Rig,
    kernels: Vec<(ComputePipeline, [&'static [u32]; 3])>,
    camera: CameraState,
    pixels: PixelState,
    output: UploadBuffer,
    cell: u32,
    rate: u32,
    mask: u32,
}

const GRID: u32 = 4;

impl Far {
    fn new(cell: u32, rate: u32) -> Self {
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
            label: Some("Far tier"), source: ShaderSource::Wgsl(rig.source.clone().into()),
        });
        let kernels = [("classify_pixels", PIXEL_CLASSIFY_BINDINGS), ("size_pixel_queue", PIXEL_ARGS_BINDINGS),
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
        let mut camera = CameraState::new(device, UVec2::splat(GRID));
        let mask = camera.history[0].hash.capacity as u32 / 4 - 1;
        camera.params.upload(device, queue, &ShaderBuffer::from(vec![scene.params]).data.unwrap()).unwrap();
        let cells = UVec2::splat(16u32.div_ceil(cell));
        let pixels = PixelState::new(device, cells);
        let output = storage(device, "pixel output", PIXEL_HEADER_BYTES as usize + cells.element_product() as usize * 16);
        let depths: Vec<u8> = (0..256).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        let normals: Vec<u8> = (0..256).flat_map(|_| [0.5f32, 1., 0.5, 1.]).flat_map(f32::to_le_bytes).collect();
        rig.upload(queue, &depths, &normals);
        Self { gpu, scene, rig, kernels, camera, pixels, output, cell, rate, mask }
    }

    fn cells(&self) -> u32 { self.pixels.cells.element_product() }

    fn control(&self, frame: u32, world: Mat4, previous: Option<Mat4>, enabled: bool) -> Control {
        Control {
            world_from_clip: world, clip_from_world: world.inverse(),
            camera: Vec4::new(16., 100., 16., 0.),
            viewport: UVec4::new(0, 0, 16, 16),
            grid: UVec4::new(GRID, GRID, GRID * GRID, self.mask),
            history: UVec4::new(0, self.mask, frame, 1),
            pixel_axis: Vec4::new(0., -1., 0., 4.),
            pixel_grid: UVec4::new(self.pixels.cells.x, self.pixels.cells.y, self.cell, u32::from(enabled)),
            pixel_tune: Vec4::new(4., 32., self.rate as f32,
                f32::from_bits(if previous.is_some() { frame - 1 } else { u32::MAX })),
            previous_clip_from_world: previous.unwrap_or(world.inverse()),
            ..default()
        }
    }

    /// Discovery's receivers for frame 0 with the tier on or off.
    fn receivers(&mut self, enabled: bool) -> usize {
        let control = self.control(0, view_at(0.), None, enabled);
        let (device, queue) = (&self.gpu.context.device, &self.gpu.context.queue);
        self.camera.control.upload(device, queue, &ShaderBuffer::from(vec![control]).data.unwrap()).unwrap();
        self.rig.run(device, queue, &self.scene, &self.camera, GRID, 0);
        let words = crate::gpu::maintenance_tests::read_words(
            &self.gpu.context, &self.camera.history[0].receivers.buffer, u64::from(GRID * GRID) * 32);
        words.chunks_exact(8).filter(|w| w[3] < 6).count()
    }

    /// One far-tier frame; returns the material's output words and the
    /// written cell histories (12 words each).
    fn frame(&mut self, frame: u32, world: Mat4, previous: Option<Mat4>) -> (Vec<u32>, Vec<u32>) {
        let (cur, prev) = (frame as usize % 2, 1 - frame as usize % 2);
        let control = self.control(frame, world, previous, true);
        let (device, queue) = (&self.gpu.context.device, &self.gpu.context.queue);
        self.camera.control.upload(device, queue, &ShaderBuffer::from(vec![control]).data.unwrap()).unwrap();
        let camera = &self.camera;
        let scene = &self.scene;
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
                    (2, 8) => wgpu::BindingResource::TextureView(&self.rig.depth_view),
                    (2, 9) => wgpu::BindingResource::TextureView(&self.rig.normal_view),
                    (2, 19) => self.pixels.history[prev].buffer.as_entire_binding(),
                    (2, 20) => self.pixels.history[cur].buffer.as_entire_binding(),
                    (2, 21) => self.pixels.stats.buffer.as_entire_binding(),
                    (2, 22) => self.output.buffer.as_entire_binding(),
                    (2, 23) => self.pixels.queue.buffer.as_entire_binding(),
                    (2, 24) => self.pixels.args.as_entire_binding(),
                    _ => unreachable!("binding {group}/{binding}"),
                };
                BindGroupEntry { binding, resource }
            }).collect();
            device.create_bind_group("Far tier", &pipeline.get_bind_group_layout(group as u32).into(), &entries)
        }).collect() };
        let mut encoder = device.create_command_encoder(&default());
        encoder.clear_buffer(&self.pixels.stats.buffer, 0, None);
        encoder.clear_buffer(&self.pixels.queue.buffer, 0, Some(4));
        {
            let mut pass = encoder.begin_compute_pass(&default());
            for (stage, (pipeline, sets)) in self.kernels.iter().enumerate() {
                pass.set_pipeline(pipeline);
                for (i, group) in make(pipeline, *sets).iter().enumerate() { pass.set_bind_group(i as u32, &**group, &[]); }
                if stage == 2 { pass.dispatch_workgroups_indirect(&self.pixels.args, 0); } else { pass.dispatch_workgroups(1, 1, 1); }
            }
        }
        queue.submit([encoder.finish()]);
        let cells = u64::from(self.cells());
        (crate::gpu::maintenance_tests::read_words(&self.gpu.context, &self.output.buffer, PIXEL_HEADER_BYTES + cells * 16),
         crate::gpu::maintenance_tests::read_words(&self.gpu.context, &self.pixels.history[cur].buffer, cells * 48))
    }
}

/// Sample counts of the output cells, checking face and stamp.
fn counts(words: &[u32]) -> Vec<f32> {
    counts_at(words, &(0..(words.len() - 16) / 4).collect::<Vec<_>>())
}
fn counts_at(words: &[u32], cells: &[usize]) -> Vec<f32> {
    cells.iter().map(|&index| {
        let c = &words[16 + index * 4..20 + index * 4];
        assert_eq!(c[2] & 7, 2, "owned top faces");
        assert_eq!(c[2] >> 3, words[15], "stamped with this frame");
        f16((c[1] >> 16) as u16)
    }).collect()
}

#[test]
#[ignore = "native per-pixel tier"]
fn native_small_faces_are_lit_per_pixel_without_receivers() {
    let mut far = Far::new(2, 1);
    assert!(far.receivers(false) > 0, "control view allocates receivers");
    assert_eq!(far.receivers(true), 0, "per-pixel faces own no receiver");
    let mut previous = None;
    for frame in 0..6 {
        let (words, _) = far.frame(frame, view_at(0.), previous);
        assert_eq!(&words[8..12], &[8, 8, 2, 1], "header grid");
        assert_eq!(words[15], frame, "header stamp");
        let counts = counts(&words);
        assert!(counts.iter().all(|&n| n == frame as f32 + 1.), "frame {frame}: {counts:?}");
        assert!(f16((words[16] & 0xffff) as u16) > 0.0, "the floor sees the sky");
        previous = Some(view_at(0.).inverse());
    }
    // A one-cell pan (four voxels): history follows the world point.
    let (words, _) = far.frame(6, view_at(4.), previous);
    let counts = counts(&words);
    let kept = counts.iter().filter(|&&n| n > 1.).count();
    assert!(kept >= 48, "reprojected history: {kept} of 64");
}

#[test]
#[ignore = "native per-pixel tier"]
fn native_far_cells_at_default_settings_sample_every_pixel_and_index() {
    // The defaults: 3-pixel cells, one trace in four frames after warmup.
    let defaults = FarTierSettings::default();
    let mut far = Far::new(defaults.cell, defaults.rate);
    let cells = far.cells() as usize;
    let width = far.pixels.cells.x as usize;
    // The 16-pixel view leaves the last row and column partly outside it.
    let interior: Vec<usize> = (0..cells).filter(|i| i % width < width - 1 && i / width < width - 1).collect();
    let mut visited = vec![std::collections::BTreeSet::new(); cells];
    let mut sequences = vec![Vec::new(); cells];
    let mut previous = None;
    for frame in 0..36 {
        let (words, history) = far.frame(frame, view_at(0.), previous);
        let counts = counts_at(&words, &interior);
        for &index in &interior {
            let cell = &history[index * 12..index * 12 + 12];
            // The sampled pixel, from the stored surface point (x = 2 px + 1).
            let x = f32::from_bits(cell[4]);
            let z = f32::from_bits(cell[6]);
            visited[index].insert((((x - 1.) / 2.).round() as i32, ((z - 1.) / 2.).round() as i32));
            sequences[index].push(cell[8]);
        }
        assert!(counts.iter().all(|&n| n > 0.), "frame {frame}: every cell holds light");
        previous = Some(view_at(0.).inverse());
    }
    for &index in &interior {
        assert_eq!(visited[index].len(), 9, "cell {index} samples all nine pixels: {:?}", visited[index]);
    }
    for &index in &interior {
        let sequence = &sequences[index];
        // One index per traced path, independent of the frame residue the
        // rate schedule picks: four warmup traces, then one in four frames.
        assert!(sequence.windows(2).all(|w| w[1] == w[0] || w[1] == w[0] + 1), "cell {index}: {sequence:?}");
        // Consecutive indices span every residue the base-2 dimension needs,
        // whichever frames the schedule traced on (4 warmup paths, then 8
        // or 9 in 32 frames by the cell's phase).
        let last = *sequence.last().unwrap();
        assert!((12..=13).contains(&last), "cell {index}: {sequence:?}");
    }
}
