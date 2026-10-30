//! Native shared-pool coverage. Crowded tiles borrow physically unrelated slots;
//! retained witnesses can cross tile boundaries without losing face identity.
use super::*;
use crate::{
    emission::EmissionDistribution,
    naadf::{linear_index, volume_len},
    work::TraceWork,
    world::{AIR, PAGE_SIZE},
};
use std::collections::BTreeSet;

#[test]
fn pool_tile_directory_covers_supported_viewports() {
    for limit in [8_192, 32_768, 65_536, MAX_VIEW_RECEIVERS] {
        for width in [1, 3, 720, 1280, 2560, 3840, 8192, 16384] {
            for height in [1, 3, 720, 1280, 1440, 2560, 8192, 16384] {
                let grid = grid_size(UVec2::new(width, height), limit);
                assert!(grid.element_product() <= limit);
                assert!(grid.x.div_ceil(2) * grid.y.div_ceil(2) <= 65536);
            }
        }
    }
}

#[test]
#[ignore = "native global receiver-pool and grazing-plane coverage"]
fn native_view_discovery_keeps_thin_surfaces_without_growing_storage() {
    let mut gpu = crate::gpu::tests::native_tracer();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for x in 8..24 {
        for y in 8..24 {
            page[linear_index(UVec3::new(x, y, 7 + (x - 8) % 6), PAGE_SIZE)] = 2;
        }
        page[linear_index(UVec3::new(7, x, 14), PAGE_SIZE)] = 2;
        page[linear_index(UVec3::new(x, 7, 14), PAGE_SIZE)] = 2;
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(), vec![(IVec3::ZERO, page)], &[],
    ));
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
    gpu.submit_compact(scene, &TransportSources::default(), &materials,
        &mut TraceWork::default(), 0, &emitters).unwrap();
    let scene = gpu.view_scene().unwrap();
    let device = &gpu.context.device;
    let queue = &gpu.context.queue;
    let rig = Rig::new(device);
    for grid in [4u32, 3] {
        for (thin, delayed_scene) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut camera = CameraState::new(device, UVec2::splat(grid)).unwrap();
            let count = grid * grid;
            let mask = camera.history[0].hash.capacity as u32 / 4 - 1;
            camera.params.upload(device, queue, &ShaderBuffer::from(vec![scene.params]).data.unwrap()).unwrap();
            let mut previous_keys = BTreeSet::new();
            let mut previous_shift = 0.;
            for (frame, shift) in [0., 0., 0.1, -4., -4., -4., 0.].into_iter().enumerate() {
                // Raster geometry is present before the transport model. The
                // same face records must survive until publication catches up,
                // without admitting any rays into the stale/absent scene.
                let waiting = delayed_scene && frame < 3;
                let mut params = scene.params;
                if waiting {
                    params.minimum = IVec3::splat(128).extend(params.minimum.w);
                    params.maximum = IVec3::splat(129).extend(params.maximum.w);
                }
                camera.params.upload(device, queue, &ShaderBuffer::from(vec![params]).data.unwrap()).unwrap();
                let hidden = frame == 5;
                let world = Mat4::from_cols(Vec4::new(8., 0., 0., 0.),
                    Vec4::new(0., -8., 0., 0.), Vec4::new(0., 0., 16., 0.),
                    Vec4::new(15.5 + shift, 15.5, 7.5, 1.));
                let mut depths = Vec::new();
                let mut normal_bytes = Vec::new();
                for y in 0..16 {
                    for x in 0..16 {
                        let wx = (8. + x as f32 + shift).floor() as i32;
                        let visible = !hidden && (8..16).contains(&wx) && y < 8;
                        let (normal, z) = if !visible { ([0.5f32; 4], 0.) }
                            else if thin && wx == 8 { ([1., 0.5, 0.5, 1.], 14.5) }
                            else if thin && y == 0 { ([0.5, 1., 0.5, 1.], 14.5) }
                            else { ([0.5, 0.5, 1., 1.], 8. + ((wx - 8) % 6) as f32) };
                        let z = if visible { (z - 7.5) / 16. } else { 0. };
                        depths.extend(z.to_le_bytes());
                        normal_bytes.extend(normal.into_iter().flat_map(f32::to_le_bytes));
                    }
                }
                rig.upload(queue, &depths, &normal_bytes);
                let control = Control {
                    world_from_clip: world, clip_from_world: world.inverse(),
                    viewport: UVec4::new(0, 0, 16, 16),
                    grid: UVec4::new(grid, grid, count, mask),
                    history: UVec4::new(if frame == 0 { 0 } else { count }, mask, frame as u32, 1),
                    discovery: UVec4::new(u32::from(frame > 0 && shift == previous_shift),
                        u32::from(frame > 0), 0, 0), ..default()
                };
                previous_shift = shift;
                camera.control.upload(device, queue, &ShaderBuffer::from(vec![control]).data.unwrap()).unwrap();
                rig.run(device, queue, &scene, &camera, grid, frame);
                let current = &camera.history[frame % 2];
                let words = crate::gpu::maintenance_tests::read_words(
                    &gpu.context, &current.receivers.buffer, u64::from(count) * 32);
                let metadata = crate::gpu::maintenance_tests::read_words(
                    &gpu.context, &current.pool.buffer, (POOL_PREFIX_BYTES + count as usize * 16) as u64);
                let keys: BTreeSet<_> = words.chunks_exact(8).filter(|w| w[3] < 6)
                    .map(|w| (w[3], [w[4], w[5], w[6]])).collect();
                let mut estimates = crate::gpu::maintenance_tests::read_words(
                    &gpu.context, &current.estimates.buffer, u64::from(count) * 64);
                let ready = crate::gpu::maintenance_tests::read_words(
                    &gpu.context, &camera.queue.buffer, 4)[0];
                if waiting { assert_eq!(ready, 0, "raster-only receivers cannot trace stale geometry"); }
                for (index, receiver) in words.chunks_exact(8).enumerate().filter(|(_, w)| w[3] < 6) {
                    let key = (receiver[3], [receiver[4], receiver[5], receiver[6]]);
                    let mean = [receiver[3] as f32 + 1., receiver[4] as f32 + 2., receiver[6] as f32 + 3., 128.];
                    let tag = mean.map(f32::to_bits);
                    let estimate = &mut estimates[index * 16..(index + 1) * 16];
                    if waiting { assert_ne!(estimate[13] & 0x80000000, 0, "explicit pending identity"); }
                    if previous_keys.contains(&key) {
                        assert_eq!(&estimate[..4], &tag, "raw history follows the face, not its spatial bin");
                        assert_eq!(estimate[12], 177, "QMC sequence survives reprojection");
                    } else {
                        assert_eq!(estimate[3], 0, "a reused physical slot cannot inherit unrelated history");
                    }
                    estimate[..4].copy_from_slice(&tag);
                    estimate[12] = 177;
                }
                queue.write_buffer(&current.estimates.buffer, 0,
                    &estimates.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
                let planes: BTreeSet<_> = keys.iter().map(|(face, voxel)| (*face, voxel[*face as usize / 2])).collect();
                eprintln!("shared pool grid={grid} thin={thin} delayed={delayed_scene} frame={frame}: {} keys, {planes:?}, counters {:?}", keys.len(), &metadata[..4]);
                assert!(keys.len() <= count as usize);
                if hidden { assert!(keys.is_empty(), "occluded geometry cannot retain a receiver"); }
                else {
                    for z in 7..13 { assert!(planes.contains(&(4, z)), "all six crowded planes need support"); }
                    if thin {
                        assert!(planes.contains(&(0, 7)), "one-pixel grazing X face");
                        assert!(planes.contains(&(2, 7)), "one-pixel grazing Y face");
                    }
                    if frame == 1 { assert_eq!(keys, previous_keys, "stationary witnesses stay stable"); }
                    if frame == 3 {
                        assert!(previous_keys.is_subset(&keys), "cross-tile reprojection keeps all visible old identities");
                        assert!(metadata[3] >= previous_keys.len() as u32);
                    }
                    if frame == 0 {
                        let mut entry = metadata[4]; // first tile head
                        let mut slots = BTreeSet::new();
                        while entry != 0 {
                            assert!(entry <= count && slots.insert(entry - 1), "unique, in-bounds spatial links");
                            entry = metadata[POOL_PREFIX_BYTES / 4 + (entry as usize - 1) * 4 + 2];
                        }
                        assert!(slots.len() > 4, "crowded tile must borrow from empty tiles");
                    }
                }
                previous_keys = keys;
            }
        }
    }
}


#[test]
#[ignore = "native coarse-page receivers"]
fn native_coarse_page_faces_are_one_receiver_per_block() {
    // A stride-8 page (`PageSample`): a floor one block thick, seen from
    // above over its whole 32x32 top. Unit faces would need 1024 receivers;
    // blocks need 16, each keyed by its minimum voxel at its face centre,
    // and witnesses reproject over the block, not one voxel of it.
    let mut gpu = crate::gpu::tests::native_tracer();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for x in 0..32 {
        for y in 0..8 {
            for z in 0..32 {
                page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 2;
            }
        }
    }
    let scene = Arc::new(TransportScene::with_strided_page_delta(
        &TransportScene::default(), vec![(IVec3::ZERO, page, 3)], &[],
    ));
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
    gpu.submit_compact(scene, &TransportSources::default(), &materials,
        &mut TraceWork::default(), 0, &emitters).unwrap();
    let scene = gpu.view_scene().unwrap();
    let device = &gpu.context.device;
    let queue = &gpu.context.queue;
    let rig = Rig::new(device);
    let grid = 4u32;
    let count = grid * grid;
    let mut camera = CameraState::new(device, UVec2::splat(grid)).unwrap();
    let mask = camera.history[0].hash.capacity as u32 / 4 - 1;
    camera.params.upload(device, queue, &ShaderBuffer::from(vec![scene.params]).data.unwrap()).unwrap();
    let mut previous = BTreeSet::new();
    for (frame, shift) in [0f32, 0., 1.5].into_iter().enumerate() {
        // Looking down: pixel x -> world x, pixel y -> world z, depth -> y.
        let world = Mat4::from_cols(Vec4::new(16., 0., 0., 0.), Vec4::new(0., 0., -16., 0.),
            Vec4::new(0., 16., 0., 0.), Vec4::new(16. + shift, 0., 16., 1.));
        let depths: Vec<u8> = (0..256).flat_map(|_| 0.5f32.to_le_bytes()).collect();
        let normals: Vec<u8> = (0..256).flat_map(|_| [0.5f32, 1., 0.5, 1.]).flat_map(f32::to_le_bytes).collect();
        rig.upload(queue, &depths, &normals);
        let control = Control {
            world_from_clip: world, clip_from_world: world.inverse(),
            viewport: UVec4::new(0, 0, 16, 16),
            grid: UVec4::new(grid, grid, count, mask),
            history: UVec4::new(if frame == 0 { 0 } else { count }, mask, frame as u32, 1),
            discovery: UVec4::new(u32::from(frame == 1), u32::from(frame > 0), 0, 0), ..default()
        };
        camera.control.upload(device, queue, &ShaderBuffer::from(vec![control]).data.unwrap()).unwrap();
        rig.run(device, queue, &scene, &camera, grid, frame);
        let words = crate::gpu::maintenance_tests::read_words(
            &gpu.context, &camera.history[frame % 2].receivers.buffer, u64::from(count) * 32);
        let mut keys = BTreeSet::new();
        for w in words.chunks_exact(8).filter(|w| w[3] < 6) {
            let surface = Vec3::from_array([w[0], w[1], w[2]].map(f32::from_bits));
            let voxel = IVec3::new(w[4] as i32, w[5] as i32, w[6] as i32);
            assert_eq!(w[3], 2, "top faces only");
            assert_eq!(voxel % 8, IVec3::ZERO, "block key is its minimum voxel: {voxel:?}");
            assert_eq!(surface, voxel.as_vec3() + Vec3::new(4., 8., 4.), "block face centre");
            keys.insert(voxel.to_array());
        }
        eprintln!("coarse frame={frame} shift={shift}: {} block receivers", keys.len());
        if frame == 0 { assert_eq!(keys.len(), 16, "every visible block face owns a receiver"); }
        if frame == 1 { assert_eq!(keys, previous, "stationary blocks keep their receivers"); }
        if frame == 2 {
            // 1.5 voxels of pan keeps most of every block on screen: its
            // witness must still land on it.
            let kept = previous.intersection(&keys).count();
            assert!(kept >= 12, "panned blocks keep their receivers: {kept} of {}", previous.len());
        }
        previous = keys;
    }
}

/// Prepare, discover, index and reproject on synthetic 16x16 R32Float depth
/// and RGBA32Float normals. Synthetic depth uploads use R32Float; the
/// ordinary/MSAA depth pipeline layouts are separately covered by
/// native_view_runtime_pipelines.
pub(super) struct Rig {
    pub source: String,
    kernels: [Kernel; 4],
    depth: bevy::render::render_resource::Texture,
    normals: bevy::render::render_resource::Texture,
    pub depth_view: bevy::render::render_resource::TextureView,
    pub normal_view: bevy::render::render_resource::TextureView,
}

const RIG_SIZE: wgpu::Extent3d = wgpu::Extent3d { width: 16, height: 16, depth_or_array_layers: 1 };

impl Rig {
    pub(super) fn new(device: &RenderDevice) -> Self {
        let source = shader(false).replace("texture_depth_2d", "texture_2d<f32>")
            .replace("let z = textureLoad(depth, pixel, 0);",
                     "let z = textureLoad(depth, pixel, 0).x;");
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("Shared pool coverage"), source: ShaderSource::Wgsl(source.clone().into()),
        });
        let reflection = crate::gpu::reflect::Reflection::new(&source);
        let names = ["prepare_receiver_pool", "discover_receivers", "index_receivers", "reproject_paths"];
        let kernels = names.map(|name| Kernel {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name), layout: None, module: &module, entry_point: Some(name),
                compilation_options: default(), cache: None,
            }),
            bindings: reflection.bindings(name),
        });
        let texture = |format| device.create_texture(&wgpu::TextureDescriptor {
            label: Some("pool coverage prepass"), size: RIG_SIZE, mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2, format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let depth = texture(wgpu::TextureFormat::R32Float);
        let normals = texture(wgpu::TextureFormat::Rgba32Float);
        let depth_view = depth.create_view(&default());
        let normal_view = normals.create_view(&default());
        Self { source, kernels, depth, normals, depth_view, normal_view }
    }

    pub(super) fn upload(&self, queue: &RenderQueue, depths: &[u8], normals: &[u8]) {
        for (texture, bytes, row) in [(&self.depth, depths, 64), (&self.normals, normals, 256)] {
            queue.write_texture(wgpu::TexelCopyTextureInfo { texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All }, bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(16) }, RIG_SIZE);
        }
    }

    pub(super) fn run(&self, device: &RenderDevice, queue: &RenderQueue, scene: &SceneBindings,
        camera: &CameraState, grid: u32, frame: usize) {
        let count = grid * grid;
        let current = &camera.history[frame % 2];
        let previous = &camera.history[1 - frame % 2];
        let mut encoder = device.create_command_encoder(&default());
        encoder.clear_buffer(&current.pool.buffer, 0, Some(POOL_PREFIX_BYTES as u64));
        encoder.clear_buffer(&current.hash.buffer, 0, None);
        encoder.clear_buffer(&camera.queue.buffer, 0, Some(48));
        for (stage, kernel) in self.kernels.iter().enumerate() {
            let groups: Vec<_> = kernel.bindings.iter().enumerate().map(|(group, bindings)| {
                let entries: Vec<_> = bindings.iter().map(|&binding| {
                    let resource = match (group, binding) {
                        (0, 0) => camera.params.buffer.as_entire_binding(),
                        (0, 1..=3) => scene.buffers[binding as usize - 1].as_entire_binding(),
                        (0, 4) => scene.buffers[3].as_entire_binding(),
                        (0, 5) => scene.buffers[4].as_entire_binding(),
                        (0, 6) => current.receivers.buffer.as_entire_binding(),
                        (0, 7) => current.paths.buffer.as_entire_binding(),
                        (1, 0) => current.tails.buffer.as_entire_binding(),
                        (2, 0) => camera.control.buffer.as_entire_binding(),
                        (2, 1) => previous.receivers.buffer.as_entire_binding(),
                        (2, 2) => previous.paths.buffer.as_entire_binding(),
                        (2, 3) => previous.tails.buffer.as_entire_binding(),
                        (2, 4) => previous.estimates.buffer.as_entire_binding(),
                        (2, 5) => current.estimates.buffer.as_entire_binding(),
                        (2, 6) => previous.hash.buffer.as_entire_binding(),
                        (2, 7) => current.hash.buffer.as_entire_binding(),
                        (2, 8) => wgpu::BindingResource::TextureView(&self.depth_view),
                        (2, 9) => wgpu::BindingResource::TextureView(&self.normal_view),
                        (2, 10) => camera.queue.buffer.as_entire_binding(),
                        (2, 17) => current.pool.buffer.as_entire_binding(),
                        (2, 18) => previous.pool.buffer.as_entire_binding(),
                        _ => unreachable!(),
                    };
                    BindGroupEntry { binding, resource }
                }).collect();
                device.create_bind_group("Pool coverage", &kernel.pipeline.get_bind_group_layout(group as u32).into(), &entries)
            }).collect();
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&kernel.pipeline);
            for (i, group) in groups.iter().enumerate() { pass.set_bind_group(i as u32, &**group, &[]); }
            pass.dispatch_workgroups(if stage == 1 { grid.div_ceil(2).pow(2) } else { count.div_ceil(64) }, 1, 1);
        }
        queue.submit([encoder.finish()]);
    }
}
