//! Synthetic prepass coverage, not a lighting-quality screenshot. Single-pixel
//! side faces between coarse sample centres must get real receiver identities.
use super::*;
use crate::voxel_gi::{
    emission::EmissionDistribution,
    naadf::{linear_index, volume_len},
    work::TraceWork,
    world::{AIR, PAGE_SIZE},
};

#[test]
fn discovery_rank_fits_the_supported_viewport_range() {
    for width in [1u32, 3, 720, 1280, 1440, 2560, 3840, 8192, 16384] {
        for height in [1u32, 3, 720, 1280, 1440, 2560, 3840, 8192, 16384] {
            let viewport = UVec2::new(width, height);
            let grid = grid_size(viewport);
            let extent = UVec2::new((width * 2).div_ceil(grid.x), (height * 2).div_ceil(grid.y));
            let pixels = extent.as_u64vec2().element_product();
            let distance_bound = extent.as_u64vec2().length_squared();
            assert!((distance_bound * 4 + 3) * pixels + pixels - 1 < u64::from(u32::MAX));
        }
    }
}

#[test]
#[ignore = "native thin-surface discovery with real transport geometry"]
fn native_view_discovery_keeps_thin_surfaces_without_growing_storage() {
    let mut gpu = crate::voxel_gi::gpu::tests::native_tracer();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for a in 8..24 {
        for b in 8..24 {
            page[linear_index(UVec3::new(a, b, 7), PAGE_SIZE)] = 2;
        }
        page[linear_index(UVec3::new(7, a, 8), PAGE_SIZE)] = 2;
        page[linear_index(UVec3::new(a, 7, 8), PAGE_SIZE)] = 2;
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page)],
        &[],
    ));
    let materials = VoxelMaterialTransportTable::default();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &scene,
        &materials,
        VoxelGiSettings::default().sampling_importance(),
    ));
    gpu.submit_compact(
        scene,
        &TransportSources::default(),
        &materials,
        &mut TraceWork::default(),
        0,
        &emitters,
        true,
    )
    .unwrap();
    let scene = gpu.view_scene().unwrap();
    let device = &gpu.context.device;
    let queue = &gpu.context.queue;
    // Upload the exact depth values through R32Float in this synthetic fixture.
    // Production depth/MSAA types are separately pipeline/composition tested.
    let source = shader(false);
    let mut sources = vec![("surface-aware", source.clone())];
    if let Some(path) = std::env::var_os("VOXEL_GI_DISCOVERY_BASELINE") {
        let old = std::fs::read_to_string(path).unwrap();
        let (label, old) = if old.contains("// Receiver keys") {
            let start = old
                .find("@compute @workgroup_size(64)\nfn discover_receivers")
                .unwrap();
            let end = old[start..].find("// Receiver keys").unwrap() + start;
            ("fixed-centre control", old[start..end].to_owned())
        } else {
            ("surface-aware control", old)
        };
        assert!(source.contains(include_str!(
            "../../../../assets/shaders/voxel_view_discovery.wgsl"
        )));
        sources.push((
            label,
            source.replace(
                include_str!("../../../../assets/shaders/voxel_view_discovery.wgsl"),
                &old,
            ),
        ));
    }
    let size = wgpu::Extent3d {
        width: 16,
        height: 16,
        depth_or_array_layers: 1,
    };
    let texture = |format| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("coverage prepass"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        })
    };
    let depth = texture(wgpu::TextureFormat::R32Float);
    let normals = texture(wgpu::TextureFormat::Rgba32Float);
    let depth_view = depth.create_view(&default());
    let normal_view = normals.create_view(&default());
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &depth,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &vec![0.5f32; 256]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(64),
            rows_per_image: Some(16),
        },
        size,
    );
    let mut expected = Vec::new();
    for (label, source) in sources {
        let memoized = label == "surface-aware" || source.contains("let cached = discovery_tiles[id.x];");
        let source = source
            .replace("texture_depth_2d", "texture_2d<f32>")
            .replace(
                "let z = textureLoad(depth, pixel, 0);",
                "let z = textureLoad(depth, pixel, 0).x;",
            );
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some(label),
            source: ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("discover_receivers"),
            compilation_options: default(),
            cache: None,
        });
        for grid in [4u32, 3] {
            let mut camera = CameraState::new(device, UVec2::splat(grid));
            camera
                .params
                .upload(
                    device,
                    queue,
                    &ShaderBuffer::from(vec![scene.params]).data.unwrap(),
                )
                .unwrap();
            let mut control = Control {
                world_from_clip: Mat4::from_cols(
                    Vec4::new(8., 0., 0., 0.),
                    Vec4::new(0., -8., 0., 0.),
                    Vec4::new(0., 0., 1., 0.),
                    Vec4::new(15.5, 15.5, 7.5, 1.),
                ),
                viewport: UVec4::new(0, 0, 16, 16),
                grid: UVec4::new(grid, grid, grid * grid, 0),
                ..default()
            };
            camera
                .control
                .upload(
                    device,
                    queue,
                    &ShaderBuffer::from(vec![control]).data.unwrap(),
                )
                .unwrap();
            let view_bindings: &[u32] = if memoized {
                &[0, 1, 5, 8, 9, 17]
            } else {
                &[0, 5, 8, 9]
            };
            let groups: Vec<_> = [&[0u32, 1, 2, 3, 6][..], &[], view_bindings]
                .iter()
                .enumerate()
                .map(|(group, bindings)| {
                    let entries: Vec<_> = bindings
                        .iter()
                        .map(|&binding| {
                            let resource = match (group, binding) {
                                (0, 0) => camera.params.buffer.as_entire_binding(),
                                (0, 1..=3) => {
                                    scene.buffers[binding as usize - 1].as_entire_binding()
                                }
                                (0, 6) => camera.history[0].receivers.buffer.as_entire_binding(),
                                (2, 0) => camera.control.buffer.as_entire_binding(),
                                (2, 1) => camera.history[1].receivers.buffer.as_entire_binding(),
                                (2, 5) => camera.history[0].estimates.buffer.as_entire_binding(),
                                (2, 8) => wgpu::BindingResource::TextureView(&depth_view),
                                (2, 9) => wgpu::BindingResource::TextureView(&normal_view),
                                (2, 17) => camera.discovery_tiles.buffer.as_entire_binding(),
                                _ => unreachable!(),
                            };
                            BindGroupEntry { binding, resource }
                        })
                        .collect();
                    device.create_bind_group(
                        label,
                        &pipeline.get_bind_group_layout(group as u32).into(),
                        &entries,
                    )
                })
                .collect();
            // Case 3 has raster-only geometry; case 4 hides the old surfaces
            // in the prepass while they still exist in transport geometry.
            for strips in 0..=4 {
                let pixels: Vec<u8> = (0..256)
                    .flat_map(|i| {
                        let normal = if strips == 4 {
                            [0.5f32, 0.5, 0.5, 1.]
                        } else if strips == 3 && i % 16 == 0 {
                            [0f32, 0.5, 0.5, 1.]
                        } else if strips >= 1 && i % 16 == 0 {
                            [1f32, 0.5, 0.5, 1.]
                        } else if strips == 2 && i / 16 == 0 {
                            [0.5, 1., 0.5, 1.]
                        } else {
                            [0.5, 0.5, 1., 1.]
                        };
                        normal.into_iter().flat_map(f32::to_le_bytes)
                    })
                    .collect();
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &normals,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &pixels,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(256),
                        rows_per_image: Some(16),
                    },
                    size,
                );
                let mut original = Vec::new();
                let mut previous_refresh = 0;
                for frame in 0u32..12 {
                    control.world_from_clip.w_axis.x = 15.5 + 0.1 * frame.saturating_sub(8) as f32;
                    control.clip_from_world = control.world_from_clip.inverse();
                    control.history.z = strips * 128
                        + match frame {
                            7 => 64,
                            8 => 65,
                            9.. => frame + 57,
                            _ => frame + 1,
                        };
                    control.discovery.x = u32::from(control.history.z > 1 && frame < 8);
                    control.discovery.y = u32::from(frame >= 9);
                    camera
                        .control
                        .upload(
                            device,
                            queue,
                            &ShaderBuffer::from(vec![control]).data.unwrap(),
                        )
                        .unwrap();
                    let mut encoder = device.create_command_encoder(&default());
                    encoder.copy_buffer_to_buffer(
                        &camera.history[0].receivers.buffer,
                        0,
                        &camera.history[1].receivers.buffer,
                        0,
                        u64::from(grid * grid) * 32,
                    );
                    {
                        let mut pass = encoder.begin_compute_pass(&default());
                        pass.set_pipeline(&pipeline);
                        for (i, group) in groups.iter().enumerate() {
                            pass.set_bind_group(i as u32, &**group, &[]);
                        }
                        let groups = if label != "fixed-centre control" {
                            grid.div_ceil(2).pow(2)
                        } else {
                            1
                        };
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                    queue.submit([encoder.finish()]);
                    let words = crate::voxel_gi::gpu::maintenance_tests::read_words(
                        &gpu.context,
                        &camera.history[0].receivers.buffer,
                        u64::from(grid * grid) * 32,
                    );
                    let faces: Vec<_> = words.chunks_exact(8).map(|w| w[3]).collect();
                    if strips == 4 {
                        if frame >= 3 {
                            assert!(faces.iter().all(|&face| face >= 6),
                                "reprojection must discard faces absent from the current prepass");
                        }
                        continue;
                    }
                    if frame == 3 {
                        original = words.clone();
                    } else if (4..8).contains(&frame) || (frame >= 9 && strips == 0) {
                        assert_eq!(
                            words, original,
                            "coverage must remain identical between scans"
                        );
                    }
                    if label == "surface-aware" {
                        let metadata = crate::voxel_gi::gpu::maintenance_tests::read_words(
                            &gpu.context,
                            &camera.discovery_tiles.buffer,
                            16,
                        );
                        if frame == 6 && strips < 3 {
                            assert_eq!(metadata[2] & 1, 1, "valid footprint is reusable");
                            assert_eq!(
                                metadata[3], previous_refresh,
                                "unchanged footprint must take the reuse path"
                            );
                        }
                        if strips == 3 && frame >= 3 {
                            assert_eq!(
                                metadata[2] & 1,
                                0,
                                "pending geometry must remain discoverable"
                            );
                        }
                        if (7..=8).contains(&frame) {
                            assert_eq!(
                                metadata[3],
                                control.history.z + 1,
                                "periodic refresh and camera change must perform a full scan"
                            );
                        }
                        if frame == 9 && strips == 0 {
                            assert_eq!(metadata[3], previous_refresh,
                                "small camera motion must reuse validated face witnesses");
                        }
                        previous_refresh = metadata[3];
                    }
                    if label == "surface-aware" {
                        if frame == 3 {
                            expected.push(words.clone());
                        }
                    } else if label == "surface-aware control" && frame >= 3 && strips == 0 {
                        let case = usize::from(grid == 3) * 4 + strips as usize;
                        assert_eq!(
                            words, expected[case],
                            "optimized discovery must preserve exact receiver records"
                        );
                    }
                    eprintln!("{label} grid={grid} strips={strips} frame={frame}: {faces:?}");
                    assert!(faces.iter().all(|&face| face < 6));
                    assert!(faces.contains(&4), "the background must remain represented");
                    if strips == 2 && label != "fixed-centre control" {
                        assert!(
                            faces.contains(&0),
                            "existing thin planes must not blink between checks"
                        );
                    }
                    if frame < 3 {
                        continue;
                    }
                    if label == "fixed-centre control" || strips == 0 || strips == 3 {
                        assert!(faces.iter().all(|&face| face == 4));
                        for (i, words) in words.chunks_exact(8).enumerate() {
                            let center = |q: u32| ((q as f32 + 0.5) / grid as f32 * 16.) as u32;
                            assert_eq!(
                                &words[4..7],
                                &[center(i as u32 % grid) + 8, center(i as u32 / grid) + 8, 7],
                                "flat surfaces retain the original centre identities"
                            );
                        }
                    } else {
                        assert!(
                            faces.contains(&0),
                            "one-pixel side face must receive a real sample"
                        );
                        assert!(faces.contains(&4), "the background must remain represented");
                        if strips == 2 {
                            assert!(faces.contains(&2));
                        }
                    }
                }
            }
        }
    }
}
