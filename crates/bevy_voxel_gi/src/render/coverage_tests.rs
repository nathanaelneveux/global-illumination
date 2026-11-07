//! Raster coverage is not necessarily pixel-centre coverage. Face-key lookups
//! must stay on the rendered primitive even for subpixel voxel silhouettes.
use bevy::tasks::block_on;

#[test]
#[ignore = "native MSAA raster/voxel-identity regression"]
fn native_msaa_voxel_identity_interpolation() {
    let adapter = crate::gpu::tests::native_adapter();
    let (device, queue) = block_on(adapter.request_device(&Default::default())).unwrap();
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut results = Vec::new();
    for (interpolation, offset) in [("center", 0.0625), ("centroid", 0.0625), ("centroid", 0.0)] {
        // Six orientations, each translated through eight subpixel phases.
        // Every quad represents one exposed face of voxel (0,0,0). The red
        // output marks a fragment that looks up some other voxel instead.
        let source = r#"
struct Out {
    @builtin(position) clip: vec4<f32>,
    @location(0) @interpolate(perspective, SAMPLING) world_position: vec4<f32>,
    @location(1) normal: vec3<f32>,
}
@vertex fn vertex(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> Out {
    let corners = array(vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(0.0, 1.0),
                        vec2(0.0, 1.0), vec2(1.0, 0.0), vec2(1.0, 1.0));
    let uv = corners[vertex];
    let face = instance / 8u;
    let axis = face / 2u;
    let negative = (face & 1u) != 0u;
    var out: Out;
    out.normal[axis] = select(1.0, -1.0, negative);
    out.world_position[axis] = select(1.0, 0.0, negative);
    out.world_position[(axis + 1u) % 3u] = uv.x;
    out.world_position[(axis + 2u) % 3u] = uv.y;
    out.world_position.w = 1.0;
    let pixel = vec2(10.0 + OFFSET + f32(instance % 8u) * 0.125, 8.0 + f32(instance)) + uv * vec2(0.6, 0.7);
    out.clip = vec4(pixel / 64.0 * vec2(2.0, -2.0) + vec2(-1.0, 1.0), 0.5, 1.0);
    return out;
}
@fragment fn fragment(in: Out) -> @location(0) vec4<f32> {
    let voxel = vec3<i32>(floor(in.world_position.xyz - in.normal * 0.01));
    return select(vec4(1.0, 0.0, 0.0, 1.0), vec4(0.0, 1.0, 0.0, 1.0), all(voxel == vec3(0)));
}
"#.replace("SAMPLING", interpolation).replace("OFFSET", &format!("{offset:.4}"));
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MSAA face identity"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(interpolation),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 4,
                ..Default::default()
            },
            multiview_mask: None,
            cache: None,
        });
        let texture = |samples, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MSAA face identity"),
                size: wgpu::Extent3d {
                    width: 64,
                    height: 64,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: samples,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage,
                view_formats: &[],
            })
        };
        let msaa = texture(4, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolved = texture(
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        );
        let msaa_view = msaa.create_view(&Default::default());
        let resolved_view = resolved.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MSAA face identity readback"),
            size: 64 * 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &msaa_view,
                    resolve_target: Some(&resolved_view),
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..6, 0..48);
        }
        encoder.copy_texture_to_buffer(
            resolved.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            resolved.size(),
        );
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap()
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap()
            .unwrap();
        let pixels = readback
            .slice(..)
            .get_mapped_range()
            .expect("the buffer is mapped once map_async completes");
        let wrong = pixels.chunks_exact(4).filter(|rgba| rgba[0] > 0).count();
        let correct = pixels.chunks_exact(4).filter(|rgba| rgba[1] > 0).count();
        eprintln!(
            "{interpolation} offset={offset}: {wrong} wrong-voxel pixels, {correct} correct pixels"
        );
        assert!(wrong + correct > 0, "the subpixel fixture must rasterize");
        results.push(wrong);
        drop(pixels);
        readback.unmap();
    }
    assert!(block_on(scope.pop()).is_none());
    assert!(
        results[0] > 0,
        "the pixel-centre control must reproduce the extrapolation bug"
    );
    assert_eq!(
        results[1], 0,
        "centroid must stay inside faces away from exact raster edge ties"
    );
    // Exact edge samples can still round to the adjacent voxel at a shared
    // integer boundary. Report this separately; centroid is not by itself a
    // complete proof of valid face identity at every floating-point boundary.
}
