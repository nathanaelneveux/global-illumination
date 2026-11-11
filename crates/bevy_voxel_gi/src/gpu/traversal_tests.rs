//! Native boundary checks for the production box-step function, independent of
//! path sampling. Keep a scalar XYZ oracle to make tie/epsilon behavior explicit.

use super::*;

#[derive(Clone, Copy, ShaderType)]
struct Case {
    position: Vec4,
    direction: Vec4,
    cell: IVec4,
    negative: IVec4,
    positive: IVec4,
}

fn reference(case: &Case) -> Vec4 {
    let size = case.position.w;
    let minimum = (case.cell - case.negative).xyz().as_vec3() * size;
    let maximum = (case.cell + case.positive + IVec4::ONE).xyz().as_vec3() * size;
    let mut distance = 1e30;
    let mut normal = Vec3::ZERO;
    for axis in 0..3 {
        let direction = case.direction[axis];
        if direction.abs() <= 1e-8 {
            continue;
        }
        let plane = if direction > 0.0 {
            maximum[axis]
        } else {
            minimum[axis]
        };
        let travel = (plane - case.position[axis]) / direction;
        if travel >= -0.001 && travel < distance {
            distance = travel.max(0.0);
            normal = Vec3::ZERO;
            normal[axis] = if direction > 0.0 { -1.0 } else { 1.0 };
        }
    }
    normal.extend((distance + 0.001).max(0.001))
}

#[test]
#[ignore = "native Metal/Vulkan box-step boundary regression"]
fn native_box_steps_match_scalar_boundary_policy() {
    let mut cases = Vec::new();
    let components = [
        -1.0, -1.01e-8, -1e-8, -0.99e-8, -0.0, 0.0, 0.99e-8, 1e-8, 1.01e-8, 1.0,
    ];
    for size in [1.0, 4.0, 16.0] {
        for base in [-4096, -16, -1, 0, 1, 16, 4096] {
            let cell = IVec3::new(base, -base, base / 2);
            for fraction in [-0.0005, 0.0, 0.5, 1.0, 1.0005] {
                for extent in [IVec3::ZERO, IVec3::new(3, 2, 1)] {
                    for x in components {
                        for y in components {
                            for z in components {
                                cases.push(Case {
                                    position: ((cell.as_vec3() + Vec3::splat(fraction)) * size)
                                        .extend(size),
                                    direction: Vec4::new(x, y, z, 0.0),
                                    cell: cell.extend(0),
                                    negative: extent.extend(0),
                                    positive: extent.extend(0),
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    // Unequal negative crossings exercise clamping before the next axis: the
    // last eligible negative crossing wins, not the most-negative component.
    cases.push(Case {
        position: Vec4::new(1.0009, 1.0001, 1.0002, 1.0),
        direction: Vec4::ONE,
        cell: IVec4::ZERO,
        negative: IVec4::ZERO,
        positive: IVec4::ZERO,
    });
    let gpu = super::tests::native_tracer();
    let device = &gpu.context.device;
    let queue = &gpu.context.queue;
    let function = SHADER
        .split("fn advance_box(")
        .nth(1)
        .unwrap()
        .split("fn geometry_ahead(")
        .next()
        .unwrap();
    let source = format!(
        r#"
struct Case {{ position: vec4<f32>, direction: vec4<f32>, cell: vec4<i32>, negative: vec4<i32>, positive: vec4<i32> }}
@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> output: array<vec4<f32>>;
fn advance_box({function}
@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= arrayLength(&cases) {{ return; }}
    let c = cases[id.x];
    output[id.x] = advance_box(c.position.xyz, c.direction.xyz, c.cell.xyz, c.negative.xyz, c.positive.xyz, c.position.w);
}}
"#
    );
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("GI box-step boundary checks"),
        source: ShaderSource::Wgsl(Cow::Owned(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("GI box-step boundary checks"),
        layout: None,
        module: &module,
        entry_point: Some("check"),
        compilation_options: default(),
        cache: None,
    });
    let mut input = UploadBuffer::new(
        device,
        "Box cases",
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    input
        .upload(device, queue, &crate::gpu::std430(&cases.clone()))
        .unwrap();
    let bytes = cases.len() * 16;
    let mut output = UploadBuffer::new(
        device,
        "Box results",
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    output.ensure(device, bytes).unwrap();
    let mut readback = UploadBuffer::new(
        device,
        "Box readback",
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    );
    readback.ensure(device, bytes).unwrap();
    let bindings = device.create_bind_group(
        "Box cases",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: input.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: output.buffer.as_entire_binding(),
            },
        ],
    );
    let mut encoder = device.create_command_encoder(&default());
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Box cases"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &*bindings, &[]);
        pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output.buffer, 0, &readback.buffer, 0, bytes as u64);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback
        .buffer
        .slice(..bytes as u64)
        .map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(5)),
        })
        .unwrap();
    receiver
        .recv_timeout(Duration::from_secs(1))
        .unwrap()
        .unwrap();
    let view = readback
        .buffer
        .slice(..bytes as u64)
        .get_mapped_range()
        .expect("the buffer is mapped once map_async completes");
    for (index, (bytes, case)) in view.chunks_exact(16).zip(&cases).enumerate() {
        let actual = Vec4::from_array(std::array::from_fn(|i| {
            f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
        }));
        let expected = reference(case);
        assert_eq!(
            actual.xyz(),
            expected.xyz(),
            "normal in case {index}: position={:?} direction={:?}",
            case.position,
            case.direction
        );
        assert!(
            (actual.w - expected.w).abs() <= 1e-5 * (1.0 + expected.w.abs()),
            "distance in case {index}: {actual:?} vs {expected:?}"
        );
    }
    eprintln!(
        "{} native box-step boundary cases match the scalar policy",
        cases.len()
    );
    drop(view);
    readback.buffer.unmap();
}
