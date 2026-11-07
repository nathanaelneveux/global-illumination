//! Native regressions for geometry shared with extracted rendering frames.
use super::*;
use crate::{
    self as gi,
    naadf::{linear_index, volume_len},
    world::{AIR, PAGE_SIZE},
};

fn shared_visibility(
    context: &GpuTransportDevice,
    geometry: &([Buffer; 3], gi::render::GpuTraversalParams),
) -> u32 {
    let device = &context.device;
    let queue = &context.queue;
    let source = include_traversal(
        r#"
struct Params { counts: vec4<u32>, minimum: vec4<i32>, maximum: vec4<i32> }
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> directory: array<vec4<i32>>;
@group(0) @binding(2) var<storage, read> regions: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
// #include voxel_traversal.wgsl
@compute @workgroup_size(1)
fn check() { output[0] = trace_domain(vec3(8.5, 5.01, 8.5), vec3(0.0, 1.0, 0.0), 768.0, true).status; }
"#,
    );
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("shared geometry lifetime"),
        source: ShaderSource::Wgsl(Cow::Owned(source)),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("shared geometry lifetime"),
        layout: None,
        module: &module,
        entry_point: Some("check"),
        compilation_options: default(),
        cache: None,
    });
    let mut params = UploadBuffer::new(
        device,
        "geometry test params",
        BufferUsages::UNIFORM | BufferUsages::COPY_DST,
    );
    let mut bytes = Vec::new();
    for word in geometry
        .1
        .counts
        .to_array()
        .into_iter()
        .chain(geometry.1.minimum.to_array().map(|x| x as u32))
        .chain(geometry.1.maximum.to_array().map(|x| x as u32))
    {
        bytes.extend(word.to_le_bytes());
    }
    params.upload(device, queue, &bytes).unwrap();
    let output = UploadBuffer::new(
        device,
        "geometry test output",
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let readback = UploadBuffer::new(
        device,
        "geometry test result",
        BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    );
    let group = device.create_bind_group(
        "shared geometry lifetime",
        &pipeline.get_bind_group_layout(0).into(),
        &[
            BindGroupEntry {
                binding: 0,
                resource: params.buffer.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: geometry.0[0].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: geometry.0[1].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: geometry.0[2].as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: output.buffer.as_entire_binding(),
            },
        ],
    );
    let mut encoder = device.create_command_encoder(&default());
    {
        let mut pass = encoder.begin_compute_pass(&default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &*group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output.buffer, 0, &readback.buffer, 0, 4);
    let submission = queue.submit([encoder.finish()]);
    let (sender, receiver) = mpsc::sync_channel(1);
    readback
        .buffer
        .slice(..4)
        .map_async(wgpu::MapMode::Read, move |r| {
            sender.send(r).unwrap();
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
        .slice(..4)
        .get_mapped_range()
        .expect("the buffer is mapped once map_async completes");
    let status = u32::from_le_bytes(view[..4].try_into().unwrap());
    drop(view);
    readback.buffer.unmap();
    status
}

#[test]
#[ignore = "native shared-geometry allocation lifetime regression"]
fn cells_growth_preserves_published_geometry_generation() {
    let mut gpu = super::tests::native_tracer();
    let mut before = vec![AIR; volume_len(PAGE_SIZE)];
    before[linear_index(UVec3::new(8, 4, 8), PAGE_SIZE)] = 2;
    for z in 0..32 {
        for y in 16..32 {
            for x in 0..32 {
                before[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 2;
            }
        }
    }
    let mut after = vec![AIR; volume_len(PAGE_SIZE)];
    after[linear_index(UVec3::new(8, 4, 8), PAGE_SIZE)] = 2;
    after[linear_index(UVec3::new(8, 20, 8), PAGE_SIZE)] = 2;
    let scene = Arc::new(TransportScene::with_page_delta(
        &default(),
        vec![(IVec3::ZERO, before)],
        &[],
    ));
    let edited = Arc::new(TransportScene::with_page_delta(
        &scene,
        vec![(IVec3::ZERO, after)],
        &[],
    ));
    let emitters = Arc::new(EmissionDistribution::default());
    let mut probes = vec![SurfaceProbe::new(IVec3::new(8, 4, 8), 4)];
    gpu.trace(scene, &default(), &default(), &mut probes, 1, &emitters)
        .unwrap();
    let held = gpu.geometry_display().unwrap();
    let before_status = shared_visibility(&gpu.context, &held);
    gpu.trace(edited, &default(), &default(), &mut probes, 1, &emitters)
        .unwrap();
    let current = gpu.geometry_display().unwrap();
    assert_ne!(held.0[0].id(), current.0[0].id());
    assert_ne!(held.0[1].id(), current.0[1].id());
    assert_ne!(held.0[2].id(), current.0[2].id());
    let held_status = shared_visibility(&gpu.context, &held);
    let current_status = shared_visibility(&gpu.context, &current);
    eprintln!(
        "SHARED_GEOMETRY old_cells={} new_cells={} old_before_status={before_status} held_after_status={held_status} current_status={current_status}",
        held.0[2].size(),
        current.0[2].size()
    );
    assert_eq!(before_status, 1);
    assert_eq!(current_status, 1);
    assert_eq!(held_status, 1, "published old tuple must remain coherent");
}

#[test]
#[ignore = "native shared-geometry bounds and sparse-edit regression"]
fn geometry_bounds_detach_but_ordinary_edits_reuse_allocations() {
    let mut gpu = super::tests::native_tracer();
    let solid = vec![2; volume_len(PAGE_SIZE)];
    let base = Arc::new(TransportScene::with_page_delta(
        &default(),
        vec![(IVec3::ZERO, solid.clone())],
        &[],
    ));
    gpu.synchronize_geometry(&base).unwrap();
    let held = gpu.geometry_display().unwrap();
    assert_eq!(shared_visibility(&gpu.context, &held), 1);
    let moved = Arc::new(TransportScene::with_page_delta(
        &base,
        vec![(IVec3::new(2, 0, 0), solid.clone())],
        &[IVec3::ZERO],
    ));
    gpu.synchronize_geometry(&moved).unwrap();
    let current = gpu.geometry_display().unwrap();
    for i in 0..3 {
        assert_eq!(held.0[i].size(), current.0[i].size(), "no growth needed");
        assert_ne!(
            held.0[i].id(),
            current.0[i].id(),
            "bounds require a new generation"
        );
    }
    assert_eq!(shared_visibility(&gpu.context, &held), 1);
    assert_eq!(shared_visibility(&gpu.context, &current), 0);

    // Uniform material replacement changes headers, not directory or bounds.
    let edited = Arc::new(TransportScene::with_page_delta(
        &moved,
        vec![(IVec3::new(2, 0, 0), vec![3; solid.len()])],
        &[],
    ));
    let uploaded = gpu.uploaded_geometry_bytes;
    gpu.synchronize_geometry(&edited).unwrap();
    let changed = gpu.geometry_display().unwrap();
    for i in 0..3 {
        assert_eq!(current.0[i].id(), changed.0[i].id());
    }
    assert_eq!(gpu.uploaded_geometry_bytes - uploaded, 8 * 16);
    gpu.synchronize_geometry(&edited).unwrap();
    assert_eq!(
        gpu.uploaded_geometry_bytes - uploaded,
        8 * 16,
        "unchanged scenes do not upload"
    );
}
