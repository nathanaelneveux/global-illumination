//! Camera-scheduled far transport. No lighting buffer is read back, and no
//! gameplay/render system waits for a GPU submission. Per-view ping-pong state
//! retains face identities, path records, unfinished tails and raw statistics.
use super::*;
use crate::voxel_gi::{
    VoxelGiSettings,
    render::{VoxelTransportAssets, VoxelTransportMaterial},
};
use bevy::{
    core_pipeline::{
        prepass::{DepthPrepass, NormalPrepass, ViewPrepassTextures},
        schedule::{Core3d, Core3dSystems},
    },
    platform::collections::HashMap,
    render::{
        GpuResourceAppExt, Render, RenderApp, RenderSystems,
        extract_resource::ExtractResourcePlugin,
        render_asset::RenderAssets,
        renderer::{RenderContext, ViewQuery},
        storage::GpuShaderBuffer,
        view::{ExtractedView, Msaa, RetainedViewEntity},
    },
};

const MAX_PIXELS: u32 = 65536;
const DISPLAY_BYTES: u64 = OUTPUT_PREFIX_BYTES + MAX_PIXELS as u64 * 48;
#[derive(Resource, Clone, ExtractResource)]
struct ViewSettings {
    output: Handle<ShaderBuffer>,
    enabled: bool,
    inner_radius: f32,
    view_first: bool,
    share: f32,
    evidence: u32,
    spatial: f32,
    response: f32,
    measured_retention: f32,
}

#[derive(Component)]
struct AddedPrepasses {
    depth: bool,
    normal: bool,
}

pub(crate) fn install(app: &mut App, output: Handle<ShaderBuffer>) {
    app.init_resource::<VoxelGiSettings>();
    app.insert_resource(ViewSettings {
        output,
        enabled: false,
        inner_radius: 0.0,
        view_first: false,
        share: 0.0,
        evidence: 1024,
        spatial: 1.0,
        response: 0.075,
        measured_retention: 1.0,
    });
    app.add_plugins(ExtractResourcePlugin::<ViewSettings>::default());
    app.add_systems(Update, configure_views);
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.init_gpu_resource::<ViewRenderer>();
        render_app.add_systems(
            Core3d,
            render_view
                .after(Core3dSystems::Prepass)
                .before(Core3dSystems::MainPass),
        );
        render_app.add_systems(Render, retire_views.in_set(RenderSystems::Cleanup));
    }
}

fn retire_views(mut renderer: ResMut<ViewRenderer>, frame: Res<bevy::diagnostic::FrameCount>) {
    // Also release state when the last camera disappears (no Core3d system
    // runs then). Recent views retain only their own bounded history.
    renderer
        .cameras
        .retain(|_, camera| frame.0.wrapping_sub(camera.last_frame) <= 1);
}

fn configure_views(
    settings: Res<VoxelGiSettings>,
    mut view: ResMut<ViewSettings>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut assets: ResMut<VoxelTransportAssets>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    cameras: Query<
        (
            Entity,
            Has<DepthPrepass>,
            Has<NormalPrepass>,
            Option<&AddedPrepasses>,
        ),
        With<Camera3d>,
    >,
    mut commands: Commands,
) {
    let enabled = settings.view_enabled();
    if enabled != view.enabled {
        let mut asset = ShaderBuffer::default();
        asset.buffer_description.size = if enabled { DISPLAY_BYTES } else { EMPTY_OUTPUT_BYTES };
        asset.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        view.output = buffers.add(asset);
        assets.far = view.output.clone();
        for (_, material) in materials.iter_mut() {
            material.extension.far = view.output.clone();
        }
    }
    for (entity, depth, normal, added) in &cameras {
        if enabled && added.is_none() {
            let mut entity = commands.entity(entity);
            entity.insert(AddedPrepasses {
                depth: !depth,
                normal: !normal,
            });
            if !depth {
                entity.insert(DepthPrepass);
            }
            if !normal {
                entity.insert(NormalPrepass);
            }
        } else if !enabled
            && let Some(added) = added
        {
            let mut entity = commands.entity(entity);
            if added.depth {
                entity.remove::<DepthPrepass>();
            }
            if added.normal {
                entity.remove::<NormalPrepass>();
            }
            entity.remove::<AddedPrepasses>();
        }
    }
    view.enabled = enabled;
    view.inner_radius = settings.view_inner_radius();
    view.view_first = settings.view_first();
    view.share = settings.gpu_share() * (1.0 - settings.world_budget_fraction());
    view.evidence = settings.spatial_evidence_target;
    view.spatial = settings.spatial_reconstruction.clamp(0.0, 1.0);
    view.response = settings.display_response_seconds();
    view.measured_retention = settings.measured_retention();
}

#[derive(Clone, Copy, Default, ShaderType)]
struct Control {
    world_from_clip: Mat4,
    camera: Vec4,
    viewport: UVec4,
    grid: UVec4,
    history: UVec4,
    display: Vec4,
}
#[derive(Clone, Copy, Default, ShaderType)]
struct Wave {
    range: UVec4,
    ticket: UVec4,
}

fn storage(device: &RenderDevice, label: &'static str, bytes: usize) -> UploadBuffer {
    let mut buffer = UploadBuffer::new(
        device,
        label,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    buffer
        .ensure(device, bytes)
        .expect("far view storage exceeds device limits");
    buffer
}
struct History {
    receivers: UploadBuffer,
    paths: UploadBuffer,
    tails: UploadBuffer,
    estimates: UploadBuffer,
    hash: UploadBuffer,
}
impl History {
    fn new(device: &RenderDevice, count: u32) -> Self {
        Self {
            receivers: storage(device, "Far receiver identities", count as usize * 32),
            paths: storage(device, "Far retained path records", count as usize * 128),
            tails: storage(device, "Far unfinished paths", count as usize * 128),
            estimates: storage(device, "Far raw estimates", count as usize * 64),
            hash: storage(
                device,
                "Far reprojection index",
                (count * 2).next_power_of_two() as usize * 4,
            ),
        }
    }
}
struct CameraState {
    history: [History; 2],
    current: usize,
    count: u32,
    valid: bool,
    control: UploadBuffer,
    params: UploadBuffer,
    wave: UploadBuffer,
    queue: UploadBuffer,
    cursor: u32,
    last_frame: u32,
    binding_keys: [Option<BindingKey>; 2],
    groups: [Vec<Vec<BindGroup>>; 2],
}
#[derive(PartialEq, Eq)]
struct BindingKey {
    scene: [BufferId; 8],
    depth: bevy::render::render_resource::TextureViewId,
    normals: bevy::render::render_resource::TextureViewId,
    output: BufferId,
    multisampled: bool,
    limited: bool,
}
impl CameraState {
    fn new(device: &RenderDevice, count: u32) -> Self {
        let uniform =
            |name| UploadBuffer::new(device, name, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        Self {
            history: [History::new(device, count), History::new(device, count)],
            current: 0,
            count,
            valid: false,
            control: uniform("Far view control"),
            params: uniform("Far transport parameters"),
            wave: uniform("Far wave allowance"),
            queue: storage(device, "Far visible owner queue", (count as usize + 12) * 4),
            cursor: 0,
            last_frame: 0,
            binding_keys: [None, None],
            groups: [Vec::new(), Vec::new()],
        }
    }
}

struct Kernel {
    pipeline: ComputePipeline,
    /// Explicit entry-point resource sets: automatic layouts omit globals the
    /// entry does not use. Group numbers remain common across all consumers.
    bindings: [Vec<u32>; 3],
}
fn shader(multisampled: bool) -> String {
    let source = include_str!("../../../../assets/shaders/voxel_view_runtime.wgsl").replace(
        "// #include voxel_view_trace.wgsl\n",
        include_str!("../../../../assets/shaders/voxel_view_trace.wgsl"),
    );
    let source = include_retention(&include_paths(&source));
    if multisampled {
        source
            .replace("texture_depth_2d", "texture_depth_multisampled_2d")
            .replace(
                "var normals: texture_2d<f32>",
                "var normals: texture_multisampled_2d<f32>",
            )
    } else {
        source
    }
}

#[derive(Resource)]
struct ViewRenderer {
    discover: [Kernel; 2],
    index: Kernel,
    reproject: Kernel,
    trace: [Kernel; 2],
    retain: Kernel,
    reconstruct: Kernel,
    publish: Kernel,
    cameras: HashMap<RetainedViewEntity, CameraState>,
    frame: u32,
    at: Instant,
    frame_seconds: f32,
    budget: super::timing::Budget,
    changes: Vec<ViewChange>,
}
impl FromWorld for ViewRenderer {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let queue = world.resource::<RenderQueue>();
        let modules = [false, true].map(|msaa| {
            device.create_and_validate_shader_module(ShaderModuleDescriptor {
                label: Some("View-owned voxel GI"),
                source: ShaderSource::Wgsl(shader(msaa).into()),
            })
        });
        let make = |name, msaa: usize, limited: bool, bindings: [&[u32]; 3]| Kernel {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &modules[msaa],
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("LIMITED_REACH", f64::from(limited))],
                    ..default()
                },
                cache: None,
            }),
            bindings: bindings.map(|indices| indices.to_vec()),
        };
        Self {
            discover: [0, 1].map(|m| {
                make(
                    "discover_receivers",
                    m,
                    false,
                    [&[0, 1, 2, 3, 6], &[], &[0, 5, 8, 9]],
                )
            }),
            index: make("index_receivers", 0, false, [&[6], &[], &[0, 7, 10]]),
            reproject: make(
                "reproject_paths",
                0,
                false,
                [&[0, 1, 2, 3, 5, 6, 7], &[0], &[0, 1, 2, 3, 4, 5, 6, 7, 10]],
            ),
            trace: [false, true].map(|limited| {
                make(
                    "trace_visible_paths",
                    0,
                    limited,
                    [&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], &[0, 1], &[0, 5, 10]],
                )
            }),
            retain: make(
                "retain_view_history",
                0,
                false,
                [&[6], &[0], &[0, 5, 7, 12, 13]],
            ),
            reconstruct: make("reconstruct_view", 0, false, [&[6], &[], &[0, 5, 7]]),
            publish: make("publish_view", 0, false, [&[0, 1, 2, 3, 5, 6], &[0], &[0, 5, 7, 10, 11]]),
            cameras: default(),
            frame: 0,
            at: Instant::now(),
            frame_seconds: 1.0 / 60.0,
            budget: super::timing::Budget::new(device, queue),
            changes: Vec::new(),
        }
    }
}

fn grid_size(viewport: UVec2) -> UVec2 {
    // Four-pixel cells preserve thin voxel sides at the normal 720p window.
    // Larger viewports retain the same bounded storage/work ceiling.
    let mut grid = UVec2::new(viewport.x.div_ceil(4), viewport.y.div_ceil(4)).max(UVec2::ONE);
    if grid.x * grid.y > MAX_PIXELS {
        let scale = (MAX_PIXELS as f32 / (grid.x * grid.y) as f32).sqrt();
        grid = (grid.as_vec2() * scale).floor().as_uvec2().max(UVec2::ONE);
    }
    grid
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod batch_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn view_runtime_shader_is_valid() {
        for multisampled in [false, true] {
            let source = shader(multisampled);
            let module = naga::front::wgsl::parse_str(&source)
                .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&source)));
            let layout = &module.types.iter().find(|(_, ty)| ty.name.as_deref() == Some("FarOutput")).unwrap().1.inner;
            let naga::TypeInner::Struct { members, span } = layout else { panic!("FarOutput must be a struct"); };
            assert_eq!(members.last().unwrap().offset as u64, OUTPUT_PREFIX_BYTES);
            assert_eq!(*span as u64, EMPTY_OUTPUT_BYTES);
        }
    }

    #[test]
    #[ignore = "native far-view pipeline qualification"]
    fn native_view_runtime_pipelines() {
        let context = crate::voxel_gi::gpu::tests::native_tracer().context;
        let scope = context
            .device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let mut world = World::new();
        world.insert_resource(context.device.clone());
        world.insert_resource(context.queue.clone());
        let _renderer = ViewRenderer::from_world(&mut world);
        if let Some(error) = bevy::tasks::block_on(scope.pop()) {
            panic!("far pipeline: {error}");
        }
    }
}

fn render_view(
    view: ViewQuery<(&ExtractedView, Option<&ViewPrepassTextures>, &Msaa)>,
    scene: Res<ViewTransport>,
    settings: Res<ViewSettings>,
    assets: Res<RenderAssets<GpuShaderBuffer>>,
    queue: Res<RenderQueue>,
    frame: Res<bevy::diagnostic::FrameCount>,
    mut renderer: ResMut<ViewRenderer>,
    mut ctx: RenderContext,
) {
    let Some(output) = assets.get(&settings.output) else {
        return;
    };
    // The scratch output belongs to this camera's encoder interval. Even a
    // camera without prepasses clears validity before its opaque draws.
    ctx.command_encoder()
        .clear_buffer(&output.buffer, 0, Some(32));
    if !settings.enabled {
        renderer.cameras.clear();
        scene.1.lock().unwrap().clear();
        return;
    }
    let changes = scene.1.clone();
    let Some(scene) = &scene.0 else {
        return;
    };
    let (view, prepass, msaa) = view.into_inner();
    let Some(prepass) = prepass else {
        return;
    };
    let (Some(depth), Some(normals)) = (prepass.depth_view(), prepass.normal_view()) else {
        return;
    };
    if view.viewport.z == 0 || view.viewport.w == 0 {
        return;
    }
    let device = ctx.render_device().clone();
    let now = Instant::now();
    let elapsed = (now - renderer.at).as_secs_f32();
    if renderer.frame != frame.0 {
        renderer.frame = frame.0;
        renderer.at = now;
        renderer.frame_seconds = elapsed;
        renderer
            .budget
            .begin_frame(&device, elapsed, settings.share, settings.view_first);
        if frame.0 % 120 == 0 {
            info!(
                far_active = renderer.budget.active,
                far_trace_ms = renderer.budget.trace_ms,
                far_presentation_ms = renderer.budget.presentation_ms,
                far_completed = renderer.budget.completed,
                far_known = renderer.budget.known,
                far_history = renderer.budget.history,
                far_max_history = renderer.budget.max_history,
                far_uncovered = renderer.budget.uncovered,
                far_displayed = renderer.budget.displayed,
                far_pending = renderer.budget.paused,
                "GI view transport"
            );
        }
        renderer
            .cameras
            .retain(|_, camera| frame.0.wrapping_sub(camera.last_frame) <= 1);
        renderer.changes = std::mem::take(&mut *changes.lock().unwrap());
    }
    let (work, ticket) = renderer.budget.reserve();
    let grid = grid_size(view.viewport.zw());
    let count = grid.x * grid.y;
    let key = view.retained_view_entity;
    let mut camera = renderer
        .cameras
        .remove(&key)
        .filter(|camera| camera.count == count)
        .unwrap_or_else(|| CameraState::new(&device, count));
    let current = &camera.history[camera.current];
    let previous = &camera.history[1 - camera.current];
    let hash_mask = current.hash.capacity as u32 / 4 - 1;
    let clip = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.affine().inverse());
    let control = Control {
        world_from_clip: clip.inverse(),
        camera: view
            .world_from_view
            .translation()
            .extend(settings.inner_radius),
        viewport: view.viewport,
        grid: UVec4::new(grid.x, grid.y, count, hash_mask),
        history: UVec4::new(if camera.valid { count } else { 0 }, hash_mask, frame.0, u32::from(settings.view_first)),
        display: Vec4::new(
            settings.evidence as f32,
            settings.spatial,
            1.0 - (-renderer.frame_seconds.max(1.0 / 240.0) / settings.response).exp(),
            settings.measured_retention,
        ),
    };
    camera
        .control
        .upload(
            &device,
            &queue,
            &ShaderBuffer::from(vec![control]).data.unwrap(),
        )
        .unwrap();
    let mut params = scene.params;
    params.counts.x = count;
    camera
        .params
        .upload(
            &device,
            &queue,
            &ShaderBuffer::from(vec![params]).data.unwrap(),
        )
        .unwrap();
    camera
        .wave
        .upload(
            &device,
            &queue,
            &ShaderBuffer::from(vec![Wave {
                range: UVec4::new(camera.cursor, work, 3, 0),
                ticket: UVec4::ZERO,
            }])
            .data
            .unwrap(),
        )
        .unwrap();
    camera.cursor = camera.cursor.wrapping_add(work);
    let group0 = [
        &camera.params.buffer,
        &scene.buffers[0],
        &scene.buffers[1],
        &scene.buffers[2],
        &scene.buffers[3],
        &scene.buffers[4],
        &current.receivers.buffer,
        &current.paths.buffer,
        &scene.buffers[5],
        &scene.buffers[6],
        &scene.buffers[7],
    ];
    let group1 = [&current.tails.buffer, &camera.wave.buffer];
    let group2 = [
        &camera.control.buffer,
        &previous.receivers.buffer,
        &previous.paths.buffer,
        &previous.tails.buffer,
        &previous.estimates.buffer,
        &current.estimates.buffer,
        &previous.hash.buffer,
        &current.hash.buffer,
    ];
    let encoder = ctx.command_encoder();
    encoder.copy_buffer_to_buffer(&scene.buffers[3], 0, &output.buffer, 32, 16384);
    encoder.clear_buffer(&current.hash.buffer, 0, None);
    encoder.clear_buffer(&camera.queue.buffer, 0, Some(48));
    let multisampled = msaa.samples() > 1;
    let limited = scene.params.transport.x < crate::voxel_gi::tracer::MAX_PATH_DISTANCE;
    let kernels = [
        &renderer.discover[usize::from(multisampled)],
        &renderer.index,
        &renderer.reproject,
        &renderer.trace[usize::from(limited)],
        &renderer.reconstruct,
        &renderer.publish,
    ];
    let make_groups = |kernel: &Kernel, change: Option<&ViewChange>| -> Vec<BindGroup> {
        kernel
            .bindings
            .iter()
            .enumerate()
            .map(|(group, bindings)| {
                let entries: Vec<_> = bindings
                    .iter()
                    .map(|&binding| {
                        let resource = match group {
                            0 => group0[binding as usize].as_entire_binding(),
                            1 => group1[binding as usize].as_entire_binding(),
                            _ => match binding {
                                8 => wgpu::BindingResource::TextureView(depth),
                                9 => wgpu::BindingResource::TextureView(normals),
                                10 => camera.queue.buffer.as_entire_binding(),
                                11 => output.buffer.as_entire_binding(),
                                12 => change.unwrap().events.as_entire_binding(),
                                13 => change.unwrap().control.as_entire_binding(),
                                _ => group2[binding as usize].as_entire_binding(),
                            },
                        };
                        BindGroupEntry { binding, resource }
                    })
                    .collect();
                device.create_bind_group(
                    "Far view entry bindings",
                    &kernel.pipeline.get_bind_group_layout(group as u32).into(),
                    &entries,
                )
            })
            .collect()
    };
    let key_bindings = BindingKey {
        scene: scene.buffers.each_ref().map(Buffer::id),
        depth: depth.id(),
        normals: normals.id(),
        output: output.buffer.id(),
        multisampled,
        limited,
    };
    if camera.binding_keys[camera.current].as_ref() != Some(&key_bindings) {
        camera.groups[camera.current] = kernels
            .iter()
            .map(|kernel| make_groups(kernel, None))
            .collect();
        camera.binding_keys[camera.current] = Some(key_bindings);
    }
    let changes: Vec<_> = renderer
        .changes
        .iter()
        .map(|change| make_groups(&renderer.retain, Some(change)))
        .collect();
    let mut executed = 0u32;
    // Each phase has a timestamp pair in the SAME pass. Cross-pass timestamp
    // pairs can collapse on Metal and falsely report zero or negative costs.
    for stage in 0..7usize {
        let retained = stage == 3;
        let dispatch_count = if stage == 4 { work } else { count };
        if dispatch_count == 0 || (retained && changes.is_empty()) {
            continue;
        }
        executed |= 1 << stage;
        let index = if stage > 3 { stage - 1 } else { stage };
        let kernel = if retained {
            &renderer.retain
        } else {
            kernels[index]
        };
        let timestamps = ticket
            .as_ref()
            .and(renderer.budget.queries.as_ref())
            .map(|queries| wgpu::ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(stage as u32 * 2),
                end_of_pass_write_index: Some(stage as u32 * 2 + 1),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Far view GI"),
            timestamp_writes: timestamps,
        });
        pass.set_pipeline(&kernel.pipeline);
        if retained {
            for groups in &changes {
                for (group, binding) in groups.iter().enumerate() {
                    pass.set_bind_group(group as u32, &**binding, &[]);
                }
                pass.dispatch_workgroups(dispatch_count.div_ceil(64), 1, 1);
            }
        } else {
            for (group, binding) in camera.groups[camera.current][index].iter().enumerate() {
                pass.set_bind_group(group as u32, &**binding, &[]);
            }
            pass.dispatch_workgroups(dispatch_count.div_ceil(64), 1, 1);
        }
    }
    if let Some(ticket) = ticket {
        renderer
            .budget
            .finish(encoder, &camera.queue.buffer, ticket, executed);
    }
    camera.current = 1 - camera.current;
    camera.valid = true;
    camera.last_frame = frame.0;
    renderer.cameras.insert(key, camera);
}
