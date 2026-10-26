//! Camera-scheduled far transport. No lighting buffer is read back, and no
//! gameplay/render system waits for a GPU submission. Per-view ping-pong state
//! retains face identities, path records, unfinished tails and raw statistics.
use super::*;
use crate::{PixelTierSettings, VoxelGiSettings, render::VoxelGiMaterialData};
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

const POOL_PREFIX_BYTES: usize = 16 + 65536 * 20;
const DISPLAY_BYTES: u64 = OUTPUT_PREFIX_BYTES + MAX_VIEW_RECEIVERS as u64 * 48;
#[derive(Resource, Clone, ExtractResource)]
struct ViewSettings {
    output: Handle<ShaderBuffer>,
    /// EXPERIMENT: per-pixel tier cells read by the material.
    pixel_output: Handle<ShaderBuffer>,
    pixel: PixelTierSettings,
    pixel_cell_allocated: u32,
    enabled: bool,
    inner_radius: f32,
    view_first: bool,
    share: f32,
    noise_threshold: f32,
    spatial: f32,
    response: f32,
    measured_retention: f32,
    receivers: u32,
    /// Last frame's view-tier GPU milliseconds, written by the render world.
    receipts: std::sync::Arc<std::sync::Mutex<ViewReceipts>>,
    /// Corner-sunlight flips read back by the render world (min, max, flux).
    sun_flips: std::sync::Arc<std::sync::Mutex<Vec<(IVec3, IVec3, f32)>>>,
}

/// View-tier GPU time per frame (timestamp queries; zero when unavailable).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewReceipts {
    pub trace_ms: f32,
    pub presentation_ms: f32,
    pub discovery_ms: f32,
}

#[derive(Component)]
struct AddedPrepasses {
    depth: bool,
    normal: bool,
}

pub(crate) fn install(app: &mut App, output: Handle<ShaderBuffer>, pixel_output: Handle<ShaderBuffer>) {
    app.init_resource::<VoxelGiSettings>();
    app.init_resource::<crate::VoxelGiStats>();
    app.insert_resource(ViewSettings {
        output,
        pixel_output,
        pixel: PixelTierSettings::default(),
        pixel_cell_allocated: 0,
        enabled: false,
        inner_radius: 0.0,
        view_first: false,
        share: 0.0,
        noise_threshold: 0.031,
        spatial: 1.0,
        response: 0.075,
        measured_retention: 1.0,
        receivers: 65_536,
        receipts: default(),
        sun_flips: default(),
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
    mut stats: ResMut<crate::VoxelGiStats>,
    mut view: ResMut<ViewSettings>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut material_data: ResMut<VoxelGiMaterialData>,
    runtime: Option<ResMut<crate::VoxelGiRuntime>>,
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
    let receipts = *view.receipts.lock().unwrap();
    if stats.view_gpu != receipts {
        stats.view_gpu = receipts;
    }
    let flips = std::mem::take(&mut *view.sun_flips.lock().unwrap());
    if let Some(mut runtime) = runtime {
        let runtime = &mut *runtime;
        runtime.sun_changes.add(flips);
        if let Some(change) = runtime.sun_changes.take(&runtime.scene, std::time::Instant::now()) {
            // Events carry no page allocations: retired world pages must not
            // suppress the change for view-owned receivers.
            runtime.spatial_history.record(change, &[]);
        }
    }
    let enabled = true;
    if enabled != view.enabled {
        let mut asset = ShaderBuffer::default();
        asset.buffer_description.size = if enabled { DISPLAY_BYTES } else { EMPTY_OUTPUT_BYTES };
        asset.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        view.output = buffers.add(asset);
        material_data.far = view.output.clone();
    }
    let pixel = settings.pixel_tier.normalized();
    if enabled && view.pixel_cell_allocated != pixel.cell {
        let mut asset = ShaderBuffer::default();
        asset.buffer_description.size = pixel_output_bytes(pixel.cell);
        asset.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        view.pixel_output = buffers.add(asset);
        material_data.far_pixels = view.pixel_output.clone();
        view.pixel_cell_allocated = pixel.cell;
    }
    if view.pixel != pixel {
        view.pixel = pixel;
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
    view.inner_radius = 0.0;
    view.view_first = true;
    view.share = settings.gpu_share() * (1.0 - settings.world_budget_fraction());
    view.noise_threshold = settings.noise_threshold();
    view.spatial = settings.spatial_reconstruction.clamp(0.0, 1.0);
    view.response = settings.display_response_seconds();
    view.measured_retention = settings.measured_retention();
    view.receivers = settings.view_receiver_limit();
}

#[derive(Clone, Copy, Default, ShaderType)]
struct Control {
    world_from_clip: Mat4,
    clip_from_world: Mat4,
    camera: Vec4,
    viewport: UVec4,
    grid: UVec4,
    history: UVec4,
    display: Vec4,
    cache: UVec4,
    discovery: UVec4,
    /// EXPERIMENT per-pixel tier: view forward (xyz) and pixels per world
    /// unit at unit view depth (w).
    pixel_axis: Vec4,
    /// Cells x, y, pixels per cell side, enabled.
    pixel_grid: UVec4,
    /// Voxel-size threshold (px), maximum history, trace rate, previous valid.
    pixel_tune: Vec4,
    previous_clip_from_world: Mat4,
}
/// Byte offsets of `Control` words that native tests rewrite in place
/// (`history.w`: view-first; `cache.x`: writable epoch).
#[cfg(test)]
pub(super) const CONTROL_VIEW_FIRST_OFFSET: u64 = 64 * 2 + 16 * 3 + 12;
#[cfg(test)]
pub(super) const CONTROL_EPOCH_OFFSET: u64 = 64 * 2 + 16 * 5;
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
    pool: UploadBuffer,
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
            pool: storage(device, "Far shared receiver pool and spatial index", POOL_PREFIX_BYTES + count as usize * 16),
        }
    }
}
/// EXPERIMENT per-pixel tier state: ping-pong cell histories (mean RGB and
/// count, surface point and face; 32 bytes) and four counters.
struct PixelState {
    cells: UVec2,
    history: [UploadBuffer; 2],
    stats: UploadBuffer,
    /// Cells to trace (count, indices) and the trace's indirect dispatch.
    queue: UploadBuffer,
    args: Buffer,
}
impl PixelState {
    fn new(device: &RenderDevice, cells: UVec2) -> Self {
        let bytes = cells.element_product().max(1) as usize * 32;
        Self {
            cells,
            history: [0, 1].map(|_| storage(device, "Far per-pixel cell history", bytes)),
            stats: storage(device, "Far per-pixel counters", 16),
            queue: storage(device, "Far per-pixel trace queue", 4 + cells.element_product().max(1) as usize * 4),
            args: device.create_buffer(&BufferDescriptor {
                label: Some("Far per-pixel trace dispatch"),
                size: 16,
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
        }
    }
}
struct CameraState {
    history: [History; 2],
    pixel: Option<PixelState>,
    previous_clip: Option<Mat4>,
    current: usize,
    grid: UVec2,
    discovery_view: Option<(Mat4, UVec4, f32, u32)>,
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
    cache: [BufferId; 2],
    depth: bevy::render::render_resource::TextureViewId,
    normals: bevy::render::render_resource::TextureViewId,
    output: BufferId,
    pixel: [BufferId; 3],
    multisampled: bool,
    limited: bool,
}
impl CameraState {
    fn new(device: &RenderDevice, grid: UVec2) -> Self {
        let count = grid.element_product();
        let uniform =
            |name| UploadBuffer::new(device, name, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        Self {
            history: [History::new(device, count), History::new(device, count)],
            pixel: None,
            previous_clip: None,
            current: 0,
            grid,
            discovery_view: None,
            valid: false,
            control: uniform("Far view control"),
            params: uniform("Far transport parameters"),
            wave: uniform("Far wave allowance"),
            queue: storage(device, "Far visible owner queue and sequence reservations", (count as usize * 2 + 12) * 4 + SUN_FLIP_BYTES as usize),
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
    let source = include_str!("../../../shaders/voxel_view_runtime.wgsl").replace(
        "// #include voxel_view_trace.wgsl\n",
        include_str!("../../../shaders/voxel_view_trace.wgsl"),
    );
    let source = include_retention(&include_paths(&source));
    let source = source.replace("// #include voxel_view_cache.wgsl\n",
        include_str!("../../../shaders/voxel_view_cache.wgsl"));
    let source = source.replace("// #include voxel_view_discovery.wgsl\n",
        include_str!("../../../shaders/voxel_view_discovery.wgsl"));
    let source = source.replace("// #include voxel_view_pixels.wgsl\n",
        include_str!("../../../shaders/voxel_view_pixels.wgsl"));
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
    prepare_pool: [Kernel; 2],
    discover: [Kernel; 2],
    index: Kernel,
    reproject: Kernel,
    trace: [Kernel; 2],
    retain: Kernel,
    load_cache: Kernel,
    store_cache: Kernel,
    reconstruct: Kernel,
    publish: Kernel,
    /// Per-pixel tier: classify (by MSAA), size the queue, trace it.
    pixels: [Kernel; 2],
    pixel_args: Kernel,
    pixel_trace: Kernel,
    /// The step budget and bounces `pixel_trace` was built with.
    pixel_steps: (u32, u32),
    modules: [bevy::render::render_resource::ShaderModule; 2],
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
            prepare_pool: [0, 1].map(|m| make("prepare_receiver_pool", m, false,
                [&[0, 1, 2, 3, 6], &[], &[0, 1, 5, 6, 7, 8, 9, 10, 17, 18]])),
            discover: [0, 1].map(|m| {
                make(
                    "discover_receivers",
                    m,
                    false,
                    [&[0, 1, 2, 3, 6], &[], &[0, 5, 7, 8, 9, 17]],
                )
            }),
            index: make("index_receivers", 0, false, [&[6], &[], &[0, 7, 10]]),
            reproject: make(
                "reproject_paths",
                0,
                false,
                [&[0, 1, 2, 3, 4, 5, 6, 7], &[0], &[0, 1, 2, 3, 4, 5, 6, 7, 10]],
            ),
            trace: [false, true].map(|limited| {
                make(
                    "trace_visible_paths",
                    0,
                    limited,
                    [&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], &[0, 1], &[0, 5, 10, 16]],
                )
            }),
            retain: make(
                "retain_view_history",
                0,
                false,
                [&[6, 10], &[0], &[0, 5, 7, 12, 13, 15, 16]],
            ),
            load_cache: make("load_shared_faces", 0, false,
                [&[0, 1, 2, 3, 4, 5, 6, 7, 10], &[0, 1], &[0, 1, 4, 5, 6, 7, 10, 14, 15, 16]]),
            store_cache: make("store_shared_faces", 0, false,
                [&[6, 10], &[], &[0, 5, 7, 10, 14, 15, 16]]),
            reconstruct: make("reconstruct_view", 0, false, [&[6], &[], &[0, 5, 7, 17]]),
            publish: make("publish_view", 0, false, [&[0, 1, 2, 3, 5, 6], &[0], &[0, 5, 7, 10, 11, 17]]),
            pixels: [0, 1].map(|m| make("classify_pixels", m, false, PIXEL_CLASSIFY_BINDINGS)),
            pixel_args: make("size_pixel_queue", 0, false, [&[], &[], &[23, 24]]),
            pixel_trace: pixel_kernel(device, &modules, (PixelTierSettings::default().steps, PixelTierSettings::default().bounces)),
            pixel_steps: (PixelTierSettings::default().steps, PixelTierSettings::default().bounces),
            modules,
            cameras: default(),
            frame: 0,
            at: Instant::now(),
            frame_seconds: 1.0 / 60.0,
            budget: super::timing::Budget::new(device, queue),
            changes: Vec::new(),
        }
    }
}

/// Corner-sunlight flips (`report_sun_flips`) follow the owner queue: a count,
/// up to this many (voxel, flux) records, then an overflow box and flux.
/// They persist until a timing receipt copies them (`Budget::finish` clears
/// them after the copy).
pub(super) const MAX_SUN_FLIPS: u64 = 2048;
pub(super) const SUN_FLIP_BYTES: u64 = 16 + MAX_SUN_FLIPS * 16 + 32;
pub(super) const SUN_FLIP_BIAS: i64 = 1 << 30;

/// The per-pixel tier's trace under its ray budget: `steps` traversal
/// steps per ray, exhaustion a miss (NAADF's rule).
fn pixel_kernel(device: &RenderDevice, modules: &[bevy::render::render_resource::ShaderModule; 2], (steps, bounces): (u32, u32)) -> Kernel {
    Kernel {
        pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace_pixel_queue"),
            layout: None,
            module: &modules[0],
            entry_point: Some("trace_pixel_queue"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("PATH_STEPS", f64::from(steps)), ("BUDGETED_RAYS", 1.0),
                    ("DIFFUSE_HITS", f64::from(bounces))],
                ..default()
            },
            cache: None,
        }),
        bindings: PIXEL_TRACE_BINDINGS.map(|indices| indices.to_vec()),
    }
}

/// Per-pixel tier resources (automatic layouts must match exactly).
const PIXEL_CLASSIFY_BINDINGS: [&[u32]; 3] = [&[0, 1, 2], &[], &[0, 8, 9, 19, 20, 21, 22, 23]];
const PIXEL_TRACE_BINDINGS: [&[u32]; 3] = [&[0, 1, 2, 3, 4, 5, 8, 9, 10], &[], &[0, 20, 21, 22, 23]];

fn sun_flip_offset(count: u32) -> u64 {
    (12 + 2 * u64::from(count)) * 4
}

fn grid_size(viewport: UVec2, receivers: u32) -> UVec2 {
    // Nominal four-pixel cells, capped at `receivers` (the view_receivers
    // setting) at larger viewports. Discovery uses 2x2-cell spatial bins;
    // physical receiver slots are shared globally.
    let limit = receivers.min(MAX_VIEW_RECEIVERS);
    let mut grid = UVec2::new(viewport.x.div_ceil(4), viewport.y.div_ceil(4)).max(UVec2::ONE);
    if grid.x * grid.y > limit {
        let scale = (limit as f32 / (grid.x * grid.y) as f32).sqrt();
        grid = (grid.as_vec2() * scale).floor().as_uvec2().max(UVec2::ONE);
    }
    grid
}

#[cfg(test)]
#[path = "batch_tests.rs"]
mod batch_tests;

#[cfg(test)]
#[path = "cache_tests.rs"]
mod cache_tests;

#[cfg(test)]
#[path = "coverage_tests.rs"]
mod coverage_tests;

#[cfg(test)]
#[path = "pixel_tests.rs"]
mod pixel_tests;

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
            let pool = &module.types.iter().find(|(_, ty)| ty.name.as_deref() == Some("ReceiverPool")).unwrap().1.inner;
            let naga::TypeInner::Struct { members, .. } = pool else { panic!("ReceiverPool must be a struct"); };
            assert_eq!(members.last().unwrap().offset as usize, POOL_PREFIX_BYTES);
        }
    }

    #[test]
    #[ignore = "native far-view pipeline qualification"]
    fn native_view_runtime_pipelines() {
        let context = crate::gpu::tests::native_tracer().context;
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
    let Some(pixel_output) = assets.get(&settings.pixel_output) else {
        return;
    };
    // The scratch output belongs to this camera's encoder interval. Even a
    // camera without prepasses clears validity before its opaque draws.
    ctx.command_encoder()
        .clear_buffer(&output.buffer, 0, Some(32));
    ctx.command_encoder()
        .clear_buffer(&pixel_output.buffer, 0, Some(PIXEL_HEADER_BYTES));
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
        settings.sun_flips.lock().unwrap().append(&mut renderer.budget.sun_flips);
        *settings.receipts.lock().unwrap() = ViewReceipts {
            trace_ms: renderer.budget.trace_ms,
            presentation_ms: renderer.budget.presentation_ms,
            discovery_ms: renderer.budget.discovery_ms,
        };
        if frame.0.is_multiple_of(120) {
            info!(
                far_active = renderer.budget.active,
                far_trace_ms = renderer.budget.trace_ms,
                far_presentation_ms = renderer.budget.presentation_ms,
                far_discovery_ms = renderer.budget.discovery_ms,
                far_discovery_mean_ms = renderer.budget.discovery_sum_ms / f64::from(renderer.budget.discovery_count.max(1)),
                far_discovery_receipts = renderer.budget.discovery_count,
                far_pool_prepare_ms = renderer.budget.pool_prepare_ms,
                far_pool_scan_ms = renderer.budget.pool_scan_ms,
                far_pool_free = renderer.budget.pool_free,
                far_pool_claimed = renderer.budget.pool_claimed,
                far_pool_exhausted = renderer.budget.pool_exhausted,
                far_pool_retained = renderer.budget.pool_retained,
                far_cache_ms = renderer.budget.cache_ms,
                far_cache_loads = renderer.budget.cache_loads,
                far_cache_promotions = renderer.budget.cache_promotions,
                far_completed = renderer.budget.completed,
                far_known = renderer.budget.known,
                far_history = renderer.budget.history,
                far_max_history = renderer.budget.max_history,
                far_uncovered = renderer.budget.uncovered,
                far_displayed = renderer.budget.displayed,
                far_pending = renderer.budget.paused,
                far_pixel_ms = renderer.budget.pixel_ms,
                far_pixel_owned = renderer.budget.pixel_owned,
                far_pixel_traced = renderer.budget.pixel_traced,
                far_pixel_reprojected = renderer.budget.pixel_reprojected,
                far_pixel_history = renderer.budget.pixel_history,
                far_stages_ms = ?renderer.budget.stage_ms.map(|ms| (ms * 100.0).round() / 100.0),
                "GI view transport"
            );
            renderer.budget.discovery_sum_ms = 0.0;
            renderer.budget.discovery_count = 0;
        }
        renderer
            .cameras
            .retain(|_, camera| frame.0.wrapping_sub(camera.last_frame) <= 1);
        renderer.changes = std::mem::take(&mut *changes.lock().unwrap());
    }
    let pixel_budget = (settings.pixel.steps, settings.pixel.bounces);
    if renderer.pixel_steps != pixel_budget {
        renderer.pixel_trace = pixel_kernel(&device, &renderer.modules, pixel_budget);
        renderer.pixel_steps = pixel_budget;
        for camera in renderer.cameras.values_mut() {
            camera.binding_keys = [None, None];
        }
    }
    let (work, ticket) = renderer.budget.reserve();
    let grid = grid_size(view.viewport.zw(), settings.receivers);
    let count = grid.x * grid.y;
    let key = view.retained_view_entity;
    let mut camera = renderer
        .cameras
        .remove(&key)
        .filter(|camera| camera.grid == grid)
        .unwrap_or_else(|| CameraState::new(&device, grid));
    let pixel_size = settings.pixel.cell;
    let pixel_cells = view.viewport.zw().map(|v| v.div_ceil(pixel_size));
    let pixel_enabled = settings.pixel.voxel_pixels > 0.0
        && u64::from(pixel_cells.element_product()) * 16 + PIXEL_HEADER_BYTES <= pixel_output.buffer.size();
    if camera.pixel.as_ref().is_none_or(|state| state.cells != pixel_cells) {
        camera.pixel = Some(PixelState::new(&device, pixel_cells));
        camera.previous_clip = None;
    }
    let pixel_state = camera.pixel.as_ref().unwrap();
    let current = &camera.history[camera.current];
    let previous = &camera.history[1 - camera.current];
    let hash_mask = current.hash.capacity as u32 / 4 - 1;
    let clip = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.affine().inverse());
    let discovery_view = (clip, view.viewport, settings.inner_radius, msaa.samples());
    let discovery_stable = camera.valid && camera.discovery_view == Some(discovery_view);
    let discovery_compatible = camera.valid && camera.discovery_view.is_some_and(|(_, viewport, radius, samples)|
        viewport == view.viewport && radius == settings.inner_radius && samples == msaa.samples());
    camera.discovery_view = Some(discovery_view);
    let control = Control {
        world_from_clip: clip.inverse(),
        clip_from_world: clip,
        camera: view
            .world_from_view
            .translation()
            .extend(settings.inner_radius),
        viewport: view.viewport,
        grid: UVec4::new(grid.x, grid.y, count, hash_mask),
        history: UVec4::new(if camera.valid { count } else { 0 }, hash_mask, frame.0, u32::from(settings.view_first)),
        display: Vec4::new(
            settings.noise_threshold,
            settings.spatial,
            1.0 - (-renderer.frame_seconds.max(1.0 / 240.0) / settings.response).exp(),
            settings.measured_retention,
        ),
        cache: scene.cache.as_ref().map_or(UVec4::ZERO, |cache|
            UVec4::new(cache.epoch, cache.origin.elapsed().as_millis() as u32,
                u32::from(cache.enabled && settings.view_first), 0)),
        discovery: UVec4::new(u32::from(discovery_stable), u32::from(discovery_compatible), 0, 0),
        pixel_axis: {
            let forward = view.world_from_view.forward();
            // clip_from_view[1][1]: NDC y per unit of y / view depth.
            let scale = view.clip_from_view.y_axis.y * view.viewport.w as f32 * 0.5;
            forward.extend(scale)
        },
        // w: enabled (bit 0), measure (bits 1-2; `pixel_owned`).
        pixel_grid: UVec4::new(pixel_cells.x, pixel_cells.y, pixel_size,
            u32::from(pixel_enabled) | (settings.pixel.measure as u32) << 1),
        pixel_tune: Vec4::new(
            settings.pixel.voxel_pixels,
            settings.pixel.history,
            settings.pixel.rate as f32,
            f32::from(u8::from(camera.previous_clip.is_some())),
        ),
        previous_clip_from_world: camera.previous_clip.unwrap_or(clip),
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
    encoder.clear_buffer(&current.pool.buffer, 0, Some(POOL_PREFIX_BYTES as u64));
    encoder.clear_buffer(&camera.queue.buffer, 0, Some(48));
    encoder.clear_buffer(&pixel_state.stats.buffer, 0, None);
    encoder.clear_buffer(&pixel_state.queue.buffer, 0, Some(4));
    let multisampled = msaa.samples() > 1;
    let limited = scene.params.transport.x < crate::tracer::MAX_PATH_DISTANCE;
    let kernels = [
        &renderer.discover[usize::from(multisampled)],
        &renderer.index,
        &renderer.reproject,
        &renderer.load_cache,
        &renderer.trace[usize::from(limited)],
        &renderer.reconstruct,
        &renderer.store_cache,
        &renderer.publish,
        &renderer.prepare_pool[usize::from(multisampled)],
        &renderer.pixels[usize::from(multisampled)],
        &renderer.pixel_args,
        &renderer.pixel_trace,
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
                                14 => scene.buffers[6].as_entire_binding(),
                                15 => scene.cache.as_ref().map_or(&scene.buffers[6], |cache| &cache.meta).as_entire_binding(),
                                16 => scene.cache.as_ref().map_or(&scene.buffers[6], |cache| &cache.guard).as_entire_binding(),
                                17 => current.pool.buffer.as_entire_binding(),
                                18 => previous.pool.buffer.as_entire_binding(),
                                19 => pixel_state.history[1 - camera.current].buffer.as_entire_binding(),
                                20 => pixel_state.history[camera.current].buffer.as_entire_binding(),
                                21 => pixel_state.stats.buffer.as_entire_binding(),
                                22 => pixel_output.buffer.as_entire_binding(),
                                23 => pixel_state.queue.buffer.as_entire_binding(),
                                24 => pixel_state.args.as_entire_binding(),
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
        cache: scene.cache.as_ref().map_or([scene.buffers[6].id(); 2], |cache| [cache.meta.id(), cache.guard.id()]),
        depth: depth.id(),
        normals: normals.id(),
        output: output.buffer.id(),
        pixel: [
            pixel_state.history[1 - camera.current].buffer.id(),
            pixel_state.history[camera.current].buffer.id(),
            pixel_output.buffer.id(),
        ],
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
    for stage in [9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10] {
        let retained = stage == 3;
        let dispatch_count = if stage == 10 {
            if pixel_enabled { pixel_cells.element_product() } else { 0 }
        } else if stage == 5 {
            work
        } else if stage == 0 {
            // One cooperative workgroup per 2x2-cell discovery tile.
            grid.x.div_ceil(2) * grid.y.div_ceil(2)
        } else {
            count
        };
        if dispatch_count == 0 || (retained && changes.is_empty())
            || (stage == 7 && control.cache.z == 0)
        {
            continue;
        }
        executed |= 1 << stage;
        let index = if stage == 10 { 9 } else if stage == 9 { 8 } else if stage > 3 { stage - 1 } else { stage };
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
        } else if stage == 10 {
            // Classify every cell, size the queue, trace it densely.
            for (group, binding) in camera.groups[camera.current][9].iter().enumerate() {
                pass.set_bind_group(group as u32, &**binding, &[]);
            }
            pass.dispatch_workgroups(dispatch_count.div_ceil(64), 1, 1);
            pass.set_pipeline(&renderer.pixel_args.pipeline);
            for (group, binding) in camera.groups[camera.current][10].iter().enumerate() {
                pass.set_bind_group(group as u32, &**binding, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&renderer.pixel_trace.pipeline);
            for (group, binding) in camera.groups[camera.current][11].iter().enumerate() {
                pass.set_bind_group(group as u32, &**binding, &[]);
            }
            pass.dispatch_workgroups_indirect(&pixel_state.args, 0);
        } else {
            for (group, binding) in camera.groups[camera.current][index].iter().enumerate() {
                pass.set_bind_group(group as u32, &**binding, &[]);
            }
            let groups = if stage == 0 {
                dispatch_count
            } else {
                dispatch_count.div_ceil(64)
            };
            pass.dispatch_workgroups(groups, 1, 1);
        }
    }
    if let Some(ticket) = ticket {
        renderer
            .budget
            .finish(encoder, &camera.queue.buffer, sun_flip_offset(count), &current.pool.buffer,
                &pixel_state.stats.buffer, ticket, executed);
    }
    camera.previous_clip = Some(clip);
    camera.current = 1 - camera.current;
    camera.valid = true;
    camera.last_frame = frame.0;
    renderer.cameras.insert(key, camera);
}
