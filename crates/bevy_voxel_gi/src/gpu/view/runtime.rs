//! Camera-scheduled far transport. No lighting buffer is read back, and no
//! gameplay/render system waits for a GPU submission. Per-view ping-pong state
//! retains face identities, path records, unfinished tails and raw statistics.
use super::*;
use crate::{FarTierSettings, VoxelGiSettings, render::VoxelGiMaterialData};
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
    /// The far tier's cells, read by the material.
    pixel_output: Handle<ShaderBuffer>,
    pixel: FarTierSettings,
    /// Bytes of the far tier's output allocated for the current cameras.
    pixel_bytes_allocated: u64,
    /// The host's declared geometry interest in regions (`pages::Interest`).
    interest: Option<(IVec3, IVec3)>,
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

/// The view tier's latest asynchronous receipt: GPU milliseconds per stage
/// (timestamp queries) and its receiver and far-tier counters. Receipts lag
/// the frame they describe by a few frames; diagnostics, not a frame bill.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ViewReceipts {
    /// Timestamp queries are supported and at least one receipt retired.
    pub available: bool,
    /// Receipts retired so far.
    pub receipt: u32,
    /// Tracing paths for visible faces.
    pub trace_ms: f32,
    /// Discovery and presentation, excluding the far tier.
    pub presentation_ms: f32,
    /// Finding visible faces from the depth and normal prepass.
    pub discovery_ms: f32,
    /// The far tier's pass.
    pub pixel_ms: f32,
    /// Visible faces with a receiver (lit per face) this frame.
    pub active_receivers: u32,
    /// Active receivers with no sample yet.
    pub uncovered: u32,
    /// Far-tier cells showing a face small on screen.
    pub pixel_owned: u32,
    /// Far-tier cells that traced a path this frame.
    pub pixel_traced: u32,
    /// View receivers that found no index slot (unlit for that frame),
    /// summed since start; nonzero means the receiver index is too full.
    pub hash_overflows: u64,
}

#[derive(Component)]
struct AddedPrepasses {
    depth: bool,
    normal: bool,
}

pub(crate) fn install(
    app: &mut App,
    output: Handle<ShaderBuffer>,
    pixel_output: Handle<ShaderBuffer>,
) {
    app.init_resource::<VoxelGiSettings>();
    app.init_resource::<crate::VoxelGiStats>();
    app.insert_resource(ViewSettings {
        output,
        pixel_output,
        pixel: FarTierSettings::default(),
        pixel_bytes_allocated: 0,
        interest: None,
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
            &Camera,
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
        let interest = runtime.interest.regions(&runtime.loaded_pages);
        if view.interest != interest {
            view.interest = interest;
        }
        runtime.sun_changes.add(flips);
        if let Some(change) = runtime
            .sun_changes
            .take(&runtime.scene, std::time::Instant::now())
        {
            // Events carry no page allocations: retired world pages must not
            // suppress the change for view-owned receivers.
            runtime.spatial_history.record(change, &[]);
        }
    }
    let enabled = true;
    if enabled != view.enabled {
        let mut asset = ShaderBuffer::default();
        asset.buffer_description.size = if enabled {
            DISPLAY_BYTES
        } else {
            EMPTY_OUTPUT_BYTES
        };
        asset.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        view.output = buffers.add(asset);
        material_data.far = view.output.clone();
    }
    let pixel = settings.far_tier.normalized();
    // Sized for the largest current camera at its cell size; it grows with
    // the viewport (one shared scratch output, written per camera).
    let needed = cameras
        .iter()
        .filter_map(|(.., camera)| camera.physical_viewport_size())
        .map(|size| pixel_output_bytes(size, pixel.cell_for(size.y)))
        .max()
        .unwrap_or(0);
    if enabled && needed > view.pixel_bytes_allocated {
        let mut asset = ShaderBuffer::default();
        asset.buffer_description.size = needed;
        asset.buffer_description.usage =
            BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        view.pixel_output = buffers.add(asset);
        material_data.far_pixels = view.pixel_output.clone();
        view.pixel_bytes_allocated = needed;
    }
    if view.pixel != pixel {
        view.pixel = pixel;
    }
    for (entity, depth, normal, added, _) in &cameras {
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
        } else if !enabled && let Some(added) = added {
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
    /// Far tier (voxel_view_pixels.wgsl): view forward (xyz) and pixels per
    /// world unit at unit view depth (w).
    pixel_axis: Vec4,
    /// Cells x, y, pixels per cell side; enabled (bit 0) and measure
    /// (bits 1-2).
    pixel_grid: UVec4,
    /// Threshold, maximum history, trace rate, previous frame (bits; all
    /// ones when none).
    pixel_tune: Vec4,
    previous_clip_from_world: Mat4,
    /// The host's geometry interest in regions, inclusive (w: 1 when
    /// bounded); discovery and the far tier skip pixels outside it.
    interest_min: IVec4,
    interest_max: IVec4,
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

/// A view-tier storage buffer, or `None` past the device's limits (the
/// caller then runs this view without the view tier).
fn storage(device: &RenderDevice, label: &'static str, bytes: usize) -> Option<UploadBuffer> {
    let mut buffer = UploadBuffer::new(
        device,
        label,
        BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    );
    match buffer.ensure(device, bytes) {
        Ok(_) => Some(buffer),
        Err(error) => {
            warn_once!(
                "Voxel GI: {label} ({bytes} bytes) exceeds device limits ({error:?}); this view runs without view GI"
            );
            None
        }
    }
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
    fn new(device: &RenderDevice, count: u32) -> Option<Self> {
        Some(Self {
            receivers: storage(device, "Far receiver identities", count as usize * 32)?,
            paths: storage(device, "Far retained path records", count as usize * 128)?,
            tails: storage(device, "Far unfinished paths", count as usize * 128)?,
            estimates: storage(device, "Far raw estimates", count as usize * 64)?,
            hash: storage(
                device,
                "Far reprojection index",
                (count * 2).next_power_of_two() as usize * 4,
            )?,
            pool: storage(
                device,
                "Far shared receiver pool and spatial index",
                POOL_PREFIX_BYTES + count as usize * 16,
            )?,
        })
    }
}
/// Far tier state: ping-pong cell histories (mean RGB and count, surface
/// point, face and frame stamp, sample sequence; 48 bytes) and its counters.
struct PixelState {
    cells: UVec2,
    history: [UploadBuffer; 2],
    stats: UploadBuffer,
    /// Cells to trace (count, indices) and the trace's indirect dispatch.
    queue: UploadBuffer,
    args: Buffer,
}
impl PixelState {
    fn new(device: &RenderDevice, cells: UVec2) -> Option<Self> {
        let bytes = cells.element_product().max(1) as usize * 48;
        Some(Self {
            cells,
            history: [
                storage(device, "Far per-pixel cell history", bytes)?,
                storage(device, "Far per-pixel cell history", bytes)?,
            ],
            stats: storage(
                device,
                "Far per-pixel counters",
                pixel_words::COUNT as usize * 4,
            )?,
            queue: storage(
                device,
                "Far per-pixel trace queue",
                4 + cells.element_product().max(1) as usize * 4,
            )?,
            args: device.create_buffer(&BufferDescriptor {
                label: Some("Far per-pixel trace dispatch"),
                size: 16,
                usage: BufferUsages::STORAGE | BufferUsages::INDIRECT,
                mapped_at_creation: false,
            }),
        })
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
    fn new(device: &RenderDevice, grid: UVec2) -> Option<Self> {
        let count = grid.element_product();
        let uniform =
            |name| UploadBuffer::new(device, name, BufferUsages::UNIFORM | BufferUsages::COPY_DST);
        Some(Self {
            history: [History::new(device, count)?, History::new(device, count)?],
            pixel: None,
            previous_clip: None,
            current: 0,
            grid,
            discovery_view: None,
            valid: false,
            control: uniform("Far view control"),
            params: uniform("Far transport parameters"),
            wave: uniform("Far wave allowance"),
            queue: storage(
                device,
                "Far visible owner queue and sequence reservations",
                (count as usize * 2 + queue_words::PREFIX as usize) * 4 + SUN_FLIP_BYTES as usize,
            )?,
            cursor: 0,
            last_frame: 0,
            binding_keys: [None, None],
            groups: [Vec::new(), Vec::new()],
        })
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
    let source = source.replace(
        "// #include voxel_view_cache.wgsl\n",
        include_str!("../../../shaders/voxel_view_cache.wgsl"),
    );
    let source = source.replace(
        "// #include voxel_view_discovery.wgsl\n",
        include_str!("../../../shaders/voxel_view_discovery.wgsl"),
    );
    let source = source.replace(
        "// #include voxel_view_pixels.wgsl\n",
        include_str!("../../../shaders/voxel_view_pixels.wgsl"),
    );
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
    /// Far tier: classify (by MSAA), size the queue, trace it.
    pixels: [Kernel; 2],
    pixel_args: Kernel,
    pixel_trace: Kernel,
    /// The step budget and reach policy `pixels` and `pixel_trace` were
    /// built with (both draw custom-light visibility, so they share it).
    pixel_steps: (u32, bool),
    modules: [bevy::render::render_resource::ShaderModule; 2],
    cameras: HashMap<RetainedViewEntity, CameraState>,
    frame: u32,
    at: Instant,
    frame_seconds: f32,
    budget: super::timing::Budget,
    changes: Vec<ViewChange>,
    /// A view-tier pipeline failed to build (e.g. FXC on DX12 rejects the
    /// kernels). Every view then takes the allocation-failure path: cleared
    /// outputs, the material's fallback, no view GI.
    unavailable: bool,
}

/// Builds view-tier pipelines inside error scopes (`gpu::scoped`), so a
/// pipeline the backend's compiler rejects disables the view tier with one
/// warning instead of reaching the app's uncaptured-error handler.
fn build_view_pipelines<T>(device: &RenderDevice, build: impl FnOnce() -> T) -> (T, bool) {
    let mut built = None;
    let result = crate::gpu::scoped(device, || built = Some(build()));
    if let Err(error) = &result {
        warn!(
            "Voxel GI: view-tier pipelines failed to build; visible faces fall back to the material without view GI. {error}"
        );
    }
    (built.unwrap(), result.is_err())
}

impl FromWorld for ViewRenderer {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>();
        let queue = world.resource::<RenderQueue>();
        let (renderer, unavailable) = build_view_pipelines(device, || Self::build(device, queue));
        Self {
            unavailable,
            ..renderer
        }
    }
}

impl ViewRenderer {
    fn build(device: &RenderDevice, queue: &RenderQueue) -> Self {
        let modules = [false, true].map(|msaa| {
            device.create_and_validate_shader_module(ShaderModuleDescriptor {
                label: Some("View-owned voxel GI"),
                source: ShaderSource::Wgsl(shader(msaa).into()),
            })
        });
        // Every kernel binds exactly what its entry point uses (automatic
        // layouts); the MSAA variant differs only in texture types.
        let reflection = crate::gpu::reflect::Reflection::new(&shader(false));
        let make = |name, msaa: usize| Kernel {
            pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &modules[msaa],
                entry_point: Some(name),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[("LIMITED_REACH", 0.0)],
                    ..default()
                },
                cache: None,
            }),
            bindings: reflection.bindings(name),
        };
        let steps = FarTierSettings::default().steps;
        let far = |name, msaa: usize| {
            pixel_kernel(
                device,
                &modules[msaa],
                name,
                reflection.bindings(name),
                steps,
                false,
            )
        };
        Self {
            prepare_pool: [0, 1].map(|m| make("prepare_receiver_pool", m)),
            discover: [0, 1].map(|m| make("discover_receivers", m)),
            index: make("index_receivers", 0),
            reproject: make("reproject_paths", 0),
            trace: [false, true].map(|limited| Kernel {
                pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("trace_visible_paths"),
                    layout: None,
                    module: &modules[0],
                    entry_point: Some("trace_visible_paths"),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &[("LIMITED_REACH", f64::from(limited))],
                        ..default()
                    },
                    cache: None,
                }),
                bindings: reflection.bindings("trace_visible_paths"),
            }),
            retain: make("retain_view_history", 0),
            load_cache: make("load_shared_faces", 0),
            store_cache: make("store_shared_faces", 0),
            reconstruct: make("reconstruct_view", 0),
            publish: make("publish_view", 0),
            pixels: [0, 1].map(|m| far("classify_pixels", m)),
            pixel_args: make("size_pixel_queue", 0),
            pixel_trace: far("trace_pixel_queue", 0),
            pixel_steps: (steps, false),
            modules,
            cameras: default(),
            frame: 0,
            at: Instant::now(),
            frame_seconds: 1.0 / 60.0,
            budget: super::timing::Budget::new(device, queue),
            changes: Vec::new(),
            unavailable: false,
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

/// The far tier's pipeline constants: its ray budget (`steps` traversal
/// steps per ray, exhaustion a miss: NAADF's rule) and the same transport
/// reach policy the receiver trace specializes on. Classify and trace both
/// take them: each writes cells with custom-light visibility
/// (`custom_direct`), and a stationary cell must not flicker between two
/// traversal rules as the trace schedule alternates.
pub(super) fn pixel_trace_constants(steps: u32, limited: bool) -> [(&'static str, f64); 3] {
    [
        ("PATH_STEPS", f64::from(steps)),
        ("BUDGETED_RAYS", 1.0),
        ("LIMITED_REACH", f64::from(limited)),
    ]
}

fn pixel_kernel(
    device: &RenderDevice,
    module: &bevy::render::render_resource::ShaderModule,
    name: &'static str,
    bindings: [Vec<u32>; 3],
    steps: u32,
    limited: bool,
) -> Kernel {
    Kernel {
        pipeline: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: None,
            module,
            entry_point: Some(name),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &pixel_trace_constants(steps, limited),
                ..default()
            },
            cache: None,
        }),
        bindings,
    }
}

/// The view tier's passes in dispatch order. Each is one compute pass with
/// one timestamp pair; `slot` is that pair and the stage's index in the
/// receipt's costs (`timing::Budget`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    PreparePool,
    Discover,
    Index,
    Reproject,
    Retain,
    LoadCache,
    Trace,
    Reconstruct,
    StoreCache,
    Publish,
    Far,
}

/// How a stage's kernel is dispatched: the stage's workgroup count, one
/// workgroup, or the far queue's indirect arguments.
#[derive(Clone, Copy)]
enum Dispatch {
    Stage,
    One,
    Indirect,
}

impl Stage {
    pub const ORDER: [Stage; 11] = [
        Stage::PreparePool,
        Stage::Discover,
        Stage::Index,
        Stage::Reproject,
        Stage::Retain,
        Stage::LoadCache,
        Stage::Trace,
        Stage::Reconstruct,
        Stage::StoreCache,
        Stage::Publish,
        Stage::Far,
    ];
    pub const COUNT: usize = 11;

    pub const fn slot(self) -> usize {
        match self {
            Stage::Discover => 0,
            Stage::Index => 1,
            Stage::Reproject => 2,
            Stage::Retain => 3,
            Stage::LoadCache => 4,
            Stage::Trace => 5,
            Stage::Reconstruct => 6,
            Stage::StoreCache => 7,
            Stage::Publish => 8,
            Stage::PreparePool => 9,
            Stage::Far => 10,
        }
    }

    /// The stage's kernels (indices into `render_view`'s kernel list) and
    /// their dispatch; `Retain` dispatches per retention event instead.
    fn kernels(self) -> &'static [(usize, Dispatch)] {
        match self {
            Stage::Discover => &[(0, Dispatch::Stage)],
            Stage::Index => &[(1, Dispatch::Stage)],
            Stage::Reproject => &[(2, Dispatch::Stage)],
            Stage::LoadCache => &[(3, Dispatch::Stage)],
            Stage::Trace => &[(4, Dispatch::Stage)],
            Stage::Reconstruct => &[(5, Dispatch::Stage)],
            Stage::StoreCache => &[(6, Dispatch::Stage)],
            Stage::Publish => &[(7, Dispatch::Stage)],
            Stage::PreparePool => &[(8, Dispatch::Stage)],
            // Classify every cell, size the queue, trace it densely.
            Stage::Far => &[
                (9, Dispatch::Stage),
                (10, Dispatch::One),
                (11, Dispatch::Indirect),
            ],
            Stage::Retain => &[],
        }
    }
}

fn sun_flip_offset(count: u32) -> u64 {
    (queue_words::PREFIX + 2 * u64::from(count)) * 4
}

/// Words of the per-camera `queue` buffer (voxel_view_runtime.wgsl's
/// `QUEUE_*`): receipt counters, then from `PREFIX` the visible-owner queue,
/// each owner's rank, and the sun flips.
pub(super) mod queue_words {
    pub const OWNERS: u64 = 0;
    pub const OVERFLOW: u64 = 1;
    pub const COMPLETED: u64 = 2;
    pub const KNOWN: u64 = 3;
    pub const HISTORY: u64 = 4;
    pub const MAX_HISTORY: u64 = 5;
    pub const UNCOVERED: u64 = 6;
    pub const DISPLAYED: u64 = 7;
    pub const PAUSED: u64 = 8;
    pub const ATTEMPTED: u64 = 9;
    pub const CACHE_LOADS: u64 = 10;
    pub const CACHE_PROMOTIONS: u64 = 11;
    pub const PREFIX: u64 = 12;
}

/// Words of the far tier's counter buffer (voxel_view_pixels.wgsl's
/// `PIXEL_*`).
pub(super) mod pixel_words {
    pub const OWNED: u64 = 0;
    pub const TRACED: u64 = 1;
    pub const HISTORY: u64 = 2;
    pub const REPROJECTED: u64 = 3;
    pub const SEEDED: u64 = 4;
    pub const RECEIVERS_SEEDED: u64 = 5;
    pub const COUNT: u64 = 8;
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
    if !settings.enabled || renderer.unavailable {
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
        settings
            .sun_flips
            .lock()
            .unwrap()
            .append(&mut renderer.budget.sun_flips);
        *settings.receipts.lock().unwrap() = ViewReceipts {
            available: renderer.budget.queries.is_some() && renderer.budget.receipts > 0,
            receipt: renderer.budget.receipts,
            trace_ms: renderer.budget.trace_ms,
            presentation_ms: renderer.budget.presentation_ms,
            discovery_ms: renderer.budget.discovery_ms,
            pixel_ms: renderer.budget.pixel_ms,
            active_receivers: renderer.budget.active,
            uncovered: renderer.budget.uncovered,
            pixel_owned: renderer.budget.pixel_owned,
            pixel_traced: renderer.budget.pixel_traced,
            hash_overflows: renderer.budget.hash_overflows,
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
                far_pixel_seeded = renderer.budget.pixel_seeded,
                far_receivers_seeded = renderer.budget.receivers_seeded,
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
    let limited = scene.params.transport.x < crate::tracer::MAX_PATH_DISTANCE;
    let pixel_policy = (settings.pixel.steps, limited);
    if renderer.pixel_steps != pixel_policy {
        let renderer = &mut *renderer;
        let rebuild = |kernel: &Kernel, name, msaa: usize| {
            let (module, bindings) = (&renderer.modules[msaa], kernel.bindings.clone());
            pixel_kernel(&device, module, name, bindings, pixel_policy.0, limited)
        };
        let ((pixels, trace), failed) = build_view_pipelines(&device, || {
            (
                [0, 1].map(|m| rebuild(&renderer.pixels[m], "classify_pixels", m)),
                rebuild(&renderer.pixel_trace, "trace_pixel_queue", 0),
            )
        });
        if failed {
            renderer.unavailable = true;
            renderer.cameras.clear();
            return;
        }
        (renderer.pixels, renderer.pixel_trace) = (pixels, trace);
        renderer.pixel_steps = pixel_policy;
        for camera in renderer.cameras.values_mut() {
            camera.binding_keys = [None, None];
        }
    }
    let grid = grid_size(view.viewport.zw(), settings.receivers);
    let count = grid.x * grid.y;
    let key = view.retained_view_entity;
    // Allocation failures (past device limits) leave this view without
    // view GI: the cleared output headers select the material's fallback.
    let Some(mut camera) = renderer
        .cameras
        .remove(&key)
        .filter(|camera| camera.grid == grid)
        .or_else(|| CameraState::new(&device, grid))
    else {
        return;
    };
    let pixel_size = settings.pixel.cell_for(view.viewport.w);
    let pixel_cells = view.viewport.zw().map(|v| v.div_ceil(pixel_size));
    let pixel_enabled = settings.pixel.voxel_pixels > 0.0
        && u64::from(pixel_cells.element_product()) * 16 + PIXEL_HEADER_BYTES
            <= pixel_output.buffer.size();
    if camera
        .pixel
        .as_ref()
        .is_none_or(|state| state.cells != pixel_cells)
    {
        let Some(state) = PixelState::new(&device, pixel_cells) else {
            return;
        };
        camera.pixel = Some(state);
        camera.previous_clip = None;
    }
    let (work, ticket) = renderer.budget.reserve();
    let pixel_state = camera.pixel.as_ref().unwrap();
    let current = &camera.history[camera.current];
    let previous = &camera.history[1 - camera.current];
    let hash_mask = current.hash.capacity as u32 / 4 - 1;
    let clip = view
        .clip_from_world
        .unwrap_or_else(|| view.clip_from_view * view.world_from_view.affine().inverse());
    let discovery_view = (clip, view.viewport, settings.inner_radius, msaa.samples());
    let discovery_stable = camera.valid && camera.discovery_view == Some(discovery_view);
    let discovery_compatible = camera.valid
        && camera
            .discovery_view
            .is_some_and(|(_, viewport, radius, samples)| {
                viewport == view.viewport
                    && radius == settings.inner_radius
                    && samples == msaa.samples()
            });
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
        history: UVec4::new(
            if camera.valid { count } else { 0 },
            hash_mask,
            frame.0,
            u32::from(settings.view_first),
        ),
        display: Vec4::new(
            settings.noise_threshold,
            settings.spatial,
            1.0 - (-renderer.frame_seconds.max(1.0 / 240.0) / settings.response).exp(),
            settings.measured_retention,
        ),
        cache: scene.cache.as_ref().map_or(UVec4::ZERO, |cache| {
            UVec4::new(
                cache.epoch,
                cache.origin.elapsed().as_millis() as u32,
                u32::from(cache.enabled && settings.view_first),
                0,
            )
        }),
        discovery: UVec4::new(
            u32::from(discovery_stable),
            u32::from(discovery_compatible),
            0,
            0,
        ),
        pixel_axis: {
            let forward = view.world_from_view.forward();
            // clip_from_view[1][1]: NDC y per unit of y / view depth.
            let scale = view.clip_from_view.y_axis.y * view.viewport.w as f32 * 0.5;
            forward.extend(scale)
        },
        pixel_grid: UVec4::new(
            pixel_cells.x,
            pixel_cells.y,
            pixel_size,
            u32::from(pixel_enabled) | (settings.pixel.measure as u32) << 1,
        ),
        pixel_tune: Vec4::new(
            settings.pixel.voxel_pixels,
            settings.pixel.history,
            settings.pixel.rate as f32,
            // The frame that wrote the previous cells (their stamp); all
            // ones when there are none.
            f32::from_bits(if camera.previous_clip.is_some() {
                camera.last_frame
            } else {
                u32::MAX
            }),
        ),
        previous_clip_from_world: camera.previous_clip.unwrap_or(clip),
        interest_min: settings
            .interest
            .map_or(IVec4::ZERO, |(min, _)| min.extend(1)),
        interest_max: settings
            .interest
            .map_or(IVec4::ZERO, |(_, max)| max.extend(1)),
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
    encoder.clear_buffer(&camera.queue.buffer, 0, Some(queue_words::PREFIX * 4));
    encoder.clear_buffer(&pixel_state.stats.buffer, 0, None);
    encoder.clear_buffer(&pixel_state.queue.buffer, 0, Some(4));
    let multisampled = msaa.samples() > 1;
    // Indexed by `Stage::kernels`; bind groups are cached in this order.
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
                                15 => scene
                                    .cache
                                    .as_ref()
                                    .map_or(&scene.buffers[6], |cache| &cache.meta)
                                    .as_entire_binding(),
                                16 => scene
                                    .cache
                                    .as_ref()
                                    .map_or(&scene.buffers[6], |cache| &cache.guard)
                                    .as_entire_binding(),
                                17 => current.pool.buffer.as_entire_binding(),
                                18 => previous.pool.buffer.as_entire_binding(),
                                19 => pixel_state.history[1 - camera.current]
                                    .buffer
                                    .as_entire_binding(),
                                20 => pixel_state.history[camera.current]
                                    .buffer
                                    .as_entire_binding(),
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
        cache: scene
            .cache
            .as_ref()
            .map_or([scene.buffers[6].id(); 2], |cache| {
                [cache.meta.id(), cache.guard.id()]
            }),
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
    for stage in Stage::ORDER {
        let workgroups = match stage {
            // One cooperative workgroup per 2x2-cell discovery tile.
            Stage::Discover => grid.x.div_ceil(2) * grid.y.div_ceil(2),
            Stage::Trace => work.div_ceil(64),
            Stage::Retain if changes.is_empty() => 0,
            Stage::StoreCache if control.cache.z == 0 => 0,
            Stage::Far if !pixel_enabled => 0,
            Stage::Far => pixel_cells.element_product().div_ceil(64),
            _ => count.div_ceil(64),
        };
        if workgroups == 0 {
            continue;
        }
        executed |= 1 << stage.slot();
        let timestamps = ticket
            .as_ref()
            .and(renderer.budget.queries.as_ref())
            .map(|queries| wgpu::ComputePassTimestampWrites {
                query_set: queries,
                beginning_of_pass_write_index: Some(stage.slot() as u32 * 2),
                end_of_pass_write_index: Some(stage.slot() as u32 * 2 + 1),
            });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Far view GI"),
            timestamp_writes: timestamps,
        });
        if stage == Stage::Retain {
            // One dispatch per retention event batch, each with its own
            // event bindings.
            pass.set_pipeline(&renderer.retain.pipeline);
            for groups in &changes {
                for (group, binding) in groups.iter().enumerate() {
                    pass.set_bind_group(group as u32, &**binding, &[]);
                }
                pass.dispatch_workgroups(workgroups, 1, 1);
            }
            continue;
        }
        for &(kernel, dispatch) in stage.kernels() {
            pass.set_pipeline(&kernels[kernel].pipeline);
            for (group, binding) in camera.groups[camera.current][kernel].iter().enumerate() {
                pass.set_bind_group(group as u32, &**binding, &[]);
            }
            match dispatch {
                Dispatch::Stage => pass.dispatch_workgroups(workgroups, 1, 1),
                Dispatch::One => pass.dispatch_workgroups(1, 1, 1),
                Dispatch::Indirect => pass.dispatch_workgroups_indirect(&pixel_state.args, 0),
            }
        }
    }
    if let Some(ticket) = ticket {
        renderer.budget.finish(
            encoder,
            &camera.queue.buffer,
            sun_flip_offset(count),
            &current.pool.buffer,
            &pixel_state.stats.buffer,
            ticket,
            executed,
        );
    }
    camera.previous_clip = Some(clip);
    camera.current = 1 - camera.current;
    camera.valid = true;
    camera.last_frame = frame.0;
    renderer.cameras.insert(key, camera);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_view_kernel_binds_only_resources_render_view_provides() {
        // render_view resolves group 0 from `group0` (0-10), group 1 from
        // tails and wave (0-1), group 2 from `group2` (0-7) and its explicit
        // arms (8-24). A kernel using anything else would panic at runtime.
        let started = std::time::Instant::now();
        let reflection = crate::gpu::reflect::Reflection::new(&shader(false));
        eprintln!("view shader reflection: {:?}", started.elapsed());
        for name in [
            "prepare_receiver_pool",
            "discover_receivers",
            "index_receivers",
            "reproject_paths",
            "trace_visible_paths",
            "retain_view_history",
            "load_shared_faces",
            "store_shared_faces",
            "reconstruct_view",
            "publish_view",
            "classify_pixels",
            "size_pixel_queue",
            "trace_pixel_queue",
        ] {
            let [group0, group1, group2] = reflection.bindings(name);
            assert!(group0.iter().all(|&b| b <= 10), "{name} group 0 {group0:?}");
            assert!(group1.iter().all(|&b| b <= 1), "{name} group 1 {group1:?}");
            assert!(group2.iter().all(|&b| b <= 24), "{name} group 2 {group2:?}");
        }
    }

    #[test]
    #[ignore = "native device limits"]
    fn view_storage_past_device_limits_is_declined_not_fatal() {
        let context = crate::gpu::tests::native_tracer().context;
        let too_large = context.device.limits().max_storage_buffer_binding_size as usize + 4;
        assert!(storage(&context.device, "test", too_large).is_none());
        assert!(storage(&context.device, "test", 1024).is_some());
    }

    #[test]
    fn stage_table_is_a_permutation_of_slots_and_kernels() {
        let mut slots: Vec<_> = Stage::ORDER.iter().map(|stage| stage.slot()).collect();
        slots.sort_unstable();
        assert_eq!(
            slots,
            (0..Stage::COUNT).collect::<Vec<_>>(),
            "one timestamp pair per stage"
        );
        let mut kernels: Vec<_> = Stage::ORDER
            .iter()
            .flat_map(|stage| stage.kernels().iter().map(|k| k.0))
            .collect();
        kernels.sort_unstable();
        assert_eq!(
            kernels,
            (0..12).collect::<Vec<_>>(),
            "every listed kernel dispatched once"
        );
    }

    #[test]
    fn presentation_module_mirrors_the_view_tier() {
        // Host materials read buffers the view tier writes through
        // bevy_voxel_gi::presentation; its copies of their structs must keep the
        // same members at the same offsets.
        let presentation = crate::gpu::reflect::Reflection::new(
            &crate::render::presentation_shader_source()
                .replace("#define_import_path bevy_voxel_gi::presentation", "")
                .replace("#{MATERIAL_BIND_GROUP}", "2"),
        );
        let runtime = crate::gpu::reflect::Reflection::new(&shader(false));
        assert_eq!(
            presentation.layout("FarOutput"),
            runtime.layout("FarOutput")
        );
        assert_eq!(presentation.layout("FarPixel"), runtime.layout("FarPixel"));
        assert_eq!(
            presentation.layout("FarCells"),
            runtime.layout("PixelOutput"),
            "the far tier's cells"
        );
    }

    #[test]
    fn rust_and_wgsl_view_layouts_agree() {
        let reflection = crate::gpu::reflect::Reflection::new(&shader(false));
        let f = |i: u32| Vec4::splat(i as f32);
        let u = |i: u32| UVec4::splat(i);
        let m = |i: u32| Mat4::from_cols(f(i), f(i), f(i), f(i));
        let control = Control {
            world_from_clip: m(1),
            clip_from_world: m(2),
            camera: f(3),
            viewport: u(4),
            grid: u(5),
            history: u(6),
            display: f(7),
            cache: u(8),
            discovery: u(9),
            pixel_axis: f(10),
            pixel_grid: u(11),
            pixel_tune: f(12),
            previous_clip_from_world: m(13),
            interest_min: IVec4::splat(14),
            interest_max: IVec4::splat(15),
        };
        crate::gpu::reflect::assert_mirrors(
            &ShaderBuffer::from(control).data.unwrap(),
            reflection.layout("ViewControl"),
            &[
                "world_from_clip",
                "clip_from_world",
                "camera",
                "viewport",
                "grid",
                "history",
                "display",
                "cache",
                "discovery",
                "pixel_axis",
                "pixel_grid",
                "pixel_tune",
                "previous_clip_from_world",
                "interest_min",
                "interest_max",
            ],
        );
        let wave = Wave {
            range: u(1),
            ticket: u(2),
        };
        crate::gpu::reflect::assert_mirrors(
            &ShaderBuffer::from(wave).data.unwrap(),
            reflection.layout("ViewWave"),
            &["range", "ticket"],
        );
        // Far tier: cell histories are allocated at 48 bytes, the output
        // header is PIXEL_HEADER_BYTES.
        assert_eq!(reflection.layout("PixelHistory").1, 48);
        let (output, _) = reflection.layout("PixelOutput");
        assert_eq!(
            output.last().unwrap(),
            &("cells".to_string(), PIXEL_HEADER_BYTES as u32)
        );
        // Counter words read back by the timing receipt.
        for (name, word) in [
            ("QUEUE_OWNERS", queue_words::OWNERS),
            ("QUEUE_OVERFLOW", queue_words::OVERFLOW),
            ("QUEUE_COMPLETED", queue_words::COMPLETED),
            ("QUEUE_KNOWN", queue_words::KNOWN),
            ("QUEUE_HISTORY", queue_words::HISTORY),
            ("QUEUE_MAX_HISTORY", queue_words::MAX_HISTORY),
            ("QUEUE_UNCOVERED", queue_words::UNCOVERED),
            ("QUEUE_DISPLAYED", queue_words::DISPLAYED),
            ("QUEUE_PAUSED", queue_words::PAUSED),
            ("QUEUE_ATTEMPTED", queue_words::ATTEMPTED),
            ("QUEUE_CACHE_LOADS", queue_words::CACHE_LOADS),
            ("QUEUE_CACHE_PROMOTIONS", queue_words::CACHE_PROMOTIONS),
            ("QUEUE_PREFIX", queue_words::PREFIX),
            ("PIXEL_OWNED", pixel_words::OWNED),
            ("PIXEL_TRACED", pixel_words::TRACED),
            ("PIXEL_HISTORY", pixel_words::HISTORY),
            ("PIXEL_REPROJECTED", pixel_words::REPROJECTED),
            ("PIXEL_SEEDED", pixel_words::SEEDED),
            ("PIXEL_RECEIVERS_SEEDED", pixel_words::RECEIVERS_SEEDED),
        ] {
            assert_eq!(u64::from(reflection.constant(name)), word, "{name}");
        }
        const { assert!(pixel_words::RECEIVERS_SEEDED < pixel_words::COUNT) };
    }

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
            let layout = &module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some("FarOutput"))
                .unwrap()
                .1
                .inner;
            let naga::TypeInner::Struct { members, span } = layout else {
                panic!("FarOutput must be a struct");
            };
            assert_eq!(members.last().unwrap().offset as u64, OUTPUT_PREFIX_BYTES);
            assert_eq!(*span as u64, EMPTY_OUTPUT_BYTES);
            let pool = &module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some("ReceiverPool"))
                .unwrap()
                .1
                .inner;
            let naga::TypeInner::Struct { members, .. } = pool else {
                panic!("ReceiverPool must be a struct");
            };
            assert_eq!(members.last().unwrap().offset as usize, POOL_PREFIX_BYTES);
        }
    }

    #[test]
    #[ignore = "native far-view pipeline qualification"]
    fn native_view_runtime_pipelines() {
        let context = crate::gpu::tests::native_tracer().context;
        let mut world = World::new();
        world.insert_resource(context.device.clone());
        world.insert_resource(context.queue.clone());
        let renderer = ViewRenderer::from_world(&mut world);
        assert!(
            !renderer.unavailable,
            "view-tier pipelines failed (warning logged above)"
        );
    }
}
