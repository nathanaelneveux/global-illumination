//! [Quick start](#quick-start) · [API map](#api-map) · [Troubleshooting](#troubleshooting)
#![doc = include_str!("../README.md")]
//!
//! ## API map
//!
//! ### Geometry and edits
//!
//! | Task | API |
//! | --- | --- |
//! | Install GI and provide a snapshot | [`VoxelGiPlugin`], [`VoxelGiSource`], [`VoxelSource::sample_page`] |
//! | Build a page at the mesh's resolution | [`PageSample`], [`dense_page`], [`coarse_page`], [page edge](VOXEL_GI_PAGE_EDGE) |
//! | Declare page residency | [`page_loaded`](VoxelGiInput::page_loaded), [`page_unloaded`](VoxelGiInput::page_unloaded) |
//! | Send a known edit | [`edit_voxel`](VoxelGiInput::edit_voxel), [tracked input](VoxelGiInput::tracked) |
//! | Refresh from the source | [`page_changed`](VoxelGiInput::page_changed), [`mark_voxel`](VoxelGiInput::mark_voxel), [`mark_all_loaded`](VoxelGiInput::mark_all_loaded) |
//!
//! ### Materials and lights
//!
//! | Task | API |
//! | --- | --- |
//! | Assign material IDs and optical properties | [`MaterialPalette`], [`VoxelMaterialTransportTable::set`], [`MaterialTransport`], [`AIR`] |
//! | Make a glowing block | [`MaterialTransport::emission`] |
//! | Light a scene with the sun, a point light or a sky | [Bevy directional light](DirectionalLight), [Bevy point light](PointLight), [`VoxelGiEnvironment`] |
//! | Add a light for GI | [`VoxelGiEmitter`] |
//! | Light a Bevy mesh | [`VoxelGiStandardMaterialPlugin`], [`standard_material`] |
//! | Light a custom material | [`VoxelGiBindings`], [`VoxelGiMaterialData`], [`VoxelGiMaterialPlugin`], [`PRESENTATION_SHADER`], [shader reference](#shader-reference) |
//! | Compose the shader outside Bevy's asset pipeline | [`presentation_shader_source`] |
//!
//! ### Quality and diagnostics
//!
//! | Task | API |
//! | --- | --- |
//! | Start from a preset or saved settings | [`VoxelGiSettings::quality`], [`VoxelGiQuality`], [`VoxelGiSettings::from_ron`] |
//! | Set ray reach or bounce depth | [`TransportSettings`] |
//! | Set the distance tier | [`FarTierSettings`], [`FarTierMeasure`] |
//! | Compare the look or inspect coverage | [`VoxelGiDisplay`], [`GiDebugView`] |
//! | Wait for geometry or read GPU cost | [`VoxelGiStats::geometry_pending`], [`ViewReceipts`], [system ordering](VoxelGiSystems) |
//!
#![warn(missing_docs)]
mod color;
mod dispatch;
mod emission;
mod field;
mod geometry;
mod gpu;
mod gpu_scene;
mod history;
mod input;
mod naadf;
mod pages;
mod palette;
mod publication;
mod reconstruction;
pub(crate) mod render;
#[cfg(test)]
mod research;
mod retention;
mod sampling;
mod scheduling;
mod settings;
mod sky;
mod sources;
#[cfg(test)]
mod statistics;
mod sun_changes;
mod telemetry;
mod tracer;
mod work;
mod world;

use std::collections::VecDeque;
// Bounded, engine-owned integer coordinates; no untrusted string keys.
use bevy::platform::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::tasks::Task;

use field::ReceiverField;
pub use palette::{MaterialPalette, PaletteFull};
pub use settings::{
    FarTierMeasure, FarTierSettings, TransportSettings, VoxelGiQuality, VoxelGiSettings,
};
pub use sources::{VoxelGiEmitter, VoxelGiEnvironment};
pub use world::{AIR, MaterialTransport, VoxelMaterialTransportTable};

use dispatch::{apply_trace_task, start_trace_task};
pub use gpu::view::ViewReceipts;
pub use input::{
    PageSample, VOXEL_GI_PAGE_EDGE, VoxelGiInput, VoxelGiSource, VoxelSource, coarse_page,
    dense_page,
};
use pages::{ingest_input, start_scene_task};
use publication::apply_scene_task;
use render::install_render_assets;
pub use render::{
    GiDebugView, PRESENTATION_SHADER, STANDARD_SHADER, VoxelGiBindings, VoxelGiDisplay,
    VoxelGiMaterialData, VoxelGiMaterialPlugin, VoxelGiStandardMaterial,
    VoxelGiStandardMaterialPlugin, VoxelGiStandardShader, presentation_shader_source,
    standard_material,
};
use sources::{TransportSources, synchronize_material_table, synchronize_transport_sources};
use telemetry::GiWorkTimings;
pub use telemetry::MAIN_STAGE_NAMES as GI_MAIN_STAGE_NAMES;
use tracer::BOOTSTRAP_SAMPLES;
use work::TraceWork;
use world::{PAGE_EDGE, SurfaceProbe, SurfaceProbeDescriptor, TransportScene};

/// Voxel GI: page ingestion, transport and the view tier. Materials bind
/// `VoxelGiMaterialData` (the BVW connector's material does this); the debug
/// HUD and panel are the opt-in `VoxelGiDebugPlugin`.
pub struct VoxelGiPlugin;

/// Schedule host readers after this set to observe this frame's published
/// diagnostics. Worker results remain asynchronous.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelGiSystems;

impl Plugin for VoxelGiPlugin {
    fn build(&self, app: &mut App) {
        install_render_assets(app);
        app.init_resource::<VoxelGiInput>()
            .init_resource::<VoxelGiSource>()
            .init_resource::<VoxelGiRuntime>()
            .init_resource::<ReceiverField>()
            .init_resource::<VoxelGiStats>()
            .init_resource::<GiWorkTimings>()
            .init_resource::<VoxelGiSettings>()
            .init_resource::<VoxelGiEnvironment>()
            .init_resource::<VoxelMaterialTransportTable>()
            .add_systems(
                Update,
                (
                    ingest_input,
                    pages::update_receiver_residency,
                    synchronize_transport_sources,
                    synchronize_material_table,
                    settings::synchronize_transport_settings,
                    apply_scene_task,
                    start_scene_task,
                    apply_trace_task,
                    start_trace_task,
                    telemetry::finish_frame,
                )
                    .chain()
                    .in_set(VoxelGiSystems),
            );
    }

    fn finish(&self, app: &mut App) {
        let context = app
            .get_sub_app(bevy::render::RenderApp)
            .and_then(|render_app| {
                let device = render_app
                    .world()
                    .get_resource::<bevy::render::renderer::RenderDevice>()?
                    .clone();
                let queue = render_app
                    .world()
                    .get_resource::<bevy::render::renderer::RenderQueue>()?
                    .clone();
                Some(gpu::GpuTransportDevice { device, queue })
            });
        if let Some(context) = context {
            app.insert_resource(context);
        }
    }
}

/// What GI is doing, for a host's HUD or adaptive quality: coverage of the
/// world cache, its cost, and the view tier's GPU time. `scheduler` holds
/// scheduler internals for diagnostics (the debug HUD, validation); they
/// change with the implementation.
#[derive(Resource, Clone, Debug, Default)]
pub struct VoxelGiStats {
    /// Geometry the host declared is not yet published to the GPU (a scene
    /// build or publication is in flight, or pages are waiting): a host
    /// readiness gate, e.g. before captures.
    pub geometry_pending: bool,
    /// Pages the host has declared loaded.
    pub loaded_pages: usize,
    /// Faces the world cache holds (within `cache_radius` of the cameras).
    pub surface_probes: usize,
    /// World-cache progress toward eight samples per face, 0 to 1. Not
    /// convergence.
    pub bootstrap_coverage: f32,
    /// Share of world-cache faces with at least one sample, 0 to 1. Not
    /// convergence.
    pub interactive_coverage: f32,
    /// Fraction passing the adaptive stability heuristic for the current
    /// transport revision. Periodic audits remain eligible; not convergence.
    pub settled_coverage: f32,
    /// Mean GI main-world system-body time per app frame over a 250 ms window.
    /// Excludes background workers and render-world upload/draw work.
    pub main_thread_ms_per_frame: Option<f32>,
    /// Completed transport compute-pass timestamps per app frame over the
    /// same window. None until available, or on CPU/unsupported GPU backends.
    /// Excludes queue waits, transfers and graphics; do not add to main time.
    pub gpu_ms_per_frame: Option<f32>,
    /// Visible-face (view tier) GPU time last frame. Unlike `gpu_ms_per_frame`
    /// (world-cache transport), presentation and discovery are not limited by
    /// `gpu_time_share`; `view_receivers` scales them.
    pub view_gpu: ViewReceipts,
    /// The transport backend in use (`"gpu-resident"`).
    pub backend: &'static str,
    /// Scheduler internals: diagnostics only, see [`SchedulerStats`].
    pub scheduler: SchedulerStats,
}

/// World-cache scheduler internals, for diagnostics. Fields may change in
/// any release; do not drive game logic from them.
#[derive(Clone, Debug, Default)]
pub struct SchedulerStats {
    /// Last app frame's GI system-body times, in telemetry stage order.
    /// Excludes workers/render work; unlike the overlay average, retains spikes.
    pub last_main_stage_ms: [f32; 6],
    /// World-cache faces queued for urgent relighting (near edits).
    pub pending_probes: usize,
    /// World-cache paths traced since start.
    pub traced_paths: u64,
    /// Duration of the last scene build (page sampling to a new scene), on
    /// a worker.
    pub last_scene_build_ms: f32,
    /// Duration of the last world-cache trace batch, on a worker.
    pub last_trace_batch_ms: f32,
    /// World-cache trace batches completed since start.
    pub transport_batches: u64,
    /// Frames the world-cache scheduler had nothing to dispatch.
    pub idle_scheduler_frames: u64,
    /// Wall time of the last trace batch's worker task.
    pub last_worker_ms: f32,
    /// Time spent choosing the last batch's faces.
    pub last_selection_ms: f32,
    /// Time spent merging the last batch's results into the cache.
    pub last_merge_ms: f32,
    /// Time spent preparing the last batch's GPU inputs.
    pub last_prepare_ms: f32,
    /// Submission-to-delivery latency; does not imply a blocking GPU wait.
    pub last_snapshot_latency_ms: f32,
    /// Time spent decoding the last batch's readback.
    pub last_decode_ms: f32,
    /// GPU bytes of GI's packed voxel geometry.
    pub packed_geometry_bytes: usize,
    /// Every world-cache face has its bootstrap samples and no page is
    /// still publishing.
    pub fully_bootstrapped: bool,
    /// World-cache faces the next batch may select.
    pub dispatch_budget: usize,
}

#[derive(Resource)]
struct VoxelGiRuntime {
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
    transport: TransportSettings,
    loaded_pages: HashSet<IVec3>,
    /// Bounds of `loaded_pages`: the host's declared geometry interest.
    interest: pages::Interest,
    dirty_pages: HashSet<IVec3>,
    full_resample_pages: HashSet<IVec3>,
    /// Sampling time budget, checked between pages (`pages::SCENE_SAMPLE_BUDGET`).
    scene_sample_budget: Duration,
    page_edits: pages::PageEdits,
    dirty_voxels: HashSet<IVec3>,
    removed_pages: HashSet<IVec3>,
    urgent_edit: bool,
    generation: u64,
    spatial_history: retention::SpatialHistory,
    source_revision: u64,
    batch_size: usize,
    /// Last global material/model replacement. Scene edits use spatial history
    /// instead; residency of an emitting voxel must not restart the world.
    model_reset_source_revision: u64,
    urgent_pending: VecDeque<IVec3>,
    urgent_queued: HashSet<IVec3>,
    /// Batch ids for GPU selection (28 bits, never zero).
    batch_counter: u32,
    /// Probes with faces that need a resident slot (eager allocation).
    slot_requests: VecDeque<IVec3>,
    slot_requested: HashSet<IVec3>,
    active_faces: u64,
    covered_faces: u64,
    completed_samples: u64,
    settled_faces: u64,
    scene_task: Option<Task<SceneBuildResult>>,
    scene_publication: Option<SceneBuildResult>,
    trace_task: Option<Task<TraceBatchResult>>,
    gpu_pending: VecDeque<PendingTransport>,
    trace_pending: HashSet<IVec3>,
    /// The one in-flight worker returns its allocation for the next dispatch.
    trace_scratch: TraceWork,
    selection_seen: bevy::platform::collections::HashSet<IVec3>,
    gpu_tracer: Option<gpu::GpuTracer>,
    emission_cache: Option<emission::EmissionCache>,
    /// Visible faces' corner-sunlight flips, merged into Light changes.
    sun_changes: sun_changes::SunChanges,
    /// Pages allowed cache receivers under `cache_radius` (None before the
    /// first residency update: all loaded).
    receiver_pages: Option<HashSet<IVec3>>,
    /// Pages entering or leaving it: rescanned for receivers, or retired,
    /// without a geometry change.
    receiver_dirty: HashSet<IVec3>,
    receiver_retired: HashSet<IVec3>,
    /// Pages whose receivers were retired: their resident slots are released
    /// by the next worker, even when no sampling batch can run.
    slot_retirements: Vec<IVec3>,
    /// GPU milliseconds transport may still spend under `gpu_time_share`:
    /// accrues with wall time, a batch reserves its budget at submission
    /// and is charged its measured time at completion.
    gpu_allowance_ms: f32,
    gpu_allowance_at: Option<Instant>,
    /// The transport could not be built on this device (its pipelines failed,
    /// e.g. FXC on DX12): the world cache idles for good, as with no device.
    transport_unavailable: bool,
}

impl Default for VoxelGiRuntime {
    fn default() -> Self {
        Self {
            scene: Arc::new(TransportScene::default()),
            sources: TransportSources::default(),
            materials: Arc::new(VoxelMaterialTransportTable::default()),
            transport: TransportSettings::default(),
            loaded_pages: HashSet::new(),
            interest: default(),
            dirty_pages: HashSet::new(),
            full_resample_pages: HashSet::new(),
            scene_sample_budget: pages::SCENE_SAMPLE_BUDGET,
            page_edits: default(),
            dirty_voxels: HashSet::new(),
            removed_pages: HashSet::new(),
            urgent_edit: false,
            generation: 0,
            spatial_history: default(),
            source_revision: 0,
            batch_size: 256,
            model_reset_source_revision: 0,
            urgent_pending: VecDeque::new(),
            urgent_queued: HashSet::new(),
            batch_counter: 0,
            slot_requests: VecDeque::new(),
            slot_requested: HashSet::new(),
            active_faces: 0,
            covered_faces: 0,
            completed_samples: 0,
            settled_faces: 0,
            scene_task: None,
            scene_publication: None,
            trace_task: None,
            gpu_pending: default(),
            trace_pending: default(),
            trace_scratch: TraceWork::default(),
            selection_seen: default(),
            gpu_tracer: None,
            emission_cache: None,
            sun_changes: default(),
            receiver_pages: None,
            receiver_dirty: HashSet::new(),
            receiver_retired: HashSet::new(),
            slot_retirements: Vec::new(),
            gpu_allowance_ms: 0.0,
            gpu_allowance_at: None,
            transport_unavailable: false,
        }
    }
}

struct SceneBuildResult {
    scene: Arc<TransportScene>,
    changed: bool,
    /// The build changed transport (geometry, emission): a new revision.
    transport_changed: bool,
    /// Its retention event, built on the worker (`SpatialHistory::event`).
    retention: Option<retention::RetentionEvent>,
    pages: VecDeque<(IVec3, Vec<SurfaceProbeDescriptor>)>,
    /// Pages rescanned or retired for the receiver radius alone.
    receivers_changed: bool,
    exact_edits: Vec<IVec3>,
    /// Dirty pages the build left for the next one (out of dense budget),
    /// with their full-resample flags.
    deferred: Vec<(IVec3, bool)>,
    build_time: Duration,
}

struct TraceBatchResult {
    batch: Option<TransportBatch>,
    pending: VecDeque<PendingTransport>,
    bindings: Vec<gpu::ResidentBinding>,
    trace_time: Duration,
    gpu_tracer: Option<gpu::GpuTracer>,
    emission_cache: Option<emission::EmissionCache>,
    gpu_error: Option<String>,
    /// The tracer could not be built (not a failure during a run): the
    /// transport is unavailable, not fatal.
    build_error: Option<String>,
    /// Geometry buffers and bounds for the terrain material (far sun).
    geometry: Option<(
        [bevy::render::render_resource::Buffer; 3],
        render::GpuTraversalParams,
    )>,
}

struct PendingTransport {
    snapshot: Option<gpu::Readback>,
    batch: TransportBatch,
}

#[derive(Default)]
struct TransportBatch {
    work: TraceWork,
    paths: u64,
    gpu_ms: Option<f32>,
    execution_ms: Option<f32>,
    /// GPU milliseconds reserved from the allowance at submission.
    gpu_reserved_ms: f32,
}

fn page_of(voxel: IVec3) -> IVec3 {
    voxel.div_euclid(IVec3::splat(PAGE_EDGE as i32))
}

fn advance_transport_revision(runtime: &mut VoxelGiRuntime, field: &mut ReceiverField) {
    field.transport_revision = field.transport_revision.wrapping_add(1);
    // Old histories remain useful, but have not passed stability checks in the
    // new scene. Revision-aware deltas cannot subtract them from this new count.
    runtime.settled_faces = 0;
}

fn enqueue_urgent(runtime: &mut VoxelGiRuntime, voxels: impl IntoIterator<Item = IVec3>) {
    for voxel in voxels {
        if runtime.urgent_queued.insert(voxel) {
            runtime.urgent_pending.push_back(voxel);
        }
    }
}

fn probe_metrics(probe: &SurfaceProbe, transport_revision: u64) -> (u64, u64, u64, u64) {
    let mut faces = 0;
    let mut samples = 0;
    let mut covered = 0;
    for face in 0..6 {
        if probe.face_mask & (1 << face) != 0 {
            faces += 1;
            samples += u64::from(probe.bootstrap_samples(face));
            covered += u64::from(probe.bootstrap_samples(face) > 0);
        }
    }
    let settled = u64::from(scheduling::settled_face_mask(probe, transport_revision).count_ones());
    (faces, samples, covered, settled)
}

fn update_stats_from_counters(
    runtime: &VoxelGiRuntime,
    field: &ReceiverField,
    stats: &mut VoxelGiStats,
) {
    let faces = runtime.active_faces;
    stats.surface_probes = field.probes.len();
    stats.scheduler.dispatch_budget = runtime.batch_size;
    stats.scheduler.pending_probes = runtime.urgent_pending.len();
    stats.bootstrap_coverage = if faces == 0 {
        1.0
    } else {
        runtime.completed_samples as f32 / (faces * u64::from(BOOTSTRAP_SAMPLES)) as f32
    };
    stats.interactive_coverage = if faces == 0 {
        1.0
    } else {
        runtime.covered_faces as f32 / faces as f32
    };
    stats.settled_coverage = if faces == 0 {
        0.0
    } else {
        runtime.settled_faces as f32 / faces as f32
    };
    // The world sweep is continuous on the GPU (oldest faces first).
    stats.scheduler.fully_bootstrapped =
        stats.bootstrap_coverage >= 1.0 && field.publishing_pages.is_empty();
}

#[cfg(test)]
mod tests;

/// Hooks for connector crates' integration tests; not a stable API.
#[doc(hidden)]
pub mod testing {
    use bevy::prelude::*;
    use bevy::render::render_resource::Buffer;

    /// Only the material-facing render setup (`VoxelGiMaterialData`, geometry
    /// installation and the view output), without scene or transport systems.
    pub fn install_render(app: &mut App) {
        super::render::install_render_assets(app);
    }

    /// Publish `buffers` as the shared geometry, as a new GPU scene would.
    pub fn publish_geometry(world: &mut World, buffers: [Buffer; 3], node_count: u32) {
        let mut geometry = world.resource_mut::<super::render::GeometryDisplay>();
        geometry.buffers = Some(buffers);
        geometry.params.counts.x = node_count;
    }

    /// Change the presentation switches materials see (`enabled`, debug view).
    pub fn set_display(world: &mut World, enabled: bool, debug: u32) {
        let mut data = world.resource_mut::<super::VoxelGiMaterialData>();
        data.settings.enabled = enabled as u32;
        data.settings.debug_irradiance = debug;
    }

    /// A material's geometry bindings (directory, regions, cells) and node count.
    pub fn geometry_bindings(
        bindings: &super::VoxelGiBindings,
    ) -> ([Handle<bevy::render::storage::ShaderBuffer>; 3], u32) {
        (
            [
                bindings.directory.clone(),
                bindings.regions.clone(),
                bindings.cells.clone(),
            ],
            bindings.traversal.counts.x,
        )
    }
}
