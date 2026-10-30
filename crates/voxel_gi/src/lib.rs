//! Persistent diffuse global illumination for opaque axis-aligned voxel
//! surfaces in Bevy.
//!
//! A host reports loaded 32³ pages and voxel edits through `VoxelGiInput` and
//! provides a `VoxelSource` to read them; GI builds immutable scene snapshots
//! on workers, traces every visible face on the GPU (the view tier) next to a
//! persistent world cache, and publishes what a lit material binds as
//! `VoxelGiMaterialData`. `voxel_gi_bvw` connects `bevy_voxel_world`;
//! `voxel_gi_debug` is the optional HUD and tuning panel.
mod input;
mod pages;
mod dispatch;
mod emission;
mod field;
mod geometry;
mod gpu;
mod gpu_scene;
mod history;
mod naadf;
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
mod color;
mod sources;
mod sun_changes;
#[cfg(test)]
mod statistics;
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
pub use settings::{FarTierMeasure, FarTierSettings, TransportSettings, VoxelGiQuality, VoxelGiSettings};
pub use sources::{VoxelGiEmitter, VoxelGiEnvironment};
pub use world::{MaterialTransport, VoxelMaterialTransportTable};

use pages::{ingest_input, start_scene_task};
pub use input::{
    PageSample, VOXEL_GI_PAGE_EDGE, VoxelGiInput, VoxelGiSource, VoxelSource, coarse_page, dense_page,
};
pub use gpu::view::ViewReceipts;
pub use gpu::include_traversal;
pub use render::{
    GiDebugView, GpuTransportSettings, GpuTraversalParams, VoxelGiDisplay, VoxelGiMaterialData,
};
pub use telemetry::MAIN_STAGE_NAMES as GI_MAIN_STAGE_NAMES;
use dispatch::{apply_trace_task, start_trace_task};
use publication::apply_scene_task;
use render::install_render_assets;
use sources::{TransportSources, synchronize_material_table, synchronize_transport_sources};
use telemetry::GiWorkTimings;
use tracer::BOOTSTRAP_SAMPLES;
use work::TraceWork;
use world::{PAGE_EDGE, SurfaceProbe, SurfaceProbeDescriptor, TransportScene};


/// Voxel GI: page ingestion, transport and the view tier. Materials bind
/// `VoxelGiMaterialData` (the BVW connector's material does this); the debug
/// HUD and panel are the opt-in `VoxelGiDebugPlugin`.
pub struct VoxelGiPlugin;

/// Schedule simulation readers after this set to observe this frame's published
/// measurements. Workers remain asynchronous; inspect estimate freshness.
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
    pub loaded_pages: usize,
    pub surface_probes: usize,
    pub bootstrap_coverage: f32,
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
    pub backend: &'static str,
    pub scheduler: SchedulerStats,
}

/// World-cache scheduler internals, for diagnostics.
#[derive(Clone, Debug, Default)]
pub struct SchedulerStats {
    /// Last app frame's GI system-body times, in telemetry stage order.
    /// Excludes workers/render work; unlike the overlay average, retains spikes.
    pub last_main_stage_ms: [f32; 6],
    pub pending_probes: usize,
    pub traced_paths: u64,
    pub last_scene_build_ms: f32,
    pub last_trace_batch_ms: f32,
    pub transport_batches: u64,
    pub idle_scheduler_frames: u64,
    pub last_worker_ms: f32,
    pub last_selection_ms: f32,
    pub last_merge_ms: f32,
    pub last_prepare_ms: f32,
    /// Submission-to-delivery latency; does not imply a blocking GPU wait.
    pub last_snapshot_latency_ms: f32,
    pub last_decode_ms: f32,
    pub packed_geometry_bytes: usize,
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
    stats.scheduler.fully_bootstrapped = stats.bootstrap_coverage >= 1.0 && field.publishing_pages.is_empty();
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
}
