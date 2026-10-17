//! Persistent diffuse transport for opaque axis-aligned voxel surfaces.
//!
//! Immutable scene snapshots and tracing run on workers; revision-checked
//! publication and bounded scheduling own the authoritative energy field.
//! `history`/`reconstruction`/`render` are display-only: their priors never become
//! simulation measurements. `bvw` is the current host adapter, `diagnostics` the
//! optional demo tooling. This module is not yet a storage-independent crate.
mod bvw;
mod diagnostics;
mod dispatch;
mod display;
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
mod revision;
mod sampling;
mod scheduling;
mod settings;
mod sky;
mod sky_cache;
mod sources;
mod spatial;
mod spectrum;
mod startup;
mod statistics;
mod telemetry;
mod tracer;
#[cfg(debug_assertions)]
mod validation;
mod work;
mod world;

use std::collections::VecDeque;
// Bounded, engine-owned integer coordinates; no untrusted string keys.
use bevy::platform::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::tasks::Task;
use rayon::ThreadPool;

pub use field::{EnergyEstimate, VoxelEnergyField};
pub use settings::{TransportSettings, VoxelGiSettings, VoxelGiVisualWeights};
pub use sources::VoxelGiEnvironment;
pub use spectrum::{EnergyEmitter, EnergySpectrum};
pub use startup::{RetentionPolicy, RetentionSummary, TransportBackend, VoxelGiStartup};
pub use world::{MaterialTransport, VoxelMaterialTransportTable};

use bvw::{collect_chunk_changes, start_scene_task};
pub(crate) use diagnostics::{VoxelGiDiagnosticsPlugin, configure_demo_launch};
use dispatch::{apply_trace_task, start_trace_task};
use display::upload_transport_cache;
use publication::apply_scene_task;
use render::{GpuSnapshot, GpuTransportCache, PreparedGpuCache, install_render_assets};
use sources::{TransportSources, synchronize_material_table, synchronize_transport_sources};
use telemetry::GiWorkTimings;
use tracer::BOOTSTRAP_SAMPLES;
use work::TraceWork;
use world::{PAGE_EDGE, SurfaceProbe, SurfaceProbeDescriptor, TransportScene};

const GPU_UPLOAD_INTERVAL: Duration = Duration::from_millis(50);

/// Transport and display-cache integration for the prototype BVW world.
/// Debug controls, HUD and validation are opt-in via the demo diagnostics plugin.
pub struct VoxelGiPlugin {
    /// Host-owned texture array matching the terrain mesh's material indices.
    /// The host prepares/reloads the image; GI does not choose an asset or palette.
    pub voxel_texture: Handle<Image>,
}

/// Schedule simulation readers after this set to observe this frame's published
/// measurements. Workers remain asynchronous; inspect estimate freshness.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelGiSystems;

impl Plugin for VoxelGiPlugin {
    fn build(&self, app: &mut App) {
        install_render_assets(app, self.voxel_texture.clone());
        app.init_resource::<VoxelGiDirty>()
            .init_resource::<VoxelGiRuntime>()
            .init_resource::<VoxelEnergyField>()
            .init_resource::<VoxelGiStats>()
            .init_resource::<GiWorkTimings>()
            .init_resource::<VoxelGiSettings>()
            .init_resource::<VoxelGiEnvironment>()
            .init_resource::<VoxelMaterialTransportTable>()
            .init_resource::<GpuTransportCache>()
            .add_systems(
                Update,
                (
                    collect_chunk_changes,
                    synchronize_transport_sources,
                    synchronize_material_table,
                    settings::synchronize_transport_settings,
                    apply_scene_task,
                    start_scene_task,
                    apply_trace_task,
                    start_trace_task,
                    upload_transport_cache,
                    telemetry::finish_frame,
                )
                    .chain()
                    .in_set(VoxelGiSystems),
            );
        startup::initialize(app);
    }

    fn finish(&self, app: &mut App) {
        // CPU tracing still uses the same display storage bindings. Capture
        // renderer limits independently of transport backend selection.
        if let Some(limits) = app
            .get_sub_app(bevy::render::RenderApp)
            .and_then(|render_app| {
                render_app
                    .world()
                    .get_resource::<bevy::render::renderer::RenderDevice>()
                    .map(|device| render::DisplayBufferLimits::from(&device.limits()))
            })
        {
            app.insert_resource(limits);
        }
        if app.world().resource::<VoxelGiStartup>().backend == TransportBackend::Cpu {
            return;
        }
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

#[derive(Resource, Default)]
pub struct VoxelGiDirty {
    voxels: HashSet<IVec3>,
    edits: bevy::platform::collections::HashMap<IVec3, u16>,
    complete_edit_tracking: bool,
    all_loaded: bool,
}

impl VoxelGiDirty {
    /// Opt in only when every host write uses the adapter's `set_voxel`, or
    /// explicitly marks an unknown edit with `mark_voxel`/`mark_all_loaded`.
    /// BVW chunk-update messages alone cannot distinguish tracked/untracked writes.
    pub fn tracked() -> Self {
        Self {
            complete_edit_tracking: true,
            ..default()
        }
    }

    pub fn mark_voxel(&mut self, voxel: IVec3) {
        // A later untracked write supersedes an earlier exact value, including
        // when both notifications arrive before the adapter drains this frame.
        self.edits.remove(&voxel);
        self.voxels.insert(voxel);
    }
    pub fn mark_all_loaded(&mut self) {
        self.edits.clear();
        self.all_loaded = true;
    }
}

#[derive(Resource, Clone, Debug, Default)]
pub struct VoxelGiStats {
    pub loaded_pages: usize,
    pub naadf_regions: usize,
    pub surface_probes: usize,
    pub bootstrap_coverage: f32,
    pub interactive_coverage: f32,
    /// Fraction passing the adaptive stability heuristic for the current
    /// transport revision. Periodic audits remain eligible; not convergence.
    pub settled_coverage: f32,
    /// Mean GI main-world system-body time per app frame over a 250 ms window.
    /// Excludes background workers and render-world upload/draw work.
    pub main_thread_ms_per_frame: Option<f32>,
    /// Last app frame's GI system-body times, in telemetry stage order.
    /// Excludes workers/render work; unlike the overlay average, retains spikes.
    pub last_main_stage_ms: [f32; 7],
    /// Opt-in display-publication substage costs, in telemetry stage order.
    /// Zero unless frame-timing validation is enabled; included in main time.
    pub last_display_stage_ms: [f32; 7],
    /// Completed transport/sky compute-pass timestamps per app frame over the
    /// same window. None until available, or on CPU/unsupported GPU backends.
    /// Excludes queue waits, transfers and graphics; do not add to main time.
    pub gpu_ms_per_frame: Option<f32>,
    pub pending_probes: usize,
    pub global_refresh: f32,
    pub traced_paths: u64,
    pub last_scene_build_ms: f32,
    pub last_trace_batch_ms: f32,
    pub sky_corner_requests: u64,
    pub sky_corners_traced: u64,
    pub sky_rays_scheduled: u64,
    pub sky_cached_corners: usize,
    pub sky_only_batches: u64,
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
    /// Approximate mean over at most 1,024 evenly spaced receivers, populated
    /// by opt-in terminal/validation diagnostics every five seconds. Not convergence.
    pub mean_history_samples: f32,
    /// Oldest among that sampled subset, not a certified world-wide maximum.
    pub oldest_sample_age_ms: f32,
    pub backend: &'static str,
}

#[derive(Resource)]
struct VoxelGiRuntime {
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
    transport: TransportSettings,
    trace_pool: Arc<OnceLock<ThreadPool>>,
    loaded_pages: HashSet<IVec3>,
    dirty_pages: HashSet<IVec3>,
    full_resample_pages: HashSet<IVec3>,
    page_edits: bvw::PageEdits,
    dirty_voxels: HashSet<IVec3>,
    removed_pages: HashSet<IVec3>,
    last_world_change: Instant,
    dirty_since: Option<Instant>,
    urgent_edit: bool,
    generation: u64,
    geometry_history: geometry::GeometryHistory,
    spatial_history: Option<retention::SpatialHistory>,
    #[cfg(debug_assertions)]
    retention_audit: validation::retention_audit::Recorder,
    source_revision: u64,
    batch_size: usize,
    /// Last global material/model replacement. Scene edits use spatial history
    /// instead; residency of an emitting voxel must not restart the world.
    model_reset_source_revision: u64,
    source_history: sources::SourceHistory,
    nearby: Vec<IVec3>,
    regional: Vec<IVec3>,
    nearby_cursor: usize,
    adaptive_cursor: usize,
    focus_page: Option<IVec3>,
    focus_dirty: bool,
    pending: VecDeque<IVec3>,
    queued: HashSet<IVec3>,
    urgent_pending: VecDeque<IVec3>,
    urgent_queued: HashSet<IVec3>,
    refresh_cursor: usize,
    refresh_remaining: usize,
    refresh_total: usize,
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
    ranked_candidates: scheduling::RankedCandidates,
    gpu_tracer: Option<gpu::GpuTracer>,
    emission_cache: Option<emission::EmissionCache>,
    cpu_sky_cache: Option<sky_cache::SkyCache>,
    sky_pending: VecDeque<IVec3>,
    sky_queued: HashSet<IVec3>,
    sky_cursor: usize,
    sky_remaining: usize,
    sky_revision: u64,
    sky_batch_size: usize,
    gpu_topology_dirty: bool,
    gpu_rebuild_required: bool,
    gpu_added: Vec<IVec3>,
    gpu_removed: Vec<IVec3>,
    /// Values that must publish atomically with face/hash topology, not the
    /// independent backlog of ordinary radiance updates.
    gpu_topology_values: HashSet<IVec3>,
    gpu_task: Option<Task<Result<PreparedGpuCache, display::DisplayBuildFailure>>>,
    gpu_snapshot: Option<GpuSnapshot>,
    gpu_prepared: Option<PreparedGpuCache>,
    gpu_catchup_dirty: display::DisplayDirtyQueue,
    gpu_value_dirty: display::DisplayDirtyQueue,
    sky_upload_pending: bool,
    last_gpu_upload: Instant,
}

impl Default for VoxelGiRuntime {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            scene: Arc::new(TransportScene::default()),
            sources: TransportSources::default(),
            materials: Arc::new(VoxelMaterialTransportTable::default()),
            transport: TransportSettings::default(),
            trace_pool: Arc::new(OnceLock::new()),
            loaded_pages: HashSet::new(),
            dirty_pages: HashSet::new(),
            full_resample_pages: HashSet::new(),
            page_edits: default(),
            dirty_voxels: HashSet::new(),
            removed_pages: HashSet::new(),
            last_world_change: now,
            dirty_since: None,
            urgent_edit: false,
            generation: 0,
            geometry_history: default(),
            spatial_history: None,
            #[cfg(debug_assertions)]
            retention_audit: default(),
            source_revision: 0,
            batch_size: 256,
            model_reset_source_revision: 0,
            source_history: default(),
            nearby: Vec::new(),
            regional: Vec::new(),
            nearby_cursor: 0,
            adaptive_cursor: 0,
            focus_page: None,
            focus_dirty: true,
            pending: VecDeque::new(),
            queued: HashSet::new(),
            urgent_pending: VecDeque::new(),
            urgent_queued: HashSet::new(),
            refresh_cursor: 0,
            refresh_remaining: 0,
            refresh_total: 0,
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
            ranked_candidates: default(),
            gpu_tracer: None,
            emission_cache: None,
            cpu_sky_cache: None,
            sky_pending: default(),
            sky_queued: default(),
            sky_cursor: 0,
            sky_remaining: 0,
            sky_revision: 0,
            sky_batch_size: 512,
            gpu_topology_dirty: false,
            gpu_rebuild_required: false,
            gpu_added: Vec::new(),
            gpu_removed: Vec::new(),
            gpu_topology_values: default(),
            gpu_task: None,
            gpu_snapshot: None,
            gpu_prepared: None,
            gpu_catchup_dirty: default(),
            gpu_value_dirty: default(),
            sky_upload_pending: false,
            last_gpu_upload: now - GPU_UPLOAD_INTERVAL,
        }
    }
}

struct SceneBuildResult {
    scene: Arc<TransportScene>,
    changed: bool,
    geometry_change: Option<geometry::GeometryChange>,
    pages: VecDeque<(IVec3, Vec<SurfaceProbeDescriptor>)>,
    exact_edits: Vec<IVec3>,
    build_time: Duration,
}

struct TraceBatchResult {
    batch: Option<TransportBatch>,
    pending: VecDeque<PendingTransport>,
    bindings: Vec<gpu::ResidentBinding>,
    sky_stats: Option<sky_cache::SkyCacheStats>,
    trace_time: Duration,
    gpu_tracer: Option<gpu::GpuTracer>,
    emission_cache: Option<emission::EmissionCache>,
    cpu_sky_cache: sky_cache::SkyCache,
    gpu_error: Option<String>,
    sunlight_change: Option<geometry::GeometryChange>,
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
    sky_only: bool,
    transport_count: usize,
    sky_count: usize,
    scene_generation: u64,
    sky_revision: u64,
}

fn page_of(voxel: IVec3) -> IVec3 {
    voxel.div_euclid(IVec3::splat(PAGE_EDGE as i32))
}

fn advance_transport_revision(runtime: &mut VoxelGiRuntime, field: &mut VoxelEnergyField) {
    field.transport_revision = field.transport_revision.wrapping_add(1);
    // Old histories remain useful, but have not passed stability checks in the
    // new scene. Revision-aware deltas cannot subtract them from this new count.
    runtime.settled_faces = 0;
}

fn request_global_refresh(runtime: &mut VoxelGiRuntime, probes: usize) {
    // Never rewind the sweep on edits: continuous digging would otherwise
    // repeatedly service the first pages and permanently starve distant ones.
    runtime.refresh_remaining = probes;
    runtime.refresh_total = probes;
}

fn enqueue_many(runtime: &mut VoxelGiRuntime, voxels: impl IntoIterator<Item = IVec3>) {
    for voxel in voxels {
        if runtime.queued.insert(voxel) {
            runtime.pending.push_back(voxel);
        }
    }
}

fn enqueue_urgent(runtime: &mut VoxelGiRuntime, voxels: impl IntoIterator<Item = IVec3>) {
    for voxel in voxels {
        if runtime.urgent_queued.insert(voxel) {
            runtime.urgent_pending.push_back(voxel);
        }
    }
}

fn enqueue_sky(runtime: &mut VoxelGiRuntime, voxels: impl IntoIterator<Item = IVec3>) {
    for voxel in voxels {
        if runtime.sky_queued.insert(voxel) {
            runtime.sky_pending.push_back(voxel);
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
            samples += u64::from(probe.samples[face].min(BOOTSTRAP_SAMPLES));
            covered += u64::from(probe.samples[face] > 0);
        }
    }
    let settled = u64::from(scheduling::settled_face_mask(probe, transport_revision).count_ones());
    (faces, samples, covered, settled)
}

fn update_stats_from_counters(
    runtime: &VoxelGiRuntime,
    field: &VoxelEnergyField,
    stats: &mut VoxelGiStats,
) {
    let faces = runtime.active_faces;
    stats.surface_probes = field.probes.len();
    stats.pending_probes = runtime.queued.len();
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
    stats.global_refresh = if runtime.refresh_total == 0 {
        1.0
    } else {
        1.0 - runtime.refresh_remaining as f32 / runtime.refresh_total as f32
    };
    stats.fully_bootstrapped = stats.bootstrap_coverage >= 1.0 && field.publishing_pages.is_empty();
}

#[cfg(test)]
mod tests;
