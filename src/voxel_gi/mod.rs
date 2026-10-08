//! Persistent diffuse transport for opaque axis-aligned voxel surfaces.
//!
//! Immutable scene snapshots and tracing run on workers; revision-checked
//! publication and bounded scheduling own the authoritative energy field.
//! `history`/`reconstruction`/`render` are display-only: their priors never become
//! simulation measurements. `bvw` is the current host adapter, `diagnostics` the
//! optional demo tooling. This module is not yet a storage-independent crate.
mod bvw;
mod diagnostics;
mod emission;
mod field;
mod geometry;
mod gpu;
mod gpu_scene;
mod history;
mod naadf;
mod reconstruction;
pub(crate) mod render;
mod revision;
mod sampling;
mod scheduling;
mod settings;
mod sky;
mod sky_cache;
mod sources;
mod spectrum;
mod statistics;
mod telemetry;
mod tracer;
#[cfg(debug_assertions)]
mod validation;
mod work;
mod world;

use std::collections::{HashMap, VecDeque};
// Bounded, engine-owned integer coordinates; no untrusted string keys.
use bevy::platform::collections::HashSet;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub use field::{EnergyEstimate, VoxelEnergyField};
pub use settings::VoxelGiSettings;
pub use sources::VoxelGiEnvironment;
pub use spectrum::{EnergyEmitter, EnergySpectrum};
pub use world::{MaterialTransport, VoxelMaterialTransportTable};

use bvw::{collect_chunk_changes, start_scene_task};
pub(crate) use diagnostics::VoxelGiDiagnosticsPlugin;
#[cfg(test)]
use diagnostics::toggle_voxel_gi;
use render::{
    GpuCacheBuild, GpuProbePatchQueue, GpuSnapshot, GpuTransportCache, VoxelTransportAssets,
    VoxelTransportMaterial, build_gpu_cache, install_gpu_cache, install_render_assets,
    patch_gpu_topology, patch_probe_values,
};
use sources::{
    TransportSources, invalidate_sources, synchronize_material_table, synchronize_transport_sources,
};
use telemetry::GiWorkTimings;
use tracer::{BOOTSTRAP_SAMPLES, trace_work_batch};
use work::TraceWork;
use world::{PAGE_EDGE, SurfaceProbe, SurfaceProbeDescriptor, TransportScene};

const GPU_UPLOAD_INTERVAL: Duration = Duration::from_millis(50);
const SAMPLES_PER_DISPATCH: u32 = 4;
const EDIT_RELIGHT_RADIUS: i32 = geometry::RELIGHT_RADIUS;

/// Transport and display-cache integration for the prototype BVW world.
/// Debug controls, HUD and validation are opt-in via the demo diagnostics plugin.
pub struct VoxelGiPlugin;

/// Schedule simulation readers after this set to observe this frame's published
/// measurements. Workers remain asynchronous; inspect estimate freshness.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelGiSystems;

impl Plugin for VoxelGiPlugin {
    fn build(&self, app: &mut App) {
        install_render_assets(app);
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
        if std::env::var("VOXEL_GI_CONTINUOUS").is_ok_and(|value| value == "1") {
            app.world_mut()
                .resource_mut::<VoxelGiSettings>()
                .adaptive_updates = false;
        }
    }

    fn finish(&self, app: &mut App) {
        if std::env::var("VOXEL_GI_BACKEND").is_ok_and(|backend| backend == "cpu") {
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
    all_loaded: bool,
}

impl VoxelGiDirty {
    pub fn mark_voxel(&mut self, voxel: IVec3) {
        self.voxels.insert(voxel);
    }
    pub fn mark_all_loaded(&mut self) {
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
    pub last_wait_ms: f32,
    pub last_decode_ms: f32,
    pub packed_geometry_bytes: usize,
    pub fully_bootstrapped: bool,
    /// Full-field summaries are populated only by opt-in terminal/validation
    /// diagnostics, every five seconds. They are not convergence measures.
    pub mean_history_samples: f32,
    /// See `mean_history_samples`; zero while full-field diagnostics are off.
    pub oldest_sample_age_ms: f32,
    pub backend: &'static str,
}

#[derive(Resource)]
struct VoxelGiRuntime {
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
    trace_pool: Arc<OnceLock<ThreadPool>>,
    loaded_pages: HashSet<IVec3>,
    dirty_pages: HashSet<IVec3>,
    full_resample_pages: HashSet<IVec3>,
    dirty_voxels: HashSet<IVec3>,
    removed_pages: HashSet<IVec3>,
    last_world_change: Instant,
    dirty_since: Option<Instant>,
    urgent_edit: bool,
    generation: u64,
    geometry_history: geometry::GeometryHistory,
    source_revision: u64,
    batch_size: usize,
    /// Last discrete emission change: old energy must not survive as samples.
    reset_source_revision: u64,
    source_history: sources::SourceHistory,
    nearby: Vec<IVec3>,
    nearby_cursor: usize,
    adaptive_cursor: usize,
    focus_page: Option<IVec3>,
    focus_topology: u64,
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
    gpu_failed: bool,
    gpu_topology_dirty: bool,
    gpu_rebuild_required: bool,
    gpu_topology_version: u64,
    gpu_added: Vec<IVec3>,
    gpu_removed: Vec<IVec3>,
    gpu_task: Option<Task<GpuCacheBuild>>,
    gpu_snapshot: Option<GpuSnapshot>,
    gpu_catchup_dirty: HashSet<IVec3>,
    gpu_task_version: u64,
    gpu_value_dirty: HashSet<IVec3>,
    sky_upload_pending: bool,
    last_gpu_upload: Instant,
    enabled: bool,
}

impl Default for VoxelGiRuntime {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            scene: Arc::new(TransportScene::default()),
            sources: TransportSources::default(),
            materials: Arc::new(VoxelMaterialTransportTable::default()),
            trace_pool: Arc::new(OnceLock::new()),
            loaded_pages: HashSet::new(),
            dirty_pages: HashSet::new(),
            full_resample_pages: HashSet::new(),
            dirty_voxels: HashSet::new(),
            removed_pages: HashSet::new(),
            last_world_change: now,
            dirty_since: None,
            urgent_edit: false,
            generation: 0,
            geometry_history: default(),
            source_revision: 0,
            batch_size: 256,
            reset_source_revision: 0,
            source_history: default(),
            nearby: Vec::new(),
            nearby_cursor: 0,
            adaptive_cursor: 0,
            focus_page: None,
            focus_topology: u64::MAX,
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
            gpu_failed: false,
            gpu_topology_dirty: false,
            gpu_rebuild_required: false,
            gpu_topology_version: 0,
            gpu_added: Vec::new(),
            gpu_removed: Vec::new(),
            gpu_task: None,
            gpu_snapshot: None,
            gpu_catchup_dirty: HashSet::new(),
            gpu_task_version: 0,
            gpu_value_dirty: HashSet::new(),
            sky_upload_pending: false,
            last_gpu_upload: now - GPU_UPLOAD_INTERVAL,
            enabled: true,
        }
    }
}

struct SceneBuildResult {
    scene: Arc<TransportScene>,
    changed: bool,
    geometry_change: Option<geometry::GeometryChange>,
    changed_materials: [bool; 256],
    pages: VecDeque<(IVec3, Vec<SurfaceProbeDescriptor>)>,
    exact_edits: Vec<IVec3>,
    build_time: Duration,
}

struct TraceBatchResult {
    work: TraceWork,
    paths: u64,
    trace_time: Duration,
    gpu_tracer: Option<gpu::GpuTracer>,
    emission_cache: Option<emission::EmissionCache>,
    cpu_sky_cache: sky_cache::SkyCache,
    gpu_error: Option<String>,
    sky_only: bool,
    transport_count: usize,
    sky_count: usize,
    scene_generation: u64,
    sky_revision: u64,
}

fn page_of(voxel: IVec3) -> IVec3 {
    voxel.div_euclid(IVec3::splat(PAGE_EDGE as i32))
}

fn apply_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope();
    let publish_started = Instant::now();
    if let Some(task) = runtime.scene_task.as_mut()
        && let Some(mut result) = block_on(poll_once(task))
    {
        runtime.scene_task = None;
        if !result.changed {
            stats.loaded_pages = runtime.loaded_pages.len();
            stats.last_scene_build_ms = result.build_time.as_secs_f32() * 1000.0;
            return;
        }
        if result
            .changed_materials
            .iter()
            .enumerate()
            .any(|(material, changed)| {
                *changed
                    && runtime
                        .materials
                        .get(material as u8)
                        .emission
                        .max_component()
                        > 0.0
            })
        {
            // Adding/removing an emitting voxel changes the source set, not
            // merely visibility. Retaining lit samples after removal produces
            // a long 16/(16+n) (or remote 64/(64+n)) ghost, before denoising.
            invalidate_sources(&mut runtime, &mut field, &mut stats);
            runtime.reset_source_revision = runtime.source_revision;
        }
        runtime.scene = result.scene.clone();
        runtime.generation = runtime.generation.wrapping_add(1);
        if let Some(change) = result.geometry_change.take() {
            let from = field.transport_revision;
            advance_transport_revision(&mut runtime, &mut field);
            runtime
                .geometry_history
                .record(from, field.transport_revision, change);
        }
        field
            .publishing_pages
            .extend(result.pages.iter().map(|(page, _)| *page));
        // Editing may overtake a large streaming publication. Replace only the
        // intersecting queued pages, prioritizing fresh edit geometry; unrelated
        // loaded terrain remains queued and is never silently discarded.
        if let Some(older) = runtime.scene_publication.take() {
            let replaced: HashSet<_> = result.pages.iter().map(|(page, _)| *page).collect();
            result.pages.extend(
                older
                    .pages
                    .into_iter()
                    .filter(|(page, _)| !replaced.contains(page)),
            );
        }
        runtime.scene_publication = Some(result);
    }
    let Some(mut result) = runtime.scene_publication.take() else {
        return;
    };
    let budget = settings.scene_publish_budget.min(Duration::from_millis(8));
    let mut published = 0;
    while let Some((page, descriptors)) = result.pages.pop_front() {
        // Empty pages need no merge; skip them without spending a whole frame.
        if descriptors.is_empty() && !field.pages.contains_key(&page) {
            field.publishing_pages.remove(&page);
            continue;
        }
        publish_scene_page(
            &mut runtime,
            &mut field,
            &mut stats,
            &mut result,
            page,
            descriptors,
        );
        published += 1;
        // Preserve progress even with a zero budget. The count cap also bounds
        // downstream display-cache patching, which happens after this system.
        if published >= 4 || publish_started.elapsed() >= budget {
            break;
        }
    }
    update_stats_from_counters(&runtime, &field, &mut stats);
    stats.loaded_pages = runtime.loaded_pages.len();
    stats.naadf_regions = runtime.scene.regions.len();
    stats.last_scene_build_ms = result.build_time.as_secs_f32() * 1_000.0;
    if !result.pages.is_empty() {
        runtime.scene_publication = Some(result);
    }
}

fn publish_scene_page(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
    result: &mut SceneBuildResult,
    page: IVec3,
    descriptors: Vec<SurfaceProbeDescriptor>,
) {
    let publish_started = Instant::now();

    let old_count = field.probes.len();
    let mut old_affected = HashMap::new();
    {
        if let Some(voxels) = field.pages.remove(&page) {
            for voxel in voxels {
                if let Some(probe) = field.probes.remove(&voxel) {
                    let (faces, samples, covered, settled) =
                        probe_metrics(&probe, field.transport_revision);
                    runtime.active_faces = runtime.active_faces.saturating_sub(faces);
                    runtime.completed_samples = runtime.completed_samples.saturating_sub(samples);
                    runtime.covered_faces = runtime.covered_faces.saturating_sub(covered);
                    runtime.settled_faces = runtime.settled_faces.saturating_sub(settled);
                    old_affected.insert(voxel, probe);
                }
            }
        }
    }

    // Read the pre-edit field before moving any descriptors. In particular,
    // newly seeded faces must never become donors later in this same rebuild.
    let now = Instant::now();
    let mut seeds = HashMap::new();
    if old_count > 0 {
        for descriptor in &descriptors {
            let old_mask = old_affected
                .get(&descriptor.voxel)
                .map_or(0, |probe| probe.face_mask);
            let new_faces = descriptor.face_mask & !old_mask;
            if new_faces == 0 {
                continue;
            }
            let mut display = [history::FaceDisplay::default(); 6];
            for (face, value) in display.iter_mut().enumerate() {
                if new_faces & (1 << face) != 0 {
                    *value = history::seed_face(
                        &runtime.scene,
                        descriptor.voxel,
                        face,
                        runtime.source_revision,
                        now,
                        |voxel| {
                            old_affected
                                .get(&voxel)
                                .map(Box::as_ref)
                                .or_else(|| field.probes.get(&voxel))
                        },
                    );
                }
            }
            seeds.insert(descriptor.voxel, (new_faces, display));
        }
    }
    let mut invalidated = HashSet::new();
    let mut added = Vec::new();
    let mut face_topology_changed = false;
    for descriptor in descriptors {
        let mut probe = old_affected.remove(&descriptor.voxel).unwrap_or_else(|| {
            invalidated.insert(descriptor.voxel);
            added.push(descriptor.voxel);
            let mut probe = SurfaceProbe::new(descriptor.voxel, descriptor.face_mask);
            probe.revision = runtime.generation << 32;
            Box::new(probe)
        });
        let old_mask = probe.face_mask;
        let display_changed = invalidated.contains(&descriptor.voxel)
            || old_mask != descriptor.face_mask
            || probe.material != descriptor.material
            || probe.emission != runtime.materials.get(descriptor.material).emission;
        if probe.material != descriptor.material {
            probe.invalidate();
            invalidated.insert(descriptor.voxel);
        }
        probe.material = descriptor.material;
        probe.emission = runtime.materials.get(descriptor.material).emission;
        probe.update_face_mask(descriptor.face_mask);
        if let Some((mask, display)) = seeds.remove(&descriptor.voxel) {
            for (face, value) in display.into_iter().enumerate() {
                if mask & (1 << face) != 0 {
                    probe.display[face] = value;
                }
            }
        }
        if old_mask != descriptor.face_mask {
            face_topology_changed = true;
            invalidated.insert(descriptor.voxel);
        }
        let (faces, samples, covered, settled) = probe_metrics(&probe, field.transport_revision);
        runtime.active_faces += faces;
        runtime.completed_samples += samples;
        runtime.covered_faces += covered;
        runtime.settled_faces += settled;
        field
            .pages
            .entry(page_of(descriptor.voxel))
            .or_default()
            .push(descriptor.voxel);
        if display_changed {
            runtime.gpu_value_dirty.insert(descriptor.voxel);
        }
        field.probes.insert_box(descriptor.voxel, probe);
    }
    let topology_changed =
        old_count != field.probes.len() || !old_affected.is_empty() || !added.is_empty();
    if topology_changed {
        runtime.gpu_removed.extend(old_affected.keys().copied());
        runtime.gpu_added.extend(added.iter().copied());
        field.update_indices(&added, &old_affected);
    }
    if topology_changed || face_topology_changed {
        runtime.gpu_topology_dirty = true;
        runtime.gpu_topology_version = runtime.gpu_topology_version.wrapping_add(1);
    }

    let mut urgent = HashSet::new();
    for &voxel in &result.exact_edits {
        let mut nearby = HashSet::new();
        invalidate_near_voxel(runtime, field, voxel, EDIT_RELIGHT_RADIUS, &mut nearby);
        urgent.extend(nearby.iter().copied());
        invalidated.extend(nearby);
    }
    let relit_probes = invalidated.len();
    let mut urgent: Vec<_> = urgent.into_iter().collect();
    urgent.sort_unstable_by_key(|voxel| {
        result
            .exact_edits
            .iter()
            .map(|edit| (voxel.as_i64vec3() - edit.as_i64vec3()).length_squared())
            .min()
            .unwrap_or(0)
    });
    enqueue_sky(runtime, urgent.iter().copied());
    enqueue_sky(runtime, added.iter().copied());
    enqueue_urgent(runtime, urgent);
    runtime.sky_remaining = field.order.len();
    if old_count == 0 {
        enqueue_many(runtime, field.order.iter().copied());
    } else {
        enqueue_many(runtime, invalidated);
    }
    if old_count > 0 {
        request_global_refresh(runtime, field.order.len());
    }
    result.exact_edits.clear();
    field.publishing_pages.remove(&page);
    update_stats_from_counters(runtime, field, stats);
    stats.loaded_pages = runtime.loaded_pages.len();
    stats.naadf_regions = runtime.scene.regions.len();
    stats.last_scene_build_ms = result.build_time.as_secs_f32() * 1_000.0;
    if diagnostics::terminal_stats_enabled() {
        info!(
            loaded_pages = stats.loaded_pages,
            naadf_regions = stats.naadf_regions,
            surface_probes = stats.surface_probes,
            pending_probes = stats.pending_probes,
            relit_probes,
            coverage_percent = format_args!("{:.1}", stats.interactive_coverage * 100.0),
            bootstrap_percent = format_args!("{:.1}", stats.bootstrap_coverage * 100.0),
            build_ms = format_args!("{:.1}", stats.last_scene_build_ms),
            publish_ms = format_args!("{:.1}", publish_started.elapsed().as_secs_f64() * 1000.0),
            "incremental world-space voxel transport scene rebuilt"
        );
    }
}

fn invalidate_near_voxel(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    center: IVec3,
    radius: i32,
    invalidated: &mut HashSet<IVec3>,
) {
    let page_radius = (radius + PAGE_EDGE as i32 - 1) / PAGE_EDGE as i32;
    let center_page = page_of(center);
    let radius_squared = i64::from(radius) * i64::from(radius);
    let mut candidates = Vec::new();
    for z in -page_radius..=page_radius {
        for y in -page_radius..=page_radius {
            for x in -page_radius..=page_radius {
                if let Some(voxels) = field.pages.get(&(center_page + IVec3::new(x, y, z))) {
                    candidates.extend(voxels.iter().copied());
                }
            }
        }
    }
    for voxel in candidates {
        let delta = voxel.as_i64vec3() - center.as_i64vec3();
        if delta.length_squared() > radius_squared {
            continue;
        }
        if let Some(probe) = field.probes.get_mut(&voxel) {
            let (_, samples, covered, settled) = probe_metrics(probe, field.transport_revision);
            probe.request_relight();
            let (_, next_samples, next_covered, next_settled) =
                probe_metrics(probe, field.transport_revision);
            runtime.completed_samples =
                runtime.completed_samples.saturating_sub(samples) + next_samples;
            runtime.covered_faces = runtime.covered_faces.saturating_sub(covered) + next_covered;
            runtime.settled_faces = runtime.settled_faces.saturating_sub(settled) + next_settled;
            // Nothing visible changed yet: do not upload every receiver in the
            // edit radius merely because its measurements now need refreshing.
            invalidated.insert(voxel);
        }
    }
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

fn append_sky_work(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
    selected: &mut TraceWork,
) -> usize {
    // Deduplicate before copying a receiver, including overlaps with transport.
    // Do not inherit candidates rejected by transport's due-face filter.
    let seen = &mut runtime.selection_seen;
    seen.clear();
    seen.extend(selected.probes.iter().map(|probe| probe.voxel));
    let mut count = 0;
    for _ in 0..runtime.sky_batch_size {
        let voxel = if let Some(voxel) = runtime.sky_pending.pop_front() {
            runtime.sky_queued.remove(&voxel);
            voxel
        } else if runtime.sky_remaining > 0 && !field.order.is_empty() {
            let voxel = field.order[runtime.sky_cursor % field.order.len()];
            runtime.sky_cursor = (runtime.sky_cursor + 1) % field.order.len();
            runtime.sky_remaining -= 1;
            voxel
        } else {
            break;
        };
        if seen.insert(voxel)
            && let Some(probe) = field.probes.get(&voxel)
        {
            count += 1;
            selected.push(
                probe,
                0,
                probe.transport_revision,
                probe.source_revision,
                false,
                geometry::UNTRACKED_HISTORY,
                u32::MAX,
            );
        }
    }
    count
}

fn start_trace_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    gpu_device: Option<Res<gpu::GpuTransportDevice>>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope();
    if runtime.trace_task.is_some() || field.order.is_empty() {
        return;
    }
    let selection_started = Instant::now();
    let requested = std::mem::take(&mut field.requested);
    enqueue_urgent(
        &mut runtime,
        requested
            .into_iter()
            .filter(|voxel| field.probes.contains_key(voxel)),
    );
    let focus = cameras.iter().next().map_or(IVec3::ZERO, |camera| {
        page_of(camera.translation().floor().as_ivec3())
    });
    update_focus(&mut runtime, &field, focus);
    runtime.batch_size = runtime.batch_size.min(settings.batch_limit());
    runtime.sky_batch_size = runtime.sky_batch_size.min(settings.batch_limit());
    // Independent work queues, one worker/submission. Cached sun lookups must
    // never consume the frames in which indirect transport could make progress.
    let mut selected = select_transport(&mut runtime, &field, &settings);
    let transport_count = selected.probes.len();
    let sky_count = append_sky_work(&mut runtime, &field, &mut selected);
    let sky_only = transport_count == 0;
    if selected.probes.is_empty() {
        runtime.trace_scratch = selected;
        stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
        stats.idle_scheduler_frames += 1;
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    }
    let paths = selected.faces.len() as u64 * u64::from(SAMPLES_PER_DISPATCH);
    stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
    let scene = Arc::clone(&runtime.scene);
    let scene_generation = runtime.generation;
    let sky_revision = runtime.sky_revision;
    let sources = runtime.sources.clone();
    let materials = Arc::clone(&runtime.materials);
    let pool = Arc::clone(&runtime.trace_pool);
    let mut gpu_tracer = runtime.gpu_tracer.take();
    let emission_cache = runtime.emission_cache.take();
    let mut cpu_sky_cache = runtime.cpu_sky_cache.take().unwrap_or_default();
    let gpu_device = gpu_device
        .filter(|_| !runtime.gpu_failed)
        .map(|device| device.clone());
    runtime.trace_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let started = Instant::now();
        let emission_cache = if sky_only {
            emission_cache
        } else {
            Some(emission::EmissionCache::prepare(
                emission_cache,
                Arc::clone(&scene),
                Arc::clone(&materials),
            ))
        };
        let emitters = emission_cache.as_ref().map_or_else(
            || Arc::new(emission::EmissionDistribution::default()),
            |cache| Arc::clone(&cache.distribution),
        );
        let mut gpu_error = None;
        if gpu_tracer.is_none()
            && let Some(device) = gpu_device
        {
            match gpu::GpuTracer::new(device) {
                Ok(tracer) => gpu_tracer = Some(tracer),
                Err(error) => gpu_error = Some(error),
            }
        }
        let mut work = selected;
        if let Some(tracer) = gpu_tracer.as_mut()
            && let Err(error) = tracer.trace_compact(
                Arc::clone(&scene),
                &sources,
                &materials,
                &mut work,
                SAMPLES_PER_DISPATCH,
                &emitters,
                sky_only,
            )
        {
            gpu_error = Some(error);
            gpu_tracer = None;
        }
        if gpu_tracer.is_none() {
            // A GPU-only session needs no extra CPU tracing threads. Create the
            // bounded fallback pool on its worker only if it is actually used.
            let pool = pool.get_or_init(|| {
                let available = std::thread::available_parallelism().map_or(2, usize::from);
                let worker_count = available.saturating_sub(2).clamp(1, 2);
                ThreadPoolBuilder::new()
                    .num_threads(worker_count)
                    .thread_name(|index| format!("voxel-gi-{index}"))
                    .build()
                    .expect("voxel GI worker pool")
            });
            pool.install(|| {
                trace_work_batch(
                    &scene,
                    &sources,
                    &materials,
                    &mut work,
                    if sky_only { 0 } else { SAMPLES_PER_DISPATCH },
                    &emitters,
                    &mut cpu_sky_cache,
                )
            });
        }
        TraceBatchResult {
            work,
            paths,
            trace_time: started.elapsed(),
            gpu_tracer,
            emission_cache,
            cpu_sky_cache,
            gpu_error,
            sky_only,
            transport_count,
            sky_count,
            scene_generation,
            sky_revision,
        }
    }));
}

fn update_focus(runtime: &mut VoxelGiRuntime, field: &VoxelEnergyField, focus: IVec3) {
    if runtime.focus_page == Some(focus) && runtime.focus_topology == runtime.gpu_topology_version {
        return;
    }
    runtime.focus_page = Some(focus);
    runtime.focus_topology = runtime.gpu_topology_version;
    runtime.nearby.clear();
    let mut pages: Vec<_> = field
        .pages
        .keys()
        .copied()
        .filter(|page| (*page - focus).abs().max_element() <= 1)
        .collect();
    pages.sort_unstable_by_key(|page| ((*page - focus).length_squared(), page.z, page.y, page.x));
    for page in pages {
        runtime.nearby.extend_from_slice(&field.pages[&page]);
    }
    runtime.nearby_cursor = 0;
}

/// At least one quarter of dispatch slots belongs to a fair loaded-world
/// sweep. Edits, initial coverage and the camera cannot starve distant pages.
#[cfg(test)]
fn select_probes(runtime: &mut VoxelGiRuntime, field: &VoxelEnergyField) -> Vec<SurfaceProbe> {
    // Reference view for existing policy tests. Production never materializes
    // these full receiver copies; it transfers only TraceWork.
    let work = select_transport(runtime, field, &VoxelGiSettings::default());
    work.probes
        .iter()
        .map(|job| {
            let mut probe = field.probes[&job.voxel].clone();
            if probe.transport_revision != job.transport_revision {
                probe.transport_revision = job.transport_revision;
                probe.fresh_mask = 0;
                probe.stable_batches = [0; 6];
            }
            if job.restart {
                probe.restart_measurements();
            }
            probe.source_revision = job.source_revision;
            probe.trace_mask = job.trace_mask;
            for face in &work.faces[job.faces()] {
                face.write(&mut probe);
            }
            probe
        })
        .collect()
}

fn select_transport(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
    settings: &VoxelGiSettings,
) -> TraceWork {
    let now = Instant::now();
    let mut selected = std::mem::take(&mut runtime.trace_scratch);
    selected.clear();
    selected.probes.reserve(runtime.batch_size);
    selected.faces.reserve(runtime.batch_size);
    let mut seen = std::mem::take(&mut runtime.selection_seen);
    seen.clear();
    seen.reserve(runtime.batch_size);
    let mut adaptive = std::mem::take(&mut runtime.ranked_candidates);
    adaptive.clear();
    if runtime.urgent_pending.len() + runtime.pending.len() < runtime.batch_size {
        adaptive.refresh(
            field,
            &runtime.nearby,
            &mut runtime.adaptive_cursor,
            now,
            runtime.batch_size,
        );
    }
    // Search capacity is bounded independently of emitted work. In particular,
    // resting neighbors are holes to skip, not GPU work that used the budget.
    for index in 0..settings.batch_limit().min(field.order.len()) {
        if selected.probes.len() >= runtime.batch_size {
            break;
        }
        let mut voxel = None;
        let mut force = false;
        if index % 4 != 0 {
            if let Some(next) = runtime.urgent_pending.pop_front() {
                runtime.urgent_queued.remove(&next);
                voxel = Some(next);
                force = true;
            } else if let Some(next) = runtime.pending.pop_front() {
                runtime.queued.remove(&next);
                voxel = Some(next);
            } else {
                while let Some(candidate) = adaptive.pop() {
                    if !seen.contains(&candidate) {
                        voxel = Some(candidate);
                        break;
                    }
                }
                if voxel.is_none() && !runtime.nearby.is_empty() {
                    voxel = Some(runtime.nearby[runtime.nearby_cursor % runtime.nearby.len()]);
                    runtime.nearby_cursor = (runtime.nearby_cursor + 1) % runtime.nearby.len();
                }
            }
        }
        let voxel = voxel.unwrap_or_else(|| {
            let next = field.order[runtime.refresh_cursor % field.order.len()];
            runtime.refresh_cursor = (runtime.refresh_cursor + 1) % field.order.len();
            runtime.refresh_remaining = runtime.refresh_remaining.saturating_sub(1);
            next
        });
        if !seen.insert(voxel) {
            continue;
        }
        // A reserved sweep slot can encounter an explicitly requested voxel
        // before its urgent queue entry. Honor the force now, not after it has
        // already been selected (or skipped as settled).
        force |= runtime.urgent_queued.contains(&voxel);
        let Some(probe) = field.probes.get(&voxel) else {
            continue;
        };
        let mask = scheduling::due_faces(probe, field.transport_revision, now, settings, force);
        if mask == 0 {
            continue;
        }
        let mut restart = false;
        let mut history_limit = u32::MAX;
        if probe.source_revision != runtime.source_revision {
            restart = probe.source_revision < runtime.reset_source_revision;
            history_limit = runtime.source_history.history_limit(
                probe.voxel,
                probe.source_revision,
                runtime.source_revision,
            );
        }
        selected.push(
            probe,
            mask,
            field.transport_revision,
            runtime.source_revision,
            restart,
            runtime.geometry_history.history_limit(
                voxel,
                probe.transport_revision,
                field.transport_revision,
            ),
            history_limit,
        );
    }
    runtime.selection_seen = seen;
    runtime.ranked_candidates = adaptive;
    selected
}

fn probe_has_pending_samples(probe: &SurfaceProbe) -> bool {
    (0..6).any(|face| probe.face_mask & (1 << face) != 0 && probe.samples[face] < BOOTSTRAP_SAMPLES)
}

fn apply_trace_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let mut timing = timings.main_scope();
    let Some(task) = runtime.trace_task.as_mut() else {
        return;
    };
    let Some(mut result) = block_on(poll_once(task)) else {
        return;
    };
    let merge_started = Instant::now();
    runtime.trace_task = None;
    timing.record_gpu_batch(
        result
            .gpu_tracer
            .as_ref()
            .and_then(|tracer| tracer.last_gpu_ms),
    );
    if let Some(error) = result.gpu_error {
        warn!(%error, "GPU transport unavailable; continuing with bounded CPU transport");
        runtime.gpu_failed = true;
    }
    stats.backend = if result.gpu_tracer.is_some() {
        "gpu"
    } else {
        "cpu"
    };
    let measured_ms = result
        .gpu_tracer
        .as_ref()
        .map_or(result.trace_time.as_secs_f32() * 1_000.0, |tracer| {
            tracer.last_dispatch_ms
        });
    let sky = result
        .gpu_tracer
        .as_ref()
        .map_or(result.cpu_sky_cache.stats, |tracer| tracer.sky_cache.stats);
    stats.sky_corner_requests += sky.requested as u64;
    stats.sky_corners_traced += sky.traced as u64;
    stats.sky_rays_scheduled += sky.rays as u64;
    stats.sky_cached_corners = sky.resident;
    if let Some(tracer) = &result.gpu_tracer {
        stats.last_prepare_ms = tracer.last_prepare_ms;
        stats.last_wait_ms = tracer.last_wait_ms;
        stats.last_decode_ms = tracer.last_decode_ms;
        stats.packed_geometry_bytes = tracer.geometry_storage_bytes();
    }
    runtime.gpu_tracer = result.gpu_tracer;
    runtime.emission_cache = result.emission_cache;
    runtime.cpu_sky_cache = Some(result.cpu_sky_cache);
    stats.last_worker_ms = result.trace_time.as_secs_f32() * 1000.0;
    // Mixed submissions share one timestamp interval. Do not attribute the
    // indirect kernel's cost to sky and accidentally collapse its scan budget.
    if result.sky_only && result.sky_count > 0 {
        runtime.sky_batch_size = scheduling::next_budget(
            runtime.sky_batch_size,
            result.sky_count,
            measured_ms,
            settings.budget_ms().min(1.0),
            settings.batch_limit(),
        );
    }
    if result.sky_only {
        stats.sky_only_batches += 1;
    } else {
        stats.transport_batches += 1;
    }
    let sky_current = result.scene_generation == runtime.generation
        && result.sky_revision == runtime.sky_revision;
    let mut requeue = Vec::new();
    let transport_revision = field.transport_revision;
    for job in &result.work.probes {
        let voxel = job.voxel;
        let Some(old) = field.probes.get_mut(&voxel) else {
            continue;
        };
        if old.revision != job.revision
            || (job.trace_mask != 0 && job.source_revision < runtime.reset_source_revision)
        {
            if probe_has_pending_samples(old) {
                requeue.push(voxel);
            }
            continue;
        }
        if sky_current {
            let changed = (0..6).any(|face| {
                old.face_mask & (1 << face) != 0 && old.display[face].sky != job.sky[face]
            });
            for face in 0..6 {
                if old.face_mask & (1 << face) != 0 {
                    old.display[face].sky = job.sky[face];
                }
            }
            if changed {
                runtime.gpu_value_dirty.insert(voxel);
                runtime.sky_upload_pending = true;
            }
        }
        if job.trace_mask == 0 {
            continue;
        }
        let (_, old_samples, old_covered, old_settled) = probe_metrics(old, transport_revision);
        // A material-table revision can change emission without rebuilding
        // geometry. Older source results were rejected above.
        old.emission = runtime.materials.get(old.material).emission;
        job.publish(
            &result.work.faces[job.faces()],
            old,
            merge_started,
            &settings,
        );
        let (_, new_samples, new_covered, new_settled) = probe_metrics(old, transport_revision);
        runtime.completed_samples =
            runtime.completed_samples.saturating_sub(old_samples) + new_samples;
        runtime.covered_faces = runtime.covered_faces.saturating_sub(old_covered) + new_covered;
        runtime.settled_faces = runtime.settled_faces.saturating_sub(old_settled) + new_settled;
        let pending = probe_has_pending_samples(old);
        runtime.gpu_value_dirty.insert(voxel);
        if pending {
            requeue.push(voxel);
        }
    }
    result.work.clear();
    runtime.trace_scratch = result.work;
    enqueue_many(&mut runtime, requeue);
    stats.traced_paths = stats.traced_paths.saturating_add(result.paths);
    stats.last_merge_ms = merge_started.elapsed().as_secs_f32() * 1000.0;
    // Adjust work to measured execution time rather than fixing throughput to
    // 1,024 probes per display frame. Limit growth to avoid sudden GPU bursts.
    if !result.sky_only {
        stats.last_trace_batch_ms = measured_ms;
        runtime.batch_size = scheduling::next_budget(
            runtime.batch_size,
            result.transport_count,
            measured_ms,
            settings.budget_ms(),
            settings.batch_limit(),
        );
    }
    update_stats_from_counters(&runtime, &field, &mut stats);
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

fn upload_transport_cache(
    mut runtime: ResMut<VoxelGiRuntime>,
    field: Res<VoxelEnergyField>,
    mut assets: ResMut<VoxelTransportAssets>,
    mut patch_queue: ResMut<GpuProbePatchQueue>,
    mut cache: ResMut<GpuTransportCache>,
    material_handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope();
    let Some(material_handle) = material_handle else {
        return;
    };

    let mut installed = false;
    if let Some(task) = runtime.gpu_task.as_mut()
        && let Some(build) = block_on(poll_once(task))
    {
        runtime.gpu_task = None;
        let catchup = std::mem::take(&mut runtime.gpu_catchup_dirty);
        runtime.gpu_value_dirty.extend(catchup);
        // Install the snapshot even if streaming continued while it encoded;
        // the delta ledger below brings it current. Repeatedly discarding it
        // would prevent a fast-moving player from ever getting a cache.
        install_gpu_cache(
            build,
            &mut cache,
            &mut patch_queue,
            &mut assets,
            &mut buffers,
            &mut materials,
            &material_handle.handle,
        );
        runtime.gpu_rebuild_required = false;
        installed = true;
        runtime.gpu_topology_dirty = runtime.gpu_task_version != runtime.gpu_topology_version;
        runtime.last_gpu_upload = Instant::now();
        info!(
            probes = field.probe_count(),
            "world-space voxel GPU cache installed"
        );
    }

    if runtime.gpu_topology_dirty
        && cache.ready
        && runtime.gpu_task.is_none()
        && runtime.gpu_snapshot.is_none()
        && !runtime.gpu_rebuild_required
    {
        let added = std::mem::take(&mut runtime.gpu_added);
        let removed = std::mem::take(&mut runtime.gpu_removed);
        if patch_gpu_topology(
            &mut cache,
            &patch_queue,
            &field.probes,
            &removed,
            &added,
            &runtime.gpu_value_dirty,
        ) {
            runtime.gpu_topology_dirty = false;
            // The transaction included these values. No rebuild is in flight
            // on this path, so there is no replacement catch-up ledger to keep.
            runtime.gpu_value_dirty.clear();
        } else {
            // Keep rendering the old cache while the larger allocation builds.
            runtime.gpu_rebuild_required = true;
        }
    }

    if runtime.gpu_topology_dirty
        && (!cache.ready || runtime.gpu_rebuild_required)
        && runtime.gpu_task.is_none()
        && runtime.gpu_snapshot.is_none()
    {
        runtime.gpu_snapshot = Some(GpuSnapshot::new(&field.order));
        runtime.gpu_added.clear();
        runtime.gpu_removed.clear();
        runtime.gpu_task_version = runtime.gpu_topology_version;
    }

    if let Some(snapshot) = runtime.gpu_snapshot.as_mut()
        && snapshot.advance(&field.probes)
    {
        let snapshot = runtime.gpu_snapshot.take().unwrap().finish();
        runtime.gpu_task =
            Some(AsyncComputeTaskPool::get().spawn(async move { build_gpu_cache(snapshot) }));
    }

    // Never expose a worker's stale snapshot for the normal upload interval.
    // Merge every value changed since that snapshot in its installation frame.
    if !cache.ready
        || (!installed
            && !runtime.sky_upload_pending
            && runtime.last_gpu_upload.elapsed() < settings.display_upload_interval)
    {
        return;
    }
    let dirty = std::mem::take(&mut runtime.gpu_value_dirty);
    patch_probe_values(&cache, &patch_queue, &field.probes, &dirty);
    if runtime.gpu_task.is_some() || runtime.gpu_snapshot.is_some() {
        // Catch-up and live publication are separate ledgers. Re-uploading the
        // entire growing catch-up set every 50ms was itself a streaming spike.
        runtime.gpu_catchup_dirty.extend(dirty.iter().copied());
    }
    // Reuse the high-water allocation instead of rebuilding a growing hash set
    // every display upload. Catch-up owns only keys, never this scratch storage.
    runtime.gpu_value_dirty = dirty;
    runtime.gpu_value_dirty.clear();
    runtime.last_gpu_upload = Instant::now();
    runtime.sky_upload_pending = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settled_counter_follows_accepted_transport_not_sky_only_or_rejected_batches() {
        use bevy::ecs::system::RunSystemOnce;
        let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.samples = [256; 6];
        probe.stable_batches = [3; 6];
        let mut field = VoxelEnergyField::default();
        field.probes.insert(IVec3::ZERO, probe);
        field.update_indices(&[IVec3::ZERO], &HashMap::new());
        let mut world = World::new();
        world.insert_resource(field);
        world.insert_resource(VoxelGiRuntime {
            active_faces: 1,
            covered_faces: 1,
            completed_samples: 8,
            ..default()
        });
        world.init_resource::<VoxelGiStats>();
        world.init_resource::<VoxelGiSettings>();
        world.init_resource::<GiWorkTimings>();
        for (case, expected) in [(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0)] {
            let mut probe = world.resource::<VoxelEnergyField>().probes[&IVec3::ZERO].clone();
            probe.trace_mask = 4;
            if case == 1 {
                probe.trace_mask = 0;
                probe.stable_batches = [0; 6];
            } else if case == 2 {
                probe.revision += 1;
                probe.stable_batches = [0; 6];
            } else if case == 3 {
                probe.radiance[2] = EnergySpectrum::ONE;
                probe.second_moment[2] = EnergySpectrum::ONE;
            }
            let task = pool.spawn(async move {
                TraceBatchResult {
                    work: TraceWork::from_probes(&[probe], case == 1),
                    paths: 4,
                    trace_time: Duration::from_millis(1),
                    gpu_tracer: None,
                    emission_cache: None,
                    cpu_sky_cache: default(),
                    gpu_error: None,
                    sky_only: case == 1,
                    transport_count: usize::from(case != 1),
                    sky_count: usize::from(case == 1),
                    scene_generation: 0,
                    sky_revision: 0,
                }
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while !task.is_finished() {
                assert!(Instant::now() < deadline, "test batch did not finish");
                std::thread::yield_now();
            }
            world.resource_mut::<VoxelGiRuntime>().trace_task = Some(task);
            world.run_system_once(apply_trace_task).unwrap();
            assert_eq!(
                world.resource::<VoxelGiStats>().settled_coverage,
                expected,
                "case {case}"
            );
        }
    }

    #[test]
    fn settled_counter_tracks_local_edits_global_revisions_and_page_removal() {
        let mut field = VoxelEnergyField::default();
        let mut runtime = VoxelGiRuntime::default();
        let mut stats = VoxelGiStats::default();
        let far = IVec3::new(128, 0, 0);
        for voxel in [IVec3::ZERO, far] {
            let mut probe = SurfaceProbe::new(voxel, 5);
            probe.samples = [128; 6];
            probe.stable_batches = [4; 6];
            let (faces, samples, covered, settled) = probe_metrics(&probe, 0);
            runtime.active_faces += faces;
            runtime.completed_samples += samples;
            runtime.covered_faces += covered;
            runtime.settled_faces += settled;
            field.probes.insert(voxel, probe);
            field.pages.insert(page_of(voxel), vec![voxel]);
        }
        field.update_indices(&[IVec3::ZERO, far], &HashMap::new());
        update_stats_from_counters(&runtime, &field, &mut stats);
        assert_eq!(stats.settled_coverage, 1.0);
        invalidate_near_voxel(
            &mut runtime,
            &mut field,
            IVec3::ZERO,
            48,
            &mut HashSet::new(),
        );
        update_stats_from_counters(&runtime, &field, &mut stats);
        assert_eq!(stats.settled_coverage, 0.5);
        assert_eq!(stats.interactive_coverage, 1.0);
        assert_eq!(stats.bootstrap_coverage, 1.0);
        assert_eq!(field.probes[&IVec3::ZERO].samples[0], 16);

        // O(1) invalidation must not scan/clear histories. Old contributions no
        // longer count, including late worker results from that old revision.
        invalidate_sources(&mut runtime, &mut field, &mut stats);
        assert_eq!(stats.settled_coverage, 0.0);
        assert_eq!(field.probes[&far].samples[0], 128);
        assert_eq!(
            probe_metrics(&field.probes[&far], field.transport_revision).3,
            0
        );
        let current = field.probes.get_mut(&IVec3::ZERO).unwrap();
        current.transport_revision = field.transport_revision;
        current.samples = [128; 6];
        current.fresh_mask = current.face_mask;
        current.stable_batches = [4; 6];
        runtime.settled_faces = probe_metrics(current, field.transport_revision).3;
        let mut result = SceneBuildResult {
            scene: runtime.scene.clone(),
            changed: true,
            geometry_change: Some(default()),
            changed_materials: [false; 256],
            pages: VecDeque::new(),
            exact_edits: vec![],
            build_time: Duration::ZERO,
        };
        // Removing old-revision terrain must not subtract freshly settled faces
        // elsewhere. Republishing an unchanged current page must count it once.
        publish_scene_page(
            &mut runtime,
            &mut field,
            &mut stats,
            &mut result,
            page_of(far),
            vec![],
        );
        assert_eq!(runtime.settled_faces, 2);
        assert_eq!(stats.settled_coverage, 1.0);
        publish_scene_page(
            &mut runtime,
            &mut field,
            &mut stats,
            &mut result,
            IVec3::ZERO,
            vec![SurfaceProbeDescriptor {
                voxel: IVec3::ZERO,
                material: 0,
                face_mask: 5,
            }],
        );
        assert_eq!(runtime.settled_faces, 2);
        publish_scene_page(
            &mut runtime,
            &mut field,
            &mut stats,
            &mut result,
            IVec3::ZERO,
            vec![],
        );
        assert_eq!(runtime.settled_faces, 0);
        assert_eq!(stats.settled_coverage, 0.0);
    }

    #[test]
    fn scene_publication_preserves_other_pages_and_publishes_one_page_per_tick() {
        use bevy::ecs::system::RunSystemOnce;
        let mut field = VoxelEnergyField::default();
        let old = IVec3::ZERO;
        let retained = IVec3::new(32, 0, 0);
        for voxel in [old, retained] {
            field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
            field.pages.insert(page_of(voxel), vec![voxel]);
        }
        let retained_address = field.probes.get(&retained).unwrap() as *const SurfaceProbe;
        field.update_indices(&[old, retained], &HashMap::new());
        field.publishing_pages.extend([IVec3::ZERO, IVec3::X]);
        let added = IVec3::Y;
        let result = SceneBuildResult {
            scene: Arc::new(TransportScene::default()),
            changed: true,
            geometry_change: Some(default()),
            changed_materials: [false; 256],
            pages: VecDeque::from([
                (
                    IVec3::ZERO,
                    vec![SurfaceProbeDescriptor {
                        voxel: added,
                        material: 0,
                        face_mask: 4,
                    }],
                ),
                (
                    IVec3::X,
                    vec![SurfaceProbeDescriptor {
                        voxel: retained,
                        material: 0,
                        face_mask: 4,
                    }],
                ),
            ]),
            exact_edits: vec![],
            build_time: Duration::ZERO,
        };
        let mut world = World::new();
        world.insert_resource(field);
        world.insert_resource(VoxelGiRuntime {
            scene_publication: Some(result),
            ..default()
        });
        world.insert_resource(VoxelGiStats::default());
        world.init_resource::<GiWorkTimings>();
        world.insert_resource(VoxelGiSettings {
            scene_publish_budget: Duration::ZERO,
            ..default()
        });
        world.run_system_once(apply_scene_task).unwrap();
        let field = world.resource::<VoxelEnergyField>();
        assert!(!field.probes.contains_key(&old));
        assert!(field.probes.contains_key(&added));
        assert!(field.probes.contains_key(&retained));
        assert_eq!(field.publishing_pages, HashSet::from([IVec3::X]));
        for (index, voxel) in field.order.iter().enumerate() {
            assert_eq!(field.probes[voxel].order_index, index);
        }
        world.run_system_once(apply_scene_task).unwrap();
        assert_eq!(
            world
                .resource::<VoxelEnergyField>()
                .probes
                .get(&retained)
                .unwrap() as *const SurfaceProbe,
            retained_address,
            "unchanged probes must keep their allocations across page publication",
        );
        assert!(
            !world
                .resource::<VoxelGiRuntime>()
                .gpu_value_dirty
                .contains(&retained),
            "unchanged page receivers do not need display uploads"
        );
        assert!(
            world
                .resource::<VoxelEnergyField>()
                .publishing_pages
                .is_empty()
        );
        assert!(
            world
                .resource::<VoxelGiRuntime>()
                .scene_publication
                .is_none()
        );
    }

    #[test]
    fn publication_batches_cheap_pages_without_unbounded_display_work() {
        use bevy::ecs::system::RunSystemOnce;
        let pages: VecDeque<_> = (0..6)
            .map(|x| {
                let page = IVec3::new(x, 0, 0);
                (
                    page,
                    vec![SurfaceProbeDescriptor {
                        voxel: page * PAGE_EDGE as i32,
                        material: 0,
                        face_mask: 4,
                    }],
                )
            })
            .collect();
        let mut field = VoxelEnergyField::default();
        field
            .publishing_pages
            .extend(pages.iter().map(|(page, _)| *page));
        let mut world = World::new();
        world.insert_resource(field);
        world.insert_resource(VoxelGiRuntime {
            scene_publication: Some(SceneBuildResult {
                scene: Arc::new(TransportScene::default()),
                changed: true,
                geometry_change: Some(default()),
                changed_materials: [false; 256],
                pages,
                exact_edits: vec![],
                build_time: Duration::ZERO,
            }),
            ..default()
        });
        world.insert_resource(VoxelGiStats::default());
        world.init_resource::<GiWorkTimings>();
        world.insert_resource(VoxelGiSettings {
            scene_publish_budget: Duration::MAX,
            ..default()
        });
        world.run_system_once(apply_scene_task).unwrap();
        let published = world.resource::<VoxelEnergyField>().probe_count();
        assert!(
            published <= 4,
            "downstream topology patches have a page-count cap"
        );
        // A delayed test runner may exhaust the wall-clock budget sooner; every
        // invocation still advances, and the queue must preserve all six pages.
        assert!(published > 0);
        for _ in 0..6 {
            world.run_system_once(apply_scene_task).unwrap();
        }
        let field = world.resource::<VoxelEnergyField>();
        assert_eq!(field.probe_count(), 6);
        assert!(field.publishing_pages.is_empty());
        assert!(
            world
                .resource::<VoxelGiRuntime>()
                .scene_publication
                .is_none()
        );
    }

    #[test]
    fn urgent_work_cannot_starve_loaded_world_sweep() {
        let mut field = VoxelEnergyField::default();
        for x in 0..64 {
            let voxel = IVec3::new(x, 0, 0);
            field.order.push(voxel);
            field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        }
        let mut runtime = VoxelGiRuntime {
            batch_size: 8,
            nearby: vec![IVec3::ZERO],
            ..default()
        };
        let mut visited = HashSet::new();
        for _ in 0..32 {
            request_global_refresh(&mut runtime, field.order.len());
            enqueue_urgent(&mut runtime, (0..6).map(|x| IVec3::new(x, 0, 0)));
            visited.extend(
                select_probes(&mut runtime, &field)
                    .into_iter()
                    .map(|probe| probe.voxel),
            );
        }
        assert_eq!(visited.len(), 64);
    }

    #[test]
    fn in_flight_lighting_cannot_restore_indices_moved_by_streaming() {
        let mut field = VoxelEnergyField::default();
        let voxels = [IVec3::ZERO, IVec3::X, IVec3::Y, IVec3::Z];
        for voxel in voxels {
            field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        }
        field.update_indices(&voxels, &HashMap::new());
        let mut pending_result = field.probes[&IVec3::Z].clone();
        let mut removed = HashMap::new();
        removed.insert(IVec3::ZERO, field.probes.remove(&IVec3::ZERO).unwrap());
        field.update_indices(&[], &removed);
        assert_eq!(field.probes[&IVec3::Z].order_index, 0);
        pending_result.preserve_topology_from(&field.probes[&IVec3::Z]);
        field.probes.insert(IVec3::Z, pending_result);
        removed.clear();
        removed.insert(IVec3::Z, field.probes.remove(&IVec3::Z).unwrap());
        field.update_indices(&[], &removed);
        assert_eq!(field.order.len(), field.probes.len());
        for (index, voxel) in field.order.iter().enumerate() {
            assert_eq!(field.probes[voxel].order_index, index);
        }
    }

    #[test]
    fn moving_sources_preserve_history_and_work_queues() {
        let voxel = IVec3::ZERO;
        let mut probe = SurfaceProbe::new(voxel, 4);
        probe.samples[2] = 64;
        probe.sequence[2] = 100;
        let mut field = VoxelEnergyField::default();
        field.probes.insert(voxel, probe);
        field.order.push(voxel);
        let mut runtime = VoxelGiRuntime::default();
        enqueue_many(&mut runtime, [voxel]);
        invalidate_sources(&mut runtime, &mut field, &mut VoxelGiStats::default());
        assert_eq!(field.probes[&voxel].samples[2], 64);
        assert_eq!(runtime.pending.len(), 1);
        let selected = select_probes(&mut runtime, &field);
        assert_eq!(selected[0].samples[2], 4);
        assert_eq!(selected[0].sequence[2], 100);
    }

    #[test]
    fn emitter_change_restarts_local_and_remote_measurements_not_presentation() {
        let mut field = VoxelEnergyField::default();
        for voxel in [IVec3::ZERO, IVec3::new(10_000, 0, 0)] {
            let mut probe = SurfaceProbe::new(voxel, 5);
            probe.samples = [128; 6];
            probe.radiance = [EnergySpectrum::ONE; 6];
            probe.second_moment = [EnergySpectrum::ONE; 6];
            probe.sequence = [1000; 6];
            probe.display[0].indirect = Vec3::ONE;
            field.probes.insert(voxel, probe);
            field.order.push(voxel);
        }
        let mut runtime = VoxelGiRuntime::default();
        invalidate_sources(&mut runtime, &mut field, &mut VoxelGiStats::default());
        runtime.reset_source_revision = runtime.source_revision;
        // A later moving-light revision must not hide the pending emitter reset.
        invalidate_sources(&mut runtime, &mut field, &mut VoxelGiStats::default());
        let selected = select_probes(&mut runtime, &field);
        assert_eq!(selected.len(), 2);
        for probe in selected {
            assert_eq!(probe.samples, [0; 6]);
            assert_eq!(probe.radiance, [EnergySpectrum::ZERO; 6]);
            assert_eq!(probe.second_moment, [EnergySpectrum::ZERO; 6]);
            assert_eq!(probe.display[0].indirect, Vec3::ONE);
            assert_eq!(probe.sequence, [1000; 6]);
            assert_eq!(probe.revision, 0, "worker result must remain mergeable");
            assert_eq!(probe.source_revision, runtime.source_revision);
        }
    }

    #[test]
    fn remote_local_light_motion_wakes_without_destroying_transport_history() {
        let voxel = IVec3::new(10_000, 0, 0);
        let mut probe = SurfaceProbe::new(voxel, 4);
        probe.samples[2] = 128;
        probe.sequence[2] = 1000;
        probe.updated_at = Some(Instant::now());
        probe.face_updated_at[2] = probe.updated_at;
        let mut field = VoxelEnergyField::default();
        field.probes.insert(voxel, probe);
        field.order.push(voxel);
        let mut runtime = VoxelGiRuntime::default();
        sources::invalidate_source_change(
            &mut runtime,
            &mut field,
            &mut VoxelGiStats::default(),
            sources::SourceChange::Local(vec![(Vec3::ZERO, 24.0)]),
        );
        let selected = select_probes(&mut runtime, &field);
        assert_eq!(selected[0].samples[2], 64);
        assert_eq!(selected[0].sequence[2], 1000);
        assert_eq!(selected[0].fresh_mask, 0);
        assert_eq!(selected[0].trace_mask, 4);
        assert!(!field.estimate(voxel, Vec3::Y).unwrap().transport_current);
        assert_eq!(field.probes[&voxel].samples[2], 128);
    }

    #[test]
    fn sky_queue_piggybacks_without_duplicating_rays_or_changing_history() {
        let initial = SurfaceProbe::new(IVec3::ZERO, 4);
        let mut transport = TraceWork::from_probes(std::slice::from_ref(&initial), false);
        let mut sky_only = SurfaceProbe::new(IVec3::X, 5);
        sky_only.samples = [128; 6];
        sky_only.sequence = [1000; 6];
        let duplicate = initial;
        let mut field = VoxelEnergyField::default();
        for probe in [duplicate, sky_only] {
            field.probes.insert(probe.voxel, probe);
        }
        let mut runtime = VoxelGiRuntime::default();
        // Includes an already-selected transport receiver and a repeated sky
        // candidate. Neither should allocate or overwrite another probe clone.
        runtime
            .sky_pending
            .extend([IVec3::ZERO, IVec3::X, IVec3::X]);
        assert_eq!(append_sky_work(&mut runtime, &field, &mut transport), 1);
        assert_eq!(transport.probes.len(), 2);
        assert_eq!(transport.probes[0].trace_mask, 4);
        assert_eq!(transport.probes[1].face_mask, 5);
        assert_eq!(transport.probes[1].trace_mask, 0);
        assert_eq!(
            transport.faces.len(),
            1,
            "sky-only work carries no transport history"
        );
        assert_eq!(field.probes[&IVec3::X].samples, [128; 6]);
        assert_eq!(field.probes[&IVec3::X].sequence, [1000; 6]);
    }

    #[test]
    fn dispatch_reuses_returned_storage_without_retaining_candidates() {
        let mut field = VoxelEnergyField::default();
        field.order.push(IVec3::ZERO);
        field
            .probes
            .insert(IVec3::ZERO, SurfaceProbe::new(IVec3::ZERO, 4));
        let mut runtime = VoxelGiRuntime {
            trace_scratch: TraceWork {
                probes: Vec::with_capacity(1024),
                faces: Vec::with_capacity(1024),
            },
            batch_size: 64,
            ..default()
        };
        let allocation = runtime.trace_scratch.probes.as_ptr();
        let face_allocation = runtime.trace_scratch.faces.as_ptr();
        for _ in 0..2 {
            let mut selected = select_transport(&mut runtime, &field, &VoxelGiSettings::default());
            assert_eq!(selected.probes.as_ptr(), allocation);
            assert_eq!(selected.faces.as_ptr(), face_allocation);
            assert_eq!(selected.probes.len(), 1);
            selected.clear();
            runtime.trace_scratch = selected;
        }
    }

    #[test]
    fn resting_candidates_do_not_consume_the_dispatch_allowance() {
        let now = Instant::now();
        let mut field = VoxelEnergyField::default();
        for x in 0..4096 {
            let voxel = IVec3::new(x, 0, 0);
            let mut probe = SurfaceProbe::new(voxel, 4);
            probe.samples[2] = 256;
            probe.stable_batches[2] = 4;
            probe.face_updated_at[2] = Some(now);
            field.order.push(voxel);
            field.probes.insert(voxel, probe);
        }
        let noisy = IVec3::new(3000, 0, 0);
        field.probes.get_mut(&noisy).unwrap().stable_batches[2] = 0;
        let mut runtime = VoxelGiRuntime {
            batch_size: 64,
            ..default()
        };
        // A previously requested bootstrap queue can contain newly settled
        // probes; this also places the distant target beyond the ranked window.
        enqueue_many(&mut runtime, field.order.iter().take(64).copied());
        let selected = select_probes(&mut runtime, &field);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected[0].voxel, noisy);
        assert_eq!(selected[0].samples[2], 256);
    }

    #[test]
    fn sky_comparison_is_display_only_and_independent_of_gi_toggle() {
        let mut app = App::new();
        let mut materials = Assets::<VoxelTransportMaterial>::default();
        let handle = materials.add(render::material_template(&VoxelTransportAssets {
            texture: default(),
            hash_entries: default(),
            probes: default(),
        }));
        let mut runtime = VoxelGiRuntime {
            source_revision: 7,
            generation: 9,
            ..default()
        };
        enqueue_urgent(&mut runtime, [IVec3::ZERO]);
        app.insert_resource(runtime)
            .insert_resource(materials)
            .insert_resource(VoxelWorldMaterialHandle {
                handle: handle.clone(),
            })
            .init_resource::<ButtonInput<KeyCode>>()
            .add_systems(Update, toggle_voxel_gi);
        let window = app
            .world_mut()
            .spawn((Window::default(), bevy::window::PrimaryWindow))
            .id();
        for (key, sky, gi) in [
            (KeyCode::F7, 0, 1),
            (KeyCode::F4, 0, 0),
            (KeyCode::F7, 1, 0),
            (KeyCode::F4, 1, 1),
        ] {
            let mut keyboard = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
            keyboard.reset_all();
            keyboard.press(key);
            app.update();
            let settings = app
                .world()
                .resource::<Assets<VoxelTransportMaterial>>()
                .get(&handle)
                .unwrap()
                .extension
                .settings;
            assert_eq!(settings.voxel_sky, sky);
            assert_eq!(settings.enabled, gi);
            assert_eq!(settings.reconstruction, 1);
            let runtime = app.world().resource::<VoxelGiRuntime>();
            assert_eq!(runtime.source_revision, 7);
            assert_eq!(runtime.generation, 9);
            assert_eq!(runtime.urgent_pending.len(), 1);
        }
        assert!(
            app.world()
                .get::<Window>(window)
                .unwrap()
                .title
                .contains("voxel corners")
        );
    }

    #[test]
    fn simulation_rejects_unmeasured_or_buried_faces_and_reports_stale_sources() {
        let voxel = IVec3::ZERO;
        let mut field = VoxelEnergyField::default();
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        assert!(field.estimate(voxel, Vec3::Y).is_none());
        let probe = field.probes.get_mut(&voxel).unwrap();
        probe.samples[2] = 8;
        probe.updated_at = Some(Instant::now());
        assert!(field.estimate(voxel, Vec3::Y).unwrap().sources_current);
        assert!(field.estimate(voxel, Vec3::X).is_none());
        field.source_revision += 1;
        assert!(!field.estimate(voxel, Vec3::Y).unwrap().sources_current);
        let probe = field.probes.get_mut(&voxel).unwrap();
        probe.sequence[2] = 100;
        probe.invalidate();
        assert_eq!(probe.sequence[2], 100);
        assert!(field.estimate(voxel, Vec3::Y).is_none());
    }

    #[test]
    fn partial_face_queries_keep_independent_age_and_revision_freshness() {
        let now = Instant::now();
        let voxel = IVec3::ZERO;
        let mut probe = SurfaceProbe::new(voxel, 5);
        probe.samples = [64; 6];
        probe.updated_at = Some(now);
        probe.face_updated_at[0] = Some(now - Duration::from_secs(10));
        probe.face_updated_at[2] = Some(now);
        probe.fresh_mask = 4;
        probe.transport_revision = 1;
        let mut field = VoxelEnergyField {
            transport_revision: 1,
            ..default()
        };
        field.probes.insert(voxel, probe);
        assert!(field.estimate(voxel, Vec3::Y).unwrap().transport_current);
        let old = field.estimate(voxel, Vec3::X).unwrap();
        assert!(!old.transport_current);
        assert!(!old.sources_current);
        assert!(old.age >= Duration::from_secs(10));
        field.transport_revision += 1;
        assert!(!field.estimate(voxel, Vec3::Y).unwrap().transport_current);
    }

    #[test]
    fn settled_scheduler_can_be_idle_and_far_geometry_invalidation_wakes_it() {
        let now = Instant::now();
        let mut field = VoxelEnergyField::default();
        for x in 0..64 {
            let voxel = IVec3::new(10_000 + x, 0, 0);
            let mut probe = SurfaceProbe::new(voxel, 5);
            probe.samples = [256; 6];
            probe.stable_batches = [4; 6];
            probe.updated_at = Some(now);
            probe.face_updated_at = [Some(now); 6];
            field.order.push(voxel);
            field.probes.insert(voxel, probe);
        }
        let mut runtime = VoxelGiRuntime::default();
        assert!(select_probes(&mut runtime, &field).is_empty());
        let requested = field.order[runtime.refresh_cursor % field.order.len()];
        enqueue_urgent(&mut runtime, [requested]);
        let forced = select_probes(&mut runtime, &field);
        assert_eq!(
            forced.len(),
            1,
            "the reserved sweep must not swallow a simulation refresh request"
        );
        assert_eq!(forced[0].voxel, requested);
        assert_eq!(forced[0].trace_mask, 5);
        field.transport_revision += 1;
        let selected = select_probes(&mut runtime, &field);
        assert!(!selected.is_empty());
        let mut visited: HashSet<_> = selected.iter().map(|probe| probe.voxel).collect();
        for _ in 0..8 {
            visited.extend(
                select_probes(&mut runtime, &field)
                    .iter()
                    .map(|probe| probe.voxel),
            );
        }
        assert_eq!(visited.len(), 64);
        assert!(
            selected.iter().all(|probe| probe.trace_mask == 5
                && probe.fresh_mask == 0
                && probe.samples[2] == 64)
        );
        assert!(
            field.probes.values().all(|probe| probe.samples[2] == 256),
            "scheduling must not mutate published histories"
        );
        enqueue_sky(&mut runtime, field.order.iter().copied());
        let mut sky = TraceWork::default();
        append_sky_work(&mut runtime, &field, &mut sky);
        assert_eq!(sky.probes.len(), 64);
        assert!(sky.faces.is_empty());
        assert!(field.probes.values().all(|probe| probe.samples[2] == 256));
    }

    #[test]
    fn energy_field_uses_directional_ambient_cube_weights() {
        let voxel = IVec3::new(2, 3, 4);
        let mut field = VoxelEnergyField::default();
        field.probes.insert(
            voxel,
            SurfaceProbe {
                voxel,
                face_mask: 0b11_1111,
                radiance: [
                    EnergySpectrum::new(Vec3::X, 1.0, 0.0),
                    EnergySpectrum::ZERO,
                    EnergySpectrum::new(Vec3::Y, 0.0, 2.0),
                    EnergySpectrum::ZERO,
                    EnergySpectrum::new(Vec3::Z, 0.0, 0.0),
                    EnergySpectrum::ZERO,
                ],
                direct: [EnergySpectrum::ZERO; 6],
                samples: [1; 6],
                updated_at: Some(Instant::now()),
                ..SurfaceProbe::new(voxel, 0b11_1111)
            },
        );
        let energy = field.sample(voxel, Vec3::new(1.0, 1.0, 0.0)).unwrap();
        assert!(energy.visible.abs_diff_eq(Vec3::new(0.5, 0.5, 0.0), 1e-6));
        assert!((energy.infrared - 0.5).abs() < 1e-6);
        assert!((energy.ionizing - 1.0).abs() < 1e-6);
    }
}
