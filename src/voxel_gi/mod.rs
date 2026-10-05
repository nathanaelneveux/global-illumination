mod diagnostics;
mod emission;
mod field;
mod gpu;
mod gpu_scene;
mod history;
mod naadf;
mod reconstruction;
pub(crate) mod render;
mod sampling;
mod scheduling;
mod settings;
mod sky;
mod sky_cache;
mod spectrum;
mod tracer;
#[cfg(debug_assertions)]
mod validation;
mod world;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::storage::ShaderBuffer;
use bevy::tasks::{AsyncComputeTaskPool, Task, block_on, poll_once};
use bevy_voxel_world::prelude::*;
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use rayon::{ThreadPool, ThreadPoolBuilder};

pub use field::{EnergyEstimate, VoxelEnergyField};
pub use settings::VoxelGiSettings;
pub use spectrum::{EnergyEmitter, EnergySpectrum};
pub use world::{MaterialTransport, VoxelMaterialTransportTable};

#[cfg(debug_assertions)]
use crate::player::Player;
use crate::terrain::PrototypeWorld;
use render::{
    GpuCacheBuild, GpuProbePatchQueue, GpuTransportCache, VoxelTransportAssets,
    VoxelTransportMaterial, build_gpu_cache, install_gpu_cache, install_render_assets,
    patch_gpu_topology, patch_probe_values, set_enabled, snapshot_gpu_probes,
};
use tracer::{
    BOOTSTRAP_SAMPLES, DirectionalTransportLight, PointTransportLight, TransportSources,
    trace_probe_batch_with_emitters,
};
use world::{
    AIR, PAGE_EDGE, PAGE_SIZE, SurfaceProbe, SurfaceProbeDescriptor, TransportScene,
    affected_probe_pages, scan_surface_probe_pages,
};

const STREAM_DEBOUNCE: Duration = Duration::from_millis(350);
const EDIT_DEBOUNCE: Duration = Duration::from_millis(40);
const MAX_REBUILD_LATENCY: Duration = Duration::from_secs(1);
const GPU_UPLOAD_INTERVAL: Duration = Duration::from_millis(50);
const SAMPLES_PER_DISPATCH: u32 = 4;
const EDIT_RELIGHT_RADIUS: i32 = 48;

pub struct VoxelGiPlugin;

impl Plugin for VoxelGiPlugin {
    fn build(&self, app: &mut App) {
        install_render_assets(app);
        diagnostics::install(app);
        app.init_resource::<VoxelGiDirty>()
            .init_resource::<VoxelGiRuntime>()
            .init_resource::<VoxelEnergyField>()
            .init_resource::<VoxelGiStats>()
            .init_resource::<VoxelGiSettings>()
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
                    toggle_voxel_gi,
                    log_voxel_gi_stats,
                )
                    .chain(),
            );
        if std::env::var("VOXEL_GI_CONTINUOUS").is_ok_and(|value| value == "1") {
            app.world_mut()
                .resource_mut::<VoxelGiSettings>()
                .adaptive_updates = false;
        }
        #[cfg(debug_assertions)]
        app.init_resource::<validation::CaptureEvidence>()
            .add_systems(
                Update,
                (
                    smoke_edit_once.after(log_voxel_gi_stats),
                    smoke_move_once.after(log_voxel_gi_stats),
                    validation::run_validation.after(log_voxel_gi_stats),
                ),
            );
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
    pub mean_history_samples: f32,
    pub oldest_sample_age_ms: f32,
    pub backend: &'static str,
}

#[derive(Resource)]
struct VoxelGiRuntime {
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: Arc<VoxelMaterialTransportTable>,
    trace_pool: Arc<ThreadPool>,
    loaded_pages: HashSet<IVec3>,
    dirty_pages: HashSet<IVec3>,
    full_resample_pages: HashSet<IVec3>,
    broad_dirty_pages: HashSet<IVec3>,
    dirty_voxels: HashSet<IVec3>,
    removed_pages: HashSet<IVec3>,
    last_world_change: Instant,
    dirty_since: Option<Instant>,
    urgent_edit: bool,
    generation: u64,
    source_revision: u64,
    batch_size: usize,
    global_source_revision: u64,
    source_influences: Vec<(Vec3, f32)>,
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
    scene_task: Option<Task<SceneBuildResult>>,
    trace_task: Option<Task<TraceBatchResult>>,
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
    gpu_task_version: u64,
    gpu_value_dirty: HashSet<IVec3>,
    sky_upload_pending: bool,
    last_gpu_upload: Instant,
    enabled: bool,
}

impl Default for VoxelGiRuntime {
    fn default() -> Self {
        let now = Instant::now();
        let available = std::thread::available_parallelism().map_or(2, usize::from);
        let worker_count = available.saturating_sub(2).clamp(1, 2);
        let trace_pool = ThreadPoolBuilder::new()
            .num_threads(worker_count)
            .thread_name(|index| format!("voxel-gi-{index}"))
            .build()
            .expect("voxel GI worker pool");
        info!(worker_count, "voxel GI worker pool ready");
        Self {
            scene: Arc::new(TransportScene::default()),
            sources: TransportSources::default(),
            materials: Arc::new(VoxelMaterialTransportTable::default()),
            trace_pool: Arc::new(trace_pool),
            loaded_pages: HashSet::new(),
            dirty_pages: HashSet::new(),
            full_resample_pages: HashSet::new(),
            broad_dirty_pages: HashSet::new(),
            dirty_voxels: HashSet::new(),
            removed_pages: HashSet::new(),
            last_world_change: now,
            dirty_since: None,
            urgent_edit: false,
            generation: 0,
            source_revision: 0,
            batch_size: 256,
            global_source_revision: 0,
            source_influences: Vec::new(),
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
            scene_task: None,
            trace_task: None,
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
    affected_pages: HashSet<IVec3>,
    descriptors: Vec<SurfaceProbeDescriptor>,
    exact_edits: Vec<IVec3>,
    broad_edits: Vec<IVec3>,
    build_time: Duration,
}

struct TraceBatchResult {
    probes: Vec<SurfaceProbe>,
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

fn collect_chunk_changes(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut explicit_dirty: ResMut<VoxelGiDirty>,
    mut spawned: MessageReader<ChunkWillSpawn<PrototypeWorld>>,
    mut updated: MessageReader<ChunkWillUpdate<PrototypeWorld>>,
    mut despawned: MessageReader<ChunkWillDespawn<PrototypeWorld>>,
) {
    let mut changed = false;
    for event in spawned.read() {
        runtime.loaded_pages.insert(event.chunk_key);
        runtime.removed_pages.remove(&event.chunk_key);
        runtime.dirty_pages.insert(event.chunk_key);
        changed = true;
    }
    for event in updated.read() {
        runtime.loaded_pages.insert(event.chunk_key);
        runtime.removed_pages.remove(&event.chunk_key);
        runtime.dirty_pages.insert(event.chunk_key);
        runtime.full_resample_pages.insert(event.chunk_key);
        runtime.broad_dirty_pages.insert(event.chunk_key);
        changed = true;
    }
    for event in despawned.read() {
        runtime.loaded_pages.remove(&event.chunk_key);
        runtime.dirty_pages.remove(&event.chunk_key);
        runtime.full_resample_pages.remove(&event.chunk_key);
        runtime.broad_dirty_pages.remove(&event.chunk_key);
        runtime.removed_pages.insert(event.chunk_key);
        changed = true;
    }
    if explicit_dirty.all_loaded {
        let pages: Vec<_> = runtime.loaded_pages.iter().copied().collect();
        runtime.dirty_pages.extend(pages.iter().copied());
        runtime.full_resample_pages.extend(pages.iter().copied());
        runtime.broad_dirty_pages.extend(pages);
        explicit_dirty.all_loaded = false;
        changed = true;
    }
    for voxel in explicit_dirty.voxels.drain() {
        let page = page_of(voxel);
        if runtime.loaded_pages.contains(&page) {
            runtime.dirty_pages.insert(page);
            runtime.full_resample_pages.insert(page);
            runtime.broad_dirty_pages.remove(&page);
            runtime.dirty_voxels.insert(voxel);
            runtime.urgent_edit = true;
            changed = true;
        }
    }
    if changed {
        let now = Instant::now();
        runtime.last_world_change = now;
        runtime.dirty_since.get_or_insert(now);
    }
}

fn start_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    generator: Res<PrototypeWorld>,
    voxel_world: VoxelWorld<PrototypeWorld>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
) {
    let pending = !runtime.dirty_pages.is_empty() || !runtime.removed_pages.is_empty();
    let debounce = if runtime.urgent_edit {
        EDIT_DEBOUNCE
    } else {
        STREAM_DEBOUNCE
    };
    let quiet = runtime.last_world_change.elapsed() >= debounce;
    let overdue = runtime
        .dirty_since
        .is_some_and(|start| start.elapsed() >= MAX_REBUILD_LATENCY);
    if runtime.scene_task.is_some() || !pending || (!quiet && !overdue) {
        return;
    }

    let mut dirty_pages: Vec<_> = runtime.dirty_pages.drain().collect();
    dirty_pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    let full_resample = std::mem::take(&mut runtime.full_resample_pages);
    let broad_edits: Vec<_> = runtime.broad_dirty_pages.drain().collect();
    let exact_edits: Vec<_> = runtime.dirty_voxels.drain().collect();
    let mut removed_pages: Vec<_> = runtime.removed_pages.drain().collect();
    removed_pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    runtime.dirty_since = None;
    runtime.urgent_edit = false;

    let affected_pages = affected_probe_pages(&dirty_pages, &removed_pages);
    let previous_scene = Arc::clone(&runtime.scene);
    let voxel_lookup = voxel_world.get_voxel_fn();
    let chunk_lookup = voxel_world.get_chunk_data_fn();
    let generator = generator.clone();
    let focus_page = cameras.iter().next().map_or(IVec3::ZERO, |camera| {
        page_of(camera.translation().floor().as_ivec3())
    });
    runtime.scene_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let started = Instant::now();
        let changed_pages: Vec<_> = dirty_pages
            .into_iter()
            .map(|page| {
                let origin = page * PAGE_EDGE as i32;
                let materials = if !full_resample.contains(&page) {
                    if let Some(chunk) = chunk_lookup(page) {
                        if chunk.is_empty() {
                            vec![AIR; PAGE_SIZE.element_product() as usize]
                        } else {
                            sample_dense_page(origin, |position| {
                                chunk
                                    .get_voxel_at_world_position(position)
                                    .unwrap_or_else(|| generator.generated_voxel(position))
                            })
                        }
                    } else {
                        sample_dense_page(origin, |position| match voxel_lookup(position) {
                            WorldVoxel::Unset => generator.generated_voxel(position),
                            voxel => voxel,
                        })
                    }
                } else {
                    sample_dense_page(origin, |position| match voxel_lookup(position) {
                        WorldVoxel::Unset => generator.generated_voxel(position),
                        voxel => voxel,
                    })
                };
                (page, materials)
            })
            .collect();
        let scene = Arc::new(TransportScene::with_page_delta(
            &previous_scene,
            changed_pages,
            &removed_pages,
        ));
        let changed = scene.regions.len() != previous_scene.regions.len()
            || scene.regions.iter().any(|(key, region)| {
                previous_scene
                    .regions
                    .get(key)
                    .is_none_or(|old| !Arc::ptr_eq(old, region))
            });
        let descriptors = if changed {
            scan_surface_probe_pages(&scene, &affected_pages, focus_page)
        } else {
            Vec::new()
        };
        SceneBuildResult {
            scene,
            changed,
            affected_pages,
            descriptors,
            exact_edits,
            broad_edits,
            build_time: started.elapsed(),
        }
    }));
}

fn sample_dense_page(origin: IVec3, sample: impl Fn(IVec3) -> WorldVoxel<u8>) -> Vec<u16> {
    let mut materials = vec![AIR; PAGE_SIZE.element_product() as usize];
    for z in 0..PAGE_EDGE {
        for y in 0..PAGE_EDGE {
            for x in 0..PAGE_EDGE {
                let local = UVec3::new(x, y, z);
                let position = origin + local.as_ivec3();
                materials[naadf::linear_index(local, PAGE_SIZE)] = match sample(position) {
                    WorldVoxel::Solid(material) => material as u16,
                    WorldVoxel::Air | WorldVoxel::Unset => AIR,
                };
            }
        }
    }
    materials
}

fn apply_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
) {
    let Some(task) = runtime.scene_task.as_mut() else {
        return;
    };
    let Some(result) = block_on(poll_once(task)) else {
        return;
    };
    runtime.scene_task = None;
    if !result.changed {
        stats.loaded_pages = runtime.loaded_pages.len();
        stats.last_scene_build_ms = result.build_time.as_secs_f32() * 1000.0;
        return;
    }
    runtime.scene = result.scene;
    runtime.generation = runtime.generation.wrapping_add(1);
    // A local-radius reset cannot certify indirect transport elsewhere. Wake
    // every loaded face lazily, without a world scan or clearing its display.
    field.transport_revision = field.transport_revision.wrapping_add(1);

    let old_count = field.probes.len();
    let mut old_affected = HashMap::new();
    for page in &result.affected_pages {
        if let Some(voxels) = field.pages.remove(page) {
            for voxel in voxels {
                if let Some(probe) = field.probes.remove(&voxel) {
                    let (faces, samples, covered) = probe_metrics(&probe);
                    runtime.active_faces = runtime.active_faces.saturating_sub(faces);
                    runtime.completed_samples = runtime.completed_samples.saturating_sub(samples);
                    runtime.covered_faces = runtime.covered_faces.saturating_sub(covered);
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
        for descriptor in &result.descriptors {
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
    for descriptor in result.descriptors {
        let mut probe = old_affected.remove(&descriptor.voxel).unwrap_or_else(|| {
            invalidated.insert(descriptor.voxel);
            added.push(descriptor.voxel);
            let mut probe = SurfaceProbe::new(descriptor.voxel, descriptor.face_mask);
            probe.revision = runtime.generation << 32;
            probe
        });
        let old_mask = probe.face_mask;
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
            invalidated.insert(descriptor.voxel);
        }
        let (faces, samples, covered) = probe_metrics(&probe);
        runtime.active_faces += faces;
        runtime.completed_samples += samples;
        runtime.covered_faces += covered;
        field
            .pages
            .entry(page_of(descriptor.voxel))
            .or_default()
            .push(descriptor.voxel);
        runtime.gpu_value_dirty.insert(descriptor.voxel);
        field.probes.insert(descriptor.voxel, probe);
    }
    let topology_changed =
        old_count != field.probes.len() || !old_affected.is_empty() || !added.is_empty();
    if topology_changed {
        runtime.gpu_removed.extend(old_affected.keys().copied());
        runtime.gpu_added.extend(added.iter().copied());
        field.update_indices(&added, &old_affected);
        runtime.gpu_topology_dirty = true;
        runtime.gpu_topology_version = runtime.gpu_topology_version.wrapping_add(1);
    }

    let mut urgent = HashSet::new();
    for &voxel in &result.exact_edits {
        let mut nearby = HashSet::new();
        invalidate_near_voxel(
            &mut runtime,
            &mut field,
            voxel,
            EDIT_RELIGHT_RADIUS,
            &mut nearby,
        );
        urgent.extend(nearby.iter().copied());
        invalidated.extend(nearby);
    }
    for page in result.broad_edits {
        let center = page * PAGE_EDGE as i32 + IVec3::splat(PAGE_EDGE as i32 / 2);
        invalidate_near_voxel(
            &mut runtime,
            &mut field,
            center,
            EDIT_RELIGHT_RADIUS,
            &mut invalidated,
        );
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
    enqueue_sky(&mut runtime, urgent.iter().copied());
    enqueue_sky(&mut runtime, added.iter().copied());
    enqueue_urgent(&mut runtime, urgent);
    runtime.sky_remaining = field.order.len();
    if old_count == 0 {
        enqueue_many(&mut runtime, field.order.iter().copied());
    } else {
        enqueue_many(&mut runtime, invalidated);
    }
    if old_count > 0 {
        request_global_refresh(&mut runtime, field.order.len());
    }
    update_stats_from_counters(&runtime, &field, &mut stats);
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
            let (_, samples, covered) = probe_metrics(probe);
            runtime.completed_samples = runtime.completed_samples.saturating_sub(samples);
            runtime.covered_faces = runtime.covered_faces.saturating_sub(covered);
            probe.invalidate();
            runtime.gpu_value_dirty.insert(voxel);
            invalidated.insert(voxel);
        }
    }
}

fn synchronize_transport_sources(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    directional_lights: Query<(&DirectionalLight, &GlobalTransform, Option<&EnergyEmitter>)>,
    point_lights: Query<(&PointLight, &GlobalTransform, Option<&EnergyEmitter>)>,
    energy_emitters: Query<
        (&EnergyEmitter, &GlobalTransform),
        (Without<PointLight>, Without<DirectionalLight>),
    >,
) {
    let mut next = TransportSources::default();
    for (light, transform, extra) in &directional_lights {
        let linear = light.color.to_linear();
        let color = Vec3::new(linear.red, linear.green, linear.blue);
        let mut radiance = EnergySpectrum::new(color * (light.illuminance / 10_000.0), 0.0, 0.0);
        if let Some(extra) = extra {
            radiance += extra.spectrum;
        }
        next.directional.push(DirectionalTransportLight {
            direction: transform.back().as_vec3(),
            radiance: radiance.bounded(1e12),
            custom_visible: extra.map_or(Vec3::ZERO, |extra| extra.spectrum.bounded(1e12).visible),
        });
    }
    for (light, transform, extra) in &point_lights {
        let linear = light.color.to_linear();
        let color = Vec3::new(linear.red, linear.green, linear.blue);
        let mut intensity = EnergySpectrum::new(color * (light.intensity / 1_000.0), 0.0, 0.0);
        if let Some(extra) = extra {
            intensity += extra.spectrum;
        }
        next.points.push(PointTransportLight {
            position: transform.translation(),
            intensity: intensity.bounded(1e12),
            range: extra.map_or(light.range, |emitter| emitter.range.max(light.range)),
            custom_visible: extra.map_or(Vec3::ZERO, |extra| extra.spectrum.bounded(1e12).visible),
        });
    }
    for (emitter, transform) in &energy_emitters {
        next.points.push(PointTransportLight {
            position: transform.translation(),
            intensity: emitter.spectrum.bounded(1e12),
            range: emitter.range,
            custom_visible: emitter.spectrum.bounded(1e12).visible,
        });
    }
    next.points
        .retain(|light| light.position.is_finite() && light.range.is_finite() && light.range > 0.0);
    next.directional
        .retain(|light| light.direction.is_finite() && light.direction.length_squared() > 0.0);
    next.directional
        .sort_by(|a, b| compare_vec3(a.direction, b.direction));
    next.points
        .sort_by(|a, b| compare_vec3(a.position, b.position));
    if next == runtime.sources {
        return;
    }
    let global = next.directional != runtime.sources.directional
        || next.environment != runtime.sources.environment;
    // Old and new influence volumes both matter when a source moves or vanishes.
    runtime.source_influences = runtime
        .sources
        .points
        .iter()
        .chain(next.points.iter())
        .map(|light| (light.position, light.range + 16.0))
        .collect();
    if sky_cache::sky_lights(&next) != sky_cache::sky_lights(&runtime.sources) {
        runtime.sky_revision = runtime.sky_revision.wrapping_add(1);
        runtime.sky_remaining = field.order.len();
    }
    runtime.sources = next;
    invalidate_sources(&mut runtime, &mut field, &mut stats);
    if global {
        runtime.global_source_revision = runtime.source_revision;
    }
}

fn compare_vec3(a: Vec3, b: Vec3) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x)
        .then_with(|| a.y.total_cmp(&b.y))
        .then_with(|| a.z.total_cmp(&b.z))
}

fn synchronize_material_table(
    mut runtime: ResMut<VoxelGiRuntime>,
    materials: Res<VoxelMaterialTransportTable>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
) {
    if !materials.is_changed() || materials.as_ref() == runtime.materials.as_ref() {
        return;
    }
    runtime.materials = Arc::new(materials.clone());
    invalidate_sources(&mut runtime, &mut field, &mut stats);
    runtime.global_source_revision = runtime.source_revision;
}

fn invalidate_sources(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
) {
    // A moving source must not discard every in-flight batch every frame.
    // Keep a versioned snapshot; shorten history lazily when each probe runs.
    runtime.source_revision = runtime.source_revision.wrapping_add(1);
    field.source_revision = runtime.source_revision;
    field.transport_revision = field.transport_revision.wrapping_add(1);
    request_global_refresh(runtime, field.order.len());
    update_stats_from_counters(runtime, field, stats);
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

fn select_sky(runtime: &mut VoxelGiRuntime, field: &VoxelEnergyField) -> Vec<SurfaceProbe> {
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
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
            selected.push(probe.clone());
        }
    }
    selected
}

fn start_trace_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    gpu_device: Option<Res<gpu::GpuTransportDevice>>,
    settings: Res<VoxelGiSettings>,
) {
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
    let transport_count = selected.len();
    let sky_probes = select_sky(&mut runtime, &field);
    let sky_count = sky_probes.len();
    merge_sky_work(&mut selected, sky_probes);
    let sky_only = transport_count == 0;
    if selected.is_empty() {
        stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
        stats.idle_scheduler_frames += 1;
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    }
    let paths = selected
        .iter()
        .map(|probe| {
            (0..6)
                .filter(|face| !sky_only && probe.face_mask & probe.trace_mask & (1 << face) != 0)
                .count() as u64
                * u64::from(SAMPLES_PER_DISPATCH)
        })
        .sum();
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
        let mut probes = selected;
        if let Some(tracer) = gpu_tracer.as_mut()
            && let Err(error) = tracer.trace_work(
                Arc::clone(&scene),
                &sources,
                &materials,
                &mut probes,
                SAMPLES_PER_DISPATCH,
                &emitters,
                sky_only,
            )
        {
            gpu_error = Some(error);
            gpu_tracer = None;
        }
        if gpu_tracer.is_none() {
            probes = pool.install(|| {
                trace_probe_batch_with_emitters(
                    scene,
                    sources,
                    materials,
                    probes,
                    if sky_only { 0 } else { SAMPLES_PER_DISPATCH },
                    &emitters,
                    &mut cpu_sky_cache,
                )
            });
        }
        TraceBatchResult {
            probes,
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

fn merge_sky_work(selected: &mut Vec<SurfaceProbe>, sky: Vec<SurfaceProbe>) {
    let mut seen: HashSet<_> = selected.iter().map(|probe| probe.voxel).collect();
    for mut probe in sky {
        if seen.insert(probe.voxel) {
            // Topology remains intact for corner lookup; no indirect rays and
            // no changes to the simulation history for a sky-only receiver.
            probe.trace_mask = 0;
            selected.push(probe);
        }
    }
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
    select_transport(runtime, field, &VoxelGiSettings::default())
}

fn select_transport(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
    settings: &VoxelGiSettings,
) -> Vec<SurfaceProbe> {
    let mut selected = Vec::with_capacity(runtime.batch_size);
    let mut seen = HashSet::with_capacity(runtime.batch_size);
    let mut adaptive = if runtime.urgent_pending.len() + runtime.pending.len() < runtime.batch_size
    {
        scheduling::ranked_candidates(
            field,
            &runtime.nearby,
            &mut runtime.adaptive_cursor,
            Instant::now(),
            runtime.batch_size,
        )
    } else {
        Vec::new()
    };
    // Search capacity is bounded independently of emitted work. In particular,
    // resting neighbors are holes to skip, not GPU work that used the budget.
    for index in 0..settings.batch_limit().min(field.order.len()) {
        if selected.len() >= runtime.batch_size {
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
        if seen.contains(&voxel) {
            continue;
        }
        // A reserved sweep slot can encounter an explicitly requested voxel
        // before its urgent queue entry. Honor the force now, not after it has
        // already been selected (or skipped as settled).
        force |= runtime.urgent_queued.contains(&voxel);
        let Some(probe) = field.probes.get(&voxel) else {
            continue;
        };
        let mask = scheduling::due_faces(
            probe,
            field.transport_revision,
            Instant::now(),
            settings,
            force,
        );
        if mask == 0 {
            continue;
        }
        seen.insert(voxel);
        let mut probe = probe.clone();
        probe.trace_mask = mask;
        if probe.transport_revision != field.transport_revision {
            probe.transport_revision = field.transport_revision;
            probe.fresh_mask = 0;
            probe.stable_batches = [0; 6];
        }
        // Refresh previously invalidated faces, including remote indirect
        // dependencies. Do not reset unrelated faces of a partially selected voxel.
        for face in 0..6 {
            if mask & !probe.fresh_mask & (1 << face) != 0 {
                // A global revision wakes all potential indirect dependencies;
                // it does not prove their energy changed. Keep a bounded useful
                // history. Exact/local edits already call invalidate(), while
                // source influence bounds below choose the faster local response.
                probe.samples[face] = probe.samples[face].min(64);
            }
        }
        if probe.source_revision != runtime.source_revision {
            let local = runtime.source_influences.iter().any(|(position, radius)| {
                probe.voxel.as_vec3().distance_squared(*position) <= radius * radius
            });
            let history = if local || probe.source_revision < runtime.global_source_revision {
                4
            } else {
                64
            };
            for (face, samples) in probe.samples.iter_mut().enumerate() {
                if mask & (1 << face) != 0 {
                    *samples = (*samples).min(history);
                }
            }
            probe.source_revision = runtime.source_revision;
        }
        selected.push(probe);
    }
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
) {
    let Some(task) = runtime.trace_task.as_mut() else {
        return;
    };
    let Some(result) = block_on(poll_once(task)) else {
        return;
    };
    let merge_started = Instant::now();
    runtime.trace_task = None;
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
    for mut probe in result.probes {
        let voxel = probe.voxel;
        let Some(old) = field.probes.get_mut(&voxel) else {
            continue;
        };
        if old.revision != probe.revision {
            if probe_has_pending_samples(old) {
                requeue.push(voxel);
            }
            continue;
        }
        if probe.trace_mask == 0 {
            if sky_current {
                let changed = (0..6).any(|face| old.display[face].sky != probe.display[face].sky);
                for face in 0..6 {
                    old.display[face].sky = probe.display[face].sky;
                }
                if changed {
                    runtime.gpu_value_dirty.insert(voxel);
                    runtime.sky_upload_pending = true;
                }
            }
            continue;
        }
        probe.preserve_topology_from(old);
        // A changed sun must never be republished by an older transport batch.
        if !sky_current {
            for face in 0..6 {
                probe.display[face].sky = old.display[face].sky;
            }
        }
        if sky_current && (0..6).any(|face| old.display[face].sky != probe.display[face].sky) {
            runtime.sky_upload_pending = true;
        }
        probe.resolve_display(old, Instant::now(), settings.display_response_seconds());
        scheduling::record_batch(&mut probe, old, Instant::now(), &settings);
        let (_, old_samples, old_covered) = probe_metrics(old);
        let (_, new_samples, new_covered) = probe_metrics(&probe);
        runtime.completed_samples =
            runtime.completed_samples.saturating_sub(old_samples) + new_samples;
        runtime.covered_faces = runtime.covered_faces.saturating_sub(old_covered) + new_covered;
        let pending = probe_has_pending_samples(&probe);
        field.probes.insert(voxel, probe);
        runtime.gpu_value_dirty.insert(voxel);
        if pending {
            requeue.push(voxel);
        }
    }
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

fn probe_metrics(probe: &SurfaceProbe) -> (u64, u64, u64) {
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
    (faces, samples, covered)
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
    stats.global_refresh = if runtime.refresh_total == 0 {
        1.0
    } else {
        1.0 - runtime.refresh_remaining as f32 / runtime.refresh_total as f32
    };
    stats.fully_bootstrapped = stats.bootstrap_coverage >= 1.0;
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
) {
    let Some(material_handle) = material_handle else {
        return;
    };

    let mut installed = false;
    if let Some(task) = runtime.gpu_task.as_mut()
        && let Some(build) = block_on(poll_once(task))
    {
        runtime.gpu_task = None;
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
            &mut materials,
            &material_handle.handle,
        ) {
            runtime.gpu_topology_dirty = false;
        } else {
            // Keep rendering the old cache while the larger allocation builds.
            runtime.gpu_rebuild_required = true;
        }
    }

    if runtime.gpu_topology_dirty
        && (!cache.ready || runtime.gpu_rebuild_required)
        && runtime.gpu_task.is_none()
    {
        let snapshot = snapshot_gpu_probes(&field.probes, &field.order);
        if cache.ready {
            // The replacement is asynchronous; keep the published generation
            // current while it builds, using its own still-valid slot map.
            patch_probe_values(
                &cache,
                &patch_queue,
                &field.probes,
                &runtime.gpu_value_dirty,
            );
        }
        // The snapshot already contains these values. Only changes made while
        // the build runs need a subsequent patch.
        runtime.gpu_value_dirty.clear();
        runtime.gpu_added.clear();
        runtime.gpu_removed.clear();
        runtime.gpu_task_version = runtime.gpu_topology_version;
        runtime.gpu_task =
            Some(AsyncComputeTaskPool::get().spawn(async move { build_gpu_cache(snapshot) }));
        return;
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
    if runtime.gpu_task.is_some() {
        // These are also the catch-up ledger for the in-flight snapshot.
        // Publishing to the OLD cache must not consume that ledger.
        patch_probe_values(
            &cache,
            &patch_queue,
            &field.probes,
            &runtime.gpu_value_dirty,
        );
    } else {
        let dirty = std::mem::take(&mut runtime.gpu_value_dirty);
        patch_probe_values(&cache, &patch_queue, &field.probes, &dirty);
    }
    runtime.last_gpu_upload = Instant::now();
    runtime.sky_upload_pending = false;
}

fn toggle_voxel_gi(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut runtime: ResMut<VoxelGiRuntime>,
    material_handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
) {
    // Scripted validation owns presentation as well as the player path.
    #[cfg(debug_assertions)]
    if std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        return;
    }
    if keyboard.just_pressed(KeyCode::F7)
        && let Some(handle) = material_handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.voxel_sky ^= 1;
        info!(
            voxel_sky = material.extension.settings.voxel_sky != 0,
            "direct sky lighting changed (F5 controls cached face/corner blending)"
        );
    }
    if keyboard.just_pressed(KeyCode::F6)
        && let Some(handle) = material_handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.debug_irradiance ^= 1;
        info!(
            irradiance_only = material.extension.settings.debug_irradiance != 0,
            "voxel GI diagnostic view changed"
        );
    }
    if keyboard.just_pressed(KeyCode::F5)
        && let Some(handle) = material_handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.reconstruction ^= 1;
        info!(
            smooth_corners = material.extension.settings.reconstruction != 0,
            "voxel GI reconstruction changed"
        );
    }
    if (keyboard.just_pressed(KeyCode::F5) || keyboard.just_pressed(KeyCode::F7))
        && let Some(handle) = material_handle.as_ref()
        && let Some(material) = materials.get(&handle.handle)
    {
        let settings = material.extension.settings;
        let sky_mode = if settings.voxel_sky == 0 {
            "per-pixel"
        } else if settings.reconstruction != 0 {
            "voxel corners"
        } else {
            "voxel faces"
        };
        for mut window in &mut windows {
            window.title = format!("Voxel GI Lab | Sun: {sky_mode} (F7)");
        }
    }
    if !keyboard.just_pressed(KeyCode::F4) {
        return;
    }
    runtime.enabled = !runtime.enabled;
    if let Some(material_handle) = material_handle {
        set_enabled(runtime.enabled, &mut materials, &material_handle.handle);
    }
    info!(enabled = runtime.enabled, "world-space voxel GI toggled");
}

fn log_voxel_gi_stats(
    mut stats: ResMut<VoxelGiStats>,
    field: Res<VoxelEnergyField>,
    runtime: Res<VoxelGiRuntime>,
    mut last_log: Local<Option<Instant>>,
) {
    let now = Instant::now();
    if last_log.is_some_and(|last| now.duration_since(last) < Duration::from_secs(5)) {
        return;
    }
    *last_log = Some(now);
    let mut samples = 0_u64;
    let mut faces = 0_u64;
    let mut oldest = Duration::ZERO;
    for probe in field.probes.values() {
        for face in 0..6 {
            if probe.face_mask & (1 << face) != 0 {
                samples += u64::from(probe.samples[face]);
                faces += 1;
                if let Some(updated) = probe.face_updated_at[face].or(probe.updated_at) {
                    oldest = oldest.max(now.saturating_duration_since(updated));
                }
            }
        }
    }
    stats.mean_history_samples = samples as f32 / faces.max(1) as f32;
    stats.oldest_sample_age_ms = oldest.as_secs_f32() * 1000.0;
    if !diagnostics::terminal_stats_enabled() {
        return;
    }
    info!(
        loaded_pages = stats.loaded_pages,
        surface_probes = stats.surface_probes,
        coverage_percent = format_args!("{:.1}", stats.interactive_coverage * 100.0),
        bootstrap_percent = format_args!("{:.1}", stats.bootstrap_coverage * 100.0),
        pending_probes = stats.pending_probes,
        traced_paths = stats.traced_paths,
        refresh_percent = format_args!("{:.1}", stats.global_refresh * 100.0),
        last_trace_ms = format_args!("{:.1}", stats.last_trace_batch_ms),
        sky_requests = stats.sky_corner_requests,
        sky_traced = stats.sky_corners_traced,
        sky_rays_scheduled = stats.sky_rays_scheduled,
        sky_cached = stats.sky_cached_corners,
        sky_batches = stats.sky_only_batches,
        transport_batches = stats.transport_batches,
        idle_scheduler_frames = stats.idle_scheduler_frames,
        dispatch_budget = runtime.batch_size,
        sky_budget = runtime.sky_batch_size,
        worker_ms = stats.last_worker_ms,
        selection_ms = stats.last_selection_ms,
        merge_ms = stats.last_merge_ms,
        prepare_ms = stats.last_prepare_ms,
        wait_ms = stats.last_wait_ms,
        decode_ms = stats.last_decode_ms,
        packed_geometry_bytes = stats.packed_geometry_bytes,
        fully_bootstrapped = stats.fully_bootstrapped,
        mean_history = stats.mean_history_samples,
        oldest_sample_ms = stats.oldest_sample_age_ms,
        backend = stats.backend,
        "voxel GI status"
    );
}

/// Opt-in, one-shot integration check for the real BVW edit path. The normal
/// application never mutates terrain automatically.
#[cfg(debug_assertions)]
fn smoke_edit_once(
    mut finished: Local<bool>,
    mut ready_since: Local<Option<Instant>>,
    stats: Res<VoxelGiStats>,
    field: Res<VoxelEnergyField>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut voxel_world: VoxelWorld<PrototypeWorld>,
    mut dirty: ResMut<VoxelGiDirty>,
) {
    if *finished || std::env::var_os("VOXEL_GI_SMOKE_EDIT").is_none() {
        return;
    }
    if !stats.fully_bootstrapped || stats.loaded_pages < 200 {
        *ready_since = None;
        return;
    }
    let ready = ready_since.get_or_insert_with(Instant::now);
    if ready.elapsed() < Duration::from_secs(10) {
        return;
    }
    let Some(camera) = cameras.iter().next() else {
        return;
    };
    let eye = camera.translation();
    let forward = camera.forward().as_vec3();
    let target = field
        .order
        .iter()
        .copied()
        .filter(|voxel| {
            let delta = voxel.as_vec3() + Vec3::splat(0.5) - eye;
            (16.0..=256.0).contains(&delta.length_squared()) && delta.normalize().dot(forward) > 0.2
        })
        .min_by(|a, b| {
            a.as_vec3()
                .distance_squared(eye)
                .total_cmp(&b.as_vec3().distance_squared(eye))
        });
    let Some(voxel) = target else {
        return;
    };
    voxel_world.set_voxel(voxel, WorldVoxel::Air);
    dirty.mark_voxel(voxel);
    *finished = true;
    info!(?voxel, "voxel GI smoke edit applied");
}

/// Opt-in streaming stress check: move far enough to cross a chunk boundary
/// after the initial transport field has settled.
#[cfg(debug_assertions)]
fn smoke_move_once(
    mut finished: Local<bool>,
    mut ready_since: Local<Option<Instant>>,
    stats: Res<VoxelGiStats>,
    mut player: Single<&mut Transform, With<Player>>,
) {
    if *finished || std::env::var_os("VOXEL_GI_SMOKE_MOVE").is_none() {
        return;
    }
    if !stats.fully_bootstrapped || stats.loaded_pages < 200 {
        *ready_since = None;
        return;
    }
    let ready = ready_since.get_or_insert_with(Instant::now);
    if ready.elapsed() < Duration::from_secs(5) {
        return;
    }
    player.translation += Vec3::new(48.0, 4.0, 0.0);
    *finished = true;
    info!(position = ?player.translation, "voxel GI smoke move applied");
}

#[cfg(test)]
mod tests {
    use super::*;

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
        runtime.global_source_revision = runtime.source_revision;
        assert_eq!(field.probes[&voxel].samples[2], 64);
        assert_eq!(runtime.pending.len(), 1);
        let selected = select_probes(&mut runtime, &field);
        assert_eq!(selected[0].samples[2], 4);
        assert_eq!(selected[0].sequence[2], 100);
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
        runtime.source_influences.push((Vec3::ZERO, 24.0));
        invalidate_sources(&mut runtime, &mut field, &mut VoxelGiStats::default());
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
        let mut transport = vec![SurfaceProbe::new(IVec3::ZERO, 4)];
        let mut sky_only = SurfaceProbe::new(IVec3::X, 5);
        sky_only.samples = [128; 6];
        sky_only.sequence = [1000; 6];
        let duplicate = transport[0].clone();
        merge_sky_work(&mut transport, vec![duplicate, sky_only]);
        assert_eq!(transport.len(), 2);
        assert_eq!(transport[0].trace_mask, 63);
        assert_eq!(transport[1].face_mask, 5);
        assert_eq!(transport[1].trace_mask, 0);
        assert_eq!(transport[1].samples, [128; 6]);
        assert_eq!(transport[1].sequence, [1000; 6]);
    }

    #[test]
    fn resting_candidates_do_not_consume_the_dispatch_allowance() {
        let now = Instant::now();
        let mut field = VoxelEnergyField::default();
        for x in 0..4096 {
            let voxel = IVec3::new(x, 0, 0);
            let mut probe = SurfaceProbe::new(voxel, 4);
            probe.samples[2] = 128;
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
        assert_eq!(selected[0].samples[2], 128);
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
            probe.samples = [128; 6];
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
            field.probes.values().all(|probe| probe.samples[2] == 128),
            "scheduling must not mutate published histories"
        );
        enqueue_sky(&mut runtime, field.order.iter().copied());
        let sky = select_sky(&mut runtime, &field);
        assert_eq!(sky.len(), 64);
        assert!(sky.iter().all(|probe| probe.samples[2] == 128));
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
