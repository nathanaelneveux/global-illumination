//! CPU event selection, worker dispatch, and revision-checked publication of
//! CPU jobs and GPU picks. Bulk selection happens on the GPU; every loaded
//! receiver remains eligible independently of camera visibility.
use super::{
    ReceiverField, TraceBatchResult, VoxelGiRuntime, VoxelGiSettings, VoxelGiStats, emission,
    enqueue_urgent, gpu, probe_metrics, scheduling, telemetry::GiWorkTimings,
    update_stats_from_counters, work::TraceWork,
};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, block_on, poll_once},
};
use std::{sync::Arc, time::Instant};

/// GPU milliseconds available after `elapsed_ms` of wall time at `share`,
/// banking at most two batches' budget.
fn accrue_gpu_allowance(allowance: f32, elapsed_ms: f32, share: f32, budget_ms: f32) -> f32 {
    (allowance + elapsed_ms * share).min(2.0 * budget_ms)
}

/// Probes per batch whose untraced faces get a resident slot and seed. GPU
/// coverage only reaches faces with a slot, so this bounds how fast newly
/// streamed terrain is traced (1,024 left a growing backlog in the hallway).
const ENSURE_PROBES_PER_BATCH: usize = 8192;

pub(super) fn start_trace_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    field: ResMut<ReceiverField>,
    mut stats: ResMut<VoxelGiStats>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    gpu_device: Option<Res<gpu::GpuTransportDevice>>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::TraceDispatch);
    if runtime.trace_task.is_some() {
        return;
    }
    // Far geometry and receiver retirement are lifecycle work, not sampling.
    // A camera above the terrain can have no near receivers at all. Likewise,
    // exhausted ray allowance must not drop retirements during readback polls.
    let maintenance = !runtime.slot_retirements.is_empty()
        || runtime.spatial_history.has_events()
        || runtime.gpu_tracer.as_ref().map_or_else(
            || !runtime.scene.regions.is_empty(),
            |tracer| {
                tracer.needs_geometry(&runtime.scene)
                    || tracer.needs_model(&runtime.sources, &runtime.materials, runtime.transport)
            },
        );
    if field.order.is_empty() && runtime.gpu_pending.is_empty() && !maintenance {
        return;
    }
    if runtime.transport_unavailable {
        return;
    }
    let Some(gpu_device) = gpu_device else {
        warn_once!("Voxel GI: no GPU transport device; the world cache is idle");
        return;
    };
    let selection_started = Instant::now();
    let focus = cameras
        .iter()
        .find(|(camera, _)| camera.is_active)
        .map_or(IVec3::ZERO, |(_, transform)| {
            transform.translation().floor().as_ivec3()
        });
    runtime.batch_size = runtime.batch_size.min(settings.batch_limit());
    // Three owned staging allocations bound queue latency and memory. When full,
    // poll without selecting more work: never turn backpressure into a CPU wait.
    // The GI budget: a share of wall time, independent of frame rate. A short
    // burst may bank up to two batches' budget; spending waits below zero.
    let now = Instant::now();
    let elapsed_ms = runtime
        .gpu_allowance_at
        .map_or(0.0, |at| (now - at).as_secs_f32() * 1000.0);
    runtime.gpu_allowance_at = Some(now);
    runtime.gpu_allowance_ms = accrue_gpu_allowance(
        runtime.gpu_allowance_ms,
        elapsed_ms,
        settings.gpu_share() * settings.world_budget_fraction(),
        settings.budget_ms(),
    );
    let can_submit = settings.world_budget_fraction() > 0.0
        && !field.order.is_empty()
        && runtime.gpu_pending.len() < 3
        && runtime.gpu_allowance_ms > 0.0;
    let mut selected = if can_submit {
        info_span!("gi_select").in_scope(|| select_transport(&mut runtime, &field))
    } else {
        TraceWork::default()
    };
    selected.transport = runtime.transport;
    selected.view_transport = true;
    selected.shared_view_cache = settings.cache_radius().is_some();
    selected.response_seconds = settings.display_response_seconds();
    selected.measured_retention = settings.measured_retention();
    // The rest of the face budget is the GPU's to pick.
    selected.picks = if can_submit {
        runtime.batch_size.saturating_sub(selected.faces.len()) as u32
    } else {
        0
    };
    selected.focus = focus;
    selected.transport_revision = field.transport_revision;
    selected.source_revision = runtime.source_revision;
    selected.stable_error = settings.stable_error();
    selected.stable_samples = settings.stable_samples();
    selected.audit_ms = settings.audit_interval().as_millis() as u32;
    selected.mirror_budget = settings.mirror_per_batch();
    let samples = settings.batch_samples();
    // Eager slots ride with any batch that prepares resident inputs.
    // Bounded per batch so streaming cannot burst.
    if !selected.faces.is_empty() || selected.picks > 0 {
        let mut drained = 0;
        while drained < ENSURE_PROBES_PER_BATCH
            && let Some(voxel) = runtime.slot_requests.pop_front()
        {
            runtime.slot_requested.remove(&voxel);
            if let Some(probe) = field.probes.get(&voxel) {
                selected.ensure_slots(probe);
                drained += 1;
            }
        }
    }
    // Batch IDs count submissions only: the GPU protects a CPU job's face for
    // four IDs and re-ranks the quarter `batch_id % 4`, so poll-only frames
    // must not advance it (they expired ownership of unpublished jobs).
    if !selected.probes.is_empty() || selected.picks > 0 {
        runtime.batch_counter = (runtime.batch_counter % 0x0fff_ffff) + 1;
        selected.batch_id = runtime.batch_counter;
    }
    if selected.probes.is_empty()
        && selected.picks == 0
        && runtime.gpu_pending.is_empty()
        && !maintenance
    {
        runtime.trace_scratch = selected;
        stats.scheduler.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
        stats.scheduler.idle_scheduler_frames += 1;
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    }
    runtime
        .trace_pending
        .extend(selected.probes.iter().map(|job| job.voxel));
    // Every published change rides with the next dispatched batch, whatever
    // it selected: the GPU applies each to all resident faces in range.
    selected.retention_events = runtime.spatial_history.take_events();
    selected.retire_pages = std::mem::take(&mut runtime.slot_retirements);
    let paths = selected.faces.len() as u64 * u64::from(samples);
    stats.scheduler.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
    let scene = Arc::clone(&runtime.scene);
    let batch_submits = !selected.probes.is_empty() || selected.picks > 0;
    let gpu_reserved_ms = if batch_submits {
        settings.budget_ms()
    } else {
        0.0
    };
    runtime.gpu_allowance_ms -= gpu_reserved_ms;
    let mut batch = super::TransportBatch {
        work: selected,
        paths,
        gpu_reserved_ms,
        ..default()
    };
    let sources = runtime.sources.clone();
    let materials = Arc::clone(&runtime.materials);
    let mut gpu_tracer = runtime.gpu_tracer.take();
    let mut pending = std::mem::take(&mut runtime.gpu_pending);
    let emission_cache = runtime.emission_cache.take();
    let gpu_device = gpu_device.clone();
    runtime.trace_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let _span = info_span!("gi_trace_worker").entered();
        let started = Instant::now();
        let emission_cache = if !batch_submits && !maintenance {
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
        let mut build_error = None;
        if gpu_tracer.is_none() {
            match gpu::GpuTracer::new(gpu_device)
                .and_then(|mut tracer| tracer.enable_resident().map(|()| tracer))
            {
                Ok(tracer) => gpu_tracer = Some(tracer),
                Err(error) => build_error = Some(error),
            }
        }
        let mut completed = None;
        if let Some(tracer) = &mut gpu_tracer {
            tracer.queue_retention(&mut batch.work.retention_events);
            let result = (|| -> Result<(), String> {
                {
                    let ready = match pending.front_mut() {
                        Some(super::PendingTransport {
                            snapshot: Some(snapshot),
                            ..
                        }) => tracer.snapshot_ready(snapshot)?,
                        Some(_) => true,
                        None => false,
                    };
                    if ready {
                        let mut oldest = pending.pop_front().unwrap();
                        if let Some(snapshot) = oldest.snapshot {
                            info_span!("gi_readback").in_scope(|| {
                                tracer.complete_readback(snapshot, &mut oldest.batch.work)
                            })?;
                            oldest.batch.gpu_ms = tracer.last_gpu_ms;
                            oldest.batch.execution_ms = Some(tracer.last_dispatch_ms);
                        } else {
                            oldest.batch.gpu_ms = Some(0.0);
                            oldest.batch.execution_ms = Some(0.0);
                        }
                        completed = Some(oldest.batch);
                    }
                    if batch_submits || maintenance {
                        let _submit = info_span!("gi_submit").entered();
                        let snapshot = tracer.submit_compact(
                            Arc::clone(&scene),
                            &sources,
                            &materials,
                            &mut batch.work,
                            samples,
                            &emitters,
                        )?;
                        if batch_submits {
                            pending.push_back(super::PendingTransport {
                                snapshot,
                                batch: std::mem::take(&mut batch),
                            });
                        } else {
                            // Maintenance has no rays or readback, so it does
                            // not occupy a staging slot or advance batch IDs.
                            debug_assert!(snapshot.is_none());
                        }
                    }
                }
                Ok(())
            })();
            if let Err(error) = result {
                gpu_error = Some(error);
                gpu_tracer = None;
                pending.clear();
            }
        }
        let geometry = gpu_tracer
            .as_ref()
            .and_then(gpu::GpuTracer::geometry_display);
        let bindings = gpu_tracer
            .as_mut()
            .map_or_else(Vec::new, gpu::GpuTracer::take_resident_bindings);
        TraceBatchResult {
            batch: completed,
            pending,
            bindings,
            trace_time: started.elapsed(),
            gpu_tracer,
            emission_cache,
            gpu_error,
            build_error,
            geometry,
        }
    }));
}

/// At least one quarter of dispatch slots belongs to a fair loaded-world
/// sweep. Coverage and refinement also have reserved opportunities, so streamed
/// bootstrap work cannot indefinitely preempt refinement near the player.
#[cfg(test)]
pub(super) fn select_probes(
    runtime: &mut VoxelGiRuntime,
    field: &ReceiverField,
) -> Vec<super::world::SurfaceProbe> {
    // Reference view for existing policy tests. Production never materializes
    // these full receiver copies; it transfers only TraceWork.
    let work = select_transport(runtime, field);
    work.probes
        .iter()
        .map(|job| {
            let mut probe = field.probes[&job.voxel].clone();
            if probe.transport_revision != job.transport_revision {
                probe.transport_revision = job.transport_revision;
                probe.fresh_mask = 0;
                if let Some(stats) = probe.stats.as_mut() {
                    stats.stable_batches = [0; 6];
                }
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

/// The CPU selects only event work: faces an edit, relight or explicit
/// request made urgent, with their revision caps and rollbacks. Bulk
/// coverage, refinement and the world sweep are picked on the GPU from its
/// own statistics (`select_*` in voxel_accumulate.wgsl), in the same batch.
pub(super) fn select_transport(runtime: &mut VoxelGiRuntime, field: &ReceiverField) -> TraceWork {
    let mut selected = std::mem::take(&mut runtime.trace_scratch);
    selected.clear();
    let mut seen = std::mem::take(&mut runtime.selection_seen);
    seen.clear();
    let mut deferred_urgent = Vec::new();
    let _choose = info_span!("gi_select_choose").entered();
    while selected.faces.len() < runtime.batch_size
        && let Some(voxel) = runtime.urgent_pending.pop_front()
    {
        runtime.urgent_queued.remove(&voxel);
        if !seen.insert(voxel) {
            continue;
        }
        if runtime.trace_pending.contains(&voxel) {
            deferred_urgent.push(voxel);
            continue;
        }
        let Some(probe) = field.probes.get(&voxel) else {
            continue;
        };
        let mask = probe.face_mask;
        if mask == 0 {
            continue;
        }
        selected.push(
            probe,
            mask,
            field.transport_revision,
            runtime.source_revision,
            false,
            u32::MAX,
            u32::MAX,
        );
    }
    runtime.selection_seen = seen;
    enqueue_urgent(runtime, deferred_urgent);
    selected
}

pub(super) fn apply_trace_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<ReceiverField>,
    mut stats: ResMut<VoxelGiStats>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
    geometry_display: Option<ResMut<super::render::GeometryDisplay>>,
    view_transport: Option<ResMut<gpu::view::ViewTransport>>,
) {
    let mut timing = timings.main_scope(super::telemetry::MainStage::TracePublication);
    let Some(task) = runtime.trace_task.as_mut() else {
        return;
    };
    let Some(mut result) = block_on(poll_once(task)) else {
        return;
    };
    let merge_started = Instant::now();
    runtime.trace_task = None;
    // A transport that cannot be built (its pipelines fail on this backend or
    // compiler) is like no device: GI idles with one warning. Only failures
    // during a run, below, are fatal.
    if let Some(error) = result.build_error {
        warn!("Voxel GI: the GPU transport could not be built; the world cache is idle. {error}");
        runtime.transport_unavailable = true;
        return;
    }
    // Fail before publishing any partial result or switching execution backend.
    // A slower CPU reference must be an explicit launch choice, never recovery
    // from a GPU validation, submission, or snapshot mapping failure.
    if let Some(error) = result.gpu_error {
        panic!("Voxel GI GPU transport failed: {error}");
    }
    timing.record_gpu_batch(result.batch.as_ref().and_then(|batch| batch.gpu_ms));
    stats.backend = result
        .gpu_tracer
        .as_ref()
        .map_or("cpu", gpu::GpuTracer::backend_name);
    let measured_ms = result
        .batch
        .as_ref()
        .and_then(|batch| batch.execution_ms)
        .unwrap_or(result.trace_time.as_secs_f32() * 1000.0);
    if let Some(tracer) = &result.gpu_tracer {
        stats.scheduler.last_prepare_ms = tracer.last_prepare_ms;
        stats.scheduler.last_snapshot_latency_ms = tracer.last_snapshot_latency_ms;
        stats.scheduler.last_decode_ms = tracer.last_decode_ms;
        stats.scheduler.packed_geometry_bytes = tracer.geometry_storage_bytes();
    }
    if let (Some((buffers, params)), Some(mut display)) = (result.geometry.take(), geometry_display)
    {
        display.buffers = Some(buffers);
        display.params = params;
    }
    if let Some(mut view) = view_transport {
        view.0 = result
            .gpu_tracer
            .as_ref()
            .and_then(gpu::GpuTracer::view_scene);
        if let Some(tracer) = result.gpu_tracer.as_mut() {
            view.1.lock().unwrap().extend(tracer.take_view_changes());
        }
    }
    runtime.gpu_tracer = result.gpu_tracer;
    runtime.emission_cache = result.emission_cache;
    stats.scheduler.last_worker_ms = result.trace_time.as_secs_f32() * 1000.0;
    runtime.gpu_pending = result.pending;
    // Address publication is independent of statistical snapshots. Once the
    // submitted GPU work completes, the renderer sees it without a CPU merge.
    for binding in result.bindings {
        if let Some(probe) = field.probes.get_mut(&binding.voxel)
            && probe.revision == binding.revision
            && binding.source_revision >= runtime.model_reset_source_revision
            && probe.resident[binding.face] != binding.address
        {
            probe.resident[binding.face] = binding.address;
        }
    }
    let Some(mut batch) = result.batch else {
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    };

    stats.scheduler.transport_batches += 1;
    // Charge the measured GPU time against what was reserved.
    runtime.gpu_allowance_ms += batch.gpu_reserved_ms - batch.gpu_ms.unwrap_or(measured_ms);
    let transport_revision = field.transport_revision;
    let _merge = info_span!("gi_merge_jobs").entered();
    for job in &batch.work.probes {
        let voxel = job.voxel;
        runtime.trace_pending.remove(&voxel);
        let Some(old) = field.probes.get_mut(&voxel) else {
            continue;
        };
        if old.revision != job.revision
            || (job.trace_mask != 0 && job.source_revision < runtime.model_reset_source_revision)
        {
            // The GPU already merged this batch; undo it at the next selection.
            old.stale_merge |= job.trace_mask;
            continue;
        }
        if job.trace_mask == 0 {
            continue;
        }
        let (_, old_samples, old_covered, old_settled) = probe_metrics(old, transport_revision);
        // A material-table revision can change emission without rebuilding
        // geometry. Older source results were rejected above.
        old.emission = runtime.materials.get(old.material).emission;
        job.publish(&batch.work.faces[job.faces()], old);
        let (_, new_samples, new_covered, new_settled) = probe_metrics(old, transport_revision);
        runtime.completed_samples =
            runtime.completed_samples.saturating_sub(old_samples) + new_samples;
        runtime.covered_faces = runtime.covered_faces.saturating_sub(old_covered) + new_covered;
        runtime.settled_faces = runtime.settled_faces.saturating_sub(old_settled) + new_settled;
    }
    // The statistics mirror of GPU-picked faces, at most mirror_budget per
    // batch (the GPU emits faces whose mirror is stale; the GPU state is the
    // authority). A face whose probe changed since its slot's revision
    // reached the GPU is skipped: its CPU job will carry the new revision. So
    // is a record older than the probe's last merge (a same-batch CPU job).
    // A GPU pick whose slot's revision is older than its probe's traced
    // pre-change light (it was in flight at the change): the probe's next CPU
    // job undoes it, unless that job is already in flight (submitted after
    // the pick, it capped it). Emitter removal leaves bright stale light
    // otherwise (response capture 1 s after removal: FLIP .38 vs .08).
    for &(voxel, face, tag) in &batch.work.pick_tags {
        if let Some(old) = field.probes.get_mut(&voxel)
            && old.revision as u32 != tag
            && !runtime.trace_pending.contains(&voxel)
        {
            old.stale_merge |= 1 << face;
        }
    }
    let picked = std::mem::take(&mut batch.work.picked);
    let _merge_picks = info_span!("gi_merge_picks").entered();
    for (face, tag) in &picked {
        let voxel = face.voxel;
        let Some(old) = field.probes.get_mut(&voxel) else {
            continue;
        };
        if old.face_mask & (1 << face.face) == 0 {
            continue;
        }
        if old.revision as u32 != *tag
            || batch.work.source_revision < runtime.model_reset_source_revision
            || face.predates(old)
        {
            continue;
        }
        let (_, old_samples, old_covered, old_settled) = probe_metrics(old, transport_revision);
        let job = super::work::ProbeWork::picked(
            old,
            face.face,
            batch.work.transport_revision,
            batch.work.source_revision,
        );
        job.publish(std::slice::from_ref(face), old);
        let (_, new_samples, new_covered, new_settled) = probe_metrics(old, transport_revision);
        runtime.completed_samples =
            runtime.completed_samples.saturating_sub(old_samples) + new_samples;
        runtime.covered_faces = runtime.covered_faces.saturating_sub(old_covered) + new_covered;
        runtime.settled_faces = runtime.settled_faces.saturating_sub(old_settled) + new_settled;
    }
    drop(_merge_picks);
    // Faces traced: the CPU's and the GPU's picks (the mirror records are a
    // bounded subset of those; counting them shrank the budget).
    let picks_traced = batch.work.pick_tags.len();
    let traced_faces = batch.work.faces.len() + picks_traced;
    stats.scheduler.traced_paths = stats
        .scheduler
        .traced_paths
        .saturating_add(picks_traced as u64 * u64::from(settings.batch_samples()));
    drop(_merge);
    batch.work.clear();
    runtime.trace_scratch = batch.work;
    stats.scheduler.traced_paths = stats.scheduler.traced_paths.saturating_add(batch.paths);
    stats.scheduler.last_merge_ms = merge_started.elapsed().as_secs_f32() * 1000.0;
    // Adjust work to measured execution time rather than fixing throughput to
    // 1,024 probes per display frame. Limit growth to avoid sudden GPU bursts.
    stats.scheduler.last_trace_batch_ms = measured_ms;
    runtime.batch_size = scheduling::next_budget(
        runtime.batch_size,
        traced_faces,
        measured_ms,
        settings.budget_ms(),
        settings.batch_limit(),
    );
    update_stats_from_counters(&runtime, &field, &mut stats);
}

#[cfg(test)]
mod budget_tests {
    use super::accrue_gpu_allowance;

    #[test]
    fn gpu_share_spends_a_fraction_of_wall_time_whatever_the_frame_rate() {
        // 20% of 1 s is 200 GPU ms: fifty 4 ms batches, at 60 or 144 Hz.
        for frame_ms in [1000.0 / 60.0, 1000.0 / 144.0] {
            let (mut allowance, mut spent) = (0.0_f32, 0.0_f32);
            for _ in 0..(1000.0 / frame_ms) as usize {
                allowance = accrue_gpu_allowance(allowance, frame_ms, 0.2, 4.0);
                if allowance > 0.0 {
                    allowance -= 4.0;
                    spent += 4.0;
                }
            }
            assert!(
                (spent - 200.0).abs() <= 8.0,
                "{frame_ms} ms frames spent {spent}"
            );
        }
        // Idle time banks at most two batches.
        assert_eq!(accrue_gpu_allowance(0.0, 10_000.0, 0.2, 4.0), 8.0);
    }
}
