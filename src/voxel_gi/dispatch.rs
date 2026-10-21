//! CPU event selection, worker dispatch, and revision-checked publication of
//! CPU jobs and GPU picks. Bulk selection happens on the GPU; every loaded
//! receiver remains eligible independently of camera visibility.
use super::{
    TraceBatchResult, VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, VoxelGiStats, emission,
    enqueue_urgent, geometry, gpu, probe_metrics, scheduling,
    telemetry::GiWorkTimings,
    update_stats_from_counters,
    work::TraceWork,
};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, block_on, poll_once},
};
use std::{sync::Arc, time::Instant};

pub(super) fn append_sky_work(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
    selected: &mut TraceWork,
) -> usize {
    // Collect explicit sky demand, then attach it to existing transport jobs.
    // Removing overlaps is linear in both queues, not a search per candidate.
    let seen = &mut runtime.selection_seen;
    seen.clear();
    let mut count = 0;
    for _ in 0..runtime.sky_batch_size {
        let voxel = if let Some(voxel) = runtime.sky_pending.pop_front() {
            runtime.sky_queued.remove(&voxel);
            voxel
        } else if let Some(voxel) = runtime.sky_band.pop() {
            voxel
        } else if runtime.sky_remaining > 0 && !field.order.is_empty() {
            let voxel = field.order[runtime.sky_cursor % field.order.len()];
            runtime.sky_cursor = (runtime.sky_cursor + 1) % field.order.len();
            runtime.sky_remaining -= 1;
            voxel
        } else {
            break;
        };
        if runtime.trace_pending.contains(&voxel) {
            runtime.sky_pending.push_back(voxel);
            runtime.sky_queued.insert(voxel);
            continue;
        }
        seen.insert(voxel);
    }
    for job in &mut selected.probes {
        if seen.remove(&job.voxel) {
            job.sky_requested = true;
        }
    }
    for voxel in seen.iter() {
        if let Some(probe) = field.probes.get(voxel) {
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
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    gpu_device: Option<Res<gpu::GpuTransportDevice>>,
    settings: Res<VoxelGiSettings>,
    frame: Option<Res<bevy::diagnostic::FrameCount>>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::TraceDispatch);
    if runtime.trace_task.is_some() || (field.order.is_empty() && runtime.gpu_pending.is_empty()) {
        return;
    }
    let gpu_device = gpu_device.expect("Voxel GI requires a GPU transport device");
    let selection_started = Instant::now();
    let requested = std::mem::take(&mut field.requested);
    enqueue_urgent(
        &mut runtime,
        requested
            .into_iter()
            .filter(|voxel| field.probes.contains_key(voxel)),
    );
    let focus = cameras
        .iter()
        .next()
        .map_or(IVec3::ZERO, |camera| camera.translation().floor().as_ivec3());
    runtime.batch_size = runtime.batch_size.min(settings.batch_limit());
    runtime.sky_batch_size = runtime.sky_batch_size.min(settings.batch_limit());
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
        settings.gpu_share(),
        settings.budget_ms(),
    );
    let can_submit = !field.order.is_empty()
        && runtime.gpu_pending.len() < 3
        && runtime.gpu_allowance_ms > 0.0;
    let mut selected = if can_submit {
        info_span!("gi_select").in_scope(|| select_transport(&mut runtime, &field, &settings))
    } else {
        TraceWork::default()
    };
    selected.transport = runtime.transport;
    selected.response_seconds = settings.display_response_seconds();
    selected.measured_retention = settings.measured_retention();
    selected.display_frame = frame.as_ref().map_or(0, |frame| frame.0);
    selected.display_transition_frames = settings.display_transition_frames.min(32);
    selected.spatial_reconstruction = settings.spatial_reconstruction;
    selected.spatial_evidence_target = settings.spatial_evidence_target.min(4096);
    selected.spatial_max_donors = settings.spatial_max_donors;
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
    let transport_count = selected.probes.len();
    let sky_count = if can_submit {
        info_span!("gi_sky_plan").in_scope(|| append_sky_work(&mut runtime, &field, &mut selected))
    } else {
        0
    };
    let sky_only = transport_count == 0 && selected.picks == 0;
    // Batch IDs count submissions only: the GPU protects a CPU job's face for
    // four IDs and re-ranks the quarter `batch_id % 4`, so poll-only frames
    // must not advance it (they expired ownership of unpublished jobs).
    if !selected.probes.is_empty() || selected.picks > 0 {
        runtime.batch_counter = (runtime.batch_counter % 0x0fff_ffff) + 1;
        selected.batch_id = runtime.batch_counter;
    }
    if selected.probes.is_empty() && selected.picks == 0 && runtime.gpu_pending.is_empty() {
        runtime.trace_scratch = selected;
        stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
        stats.idle_scheduler_frames += 1;
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    }
    runtime
        .trace_pending
        .extend(selected.probes.iter().map(|job| job.voxel));
    // Every published change rides with the next dispatched batch, whatever
    // it selected: the GPU applies each to all resident faces in range.
    selected.retention_events = runtime.spatial_history.take_events();
    let paths = selected.faces.len() as u64 * u64::from(samples);
    stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
    let scene = Arc::clone(&runtime.scene);
    let batch_submits = !selected.probes.is_empty() || selected.picks > 0;
    let gpu_reserved_ms = if batch_submits { settings.budget_ms() } else { 0.0 };
    runtime.gpu_allowance_ms -= gpu_reserved_ms;
    let mut batch = super::TransportBatch {
        work: selected,
        paths,
        sky_only,
        sky_count,
        gpu_reserved_ms,
        scene_generation: runtime.generation,
        sky_revision: runtime.sky_revision,
        ..default()
    };
    let sources = runtime.sources.clone();
    let materials = Arc::clone(&runtime.materials);
    let mut gpu_tracer = runtime.gpu_tracer.take();
    let mut pending = std::mem::take(&mut runtime.gpu_pending);
    let emission_cache = runtime.emission_cache.take();
    let sampling_importance = settings.sampling_importance();
    let gpu_device = gpu_device.clone();
    runtime.trace_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let _span = info_span!("gi_trace_worker").entered();
        let started = Instant::now();
        let focus = batch.work.focus;
        let dispatched = !batch.work.probes.is_empty();
        let emission_cache = if sky_only {
            emission_cache
        } else {
            Some(emission::EmissionCache::prepare_with_importance(
                emission_cache,
                Arc::clone(&scene),
                Arc::clone(&materials),
                sampling_importance,
            ))
        };
        let emitters = emission_cache.as_ref().map_or_else(
            || Arc::new(emission::EmissionDistribution::default()),
            |cache| Arc::clone(&cache.distribution),
        );
        let mut gpu_error = None;
        if gpu_tracer.is_none() {
            match gpu::GpuTracer::new(gpu_device) {
                Ok(mut tracer) => {
                    if let Err(error) = tracer.enable_resident() {
                        gpu_error = Some(error);
                    } else {
                        gpu_tracer = Some(tracer);
                    }
                }
                Err(error) => gpu_error = Some(error),
            }
        }
        let mut completed = None;
        if let Some(tracer) = &mut gpu_tracer {
            tracer.queue_retention(&mut batch.work.retention_events);
            tracer.sky_cache.track_changes = true;
            tracer.sky_cache.deferred_results = true;
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
                            info_span!("gi_readback")
                                .in_scope(|| tracer.complete_readback(snapshot, &mut oldest.batch.work))?;
                            oldest.batch.gpu_ms = tracer.last_gpu_ms;
                            oldest.batch.execution_ms = Some(tracer.last_dispatch_ms);
                        } else {
                            oldest.batch.gpu_ms = Some(0.0);
                            oldest.batch.execution_ms = Some(0.0);
                        }
                        completed = Some(oldest.batch);
                    }
                    if !batch.work.probes.is_empty() || batch.work.picks > 0 {
                        let _submit = info_span!("gi_submit").entered();
                        let snapshot = tracer.submit_compact(
                            Arc::clone(&scene),
                            &sources,
                            &materials,
                            &mut batch.work,
                            samples,
                            &emitters,
                            sky_only,
                        )?;
                        pending.push_back(super::PendingTransport {
                            snapshot,
                            batch: std::mem::take(&mut batch),
                        });
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
        let sunlight_change = gpu_tracer
            .as_mut()
            .and_then(|tracer| tracer.sky_cache.take_changes(&scene));
        let sky_band = gpu_tracer
            .as_mut()
            .map_or_else(Vec::new, |tracer| tracer.sky_cache.take_band(focus));
        let bindings = gpu_tracer
            .as_mut()
            .map_or_else(Vec::new, gpu::GpuTracer::take_resident_bindings);
        let sky_stats = dispatched
            .then(|| gpu_tracer.as_ref().map(|tracer| tracer.sky_cache.stats))
            .flatten();
        TraceBatchResult {
            batch: completed,
            pending,
            bindings,
            sky_stats,
            trace_time: started.elapsed(),
            gpu_tracer,
            emission_cache,
            gpu_error,
            sunlight_change,
            sky_band,
        }
    }));
}

/// At least one quarter of dispatch slots belongs to a fair loaded-world
/// sweep. Coverage and refinement also have reserved opportunities, so streamed
/// bootstrap work cannot indefinitely preempt refinement near the player.
#[cfg(test)]
pub(super) fn select_probes(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
) -> Vec<super::world::SurfaceProbe> {
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

/// The CPU selects only event work: faces an edit, relight or explicit
/// request made urgent, with their revision caps and rollbacks. Bulk
/// coverage, refinement and the world sweep are picked on the GPU from its
/// own statistics (`select_*` in voxel_accumulate.wgsl), in the same batch.
pub(super) fn select_transport(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
    settings: &VoxelGiSettings,
) -> TraceWork {
    let now = Instant::now();
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
        let mask = scheduling::due_faces(probe, field.transport_revision, now, settings, true);
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
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
    resident_display: Option<ResMut<super::render::ResidentDisplay>>,
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
    // Fail before publishing any partial result or switching execution backend.
    // A slower CPU reference must be an explicit launch choice, never recovery
    // from a GPU validation, submission, or snapshot mapping failure.
    if let Some(error) = result.gpu_error {
        panic!("Voxel GI GPU transport failed: {error}");
    }
    if !result.sky_band.is_empty() {
        // Newer requests pop first (each band is sorted nearest last). Bound
        // leftovers: the world sweep still reaches anything dropped.
        if runtime.sky_band.len() > 1 << 20 {
            runtime.sky_band.clear();
        }
        runtime.sky_band.append(&mut result.sky_band);
    }
    if let Some(mut change) = result.sunlight_change.take() {
        // Even superseded results describe a possible change, so retain their
        // conservative evidence. They cannot publish stale sky display values.
        // Jobs carry their *selection* stamp, leaving late evidence pending.
        // A superseded worker must not recreate metadata for retired pages.
        change
            .nearby
            .retain(|page, _| field.pages.contains_key(page));
        runtime.spatial_history.record(change, &[]);
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
    if let Some(sky) = result.sky_stats {
        stats.sky_corner_requests += sky.requested as u64;
        stats.sky_corners_traced += sky.traced as u64;
        stats.sky_rays_scheduled += sky.rays as u64;
        stats.sky_cached_corners = sky.resident;
    }
    if let Some(tracer) = &result.gpu_tracer {
        stats.last_prepare_ms = tracer.last_prepare_ms;
        stats.last_snapshot_latency_ms = tracer.last_snapshot_latency_ms;
        stats.last_decode_ms = tracer.last_decode_ms;
        stats.packed_geometry_bytes = tracer.geometry_storage_bytes();
    }
    if let Some(mut display) = resident_display {
        display.buffer = result
            .gpu_tracer
            .as_ref()
            .and_then(gpu::GpuTracer::resident_buffer);
    }
    runtime.gpu_tracer = result.gpu_tracer;
    runtime.emission_cache = result.emission_cache;
    stats.last_worker_ms = result.trace_time.as_secs_f32() * 1000.0;
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
            runtime.mark_display_dirty(binding.voxel);
        }
    }
    let Some(mut batch) = result.batch else {
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    };

    // Mixed submissions share one timestamp interval. Do not attribute the
    // indirect kernel's cost to sky and accidentally collapse its scan budget.
    if batch.sky_only && batch.sky_count > 0 {
        runtime.sky_batch_size = scheduling::next_budget(
            runtime.sky_batch_size,
            batch.sky_count,
            measured_ms,
            settings.budget_ms().min(1.0),
            settings.batch_limit(),
        );
    }
    if batch.sky_only {
        stats.sky_only_batches += 1;
    } else {
        stats.transport_batches += 1;
    }
    // Charge the measured GPU time against what was reserved.
    runtime.gpu_allowance_ms += batch.gpu_reserved_ms - batch.gpu_ms.unwrap_or(measured_ms);
    let sky_current =
        batch.scene_generation == runtime.generation && batch.sky_revision == runtime.sky_revision;
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
        if sky_current && job.sky_requested {
            let changed = (0..6).any(|face| {
                old.face_mask & (1 << face) != 0 && old.display[face].sky != job.sky[face]
            });
            for face in 0..6 {
                if old.face_mask & (1 << face) != 0 {
                    old.display[face].sky = job.sky[face];
                }
            }
            if changed {
                runtime.mark_display_dirty(voxel);
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
        let publish_display = batch.work.faces[job.faces()]
            .iter()
            .any(|face| face.resident.y == 0 || old.resident[face.face] != face.resident);
        job.publish(&batch.work.faces[job.faces()], old, merge_started, &settings);
        let (_, new_samples, new_covered, new_settled) = probe_metrics(old, transport_revision);
        runtime.completed_samples =
            runtime.completed_samples.saturating_sub(old_samples) + new_samples;
        runtime.covered_faces = runtime.covered_faces.saturating_sub(old_covered) + new_covered;
        runtime.settled_faces = runtime.settled_faces.saturating_sub(old_settled) + new_settled;
        if publish_display {
            runtime.mark_display_dirty(voxel);
        }
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
        let publish_display = old.resident[face.face] != face.resident;
        let job = super::work::ProbeWork::picked(
            old,
            face.face,
            batch.work.transport_revision,
            batch.work.source_revision,
        );
        job.publish(std::slice::from_ref(face), old, merge_started, &settings);
        let (_, new_samples, new_covered, new_settled) = probe_metrics(old, transport_revision);
        runtime.completed_samples =
            runtime.completed_samples.saturating_sub(old_samples) + new_samples;
        runtime.covered_faces = runtime.covered_faces.saturating_sub(old_covered) + new_covered;
        runtime.settled_faces = runtime.settled_faces.saturating_sub(old_settled) + new_settled;
        if publish_display {
            runtime.mark_display_dirty(voxel);
        }
    }
    drop(_merge_picks);
    // Faces traced: the CPU's and the GPU's picks (the mirror records are a
    // bounded subset of those; counting them shrank the budget).
    let picks_traced = batch.work.pick_tags.len();
    let traced_faces = batch.work.faces.len() + picks_traced;
    stats.traced_paths = stats
        .traced_paths
        .saturating_add(picks_traced as u64 * u64::from(settings.batch_samples()));
    drop(_merge);
    batch.work.clear();
    runtime.trace_scratch = batch.work;
    stats.traced_paths = stats.traced_paths.saturating_add(batch.paths);
    stats.last_merge_ms = merge_started.elapsed().as_secs_f32() * 1000.0;
    // Adjust work to measured execution time rather than fixing throughput to
    // 1,024 probes per display frame. Limit growth to avoid sudden GPU bursts.
    if !batch.sky_only {
        stats.last_trace_batch_ms = measured_ms;
        runtime.batch_size = scheduling::next_budget(
            runtime.batch_size,
            traced_faces,
            measured_ms,
            settings.budget_ms(),
            settings.batch_limit(),
        );
    }
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
            assert!((spent - 200.0).abs() <= 8.0, "{frame_ms} ms frames spent {spent}");
        }
        // Idle time banks at most two batches.
        assert_eq!(accrue_gpu_allowance(0.0, 10_000.0, 0.2, 4.0), 8.0);
    }
}
