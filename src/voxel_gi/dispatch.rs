//! Bounded selection, worker dispatch, and revision-checked measurement publication.
//! Every loaded receiver remains eligible independently of camera visibility.
use super::{
    TraceBatchResult, VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, VoxelGiStats, emission,
    enqueue_many, enqueue_urgent, geometry, gpu, page_of, probe_metrics, scheduling,
    telemetry::GiWorkTimings,
    tracer::{BOOTSTRAP_SAMPLES, trace_work_batch},
    update_stats_from_counters,
    work::TraceWork,
    world::SurfaceProbe,
};
use bevy::{
    prelude::*,
    tasks::{AsyncComputeTaskPool, block_on, poll_once},
};
use rayon::ThreadPoolBuilder;
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

pub(super) fn start_trace_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    gpu_device: Option<Res<gpu::GpuTransportDevice>>,
    settings: Res<VoxelGiSettings>,
    startup: Option<Res<super::VoxelGiStartup>>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::TraceDispatch);
    if runtime.trace_task.is_some() || (field.order.is_empty() && runtime.gpu_pending.is_empty()) {
        return;
    }
    let selection_started = Instant::now();
    let reuse = startup
        .as_ref()
        .is_some_and(|s| s.backend == super::TransportBackend::Reuse);
    let resident = reuse || startup.is_some_and(|s| s.backend == super::TransportBackend::Resident);
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
    // Three owned staging allocations bound queue latency and memory. When full,
    // poll without selecting more work: never turn backpressure into a CPU wait.
    let can_submit = !field.order.is_empty() && runtime.gpu_pending.len() < 3;
    let mut selected = if can_submit {
        select_transport(&mut runtime, &field, &settings)
    } else {
        TraceWork::default()
    };
    selected.transport = runtime.transport;
    selected.response_seconds = settings.display_response_seconds();
    let samples = settings.batch_samples();
    let transport_count = selected.probes.len();
    let sky_count = if can_submit {
        append_sky_work(&mut runtime, &field, &mut selected)
    } else {
        0
    };
    let sky_only = transport_count == 0;
    if selected.probes.is_empty() && runtime.gpu_pending.is_empty() {
        runtime.trace_scratch = selected;
        stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
        stats.idle_scheduler_frames += 1;
        update_stats_from_counters(&runtime, &field, &mut stats);
        return;
    }
    runtime
        .trace_pending
        .extend(selected.probes.iter().map(|job| job.voxel));
    let paths = selected.faces.len() as u64 * u64::from(samples);
    stats.last_selection_ms = selection_started.elapsed().as_secs_f32() * 1000.0;
    let scene = Arc::clone(&runtime.scene);
    let mut batch = super::TransportBatch {
        work: selected,
        paths,
        sky_only,
        transport_count,
        sky_count,
        scene_generation: runtime.generation,
        sky_revision: runtime.sky_revision,
        ..default()
    };
    let track_changes = runtime.spatial_history.is_some();
    let sources = runtime.sources.clone();
    let materials = Arc::clone(&runtime.materials);
    let pool = Arc::clone(&runtime.trace_pool);
    let mut gpu_tracer = runtime.gpu_tracer.take();
    let mut pending = std::mem::take(&mut runtime.gpu_pending);
    let emission_cache = runtime.emission_cache.take();
    let mut cpu_sky_cache = runtime.cpu_sky_cache.take().unwrap_or_default();
    let gpu_device = gpu_device
        .filter(|_| !runtime.gpu_failed)
        .map(|device| device.clone());
    runtime.trace_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let started = Instant::now();
        let dispatched = !batch.work.probes.is_empty();
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
                Ok(mut tracer) => {
                    let enabled = if reuse {
                        tracer.enable_reuse()
                    } else if resident {
                        tracer.enable_resident()
                    } else {
                        Ok(())
                    };
                    if let Err(error) = enabled {
                        gpu_error = Some(error);
                    } else {
                        gpu_tracer = Some(tracer);
                    }
                }
                Err(error) => gpu_error = Some(error),
            }
        }
        batch.work.apply_retention();
        cpu_sky_cache.track_changes = track_changes;
        let mut completed = None;
        if let Some(tracer) = &mut gpu_tracer {
            tracer.sky_cache.track_changes = track_changes;
            tracer.sky_cache.deferred_results = resident;
            let result = (|| -> Result<(), String> {
                if resident {
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
                            tracer.complete_readback(snapshot, &mut oldest.batch.work)?;
                            oldest.batch.gpu_ms = tracer.last_gpu_ms;
                            oldest.batch.execution_ms = Some(tracer.last_dispatch_ms);
                        } else {
                            oldest.batch.gpu_ms = Some(0.0);
                            oldest.batch.execution_ms = Some(0.0);
                        }
                        completed = Some(oldest.batch);
                    }
                    if !batch.work.probes.is_empty() {
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
                } else {
                    tracer.trace_compact(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut batch.work,
                        samples,
                        &emitters,
                        sky_only,
                    )?;
                    batch.gpu_ms = tracer.last_gpu_ms;
                    batch.execution_ms = Some(tracer.last_dispatch_ms);
                    completed = Some(std::mem::take(&mut batch));
                }
                Ok(())
            })();
            if let Err(error) = result {
                gpu_error = Some(error);
                gpu_tracer = None;
                pending.clear();
            }
        }
        if gpu_tracer.is_none() && !batch.work.probes.is_empty() {
            let pool = pool.get_or_init(|| {
                let available = std::thread::available_parallelism().map_or(2, usize::from);
                ThreadPoolBuilder::new()
                    .num_threads(available.saturating_sub(2).clamp(1, 2))
                    .thread_name(|index| format!("voxel-gi-{index}"))
                    .build()
                    .expect("voxel GI worker pool")
            });
            for face in &mut batch.work.faces {
                face.resident = UVec2::ZERO;
            }
            pool.install(|| {
                trace_work_batch(
                    &scene,
                    &sources,
                    &materials,
                    &mut batch.work,
                    if sky_only { 0 } else { samples },
                    &emitters,
                    &mut cpu_sky_cache,
                )
            });
            completed = Some(batch);
        }
        let sunlight_change = gpu_tracer.as_mut().map_or_else(
            || cpu_sky_cache.take_changes(&scene),
            |tracer| tracer.sky_cache.take_changes(&scene),
        );
        let bindings = gpu_tracer
            .as_mut()
            .map_or_else(Vec::new, gpu::GpuTracer::take_resident_bindings);
        let sky_stats = dispatched.then(|| {
            gpu_tracer
                .as_ref()
                .map_or(cpu_sky_cache.stats, |tracer| tracer.sky_cache.stats)
        });
        TraceBatchResult {
            batch: completed,
            pending,
            bindings,
            sky_stats,
            trace_time: started.elapsed(),
            gpu_tracer,
            emission_cache,
            cpu_sky_cache,
            gpu_error,
            sunlight_change,
        }
    }));
}

/// Only receiver membership inside the local sampling window invalidates its
/// index. Face/material changes still refresh transport, but do not change this
/// voxel list. Distant streaming must not restart a player's refinement sweep.
pub(super) fn invalidate_focus_page(runtime: &mut VoxelGiRuntime, page: IVec3) {
    if runtime
        .focus_page
        .is_some_and(|focus| (page.as_i64vec3() - focus.as_i64vec3()).abs().max_element() <= 2)
    {
        runtime.focus_dirty = true;
    }
}

pub(super) fn update_focus(runtime: &mut VoxelGiRuntime, field: &VoxelEnergyField, focus: IVec3) {
    let moved = runtime.focus_page != Some(focus);
    if !moved && !runtime.focus_dirty {
        return;
    }
    runtime.focus_page = Some(focus);
    runtime.focus_dirty = false;
    runtime.nearby.clear();
    runtime.regional.clear();
    // Constant-size page lookup, independent of total loaded-world size.
    let mut pages: Vec<_> = (-2..=2)
        .flat_map(|z| (-2..=2).flat_map(move |y| (-2..=2).map(move |x| IVec3::new(x, y, z))))
        .map(|offset| focus + offset)
        .filter(|page| field.pages.contains_key(page))
        .collect();
    pages.sort_unstable_by_key(|page| ((*page - focus).length_squared(), page.z, page.y, page.x));
    for page in pages {
        if (page - focus).abs().max_element() <= 1 {
            runtime.nearby.extend_from_slice(&field.pages[&page]);
        } else {
            runtime.regional.extend_from_slice(&field.pages[&page]);
        }
    }
    if moved {
        runtime.nearby_cursor = 0;
    }
}

/// At least one quarter of dispatch slots belongs to a fair loaded-world
/// sweep. Coverage and refinement also have reserved opportunities, so streamed
/// bootstrap work cannot indefinitely preempt refinement near the player.
#[cfg(test)]
pub(super) fn select_probes(
    runtime: &mut VoxelGiRuntime,
    field: &VoxelEnergyField,
) -> Vec<SurfaceProbe> {
    // Reference view for existing policy tests. Production never materializes
    // these full receiver copies; it transfers only TraceWork.
    let mut work = select_transport(runtime, field, &VoxelGiSettings::default());
    work.apply_retention();
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

pub(super) fn select_transport(
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
    let mut deferred_urgent = Vec::new();
    adaptive.clear();
    if runtime.urgent_pending.len() < runtime.batch_size {
        adaptive.refresh(
            field,
            &runtime.nearby,
            &runtime.regional,
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
            } else {
                // One quarter coverage, one quarter global, and half adaptive
                // refinement. Empty queues lend their slots; explicit edits
                // still take priority in all three non-global slots. Keeping a
                // refinement reservation is essential while terrain streams.
                if index % 4 == 1
                    && let Some(next) = runtime.pending.pop_front()
                {
                    runtime.queued.remove(&next);
                    voxel = Some(next);
                }
                if voxel.is_none() {
                    while let Some(candidate) = adaptive.pop() {
                        if !seen.contains(&candidate) {
                            voxel = Some(candidate);
                            break;
                        }
                    }
                }
                if voxel.is_none() && !runtime.nearby.is_empty() {
                    voxel = Some(runtime.nearby[runtime.nearby_cursor % runtime.nearby.len()]);
                    runtime.nearby_cursor = (runtime.nearby_cursor + 1) % runtime.nearby.len();
                }
                if voxel.is_none()
                    && let Some(next) = runtime.pending.pop_front()
                {
                    runtime.queued.remove(&next);
                    voxel = Some(next);
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
        if runtime.trace_pending.contains(&voxel) {
            if force {
                deferred_urgent.push(voxel);
            }
            continue;
        }
        // A reserved sweep slot can encounter an explicitly requested voxel
        // before its urgent queue entry. Honor the force now, not after it has
        // already been selected (or skipped as settled).
        force |= runtime.urgent_queued.contains(&voxel);
        let Some(probe) = field.probes.get(&voxel) else {
            continue;
        };
        let retention = runtime
            .spatial_history
            .as_ref()
            .map(|history| history.snapshot(voxel));
        // Conservative page wakeup is O(1). Exact per-receiver distances are
        // evaluated by the worker; page-edge false positives may get an audit,
        // but never an unjustified cap or a lost late sunlight notification.
        force |= retention
            .as_ref()
            .is_some_and(|snapshot| snapshot.pending(probe.retention_revision));
        let mask = scheduling::due_faces(probe, field.transport_revision, now, settings, force);
        if mask == 0 {
            continue;
        }
        let mut restart = false;
        let mut history_limit = u32::MAX;
        if probe.source_revision != runtime.source_revision {
            restart = probe.source_revision < runtime.model_reset_source_revision;
            history_limit = runtime.source_history.history_limit(
                probe.voxel,
                probe.source_revision,
                runtime.source_revision,
            );
        }
        let geometry_limit = retention.as_ref().map_or_else(
            || {
                runtime.geometry_history.history_limit(
                    voxel,
                    probe.transport_revision,
                    field.transport_revision,
                )
            },
            |_| u32::MAX,
        );
        #[cfg(debug_assertions)]
        let (geometry_limit, history_limit) = (
            super::validation::retained_history_limit(geometry_limit),
            super::validation::retained_history_limit(history_limit),
        );
        selected.push(
            probe,
            mask,
            field.transport_revision,
            runtime.source_revision,
            restart,
            geometry_limit,
            history_limit,
        );
        if let Some(snapshot) = retention {
            selected.defer_retention(probe.retention_revision, snapshot);
        }
    }
    runtime.selection_seen = seen;
    runtime.ranked_candidates = adaptive;
    enqueue_urgent(runtime, deferred_urgent);
    selected
}

pub(super) fn probe_has_pending_samples(probe: &SurfaceProbe) -> bool {
    (0..6).any(|face| probe.face_mask & (1 << face) != 0 && probe.samples[face] < BOOTSTRAP_SAMPLES)
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
    if let Some(mut change) = result.sunlight_change.take()
        && let Some(history) = &mut runtime.spatial_history
    {
        // Even superseded results describe a possible change, so retain their
        // conservative evidence. They cannot publish stale sky display values.
        // Jobs carry their *selection* stamp, leaving late evidence pending.
        // A superseded worker must not recreate metadata for retired pages.
        change
            .nearby
            .retain(|page, _| field.pages.contains_key(page));
        history.record(change, super::retention::Origin::Sunlight);
    }
    timing.record_gpu_batch(result.batch.as_ref().and_then(|batch| batch.gpu_ms));
    if let Some(error) = result.gpu_error {
        warn!(%error, "GPU transport unavailable; continuing with bounded CPU transport");
        runtime.gpu_failed = true;
        let retry: Vec<_> = runtime.trace_pending.drain().collect();
        enqueue_urgent(&mut runtime, retry);
    }
    stats.backend = if result.gpu_tracer.is_some() {
        "gpu"
    } else {
        "cpu"
    };
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
        stats.last_wait_ms = tracer.last_wait_ms;
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
    runtime.cpu_sky_cache = Some(result.cpu_sky_cache);
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
    let sky_current =
        batch.scene_generation == runtime.generation && batch.sky_revision == runtime.sky_revision;
    let mut requeue = Vec::new();
    let transport_revision = field.transport_revision;
    for job in &batch.work.probes {
        let voxel = job.voxel;
        runtime.trace_pending.remove(&voxel);
        let Some(old) = field.probes.get_mut(&voxel) else {
            continue;
        };
        if old.revision != job.revision
            || (job.trace_mask != 0 && job.source_revision < runtime.model_reset_source_revision)
        {
            if probe_has_pending_samples(old) {
                requeue.push(voxel);
            }
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
        #[cfg(debug_assertions)]
        runtime
            .retention_audit
            .record(job, old, merge_started, sky_current);
        // A material-table revision can change emission without rebuilding
        // geometry. Older source results were rejected above.
        old.emission = runtime.materials.get(old.material).emission;
        let publish_display = batch.work.faces[job.faces()]
            .iter()
            .any(|face| face.resident.y == 0 || old.resident[face.face] != face.resident);
        job.publish(
            &batch.work.faces[job.faces()],
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
        if publish_display {
            runtime.mark_display_dirty(voxel);
        }
        if pending {
            requeue.push(voxel);
        }
    }
    batch.work.clear();
    runtime.trace_scratch = batch.work;
    enqueue_many(&mut runtime, requeue);
    stats.traced_paths = stats.traced_paths.saturating_add(batch.paths);
    stats.last_merge_ms = merge_started.elapsed().as_secs_f32() * 1000.0;
    // Adjust work to measured execution time rather than fixing throughput to
    // 1,024 probes per display frame. Limit growth to avoid sudden GPU bursts.
    if !batch.sky_only {
        stats.last_trace_batch_ms = measured_ms;
        runtime.batch_size = scheduling::next_budget(
            runtime.batch_size,
            batch.transport_count,
            measured_ms,
            settings.budget_ms(),
            settings.batch_limit(),
        );
    }
    update_stats_from_counters(&runtime, &field, &mut stats);
}
