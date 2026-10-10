//! Main-world scene publication. Geometry is prepared off-thread; receiver pages
//! publish atomically with retained measurements and display-only initialization.
use super::{
    SceneBuildResult, VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, VoxelGiStats,
    advance_transport_revision, diagnostics, enqueue_many, enqueue_sky, enqueue_urgent, history,
    page_of, probe_metrics, request_global_refresh,
    telemetry::GiWorkTimings,
    update_stats_from_counters,
    world::{PAGE_EDGE, SurfaceProbe, SurfaceProbeDescriptor},
};
use bevy::{
    platform::collections::HashSet,
    prelude::*,
    tasks::{block_on, poll_once},
};
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

const EDIT_RELIGHT_RADIUS: i32 = super::geometry::RELIGHT_RADIUS;

pub(super) fn apply_scene_task(
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
        // Scene edits, including emitting voxels, share spatial history
        // retention. Source/model resets belong to changes of the global
        // material table or transport model, not residency of one source.
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

pub(super) fn publish_scene_page(
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

pub(super) fn invalidate_near_voxel(
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
