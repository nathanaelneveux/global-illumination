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
    platform::collections::{HashMap, HashSet},
    prelude::*,
    tasks::{block_on, poll_once},
};
use std::time::{Duration, Instant};

const EDIT_RELIGHT_RADIUS: i32 = super::geometry::RELIGHT_RADIUS;

/// Page-level conservative donor availability. New streamed page interiors
/// cannot inherit from another page three or more cells away; skip their empty
/// hash searches. Existing local receivers and reachable border pages retain
/// the exact same donor/connectivity tests in `seed_face`.
struct DonorPages([bool; 27]);

impl DonorPages {
    fn new(page: IVec3, local: bool, pages: &HashMap<IVec3, Vec<IVec3>>) -> Self {
        Self(std::array::from_fn(|i| {
            let offset = IVec3::new(i as i32 % 3 - 1, i as i32 / 3 % 3 - 1, i as i32 / 9 - 1);
            if offset == IVec3::ZERO {
                local
            } else {
                pages.contains_key(&(page + offset))
            }
        }))
    }

    fn can_seed(&self, voxel: IVec3) -> bool {
        if self.0[13] {
            return true;
        }
        let local = voxel.rem_euclid(IVec3::splat(PAGE_EDGE as i32));
        let lo = |v| if v < history::DONOR_REACH { -1 } else { 0 };
        let hi = |v| {
            if v >= PAGE_EDGE as i32 - history::DONOR_REACH {
                1
            } else {
                0
            }
        };
        for z in lo(local.z)..=hi(local.z) {
            for y in lo(local.y)..=hi(local.y) {
                for x in lo(local.x)..=hi(local.x) {
                    if self.0[((z + 1) * 9 + (y + 1) * 3 + x + 1) as usize] {
                        return true;
                    }
                }
            }
        }
        false
    }
}

pub(super) fn apply_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut field: ResMut<VoxelEnergyField>,
    mut stats: ResMut<VoxelGiStats>,
    settings: Res<VoxelGiSettings>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::ScenePublication);
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
        let donors = DonorPages::new(page, !old_affected.is_empty(), &field.pages);
        for descriptor in &descriptors {
            if !donors.can_seed(descriptor.voxel) {
                continue;
            }
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
            // Timestamp the new display prior independently of material ID.
            // This is not measurement evidence or a transport freshness stamp.
            probe.display_updated_at = Some(now);
            Box::new(probe)
        });
        let old_mask = probe.face_mask;
        let display_changed = invalidated.contains(&descriptor.voxel)
            || old_mask != descriptor.face_mask
            || probe.material != descriptor.material
            || probe.emission != runtime.materials.get(descriptor.material).emission;
        if probe.material != descriptor.material {
            // Incident energy is defined by position/orientation, not the
            // receiver's material ID. Treat replacement as a local scene edit;
            // retain bounded stale measurements and reject pre-edit jobs.
            probe.request_relight();
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
            runtime.mark_display_dirty(descriptor.voxel);
        }
        field.probes.insert_box(descriptor.voxel, probe);
    }
    let topology_changed =
        old_count != field.probes.len() || !old_affected.is_empty() || !added.is_empty();
    if topology_changed {
        super::dispatch::invalidate_focus_page(runtime, page);
        runtime.gpu_removed.extend(old_affected.keys().copied());
        runtime.gpu_added.extend(added.iter().copied());
        field.update_indices(&added, &old_affected);
    }
    if topology_changed || face_topology_changed {
        runtime.gpu_topology_dirty = true;
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

#[cfg(test)]
mod donor_tests {
    use super::*;

    #[test]
    fn donor_page_bounds_include_borders_and_exclude_empty_interiors() {
        let page = IVec3::new(-2, 0, 3);
        let mut pages = HashMap::new();
        for offset in [IVec3::X, IVec3::NEG_Y, IVec3::ONE] {
            pages.insert(page + offset, vec![(page + offset) * PAGE_EDGE as i32]);
        }
        let donors = DonorPages::new(page, false, &pages);
        let origin = page * PAGE_EDGE as i32;
        for local in [IVec3::splat(3), IVec3::splat(16), IVec3::splat(28)] {
            assert!(!donors.can_seed(origin + local));
        }
        for local in [
            IVec3::new(29, 16, 16),
            IVec3::new(16, 2, 16),
            IVec3::splat(31),
        ] {
            assert!(donors.can_seed(origin + local));
        }
        assert!(!donors.can_seed(origin + IVec3::new(0, 16, 16)));
        let local = DonorPages::new(page, true, &pages);
        assert!(local.can_seed(origin + IVec3::splat(16)));
        assert!(!DonorPages::new(page, false, &HashMap::new()).can_seed(origin));
    }

    #[test]
    #[ignore = "paired CPU donor-culling benchmark; no GPU/frame timing"]
    fn streamed_page_seed_culling_matches_unculled_priors() {
        use super::super::naadf::{linear_index, volume_len};
        use super::super::world::{AIR, PAGE_SIZE, TransportScene};
        let mut floor = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                floor[linear_index(UVec3::new(x, 0, z), PAGE_SIZE)] = 2;
            }
        }
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::NEG_X, floor.clone()), (IVec3::ZERO, floor)],
            &[],
        );
        let now = Instant::now();
        let mut measured = HashMap::new();
        for z in 0..32 {
            for x in -32..0 {
                let p = IVec3::new(x, 0, z);
                let mut probe = SurfaceProbe::new(p, 4);
                probe.samples[2] = 64;
                probe.face_updated_at[2] = Some(now);
                probe.display[2].indirect = Vec3::new(1.0, z as f32 / 32.0, 0.25);
                probe.display[2].confidence = 0.9;
                measured.insert(p, probe);
            }
        }
        let pages = [(IVec3::NEG_X, measured.keys().copied().collect())]
            .into_iter()
            .collect();
        let donors = DonorPages::new(IVec3::ZERO, false, &pages);
        let receivers: Vec<_> = (0..32)
            .flat_map(|z| (0..32).map(move |x| IVec3::new(x, 0, z)))
            .collect();
        let lookups = std::cell::Cell::new(0);
        let seed = |p, cull: bool| {
            if cull && !donors.can_seed(p) {
                return history::FaceDisplay::default();
            }
            history::seed_face(&scene, p, 2, 0, now, |v| {
                lookups.set(lookups.get() + 1);
                measured.get(&v)
            })
        };
        for &p in &receivers {
            let a = seed(p, false);
            let b = seed(p, true);
            assert_eq!(
                (a.indirect, a.sky, a.confidence),
                (b.indirect, b.sky, b.confidence)
            );
        }
        let mut times: [Vec<Duration>; 2] = default();
        let mut counts = [0; 2];
        for round in 0..40 {
            for mode in [round % 2, 1 - round % 2] {
                lookups.set(0);
                let start = Instant::now();
                for &p in &receivers {
                    std::hint::black_box(seed(p, mode == 1));
                }
                let elapsed = start.elapsed();
                if round >= 8 {
                    times[mode].push(elapsed);
                }
                counts[mode] = lookups.get();
            }
        }
        for values in &mut times {
            values.sort_unstable();
        }
        eprintln!(
            "SEED_CULL 1024 new faces, exact priors; lookups {:?}; unculled/culled p50 {:?}/{:?}",
            counts, times[0][16], times[1][16]
        );
        assert!(counts[1] * 4 < counts[0]);
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
