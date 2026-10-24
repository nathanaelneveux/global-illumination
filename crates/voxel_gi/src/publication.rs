//! Main-world scene publication. Geometry is prepared off-thread; receiver pages
//! publish atomically with retained measurements and display-only initialization.
use super::{
    SceneBuildResult, VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, VoxelGiStats,
    advance_transport_revision, enqueue_urgent, history, page_of,
    probe_metrics,
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
        if !result.changed && !result.receivers_changed {
            stats.loaded_pages = runtime.loaded_pages.len();
            stats.last_scene_build_ms = result.build_time.as_secs_f32() * 1000.0;
            return;
        }
        // Scene edits, including emitting voxels, share spatial history
        // retention. Source/model resets belong to changes of the global
        // material table or transport model, not residency of one source.
        // A receiver-radius change alone publishes pages, not a new scene.
        if result.changed {
            runtime.scene = result.scene.clone();
            runtime.generation = runtime.generation.wrapping_add(1);
        }
        if let Some(mut change) = result.geometry_change.take() {
            advance_transport_revision(&mut runtime, &mut field);
            finalize_change_flux(&mut change, &runtime.sources);
            // The scene's occupied pages include view-owned receivers too.
            // Filtering by world-cache residency drops edits outside that
            // cache (all edits in cacheless mode). GPU consumers apply reach.
            // The GPU measures edits' flux and wakes the faces near player
            // edits whose history the change binds (no CPU statistics read).
            runtime.spatial_history.record(change, &result.exact_edits);
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
            // An overtaken publication may not have reached its edit wakeups.
            for voxel in older.exact_edits {
                if !result.exact_edits.contains(&voxel) {
                    result.exact_edits.push(voxel);
                }
            }
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
    let started = Instant::now();
    // Worker scans include touching pages, even when their exposed topology
    // did not change. Preserve their allocations, indices and measurements
    // directly; global scene freshness and explicit edit requests still apply.
    if page_matches(field, page, &descriptors, &runtime.materials) {
        finish_scene_page(
            runtime,
            field,
            stats,
            result,
            page,
            HashSet::new(),
            Vec::new(),
            started,
        );
    } else {
        reconcile_scene_page(runtime, field, stats, result, page, descriptors);
    }
}

/// Descriptors are the worker scanner's unique voxel keys. Equal cardinality
/// and equal exposed records establish page identity independently of order.
fn page_matches(
    field: &VoxelEnergyField,
    page: IVec3,
    descriptors: &[SurfaceProbeDescriptor],
    materials: &super::world::VoxelMaterialTransportTable,
) -> bool {
    field.pages.get(&page).is_some_and(|current| {
        current.len() == descriptors.len()
            && descriptors.iter().all(|descriptor| {
                field.probes.get(&descriptor.voxel).is_some_and(|probe| {
                    probe.face_mask == descriptor.face_mask
                        && probe.material == descriptor.material
                        && probe.emission == materials.get(descriptor.material).emission
                })
            })
    })
}

fn reconcile_scene_page(
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
    for descriptor in descriptors {
        let mut probe = old_affected.remove(&descriptor.voxel).unwrap_or_else(|| {
            invalidated.insert(descriptor.voxel);
            added.push(descriptor.voxel);
            let mut probe = SurfaceProbe::new_slim(descriptor.voxel, descriptor.face_mask);
            probe.revision = runtime.generation << 32;
            // Timestamp the new display prior independently of material ID.
            // This is not measurement evidence or a transport freshness stamp.
            Box::new(probe)
        });
        let old_mask = probe.face_mask;
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
            invalidated.insert(descriptor.voxel);
        }
        // New or newly exposed faces get resident GPU state soon, not only
        // when first traced (see dispatch::ENSURE_PROBES_PER_BATCH).
        if (descriptor.face_mask & !old_mask != 0 || added.last() == Some(&descriptor.voxel))
            && runtime.slot_requested.insert(descriptor.voxel)
        {
            runtime.slot_requests.push_back(descriptor.voxel);
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
        field.probes.insert_box(descriptor.voxel, probe);
    }
    if !old_affected.is_empty() && field.pages.get(&page).is_none_or(|probes| probes.is_empty()) {
        // The page's receivers are gone (retired by the receiver radius, or
        // its surfaces removed): release its resident slots and sun corners.
        runtime.slot_retirements.push(page);
    }
    let topology_changed =
        old_count != field.probes.len() || !old_affected.is_empty() || !added.is_empty();
    if topology_changed {
        field.update_indices(&added, &old_affected);
    }

    finish_scene_page(
        runtime,
        field,
        stats,
        result,
        page,
        invalidated,
        added,
        publish_started,
    );
}

fn finish_scene_page(
    runtime: &mut VoxelGiRuntime,
    field: &mut VoxelEnergyField,
    stats: &mut VoxelGiStats,
    result: &mut SceneBuildResult,
    page: IVec3,
    invalidated: HashSet<IVec3>,
    added: Vec<IVec3>,
    publish_started: Instant,
) {
    // The light is the GPU's: retention events (and the edit's shadow
    // footprint) wake the faces whose history it binds. Enumerating every
    // probe within the relight radius here cost ~3 ms of main thread per
    // edited page.
    let relit_probes = invalidated.len();
    // Existing probes whose revision changed here (material or exposed faces:
    // relit, history capped) are event work: only a CPU job carries the new
    // revision and its cap to the GPU, and until one does, the mirror rejects
    // the GPU's picks of them. New probes need none (eager slots, coverage).
    let added: HashSet<IVec3> = added.into_iter().collect();
    let mut relit: Vec<_> = invalidated
        .iter()
        .copied()
        .filter(|voxel| !added.contains(voxel))
        .collect();
    relit.sort_unstable_by_key(|voxel| voxel.to_array());
    enqueue_urgent(runtime, relit);
    // New faces' coverage and the world sweep are the GPU's picks.
    result.exact_edits.clear();
    field.publishing_pages.remove(&page);
    update_stats_from_counters(runtime, field, stats);
    stats.loaded_pages = runtime.loaded_pages.len();
    stats.naadf_regions = runtime.scene.regions.len();
    stats.last_scene_build_ms = result.build_time.as_secs_f32() * 1_000.0;
    debug!(
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

#[cfg(test)]
mod donor_tests {
    use super::*;

    #[test]
    fn edits_are_measured_on_the_gpu_within_the_scan_budget() {
        use super::super::geometry::{Bounds, ChangeKind, ChangeWeight, GeometryChange};
        let small = Bounds::point(IVec3::ZERO);
        let huge = Bounds { min: IVec3::ZERO, max: IVec3::splat(40) };
        let mut change = GeometryChange::default();
        for b in [small, huge] {
            change.weights.insert(
                b,
                ChangeWeight { kind: ChangeKind::Edit { voxels: 1, measure: false }, flux: 0.5 },
            );
        }
        finalize_change_flux(&mut change, &Default::default());
        // A 3x3x3 scan fits; 43^3 positions exceed the budget: unknown flux
        // (the distance curve), never an underestimate.
        assert_eq!(
            change.weights[&small],
            ChangeWeight { kind: ChangeKind::Edit { voxels: 1, measure: true }, flux: 0.5 }
        );
        assert_eq!(change.weights[&huge], ChangeWeight::UNKNOWN);
    }

    fn unchanged_page_fixture() -> (
        VoxelGiRuntime,
        VoxelEnergyField,
        Vec<SurfaceProbeDescriptor>,
    ) {
        let mut field = VoxelEnergyField::default();
        let descriptors: Vec<_> = (0..32)
            .flat_map(|z| {
                (0..32).map(move |x| SurfaceProbeDescriptor {
                    voxel: IVec3::new(x, 0, z),
                    face_mask: 4,
                    material: 1,
                })
            })
            .collect();
        for d in &descriptors {
            let mut probe = SurfaceProbe::new(d.voxel, d.face_mask);
            probe.material = d.material;
            probe.stats_mut().samples[2] = 64;
            probe.display[2].indirect = Vec3::splat(2.0);
            field.probes.insert(d.voxel, probe);
        }
        let keys: Vec<_> = descriptors.iter().map(|d| d.voxel).collect();
        field.pages.insert(IVec3::ZERO, keys.clone());
        field.update_indices(&keys, &HashMap::new());
        (
            VoxelGiRuntime {
                active_faces: 1024,
                completed_samples: 65536,
                covered_faces: 1024,
                ..default()
            },
            field,
            descriptors,
        )
    }

    fn empty_result() -> SceneBuildResult {
        SceneBuildResult {
            scene: std::sync::Arc::new(super::super::world::TransportScene::default()),
            changed: true,
            geometry_change: None,
            pages: default(),
            receivers_changed: false,
            exact_edits: Vec::new(),
            build_time: Duration::ZERO,
        }
    }

    #[test]
    fn unchanged_page_check_is_exact_and_does_not_suppress_edit_refresh() {
        let (mut runtime, mut field, mut descriptors) = unchanged_page_fixture();
        assert!(page_matches(
            &field,
            IVec3::ZERO,
            &descriptors,
            &runtime.materials
        ));
        descriptors.reverse();
        assert!(page_matches(
            &field,
            IVec3::ZERO,
            &descriptors,
            &runtime.materials
        ));
        descriptors[0].face_mask = 5;
        assert!(!page_matches(
            &field,
            IVec3::ZERO,
            &descriptors,
            &runtime.materials
        ));
        descriptors[0].face_mask = 4;
        descriptors[0].material = 2;
        assert!(!page_matches(
            &field,
            IVec3::ZERO,
            &descriptors,
            &runtime.materials
        ));
        descriptors[0].material = 1;
        let voxel = descriptors[0].voxel;
        field.probes.get_mut(&voxel).unwrap().emission = super::super::Vec3::ONE;
        assert!(!page_matches(
            &field,
            IVec3::ZERO,
            &descriptors,
            &runtime.materials
        ));
        field.probes.get_mut(&voxel).unwrap().emission = super::super::Vec3::ZERO;
        assert!(!page_matches(
            &field,
            IVec3::ZERO,
            &descriptors[..1023],
            &runtime.materials
        ));
        let pointer = field.probes.get(&voxel).unwrap() as *const SurfaceProbe;
        let mut result = empty_result();
        result.exact_edits.push(IVec3::new(16, 0, 16));
        field.publishing_pages.insert(IVec3::ZERO);
        publish_scene_page(
            &mut runtime,
            &mut field,
            &mut default(),
            &mut result,
            IVec3::ZERO,
            descriptors,
        );
        assert_eq!(
            field.probes.get(&voxel).unwrap() as *const SurfaceProbe,
            pointer
        );
        assert_eq!(field.probes[&voxel].display[2].indirect, Vec3::splat(2.0));
        // Whether its light is urgent is the GPU's decision (retention
        // events): no CPU enumeration of the edit's neighbourhood.
        assert!(!runtime.urgent_queued.contains(&voxel));
        // The count stays for measured retention to cap.
        assert_eq!(field.probes[&voxel].stats().samples[2], 64);
        assert!(!field.publishing_pages.contains(&IVec3::ZERO));
        assert_eq!(runtime.active_faces, 1024);
        assert_eq!(field.order.len(), 1024);
    }

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
                probe.stats_mut().samples[2] = 64;
                probe.stats_mut().face_updated_at[2] = Some(now);
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
                (a.indirect, a.confidence),
                (b.indirect, b.confidence)
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

fn luminance(value: Vec3) -> f32 {
    0.2126 * value.x + 0.7152 * value.y + 0.0722 * value.z
}

/// GPU scan positions (each a voxel's six faces) per scene update. Edits past
/// it get unknown flux, so their receivers use the distance curve instead.
const FLUX_SCAN_BUDGET: usize = 16_384;

/// Flux bounds for measured retention. Edits keep the changed voxels' own
/// emission (scene worker) and are measured on the GPU (`measure_edits`: the
/// brightest outgoing luminance within one voxel, from the GPU's statistics)
/// while their one-voxel scans fit the budget; past it they are unknown (the
/// distance curve). Loads get the brighter of sky radiance and reflected
/// sunlight per unit of height above a receiver, times the region's width.
/// Approximate bounds, not a certified dependency analysis.
fn finalize_change_flux(
    change: &mut super::geometry::GeometryChange,
    sources: &super::sources::TransportSources,
) {
    use super::geometry::ChangeKind;
    let sky = luminance(sources.environment).max(
        sources
            .directional
            .iter()
            .map(|light| 0.5 * luminance(light.radiance))
            .fold(0.0, f32::max),
    );
    let mut budget = FLUX_SCAN_BUDGET as u64;
    for (bounds, weight) in change.weights.iter_mut() {
        match &mut weight.kind {
            ChangeKind::Load { .. } => weight.flux = sky * super::world::REGION_EDGE as f32,
            ChangeKind::Edit { measure, .. } => {
                let size = (bounds.max - bounds.min + IVec3::splat(3)).max(IVec3::ONE).as_i64vec3();
                let positions = (size.x * size.y * size.z) as u64;
                if positions <= budget {
                    budget -= positions;
                    *measure = true;
                } else {
                    *weight = super::geometry::ChangeWeight::UNKNOWN;
                }
            }
            ChangeKind::Unknown | ChangeKind::Unload | ChangeKind::Light => {}
        }
    }
}
