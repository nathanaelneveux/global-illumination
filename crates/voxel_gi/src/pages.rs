//! Page bookkeeping between host input and scene builds: residency of
//! world-cache receivers, exact edit overlays, and the worker scene build that
//! samples changed pages from the host's `VoxelSource`.
use super::input::{PageEvent, VoxelGiInput, VoxelGiSource};
use super::telemetry::GiWorkTimings;
use super::world::{
    PAGE_EDGE, PAGE_SIZE, TransportScene, affected_probe_pages, scan_surface_probe_pages,
};
use super::{SceneBuildResult, VoxelGiRuntime, naadf, page_of};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use std::sync::Arc;
use std::time::Instant;

/// Receivers are kept until a page is this much farther than the radius, so a
/// camera moving back and forth across it does not churn pages.
const RECEIVER_HYSTERESIS: f32 = PAGE_EDGE as f32;

/// Which loaded pages have world-cache receivers: those within
/// `cache_radius` of any active camera (distance to the page's nearest point),
/// kept until beyond the radius plus a hysteresis. Geometry is unaffected.
pub(super) fn update_receiver_residency(
    mut runtime: ResMut<VoxelGiRuntime>,
    settings: Res<super::VoxelGiSettings>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
) {
    let Some(radius) = settings.cache_radius() else {
        let left = runtime
            .receiver_pages
            .take()
            .unwrap_or_else(|| runtime.loaded_pages.clone());
        runtime.receiver_pages = Some(HashSet::new());
        mark_receivers(&mut runtime, Vec::new(), left.into_iter().collect());
        return;
    };
    let foci: Vec<Vec3> = cameras
        .iter()
        .filter(|(camera, _)| camera.is_active)
        .map(|(_, transform)| transform.translation())
        .collect();
    if foci.is_empty() {
        return;
    }
    let distance = |page: IVec3| {
        let minimum = (page * PAGE_EDGE as i32).as_vec3();
        let maximum = minimum + Vec3::splat(PAGE_EDGE as f32);
        foci.iter()
            .map(|focus| focus.clamp(minimum, maximum).distance(*focus))
            .fold(f32::INFINITY, f32::min)
    };
    let first = runtime.receiver_pages.is_none();
    let mut allowed = runtime.receiver_pages.take().unwrap_or_default();
    let (mut entered, mut left) = (Vec::new(), Vec::new());
    for &page in &runtime.loaded_pages {
        let d = distance(page);
        if allowed.contains(&page) {
            if d > radius + RECEIVER_HYSTERESIS {
                left.push(page);
            }
        } else if d <= radius {
            entered.push(page);
        } else if first {
            // Startup: pages outside lose their receivers.
            left.push(page);
        }
    }
    allowed.retain(|page| runtime.loaded_pages.contains(page));
    for page in &entered {
        allowed.insert(*page);
    }
    for page in &left {
        allowed.remove(page);
    }
    runtime.receiver_pages = Some(allowed);
    if first {
        // Pages already allowed keep their receivers; only changes rescan.
        entered.clear();
    }
    mark_receivers(&mut runtime, entered, left);
}

fn mark_receivers(runtime: &mut VoxelGiRuntime, entered: Vec<IVec3>, left: Vec<IVec3>) {
    if entered.is_empty() && left.is_empty() {
        return;
    }
    for page in entered {
        runtime.receiver_retired.remove(&page);
        runtime.receiver_dirty.insert(page);
    }
    for page in left {
        runtime.receiver_dirty.remove(&page);
        runtime.receiver_retired.insert(page);
    }
}

/// Sparse authoritative edits, grouped by page. Immutable worker views use
/// copy-on-write only for pages edited again while a task holds a snapshot.
/// Keep edits across unload/reload: a BVW mesh snapshot may lag current writes.
/// Explicit unknown changes discard the affected overlay and resample the host.
#[derive(Default)]
pub(super) struct PageEdits(pub(super) HashMap<IVec3, Arc<HashMap<IVec3, u16>>>);

impl PageEdits {
    pub(super) fn insert(&mut self, voxel: IVec3, material: u16) {
        Arc::make_mut(self.0.entry(page_of(voxel)).or_default()).insert(voxel, material);
    }
}

/// The bounding box of the pages the host has declared loaded: view
/// discovery and the far tier only light pixels inside it. Inside, geometry
/// the host has rendered but GI has not yet published keeps a pending
/// receiver; outside (a host rendering beyond its GI interest) there is
/// nothing to wait for. Loads widen it at once; after unloads it is
/// recomputed at most every `RECOMPUTE_FRAMES` frames (a stale box is only
/// larger, never wrong).
#[derive(Default)]
pub(super) struct Interest {
    pages: Option<(IVec3, IVec3)>,
    stale: bool,
    frames: u32,
}

impl Interest {
    const RECOMPUTE_FRAMES: u32 = 30;

    fn load(&mut self, page: IVec3) {
        self.pages = Some(
            self.pages
                .map_or((page, page), |(min, max)| (min.min(page), max.max(page))),
        );
    }

    /// The interest in 16-voxel regions (inclusive), refreshed if due.
    pub(super) fn regions(&mut self, loaded: &HashSet<IVec3>) -> Option<(IVec3, IVec3)> {
        self.frames += 1;
        if self.stale && self.frames >= Self::RECOMPUTE_FRAMES {
            self.pages = None;
            for &page in loaded {
                self.load(page);
            }
            self.stale = false;
            self.frames = 0;
        }
        let per_page = (PAGE_EDGE / 16) as i32;
        self.pages
            .map(|(min, max)| (min * per_page, max * per_page + IVec3::splat(per_page - 1)))
    }
}

pub(super) fn ingest_input(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut explicit_dirty: ResMut<VoxelGiInput>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::Ingestion);
    // Read-only check first: draining an idle input must not flag it changed
    // (hosts rebuild their `VoxelSource` only when input changed).
    if explicit_dirty.is_idle() {
        return;
    }
    let tracked = explicit_dirty.complete_edit_tracking;
    for (page, event) in std::mem::take(&mut explicit_dirty.pages) {
        match event {
            PageEvent::Loaded | PageEvent::Changed => {
                runtime.loaded_pages.insert(page);
                runtime.interest.load(page);
                runtime.removed_pages.remove(&page);
                runtime.dirty_pages.insert(page);
                if event == PageEvent::Changed && !tracked {
                    runtime.full_resample_pages.insert(page);
                    runtime.page_edits.0.remove(&page);
                }
            }
            PageEvent::Unloaded => {
                runtime.loaded_pages.remove(&page);
                runtime.interest.stale = true;
                runtime.dirty_pages.remove(&page);
                runtime.full_resample_pages.remove(&page);
                runtime.removed_pages.insert(page);
            }
        }
    }
    if explicit_dirty.all_loaded {
        runtime.page_edits.0.clear();
        let pages: Vec<_> = runtime.loaded_pages.iter().copied().collect();
        runtime.dirty_pages.extend(pages.iter().copied());
        runtime.full_resample_pages.extend(pages.iter().copied());
        explicit_dirty.all_loaded = false;
    }
    for voxel in explicit_dirty.voxels.drain() {
        let page = page_of(voxel);
        runtime.page_edits.0.remove(&page);
        if runtime.loaded_pages.contains(&page) {
            runtime.dirty_pages.insert(page);
            runtime.full_resample_pages.insert(page);
            runtime.dirty_voxels.insert(voxel);
            runtime.urgent_edit = true;
        }
    }
    for (voxel, material) in explicit_dirty.edits.drain() {
        let page = page_of(voxel);
        runtime.page_edits.insert(voxel, material);
        if runtime.loaded_pages.contains(&page) {
            runtime.dirty_pages.insert(page);
            runtime.dirty_voxels.insert(voxel);
            runtime.urgent_edit = true;
        }
    }
}

/// Dirty pages one scene build samples at most (about 48 MB of dense samples).
const MAX_SCENE_PAGES: usize = 768;

pub(super) fn start_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    source: Res<VoxelGiSource>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::SceneDispatch);
    let pending = !runtime.dirty_pages.is_empty()
        || !runtime.removed_pages.is_empty()
        || !runtime.receiver_dirty.is_empty()
        || !runtime.receiver_retired.is_empty();
    // No quiet period: a build in flight already batches whatever arrives
    // meanwhile. Waiting for streaming or an edit to go quiet only kept
    // freshly drawn terrain and edits without GI geometry (pending, black)
    // for up to a second while their meshes were already visible.
    if runtime.scene_task.is_some()
        || (runtime.scene_publication.is_some() && !runtime.urgent_edit)
        || !pending
    {
        return;
    }
    let Some(source) = source.0.clone() else {
        return;
    };

    let mut dirty_pages: Vec<_> = runtime.dirty_pages.drain().collect();
    if dirty_pages.len() > MAX_SCENE_PAGES {
        // Each dirty page is sampled densely (64 KB) in one build. Streaming a
        // far world queued tens of thousands at once: gigabytes of transient
        // samples and one very late build. Take edited pages, then the nearest;
        // the rest wait for the next build.
        let focus = cameras.iter().next().map_or(IVec3::ZERO, |camera| {
            page_of(camera.translation().floor().as_ivec3())
        });
        let edited: HashSet<IVec3> = runtime
            .dirty_voxels
            .iter()
            .map(|voxel| page_of(*voxel))
            .collect();
        dirty_pages.sort_unstable_by_key(|page| {
            (
                !edited.contains(page),
                (page.as_i64vec3() - focus.as_i64vec3()).length_squared(),
            )
        });
        let deferred = dirty_pages.split_off(MAX_SCENE_PAGES);
        runtime.dirty_pages.extend(deferred);
    }
    dirty_pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    let full_resample: HashSet<IVec3> = dirty_pages
        .iter()
        .copied()
        .filter(|page| runtime.full_resample_pages.remove(page))
        .collect();
    let edits: HashMap<_, _> = dirty_pages
        .iter()
        .filter_map(|page| {
            runtime
                .page_edits
                .0
                .get(page)
                .map(|edits| (*page, Arc::clone(edits)))
        })
        .collect();
    let exact_edits: Vec<_> = runtime.dirty_voxels.drain().collect();
    let mut removed_pages: Vec<_> = runtime.removed_pages.drain().collect();
    removed_pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    runtime.urgent_edit = false;

    let mut affected_pages = affected_probe_pages(&dirty_pages, &removed_pages);
    // Receiver radius: pages entering it are rescanned and pages leaving it
    // publish no receivers, without a geometry change. Scanning is limited to
    // allowed pages; every other affected page publishes none.
    let mut receiver_pages = std::mem::take(&mut runtime.receiver_dirty);
    receiver_pages.extend(runtime.receiver_retired.drain());
    let receivers_changed = !receiver_pages.is_empty();
    affected_pages.extend(receiver_pages);
    let allowed = runtime.receiver_pages.clone();
    let previous_scene = Arc::clone(&runtime.scene);
    let retention_radius = super::retention::INFLUENCE_RADIUS;
    let materials = Arc::clone(&runtime.materials);
    let sources = runtime.sources.clone();
    let focus_page = cameras.iter().next().map_or(IVec3::ZERO, |camera| {
        page_of(camera.translation().floor().as_ivec3())
    });
    runtime.scene_task = Some(AsyncComputeTaskPool::get().spawn(async move {
        let _span = info_span!("gi_scene_build").entered();
        let started = Instant::now();
        let changed_pages: Vec<_> = dirty_pages
            .into_iter()
            .map(|page| {
                let origin = page * PAGE_EDGE as i32;
                let sample = source.sample_page(page, full_resample.contains(&page));
                let mut stride = sample.stride().trailing_zeros() as u8;
                let mut materials = sample.into_dense();
                if let Some(edits) = edits.get(&page) {
                    apply_page_edits(&mut materials, origin, edits);
                    stride = edited_stride(&materials, origin, edits, stride);
                }
                (page, materials, stride)
            })
            .collect();
        let scene = Arc::new(TransportScene::with_strided_page_delta(
            &previous_scene,
            changed_pages,
            &removed_pages,
        ));
        let changes = super::geometry::SceneChanges::with_materials(
            &previous_scene,
            &scene,
            retention_radius,
            Some(&materials),
        );
        let descriptors = if changes.changed || receivers_changed {
            let scanned: HashSet<_> = match &allowed {
                Some(allowed) => affected_pages.intersection(allowed).copied().collect(),
                None => affected_pages.clone(),
            };
            scan_surface_probe_pages(&scene, &scanned, focus_page)
        } else {
            Vec::new()
        };
        // Publication units are spatial pages, prepared and prioritized on the
        // worker. A turn can discover hundreds of thousands of receivers; never
        // install that entire world-sized delta in a single game frame.
        let mut by_page: HashMap<_, Vec<_>> = affected_pages
            .into_iter()
            .map(|page| (page, Vec::new()))
            .collect();
        for descriptor in descriptors {
            by_page
                .entry(page_of(descriptor.voxel))
                .or_default()
                .push(descriptor);
        }
        let mut pages: Vec<_> = by_page.into_iter().collect();
        pages.sort_unstable_by_key(|(page, _)| {
            (
                !exact_edits.iter().any(|v| page_of(*v) == *page),
                (page.as_i64vec3() - focus_page.as_i64vec3()).length_squared(),
            )
        });
        SceneBuildResult {
            scene,
            changed: changes.changed,
            transport_changed: changes.transport.is_some(),
            retention: changes.transport.and_then(|mut change| {
                super::publication::finalize_change_flux(&mut change, &sources);
                super::retention::SpatialHistory::event(change, &exact_edits)
            }),
            pages: pages.into(),
            receivers_changed,
            exact_edits,
            build_time: started.elapsed(),
        }
    }));
}

pub(super) fn apply_page_edits(materials: &mut [u16], origin: IVec3, edits: &HashMap<IVec3, u16>) {
    for (&position, &material) in edits {
        let local = (position - origin).as_uvec3();
        materials[naadf::linear_index(local, PAGE_SIZE)] = material;
    }
}

/// A coarse page's stride after exact edits: kept only while every edited
/// block is still one material, the contract coarse receivers rely on
/// (`block_receiver`, `receiver_exists`). Otherwise the page is traced and
/// keyed at full resolution: removing one voxel from a stride-8 floor
/// exposes a face inside its block that the block face cannot stand for.
fn edited_stride(
    materials: &[u16],
    origin: IVec3,
    edits: &HashMap<IVec3, u16>,
    stride_log2: u8,
) -> u8 {
    if stride_log2 == 0 {
        return 0;
    }
    let size = 1u32 << stride_log2;
    // Each touched block once: an explosion edits many voxels of a block.
    let blocks: HashSet<UVec3> = edits
        .keys()
        .map(|position| (position - origin).as_uvec3() / size * size)
        .collect();
    let uniform = |block: &UVec3| {
        let first = materials[naadf::linear_index(*block, PAGE_SIZE)];
        (0..size).all(|z| {
            (0..size).all(|y| {
                (0..size).all(|x| {
                    materials[naadf::linear_index(block + UVec3::new(x, y, z), PAGE_SIZE)] == first
                })
            })
        })
    };
    if blocks.iter().all(uniform) {
        stride_log2
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::input::{VoxelGiInput, dense_page};
    use super::super::telemetry::GiWorkTimings;
    use super::super::world::{AIR, PAGE_SIZE};
    use super::super::{VoxelGiRuntime, naadf, page_of};
    use super::*;

    fn app(tracked: bool) -> App {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins)
            .insert_resource(if tracked {
                VoxelGiInput::tracked()
            } else {
                VoxelGiInput::default()
            })
            .init_resource::<VoxelGiRuntime>()
            .init_resource::<GiWorkTimings>()
            .add_systems(Update, ingest_input);
        app.update();
        app
    }

    #[test]
    fn exact_edits_are_last_write_wins_and_survive_inflight_snapshots_and_reload() {
        let mut app = app(true);
        let position = IVec3::new(-1, 31, 32);
        let page = page_of(position);
        app.world_mut()
            .resource_mut::<VoxelGiRuntime>()
            .loaded_pages
            .insert(page);
        {
            let mut input = app.world_mut().resource_mut::<VoxelGiInput>();
            input.edit_voxel(position, Some(1));
            input.edit_voxel(position, None);
            input.edit_voxel(position, Some(2));
        }
        app.update();
        app.world_mut()
            .resource_mut::<VoxelGiInput>()
            .page_changed(page);
        app.update();
        let snapshot = app.world().resource::<VoxelGiRuntime>().page_edits.0[&page].clone();
        assert_eq!(snapshot[&position], 2);
        assert!(
            !app.world()
                .resource::<VoxelGiRuntime>()
                .full_resample_pages
                .contains(&page)
        );
        {
            let mut input = app.world_mut().resource_mut::<VoxelGiInput>();
            input.edit_voxel(position, None);
            input.page_unloaded(page);
        }
        app.update();
        assert_eq!(snapshot[&position], 2, "worker view must not change");
        app.world_mut()
            .resource_mut::<VoxelGiInput>()
            .page_loaded(page);
        app.update();
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert!(runtime.dirty_pages.contains(&page));
        let mut data = dense_page(page, |_| Some(1)).into_dense();
        apply_page_edits(&mut data, page * 32, &runtime.page_edits.0[&page]);
        assert_eq!(
            data[naadf::linear_index((position - page * 32).as_uvec3(), PAGE_SIZE)],
            AIR
        );
        assert_eq!(data.iter().filter(|&&m| m == AIR).count(), 1);
    }

    #[test]
    fn interest_widens_on_load_and_shrinks_only_when_recomputed() {
        let mut interest = Interest::default();
        let mut loaded = HashSet::new();
        assert_eq!(
            interest.regions(&loaded),
            None,
            "no declared pages: no bounds"
        );
        for page in [IVec3::new(-1, 0, 2), IVec3::new(3, 1, 2)] {
            loaded.insert(page);
            interest.load(page);
        }
        let both = Some((IVec3::new(-2, 0, 4), IVec3::new(7, 3, 5)));
        assert_eq!(
            interest.regions(&loaded),
            both,
            "two 16-voxel regions per page edge"
        );
        loaded.remove(&IVec3::new(3, 1, 2));
        interest.stale = true;
        assert_eq!(
            interest.regions(&loaded),
            both,
            "a stale box is kept until due"
        );
        for _ in 0..Interest::RECOMPUTE_FRAMES {
            interest.regions(&loaded);
        }
        assert_eq!(
            interest.regions(&loaded),
            Some((IVec3::new(-2, 0, 4), IVec3::new(-1, 1, 5)))
        );
    }

    #[test]
    fn edits_that_break_a_coarse_block_drop_the_page_to_full_resolution() {
        // A stride-8 floor, one block thick (y 0-7).
        let mut page = vec![AIR; naadf::volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for y in 0..8 {
                for x in 0..32 {
                    page[naadf::linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 3;
                }
            }
        }
        let edited = |edits: &[(IVec3, u16)]| {
            let edits: HashMap<IVec3, u16> = edits.iter().copied().collect();
            let mut materials = page.clone();
            apply_page_edits(&mut materials, IVec3::ZERO, &edits);
            edited_stride(&materials, IVec3::ZERO, &edits, 3)
        };
        assert_eq!(
            edited(&[(IVec3::new(4, 7, 4), AIR)]),
            0,
            "a hole exposes a face inside the block"
        );
        assert_eq!(
            edited(&[(IVec3::new(4, 7, 4), 3)]),
            3,
            "rewriting the same material keeps blocks uniform"
        );
        let whole: Vec<_> = (0..8)
            .flat_map(|z| (0..8).flat_map(move |y| (0..8).map(move |x| (IVec3::new(x, y, z), AIR))))
            .collect();
        assert_eq!(edited(&whole), 3, "removing a whole block keeps the stride");
        assert_eq!(edited_stride(&page, IVec3::ZERO, &HashMap::new(), 0), 0);
    }

    #[test]
    fn coarse_unknown_changes_resample_and_discard_stale_overrides() {
        for tracked in [false, true] {
            let mut app = app(tracked);
            {
                let mut runtime = app.world_mut().resource_mut::<VoxelGiRuntime>();
                runtime.loaded_pages.insert(IVec3::ZERO);
                runtime.page_edits.insert(IVec3::ONE, 7);
            }
            let mut input = app.world_mut().resource_mut::<VoxelGiInput>();
            if tracked {
                input.mark_voxel(IVec3::ONE);
            } else {
                input.page_changed(IVec3::ZERO);
            }
            app.update();
            let runtime = app.world().resource::<VoxelGiRuntime>();
            assert!(runtime.full_resample_pages.contains(&IVec3::ZERO));
            assert!(!runtime.page_edits.0.contains_key(&IVec3::ZERO));
        }
    }
}
