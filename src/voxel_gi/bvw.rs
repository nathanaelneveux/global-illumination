//! Host adapter: BVW chunk notifications and prototype generator snapshots.
//! Transport workers consume immutable dense pages, not BVW chunks or meshes.
//! Replace this boundary for a different storage engine; do not teach traversal,
//! source sampling or display history about gameplay material IDs or generators.
use super::telemetry::GiWorkTimings;
use super::world::{
    AIR, PAGE_EDGE, PAGE_SIZE, TransportScene, affected_probe_pages, scan_surface_probe_pages,
};
use super::{SceneBuildResult, VoxelGiDirty, VoxelGiRuntime, naadf, page_of};
use crate::terrain::PrototypeWorld;
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use bevy_voxel_world::prelude::*;
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
        let left = runtime.receiver_pages.take().unwrap_or_else(|| runtime.loaded_pages.clone());
        runtime.receiver_pages = Some(HashSet::new());
        mark_receivers(&mut runtime, Vec::new(), left.into_iter().collect());
        return;
    };
    let foci: Vec<Vec3> = cameras.iter()
        .filter(|(camera, _)| camera.is_active)
        .map(|(_, transform)| transform.translation()).collect();
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
pub(super) struct PageEdits(HashMap<IVec3, Arc<HashMap<IVec3, u16>>>);

impl PageEdits {
    fn insert(&mut self, voxel: IVec3, material: u16) {
        Arc::make_mut(self.0.entry(page_of(voxel)).or_default()).insert(voxel, material);
    }
}

impl VoxelGiDirty {
    /// The prototype's edit gateway: forward the same value to BVW and GI.
    /// Last write wins, including multiple writes to one voxel in one frame.
    pub fn set_voxel(
        &mut self,
        world: &mut VoxelWorld<PrototypeWorld>,
        position: IVec3,
        voxel: WorldVoxel<u8>,
    ) {
        world.set_voxel(position, voxel);
        match voxel {
            WorldVoxel::Solid(material) => {
                self.edits.insert(position, material as u16);
            }
            WorldVoxel::Air => {
                self.edits.insert(position, AIR);
            }
            WorldVoxel::Unset => {
                self.edits.remove(&position);
                self.mark_voxel(position);
            }
        }
    }
}

pub(super) fn collect_chunk_changes(
    mut runtime: ResMut<VoxelGiRuntime>,
    mut explicit_dirty: ResMut<VoxelGiDirty>,
    mut spawned: MessageReader<ChunkWillSpawn<PrototypeWorld>>,
    mut updated: MessageReader<ChunkWillUpdate<PrototypeWorld>>,
    mut despawned: MessageReader<ChunkWillDespawn<PrototypeWorld>>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::Ingestion);
    for event in spawned.read() {
        runtime.loaded_pages.insert(event.chunk_key);
        runtime.removed_pages.remove(&event.chunk_key);
        runtime.dirty_pages.insert(event.chunk_key);
    }
    for event in updated.read() {
        runtime.loaded_pages.insert(event.chunk_key);
        runtime.removed_pages.remove(&event.chunk_key);
        runtime.dirty_pages.insert(event.chunk_key);
        if !explicit_dirty.complete_edit_tracking {
            runtime.full_resample_pages.insert(event.chunk_key);
            runtime.page_edits.0.remove(&event.chunk_key);
        }
    }
    for event in despawned.read() {
        runtime.loaded_pages.remove(&event.chunk_key);
        runtime.dirty_pages.remove(&event.chunk_key);
        runtime.full_resample_pages.remove(&event.chunk_key);
        runtime.removed_pages.insert(event.chunk_key);
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

pub(super) fn start_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    generator: Res<PrototypeWorld>,
    voxel_world: VoxelWorld<PrototypeWorld>,
    input: Res<VoxelGiDirty>,
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

    let mut dirty_pages: Vec<_> = runtime.dirty_pages.drain().collect();
    dirty_pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    let full_resample = std::mem::take(&mut runtime.full_resample_pages);
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
    let voxel_lookup = (!full_resample.is_empty() || !input.complete_edit_tracking)
        .then(|| voxel_world.get_voxel_fn());
    let chunk_lookup = voxel_world.get_chunk_data_fn();
    let generator = generator.clone();
    let materials = Arc::clone(&runtime.materials);
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
                let mut materials = if !full_resample.contains(&page) {
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
                    } else if let Some(lookup) = &voxel_lookup {
                        sample_dense_page(origin, |position| match lookup(position) {
                            WorldVoxel::Unset => generator.generated_voxel(position),
                            voxel => voxel,
                        })
                    } else {
                        sample_dense_page(origin, |position| generator.generated_voxel(position))
                    }
                } else {
                    let lookup = voxel_lookup
                        .as_ref()
                        .expect("unknown updates capture a host lookup");
                    sample_dense_page(origin, |position| match lookup(position) {
                        WorldVoxel::Unset => generator.generated_voxel(position),
                        voxel => voxel,
                    })
                };
                if let Some(edits) = edits.get(&page) {
                    apply_page_edits(&mut materials, origin, edits);
                }
                (page, materials)
            })
            .collect();
        let scene = Arc::new(TransportScene::with_page_delta(
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
            geometry_change: changes.transport,
            pages: pages.into(),
            receivers_changed,
            exact_edits,
            build_time: started.elapsed(),
        }
    }));
}

fn apply_page_edits(materials: &mut [u16], origin: IVec3, edits: &HashMap<IVec3, u16>) {
    for (&position, &material) in edits {
        let local = (position - origin).as_uvec3();
        materials[naadf::linear_index(local, PAGE_SIZE)] = material;
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    fn app(tracked: bool) -> App {
        let mut app = App::new();
        app.add_plugins((
            MinimalPlugins,
            VoxelWorldPlugin::<PrototypeWorld>::minimal(),
        ))
        .insert_resource(if tracked {
            VoxelGiDirty::tracked()
        } else {
            VoxelGiDirty::default()
        })
        .init_resource::<VoxelGiRuntime>()
        .init_resource::<GiWorkTimings>()
        .add_systems(Update, collect_chunk_changes);
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
        app.world_mut()
            .run_system_once(
                move |mut world: VoxelWorld<PrototypeWorld>, mut dirty: ResMut<VoxelGiDirty>| {
                    dirty.set_voxel(&mut world, position, WorldVoxel::Solid(1));
                    dirty.set_voxel(&mut world, position, WorldVoxel::Air);
                    dirty.set_voxel(&mut world, position, WorldVoxel::Solid(2));
                },
            )
            .unwrap();
        app.update();
        app.world_mut()
            .write_message(ChunkWillUpdate::<PrototypeWorld>::new(
                page,
                Entity::PLACEHOLDER,
            ));
        app.update();
        let snapshot = app.world().resource::<VoxelGiRuntime>().page_edits.0[&page].clone();
        assert_eq!(snapshot[&position], 2);
        assert!(
            !app.world()
                .resource::<VoxelGiRuntime>()
                .full_resample_pages
                .contains(&page)
        );
        app.world_mut()
            .run_system_once(
                move |mut world: VoxelWorld<PrototypeWorld>, mut dirty: ResMut<VoxelGiDirty>| {
                    assert_eq!(world.get_voxel(position), WorldVoxel::Solid(2));
                    dirty.set_voxel(&mut world, position, WorldVoxel::Air);
                },
            )
            .unwrap();
        app.world_mut()
            .write_message(ChunkWillDespawn::<PrototypeWorld>::new(
                page,
                Entity::PLACEHOLDER,
            ));
        app.update();
        assert_eq!(snapshot[&position], 2, "worker view must not change");
        app.world_mut()
            .write_message(ChunkWillSpawn::<PrototypeWorld>::new(
                page,
                Entity::PLACEHOLDER,
            ));
        app.update();
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert!(runtime.dirty_pages.contains(&page));
        let mut data = sample_dense_page(page * 32, |_| WorldVoxel::Solid(1));
        apply_page_edits(&mut data, page * 32, &runtime.page_edits.0[&page]);
        assert_eq!(
            data[naadf::linear_index((position - page * 32).as_uvec3(), PAGE_SIZE)],
            AIR
        );
        assert_eq!(data.iter().filter(|&&m| m == AIR).count(), 1);
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
            if tracked {
                app.world_mut()
                    .resource_mut::<VoxelGiDirty>()
                    .mark_voxel(IVec3::ONE);
            } else {
                app.world_mut()
                    .write_message(ChunkWillUpdate::<PrototypeWorld>::new(
                        IVec3::ZERO,
                        Entity::PLACEHOLDER,
                    ));
            }
            app.update();
            let runtime = app.world().resource::<VoxelGiRuntime>();
            assert!(runtime.full_resample_pages.contains(&IVec3::ZERO));
            assert!(!runtime.page_edits.0.contains_key(&IVec3::ZERO));
        }
    }

    #[test]
    fn unknown_notification_supersedes_an_exact_value_in_the_same_frame() {
        let mut dirty = VoxelGiDirty::tracked();
        dirty.edits.insert(IVec3::ONE, 7);
        dirty.mark_voxel(IVec3::ONE);
        assert!(dirty.edits.is_empty());
        dirty.edits.insert(IVec3::ZERO, 4);
        dirty.mark_all_loaded();
        assert!(dirty.edits.is_empty());
    }

}
