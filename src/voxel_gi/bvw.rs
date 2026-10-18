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
use bevy::platform::collections::HashMap;
use bevy::prelude::*;
use bevy::tasks::AsyncComputeTaskPool;
use bevy_voxel_world::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

// Existing coalescing windows; publication and display response are independent.
const STREAM_DEBOUNCE: Duration = Duration::from_millis(350);
const EDIT_DEBOUNCE: Duration = Duration::from_millis(40);
const MAX_REBUILD_LATENCY: Duration = Duration::from_secs(1);

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
        if !explicit_dirty.complete_edit_tracking {
            runtime.full_resample_pages.insert(event.chunk_key);
            runtime.page_edits.0.remove(&event.chunk_key);
        }
        changed = true;
    }
    for event in despawned.read() {
        runtime.loaded_pages.remove(&event.chunk_key);
        runtime.dirty_pages.remove(&event.chunk_key);
        runtime.full_resample_pages.remove(&event.chunk_key);
        runtime.removed_pages.insert(event.chunk_key);
        changed = true;
    }
    if explicit_dirty.all_loaded {
        runtime.page_edits.0.clear();
        let pages: Vec<_> = runtime.loaded_pages.iter().copied().collect();
        runtime.dirty_pages.extend(pages.iter().copied());
        runtime.full_resample_pages.extend(pages.iter().copied());
        explicit_dirty.all_loaded = false;
        changed = true;
    }
    for voxel in explicit_dirty.voxels.drain() {
        let page = page_of(voxel);
        runtime.page_edits.0.remove(&page);
        if runtime.loaded_pages.contains(&page) {
            runtime.dirty_pages.insert(page);
            runtime.full_resample_pages.insert(page);
            runtime.dirty_voxels.insert(voxel);
            runtime.urgent_edit = true;
            changed = true;
        }
    }
    for (voxel, material) in explicit_dirty.edits.drain() {
        let page = page_of(voxel);
        runtime.page_edits.insert(voxel, material);
        if runtime.loaded_pages.contains(&page) {
            runtime.dirty_pages.insert(page);
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

pub(super) fn start_scene_task(
    mut runtime: ResMut<VoxelGiRuntime>,
    generator: Res<PrototypeWorld>,
    voxel_world: VoxelWorld<PrototypeWorld>,
    input: Res<VoxelGiDirty>,
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut timings: ResMut<GiWorkTimings>,
) {
    let _timing = timings.main_scope(super::telemetry::MainStage::SceneDispatch);
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
    if runtime.scene_task.is_some()
        || (runtime.scene_publication.is_some() && !runtime.urgent_edit)
        || !pending
        || (!quiet && !overdue)
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
    runtime.dirty_since = None;
    runtime.urgent_edit = false;

    let affected_pages = affected_probe_pages(&dirty_pages, &removed_pages);
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
        let descriptors = if changes.changed {
            scan_surface_probe_pages(&scene, &affected_pages, focus_page)
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

    #[test]
    #[ignore = "isolated BVW input sampling pair, not frame timing"]
    fn immutable_edit_input_cost_study() {
        let mut app = app(true);
        app.world_mut()
            .run_system_once(
                |mut world: VoxelWorld<PrototypeWorld>, mut dirty: ResMut<VoxelGiDirty>| {
                    dirty.set_voxel(&mut world, IVec3::ONE, WorldVoxel::Air);
                },
            )
            .unwrap();
        app.update();
        app.world_mut().run_system_once(|world: VoxelWorld<PrototypeWorld>, runtime: Res<VoxelGiRuntime>| {
            let lookup = world.get_voxel_fn();
            let overlay = &runtime.page_edits.0[&IVec3::ZERO];
            let mut times: [Vec<f64>; 2] = default();
            for round in 0..24 {
                let mut data = Vec::new();
                for mode in [round % 2, 1 - round % 2] {
                    let started = Instant::now();
                    let next = if mode == 0 {
                        sample_dense_page(IVec3::ZERO, |p| match lookup(p) {
                            WorldVoxel::Unset => WorldVoxel::Solid(1), v => v,
                        })
                    } else {
                        let mut page = sample_dense_page(IVec3::ZERO, |_| WorldVoxel::Solid(1));
                        apply_page_edits(&mut page, IVec3::ZERO, overlay);
                        page
                    };
                    if round >= 4 { times[mode].push(started.elapsed().as_secs_f64() * 1000.0); }
                    if !data.is_empty() { assert_eq!(data, next); }
                    data = std::hint::black_box(next);
                }
            }
            for t in &mut times { t.sort_by(f64::total_cmp); }
            println!("INPUT_PAIR page=32768 edits=1 paths=real_BVW_lookup/generated_page_plus_overlay old_ms={:.4} new_ms={:.4} exact=true", times[0][10], times[1][10]);
        }).unwrap();
    }
}
