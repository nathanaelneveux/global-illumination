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
        runtime.full_resample_pages.insert(event.chunk_key);
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
        let pages: Vec<_> = runtime.loaded_pages.iter().copied().collect();
        runtime.dirty_pages.extend(pages.iter().copied());
        runtime.full_resample_pages.extend(pages.iter().copied());
        explicit_dirty.all_loaded = false;
        changed = true;
    }
    for voxel in explicit_dirty.voxels.drain() {
        let page = page_of(voxel);
        if runtime.loaded_pages.contains(&page) {
            runtime.dirty_pages.insert(page);
            runtime.full_resample_pages.insert(page);
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
    let exact_edits: Vec<_> = runtime.dirty_voxels.drain().collect();
    let mut removed_pages: Vec<_> = runtime.removed_pages.drain().collect();
    removed_pages.sort_unstable_by_key(|p| (p.z, p.y, p.x));
    runtime.dirty_since = None;
    runtime.urgent_edit = false;

    let affected_pages = affected_probe_pages(&dirty_pages, &removed_pages);
    let previous_scene = Arc::clone(&runtime.scene);
    let retention_radius = if runtime.spatial_history.is_some() {
        super::retention::INFLUENCE_RADIUS
    } else {
        super::geometry::RELIGHT_RADIUS
    };
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
        let changes =
            super::geometry::SceneChanges::with_radius(&previous_scene, &scene, retention_radius);
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
