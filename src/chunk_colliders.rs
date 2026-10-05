use std::collections::{HashMap, HashSet, VecDeque};

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy::tasks::{Task, TaskPool, TaskPoolBuilder, block_on, poll_once};
use bevy_voxel_world::prelude::{
    Chunk, ChunkWillChangeLod, ChunkWillDespawn, ChunkWillSpawn, ChunkWillUpdate, NeedsDespawn,
    VoxelWorld, WorldVoxel,
};

use crate::terrain::PrototypeWorld;

const CHUNK_COLLIDER_VOXEL_SIZE: f32 = 1.0;
const CHUNK_COLLIDER_CACHE_MAX_ENTRIES: usize = 512;
const MIN_PADDED_CHUNK_DIMENSION: u32 = 3;
const MAX_COLLIDER_JOBS: usize = 2;
const MAX_PUBLICATIONS_PER_FRAME: usize = 2;

type ColliderCacheKey = u64;

pub struct ChunkColliderPlugin;

impl Plugin for ChunkColliderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkColliderCache>()
            .init_resource::<ColliderWorkers>()
            .add_systems(PreUpdate, cleanup_chunk_assets)
            .add_systems(
                PostUpdate,
                (invalidate_chunk_colliders, ensure_chunk_colliders).chain(),
            );
    }
}

#[derive(Component, Default)]
struct ChunkColliderReady;

struct ColliderJob {
    entity: Entity,
    position: IVec3,
    key: ColliderCacheKey,
    task: Task<Option<Collider>>,
}

#[derive(Resource)]
struct ColliderWorkers {
    pool: TaskPool,
    jobs: Vec<ColliderJob>,
    /// BVW still exposes the previous immutable data until ChunkWillSpawn.
    awaiting_data: HashSet<Entity>,
}

impl Default for ColliderWorkers {
    fn default() -> Self {
        Self {
            pool: TaskPoolBuilder::new()
                .num_threads(MAX_COLLIDER_JOBS)
                .thread_name("voxel-colliders".into())
                .build(),
            jobs: Vec::new(),
            awaiting_data: HashSet::new(),
        }
    }
}

#[derive(Resource, Default)]
struct ChunkColliderCache {
    colliders: HashMap<ColliderCacheKey, Collider>,
    access_order: VecDeque<ColliderCacheKey>,
}

impl ChunkColliderCache {
    fn get(&mut self, key: ColliderCacheKey) -> Option<Collider> {
        let collider = self.colliders.get(&key).cloned();
        if collider.is_some() {
            self.touch(key);
        }
        collider
    }

    fn insert(&mut self, key: ColliderCacheKey, collider: Collider) {
        self.insert_with_capacity(key, collider, CHUNK_COLLIDER_CACHE_MAX_ENTRIES);
    }

    fn insert_with_capacity(&mut self, key: ColliderCacheKey, collider: Collider, capacity: usize) {
        if capacity == 0 {
            return;
        }

        let is_new = !self.colliders.contains_key(&key);
        self.colliders.insert(key, collider);
        self.touch(key);

        if is_new
            && self.colliders.len() > capacity
            && let Some(oldest) = self.access_order.pop_front()
        {
            self.colliders.remove(&oldest);
        }
    }

    fn touch(&mut self, key: ColliderCacheKey) {
        if let Some(index) = self
            .access_order
            .iter()
            .position(|existing| *existing == key)
        {
            self.access_order.remove(index);
        }

        self.access_order.push_back(key);
    }
}

fn ensure_chunk_colliders(
    mut commands: Commands,
    mut chunk_collider_cache: ResMut<ChunkColliderCache>,
    mut workers: ResMut<ColliderWorkers>,
    voxel_world: VoxelWorld<PrototypeWorld>,
    chunks: Query<(Entity, &Chunk<PrototypeWorld>, Has<ChunkColliderReady>), Without<NeedsDespawn>>,
    player: Query<&Transform, With<crate::player::Player>>,
) {
    let started = std::time::Instant::now();
    let mut published = HashSet::new();
    // Never wait for construction. Stale jobs keep occupying their bounded
    // worker slot until completion, so repeated edits cannot flood the pool.
    for index in (0..workers.jobs.len()).rev() {
        let Some(collider) = block_on(poll_once(&mut workers.jobs[index].task)) else {
            continue;
        };
        let job = workers.jobs.swap_remove(index);
        let current = chunks
            .get(job.entity)
            .ok()
            .is_some_and(|(_, chunk, _)| chunk.position == job.position && chunk.lod_level == 0)
            && !workers.awaiting_data.contains(&job.entity)
            && voxel_world
                .get_chunk_data(job.position)
                .is_some_and(|data| {
                    data.has_generated() && data.lod_level() == 0 && data.voxels_hash() == job.key
                });
        if !current {
            continue;
        }
        if let Some(collider) = collider {
            chunk_collider_cache.insert(job.key, collider.clone());
            commands
                .entity(job.entity)
                .insert((RigidBody::Static, collider, ChunkColliderReady));
        } else {
            mark_chunk_collider_ready(&mut commands, job.entity);
        }
        published.insert(job.entity);
    }
    let eye = player.iter().next().map_or(Vec3::ZERO, |p| p.translation);
    let mut pending: Vec<_> = chunks
        .iter()
        .filter(|(entity, _, ready)| {
            !ready
                && !published.contains(entity)
                && !workers.awaiting_data.contains(entity)
                && !workers.jobs.iter().any(|job| job.entity == *entity)
        })
        .collect();
    pending.sort_unstable_by_key(|(_, chunk, _)| {
        ((chunk.position.as_vec3() * 32.0 + Vec3::splat(16.0)).distance_squared(eye)) as u64
    });
    for (entity, chunk, _) in pending {
        if published.len() >= MAX_PUBLICATIONS_PER_FRAME {
            break;
        }
        if chunk.lod_level != 0 {
            mark_chunk_collider_ready(&mut commands, entity);
            published.insert(entity);
            continue;
        }

        let Some(chunk_data) = voxel_world.get_chunk_data(chunk.position) else {
            continue;
        };

        if !chunk_data.has_generated() || chunk_data.lod_level() != 0 {
            continue;
        }
        if chunk_data.is_empty() {
            mark_chunk_collider_ready(&mut commands, entity);
            published.insert(entity);
            continue;
        }

        let voxels_hash = chunk_data.voxels_hash();
        if let Some(collider) = chunk_collider_cache.get(voxels_hash) {
            commands
                .entity(entity)
                .insert((RigidBody::Static, collider, ChunkColliderReady));
            published.insert(entity);
        } else {
            if workers.jobs.len() >= MAX_COLLIDER_JOBS {
                continue;
            }
            let shape = chunk_data.data_shape();
            let [sx, sy, sz] = shape.to_array();
            if sx < MIN_PADDED_CHUNK_DIMENSION
                || sy < MIN_PADDED_CHUNK_DIMENSION
                || sz < MIN_PADDED_CHUNK_DIMENSION
            {
                continue;
            }

            let voxels = chunk_data.voxels_arc();
            let full = chunk_data.is_full();
            let task = workers.pool.spawn(async move {
                let coords = if let Some(voxels) = voxels {
                    collect_solid_voxel_coordinates(&voxels, sx, sy, sz)
                } else if full {
                    full_chunk_coordinates(sx, sy, sz)
                } else {
                    Vec::new()
                };
                (!coords.is_empty())
                    .then(|| Collider::voxels(Vec3::splat(CHUNK_COLLIDER_VOXEL_SIZE), &coords))
            });
            workers.jobs.push(ColliderJob {
                entity,
                position: chunk.position,
                key: voxels_hash,
                task,
            });
        }
    }
    if started.elapsed().as_millis() >= 4 && std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        info!(
            published = published.len(),
            main_ms = started.elapsed().as_secs_f64() * 1000.0,
            "chunk collider publication"
        );
    }
}

fn mark_chunk_collider_ready(commands: &mut Commands, entity: Entity) {
    commands
        .entity(entity)
        .remove::<(RigidBody, Collider)>()
        .insert(ChunkColliderReady);
}

fn collect_solid_voxel_coordinates<I: Copy + PartialEq>(
    voxels: &[WorldVoxel<I>],
    sx: u32,
    sy: u32,
    sz: u32,
) -> Vec<IVec3> {
    let mut coords = Vec::with_capacity(interior_voxel_count(sx, sy, sz).min(voxels.len()));
    let yz_stride = sx * sy;

    for z in 1..sz.saturating_sub(1) {
        for y in 1..sy.saturating_sub(1) {
            for x in 1..sx.saturating_sub(1) {
                let index = (x + sx * y + yz_stride * z) as usize;
                if matches!(voxels[index], WorldVoxel::Solid(_)) {
                    coords.push(IVec3::new(x as i32, y as i32, z as i32));
                }
            }
        }
    }

    coords
}

fn full_chunk_coordinates(sx: u32, sy: u32, sz: u32) -> Vec<IVec3> {
    let mut coords = Vec::with_capacity(interior_voxel_count(sx, sy, sz));

    for z in 1..sz.saturating_sub(1) {
        for y in 1..sy.saturating_sub(1) {
            for x in 1..sx.saturating_sub(1) {
                coords.push(IVec3::new(x as i32, y as i32, z as i32));
            }
        }
    }

    coords
}

fn interior_voxel_count(sx: u32, sy: u32, sz: u32) -> usize {
    (sx.saturating_sub(2) * sy.saturating_sub(2) * sz.saturating_sub(2)) as usize
}

fn cleanup_chunk_assets(
    mut commands: Commands,
    mut workers: ResMut<ColliderWorkers>,
    mut events: MessageReader<ChunkWillDespawn<PrototypeWorld>>,
) {
    for event in events.read() {
        workers.awaiting_data.remove(&event.entity);
        if let Ok(mut entity_commands) = commands.get_entity(event.entity) {
            entity_commands.remove::<(ChunkColliderReady, Collider, RigidBody)>();
        }
    }
}

fn invalidate_chunk_colliders(
    mut commands: Commands,
    mut workers: ResMut<ColliderWorkers>,
    mut updates: MessageReader<ChunkWillUpdate<PrototypeWorld>>,
    mut lod_changes: MessageReader<ChunkWillChangeLod<PrototypeWorld>>,
    mut chunk_spawns: MessageReader<ChunkWillSpawn<PrototypeWorld>>,
    mut dirty_entities: Local<HashSet<Entity>>,
) {
    dirty_entities.clear();

    for event in updates.read() {
        workers.awaiting_data.insert(event.entity);
        dirty_entities.insert(event.entity);
    }

    for event in lod_changes.read() {
        workers.awaiting_data.insert(event.entity);
        dirty_entities.insert(event.entity);
    }

    for event in chunk_spawns.read() {
        workers.awaiting_data.remove(&event.entity);
        dirty_entities.insert(event.entity);
    }

    for entity in dirty_entities.drain() {
        // Keep the previous collision surface until the replacement is ready.
        // Removing it here creates a physics hole during asynchronous remeshing.
        if let Ok(mut entity) = commands.get_entity(entity) {
            entity.remove::<ChunkColliderReady>();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalidation_keeps_collision_until_new_data_is_published() {
        let mut app = App::new();
        app.init_resource::<ColliderWorkers>()
            .add_message::<ChunkWillUpdate<PrototypeWorld>>()
            .add_message::<ChunkWillChangeLod<PrototypeWorld>>()
            .add_message::<ChunkWillSpawn<PrototypeWorld>>()
            .add_systems(Update, invalidate_chunk_colliders);
        let entity = app
            .world_mut()
            .spawn((
                Collider::cuboid(1.0, 1.0, 1.0),
                RigidBody::Static,
                ChunkColliderReady,
            ))
            .id();
        app.world_mut()
            .write_message(ChunkWillUpdate::<PrototypeWorld>::new(IVec3::ZERO, entity));
        app.update();
        assert!(app.world().get::<Collider>(entity).is_some());
        assert!(app.world().get::<ChunkColliderReady>(entity).is_none());
        assert!(
            app.world()
                .resource::<ColliderWorkers>()
                .awaiting_data
                .contains(&entity)
        );
        app.world_mut()
            .write_message(ChunkWillSpawn::<PrototypeWorld>::new(IVec3::ZERO, entity));
        app.update();
        assert!(app.world().get::<Collider>(entity).is_some());
        assert!(
            !app.world()
                .resource::<ColliderWorkers>()
                .awaiting_data
                .contains(&entity)
        );
    }

    #[test]
    fn worker_coordinates_exclude_padding_and_air() {
        let mut voxels = vec![WorldVoxel::Air; 64];
        voxels[0] = WorldVoxel::Solid(0u8);
        voxels[1 + 4 + 16] = WorldVoxel::Solid(0);
        assert_eq!(
            collect_solid_voxel_coordinates(&voxels, 4, 4, 4),
            [IVec3::ONE]
        );
        assert_eq!(full_chunk_coordinates(4, 4, 4).len(), 8);
    }

    #[test]
    fn cache_evicts_oldest_when_capacity_is_reached() {
        let mut cache = ChunkColliderCache::default();
        let collider = Collider::cuboid(1.0, 1.0, 1.0);

        cache.insert_with_capacity(1, collider.clone(), 2);
        cache.insert_with_capacity(2, collider.clone(), 2);
        cache.insert_with_capacity(3, collider, 2);

        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn cache_updates_duplicate_key_and_preserves_single_entry() {
        let mut cache = ChunkColliderCache::default();
        let collider_a = Collider::cuboid(1.0, 1.0, 1.0);
        let collider_b = Collider::cuboid(2.0, 2.0, 2.0);

        cache.insert_with_capacity(1, collider_a, 2);
        cache.insert_with_capacity(1, collider_b, 2);

        assert_eq!(cache.colliders.len(), 1);
        assert_eq!(cache.access_order.len(), 1);
        assert!(cache.get(1).is_some());
    }
}
