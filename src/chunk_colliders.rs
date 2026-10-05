use std::collections::{HashMap, HashSet, VecDeque};

use avian3d::prelude::*;
use bevy::prelude::*;
use bevy_voxel_world::prelude::{
    Chunk, ChunkWillChangeLod, ChunkWillDespawn, ChunkWillSpawn, NeedsDespawn, VoxelWorld,
    WorldVoxel,
};

use crate::terrain::PrototypeWorld;

const CHUNK_COLLIDER_VOXEL_SIZE: f32 = 1.0;
const CHUNK_COLLIDER_CACHE_MAX_ENTRIES: usize = 512;
const MIN_PADDED_CHUNK_DIMENSION: u32 = 3;

type ColliderCacheKey = u64;

pub struct ChunkColliderPlugin;

impl Plugin for ChunkColliderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ChunkColliderCache>()
            .add_systems(PreUpdate, cleanup_chunk_assets)
            .add_systems(
                PostUpdate,
                (invalidate_chunk_colliders, ensure_chunk_colliders).chain(),
            );
    }
}

#[derive(Component, Default)]
struct ChunkColliderReady;

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
    voxel_world: VoxelWorld<PrototypeWorld>,
    chunk_meshes: Query<
        (Entity, &Chunk<PrototypeWorld>),
        (
            With<Mesh3d>,
            Without<ChunkColliderReady>,
            Without<NeedsDespawn>,
        ),
    >,
) {
    let started = std::time::Instant::now();
    let mut built = 0;
    for (entity, chunk) in &chunk_meshes {
        if chunk.lod_level != 0 {
            mark_chunk_collider_ready(&mut commands, entity);
            continue;
        }

        let Some(chunk_data) = voxel_world.get_chunk_data(chunk.position) else {
            continue;
        };

        if chunk_data.is_empty() {
            mark_chunk_collider_ready(&mut commands, entity);
            continue;
        }

        let voxels_hash = chunk_data.voxels_hash();
        let collider = if let Some(cached) = chunk_collider_cache.get(voxels_hash) {
            cached
        } else {
            let shape = chunk_data.data_shape();
            let [sx, sy, sz] = shape.to_array();
            if sx < MIN_PADDED_CHUNK_DIMENSION
                || sy < MIN_PADDED_CHUNK_DIMENSION
                || sz < MIN_PADDED_CHUNK_DIMENSION
            {
                continue;
            }

            let voxel_coords = if let Some(voxels) = chunk_data.voxels_arc() {
                collect_solid_voxel_coordinates(voxels.as_ref(), sx, sy, sz)
            } else if chunk_data.is_full() {
                full_chunk_coordinates(sx, sy, sz)
            } else {
                continue;
            };

            if voxel_coords.is_empty() {
                continue;
            }

            let collider = Collider::voxels(Vec3::splat(CHUNK_COLLIDER_VOXEL_SIZE), &voxel_coords);
            built += 1;
            chunk_collider_cache.insert(voxels_hash, collider.clone());
            collider
        };

        commands
            .entity(entity)
            .insert((RigidBody::Static, collider, ChunkColliderReady));
    }
    if started.elapsed().as_millis() >= 4 && std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        info!(
            built,
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
    mut events: MessageReader<ChunkWillDespawn<PrototypeWorld>>,
) {
    for event in events.read() {
        if let Ok(mut entity_commands) = commands.get_entity(event.entity) {
            entity_commands.remove::<(ChunkColliderReady, Collider, RigidBody)>();
        }
    }
}

fn invalidate_chunk_colliders(
    mut commands: Commands,
    mut lod_changes: MessageReader<ChunkWillChangeLod<PrototypeWorld>>,
    mut chunk_spawns: MessageReader<ChunkWillSpawn<PrototypeWorld>>,
    mut dirty_entities: Local<HashSet<Entity>>,
) {
    dirty_entities.clear();

    for event in lod_changes.read() {
        dirty_entities.insert(event.entity);
    }

    for event in chunk_spawns.read() {
        dirty_entities.insert(event.entity);
    }

    for entity in dirty_entities.drain() {
        commands
            .entity(entity)
            .remove::<(ChunkColliderReady, Collider, RigidBody)>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
