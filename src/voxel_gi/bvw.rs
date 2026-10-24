//! `bevy_voxel_world` connector: forwards chunk spawn/update/despawn as GI page
//! events and samples pages from BVW chunk data. Any `VoxelWorldConfig` with
//! `u8` materials works; a BVW chunk is exactly one GI page (32³).
use super::input::{VoxelGiInput, VoxelGiSource, VoxelSource, dense_page};
use super::VoxelGiSystems;
use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

/// Voxels a loaded chunk does not store (and unloaded positions) come from
/// the world's own generator when it has one. The default treats them as air.
pub trait GiFallbackVoxel {
    fn fallback_voxel(&self, _position: IVec3) -> WorldVoxel<u8> {
        WorldVoxel::Unset
    }
}

/// Connects the `VoxelWorld<C>` to voxel GI.
pub struct VoxelGiBvwPlugin<C>(PhantomData<C>);

impl<C> Default for VoxelGiBvwPlugin<C> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<C> Plugin for VoxelGiBvwPlugin<C>
where
    C: VoxelWorldConfig<MaterialIndex = u8> + GiFallbackVoxel + Resource + Clone,
{
    fn build(&self, app: &mut App) {
        assert_eq!(
            bevy_voxel_world::custom_meshing::CHUNK_SIZE_U,
            super::input::VOXEL_GI_PAGE_EDGE,
            "a BVW chunk must be one GI page"
        );
        super::bvw_material::install(app);
        app.add_systems(
            Update,
            (forward_chunk_events::<C>, refresh_source::<C>)
                .chain()
                .before(VoxelGiSystems),
        );
    }
}

/// Edits through BVW that GI must see: the same value goes to both.
pub trait BvwVoxelEdits {
    fn set_voxel<C: VoxelWorldConfig<MaterialIndex = u8>>(
        &mut self,
        world: &mut VoxelWorld<C>,
        position: IVec3,
        voxel: WorldVoxel<u8>,
    );
}

impl BvwVoxelEdits for VoxelGiInput {
    fn set_voxel<C: VoxelWorldConfig<MaterialIndex = u8>>(
        &mut self,
        world: &mut VoxelWorld<C>,
        position: IVec3,
        voxel: WorldVoxel<u8>,
    ) {
        world.set_voxel(position, voxel);
        match voxel {
            WorldVoxel::Solid(material) => self.edit_voxel(position, Some(material)),
            WorldVoxel::Air => self.edit_voxel(position, None),
            WorldVoxel::Unset => self.mark_voxel(position),
        }
    }
}

fn forward_chunk_events<C: VoxelWorldConfig>(
    mut input: ResMut<VoxelGiInput>,
    mut spawned: MessageReader<ChunkWillSpawn<C>>,
    mut updated: MessageReader<ChunkWillUpdate<C>>,
    mut despawned: MessageReader<ChunkWillDespawn<C>>,
) {
    for event in spawned.read() {
        input.page_loaded(event.chunk_key);
    }
    for event in updated.read() {
        input.page_changed(event.chunk_key);
    }
    for event in despawned.read() {
        input.page_unloaded(event.chunk_key);
    }
}

type ChunkPage = dyn Fn(IVec3, &dyn Fn(IVec3) -> WorldVoxel<u8>) -> Option<Vec<u16>> + Send + Sync;

struct BvwSource<C: VoxelWorldConfig<MaterialIndex = u8>> {
    /// A loaded chunk's page, with unstored voxels from the fallback.
    chunk_page: Arc<ChunkPage>,
    voxels: Arc<dyn Fn(IVec3) -> WorldVoxel<u8> + Send + Sync>,
    config: C,
}

fn solid(voxel: WorldVoxel<u8>) -> Option<u8> {
    match voxel {
        WorldVoxel::Solid(material) => Some(material),
        WorldVoxel::Air | WorldVoxel::Unset => None,
    }
}

impl<C> VoxelSource for BvwSource<C>
where
    C: VoxelWorldConfig<MaterialIndex = u8> + GiFallbackVoxel + Send + Sync + 'static,
{
    fn sample_page(&self, page: IVec3, resample: bool) -> Vec<u16> {
        let lookup = |position| match (self.voxels)(position) {
            WorldVoxel::Unset => self.config.fallback_voxel(position),
            voxel => voxel,
        };
        if !resample
            && let Some(materials) =
                (self.chunk_page)(page, &|position| self.config.fallback_voxel(position))
        {
            return materials;
        }
        dense_page(page, |position| solid(lookup(position)))
    }
}

/// A new source snapshot whenever GI input changed (chunk events, edits).
fn refresh_source<C>(
    input: Res<VoxelGiInput>,
    world: VoxelWorld<C>,
    config: Res<C>,
    mut source: ResMut<VoxelGiSource>,
) where
    C: VoxelWorldConfig<MaterialIndex = u8> + GiFallbackVoxel + Resource + Clone,
{
    if source.0.is_some() && !input.is_changed() && !config.is_changed() {
        return;
    }
    let chunks = world.get_chunk_data_fn();
    let chunk_page: Arc<ChunkPage> = Arc::new(move |page, fallback| {
        let chunk = chunks(page)?;
        if chunk.is_empty() {
            return Some(dense_page(page, |_| None));
        }
        Some(dense_page(page, |position| {
            solid(chunk.get_voxel_at_world_position(position).unwrap_or_else(|| fallback(position)))
        }))
    });
    source.0 = Some(Arc::new(BvwSource {
        chunk_page,
        voxels: world.get_voxel_fn(),
        config: config.clone(),
    }));
}

#[cfg(test)]
mod tests {
    use super::super::input::{VoxelGiInput, dense_page};
    use super::super::pages::{apply_page_edits, ingest_input};
    use super::super::telemetry::GiWorkTimings;
    use super::super::world::{AIR, PAGE_SIZE};
    use super::super::{VoxelGiRuntime, naadf, page_of};
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    #[derive(Resource, Clone, Default)]
    struct TestWorld;
    impl VoxelWorldConfig for TestWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();
    }
    impl GiFallbackVoxel for TestWorld {}

    fn app(tracked: bool) -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<TestWorld>::minimal()))
            .insert_resource(if tracked { VoxelGiInput::tracked() } else { VoxelGiInput::default() })
            .init_resource::<VoxelGiRuntime>()
            .init_resource::<GiWorkTimings>()
            .add_systems(Update, (forward_chunk_events::<TestWorld>, ingest_input).chain());
        app.update();
        app
    }

    #[test]
    fn exact_edits_are_last_write_wins_and_survive_inflight_snapshots_and_reload() {
        let mut app = app(true);
        let position = IVec3::new(-1, 31, 32);
        let page = page_of(position);
        app.world_mut().resource_mut::<VoxelGiRuntime>().loaded_pages.insert(page);
        app.world_mut()
            .run_system_once(move |mut world: VoxelWorld<TestWorld>, mut input: ResMut<VoxelGiInput>| {
                input.set_voxel(&mut world, position, WorldVoxel::Solid(1));
                input.set_voxel(&mut world, position, WorldVoxel::Air);
                input.set_voxel(&mut world, position, WorldVoxel::Solid(2));
            })
            .unwrap();
        app.update();
        app.world_mut().write_message(ChunkWillUpdate::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.update();
        let snapshot = app.world().resource::<VoxelGiRuntime>().page_edits.0[&page].clone();
        assert_eq!(snapshot[&position], 2);
        assert!(!app.world().resource::<VoxelGiRuntime>().full_resample_pages.contains(&page));
        app.world_mut()
            .run_system_once(move |mut world: VoxelWorld<TestWorld>, mut input: ResMut<VoxelGiInput>| {
                assert_eq!(world.get_voxel(position), WorldVoxel::Solid(2));
                input.set_voxel(&mut world, position, WorldVoxel::Air);
            })
            .unwrap();
        app.world_mut().write_message(ChunkWillDespawn::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.update();
        assert_eq!(snapshot[&position], 2, "worker view must not change");
        app.world_mut().write_message(ChunkWillSpawn::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.update();
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert!(runtime.dirty_pages.contains(&page));
        let mut data = dense_page(page, |_| Some(1));
        apply_page_edits(&mut data, page * 32, &runtime.page_edits.0[&page]);
        assert_eq!(data[naadf::linear_index((position - page * 32).as_uvec3(), PAGE_SIZE)], AIR);
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
                app.world_mut().resource_mut::<VoxelGiInput>().mark_voxel(IVec3::ONE);
            } else {
                app.world_mut()
                    .write_message(ChunkWillUpdate::<TestWorld>::new(IVec3::ZERO, Entity::PLACEHOLDER));
            }
            app.update();
            let runtime = app.world().resource::<VoxelGiRuntime>();
            assert!(runtime.full_resample_pages.contains(&IVec3::ZERO));
            assert!(!runtime.page_edits.0.contains_key(&IVec3::ZERO));
        }
    }
}
