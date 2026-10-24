//! `bevy_voxel_world` connector: forwards chunk spawn/update/despawn as GI page
//! events and samples pages from BVW chunk data. Any `VoxelWorldConfig` with
//! `u8` materials works; a BVW chunk is exactly one GI page (32³). The mesh
//! material lit by GI is in `material`.
mod material;

pub use material::{
    VoxelTransportExtension, VoxelTransportMaterial, material_template, prepare_texture_array,
    texture_array_ready,
};

use bevy::prelude::*;
use voxel_gi::{VoxelGiInput, VoxelGiSource, VoxelGiSystems, VoxelSource, dense_page};
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
            voxel_gi::VOXEL_GI_PAGE_EDGE,
            "a BVW chunk must be one GI page"
        );
        material::install(app);
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
    use super::*;
    use bevy::ecs::system::RunSystemOnce;

    #[derive(Resource, Clone, Default)]
    struct TestWorld;
    impl VoxelWorldConfig for TestWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();
    }
    impl GiFallbackVoxel for TestWorld {}

    fn app() -> App {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<TestWorld>::minimal()))
            .insert_resource(VoxelGiInput::tracked())
            .init_resource::<VoxelGiSource>()
            .add_systems(Update, (forward_chunk_events::<TestWorld>, refresh_source::<TestWorld>).chain());
        app.update();
        app
    }

    #[test]
    fn chunk_events_and_edits_reach_both_bvw_and_gi() {
        let mut app = app();
        let position = IVec3::new(-1, 31, 32);
        let page = position.div_euclid(IVec3::splat(32));
        app.world_mut()
            .run_system_once(move |mut world: VoxelWorld<TestWorld>, mut input: ResMut<VoxelGiInput>| {
                input.set_voxel(&mut world, position, WorldVoxel::Solid(1));
                input.set_voxel(&mut world, position, WorldVoxel::Air);
                input.set_voxel(&mut world, position, WorldVoxel::Solid(2));
                input.set_voxel(&mut world, IVec3::ZERO, WorldVoxel::Unset);
            })
            .unwrap();
        app.world_mut().write_message(ChunkWillSpawn::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.world_mut().write_message(ChunkWillUpdate::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.world_mut().write_message(ChunkWillDespawn::<TestWorld>::new(IVec3::ONE, Entity::PLACEHOLDER));
        app.update();
        let mut expected = VoxelGiInput::tracked();
        expected.edit_voxel(position, Some(2));
        expected.mark_voxel(IVec3::ZERO);
        expected.page_loaded(page);
        expected.page_changed(page);
        expected.page_unloaded(IVec3::ONE);
        assert_eq!(*app.world().resource::<VoxelGiInput>(), expected);
        let source = app.world().resource::<VoxelGiSource>().0.clone().expect("source snapshot");
        let materials = source.sample_page(page, true);
        let local = (position - page * 32).as_uvec3();
        assert_eq!(materials[(local.x + 32 * (local.y + 32 * local.z)) as usize], 2);
        assert_eq!(materials.iter().filter(|&&m| m == 2).count(), 1);
    }
}
