#![doc = include_str!("../README.md")]
mod material;

pub use material::{
    VoxelTransportExtension, VoxelTransportMaterial, material_template, prepare_texture_array,
    texture_array_ready,
};

use bevy::prelude::*;
use bevy_voxel_world::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;
use voxel_gi::{
    PageSample, VoxelGiInput, VoxelGiSource, VoxelGiSystems, VoxelSource, coarse_page, dense_page,
};

/// Positions with no loaded page (or gaps in unsupported sampling shapes)
/// come from the world's own generator when it has one. The default treats
/// them as air. Loaded LOD cells cover their whole block, even on resampling;
/// the generator must not replace the space between stored sample positions.
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

type ChunkPage =
    dyn Fn(IVec3, &dyn Fn(IVec3) -> WorldVoxel<u8>) -> Option<PageSample> + Send + Sync;

struct BvwSource<C: VoxelWorldConfig<MaterialIndex = u8>> {
    /// A loaded chunk's page, including the volume covered by coarse cells.
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
    fn sample_page(&self, page: IVec3, resample: bool) -> PageSample {
        if let Some(materials) =
            (self.chunk_page)(page, &|position| self.config.fallback_voxel(position))
        {
            if !resample {
                return materials;
            }
            let origin = page * CHUNK as i32;
            // The authoritative lookup includes uncommitted writes and BVW's
            // edit map, including edits between LOD sample positions. Unset
            // means no point sample, not air: retain the covering coarse cell.
            // Expand only this resampled page so a one-voxel edit need not
            // replace an entire coarse cell. Ordinary ingestion stays strided.
            return dense_page(page, |position| match (self.voxels)(position) {
                WorldVoxel::Unset => {
                    // Solid IDs fit u8; PageSample's air value does not.
                    u8::try_from(materials.material((position - origin).as_uvec3())).ok()
                }
                voxel => solid(voxel),
            });
        }
        let lookup = |position| match (self.voxels)(position) {
            WorldVoxel::Unset => self.config.fallback_voxel(position),
            voxel => voxel,
        };
        dense_page(page, |position| solid(lookup(position)))
    }
}

const CHUNK: u32 = voxel_gi::VOXEL_GI_PAGE_EDGE;

/// The stride a chunk's padded data shape stores (`samples + 2` per axis):
/// cubic, with a power-of-two sample count dividing the chunk. Other shapes
/// sample at full resolution through the world's lookup.
fn chunk_stride(shape: UVec3) -> Option<u32> {
    let samples = shape.x.checked_sub(2)?;
    (shape == UVec3::splat(samples + 2)
        && samples > 0
        && samples.is_power_of_two()
        && CHUNK.is_multiple_of(samples))
    .then_some(CHUNK / samples)
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
            return Some(PageSample::air());
        }
        if let Some(stride) = chunk_stride(chunk.data_shape()) {
            // One stored sample per cell, past the one-sample padding.
            return Some(coarse_page(page, stride, |position| {
                let cell = (position - page * CHUNK as i32).as_uvec3() / stride;
                solid(chunk.get_voxel(cell + UVec3::ONE))
            }));
        }
        Some(dense_page(page, |position| {
            solid(
                chunk
                    .get_voxel_at_world_position(position)
                    .unwrap_or_else(|| fallback(position)),
            )
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
            .add_systems(
                Update,
                (
                    forward_chunk_events::<TestWorld>,
                    refresh_source::<TestWorld>,
                )
                    .chain(),
            );
        app.update();
        app
    }

    #[test]
    fn chunk_events_and_edits_reach_both_bvw_and_gi() {
        let mut app = app();
        let position = IVec3::new(-1, 31, 32);
        let page = position.div_euclid(IVec3::splat(32));
        app.world_mut()
            .run_system_once(
                move |mut world: VoxelWorld<TestWorld>, mut input: ResMut<VoxelGiInput>| {
                    input.set_voxel(&mut world, position, WorldVoxel::Solid(1));
                    input.set_voxel(&mut world, position, WorldVoxel::Air);
                    input.set_voxel(&mut world, position, WorldVoxel::Solid(2));
                    input.set_voxel(&mut world, IVec3::ZERO, WorldVoxel::Unset);
                },
            )
            .unwrap();
        app.world_mut()
            .write_message(ChunkWillSpawn::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.world_mut()
            .write_message(ChunkWillUpdate::<TestWorld>::new(page, Entity::PLACEHOLDER));
        app.world_mut()
            .write_message(ChunkWillDespawn::<TestWorld>::new(
                IVec3::ONE,
                Entity::PLACEHOLDER,
            ));
        app.update();
        let mut expected = VoxelGiInput::tracked();
        expected.edit_voxel(position, Some(2));
        expected.mark_voxel(IVec3::ZERO);
        expected.page_loaded(page);
        expected.page_changed(page);
        expected.page_unloaded(IVec3::ONE);
        assert_eq!(*app.world().resource::<VoxelGiInput>(), expected);
        let source = app
            .world()
            .resource::<VoxelGiSource>()
            .0
            .clone()
            .expect("source snapshot");
        let materials = source.sample_page(page, true);
        let local = (position - page * 32).as_uvec3();
        assert_eq!(materials.stride(), 1);
        assert_eq!(materials.material(local), 2);
        let solid = (0..32u32 * 32 * 32)
            .filter(|i| materials.material(UVec3::new(i % 32, i / 32 % 32, i / 1024)) == 2)
            .count();
        assert_eq!(solid, 1);
    }

    #[test]
    fn lod_chunk_shapes_map_to_page_strides() {
        assert_eq!(chunk_stride(UVec3::splat(34)), Some(1));
        assert_eq!(chunk_stride(UVec3::splat(18)), Some(2));
        assert_eq!(chunk_stride(UVec3::splat(6)), Some(8));
        assert_eq!(chunk_stride(UVec3::splat(3)), Some(32));
        assert_eq!(
            chunk_stride(UVec3::splat(14)),
            None,
            "12 samples do not divide 32"
        );
        assert_eq!(chunk_stride(UVec3::new(34, 18, 34)), None, "non-cubic");
    }

    #[derive(Resource, Clone, Default)]
    struct GeneratedWorld;
    impl VoxelWorldConfig for GeneratedWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();
    }
    impl GiFallbackVoxel for GeneratedWorld {
        fn fallback_voxel(&self, _: IVec3) -> WorldVoxel<u8> {
            WorldVoxel::Solid(7)
        }
    }

    fn stored_voxel(page: &PageSample, local: UVec3) -> WorldVoxel<u8> {
        u8::try_from(page.material(local)).map_or(WorldVoxel::Air, WorldVoxel::Solid)
    }

    #[test]
    fn authoritative_resampling_preserves_coarse_coverage_with_or_without_a_generator() {
        fn check<C: VoxelWorldConfig<MaterialIndex = u8> + GiFallbackVoxel + Clone>(config: C) {
            let key = IVec3::new(-2, 1, -1);
            let origin = key * 32;
            for stride in [1, 2, 4, 8, 16, 32] {
                let stored = coarse_page(key, stride, |p| {
                    ((p - origin).as_uvec3() / stride)
                        .element_sum()
                        .is_multiple_of(2)
                        .then_some(2)
                });
                let expected = stored.clone();
                let sparse = stored.clone();
                let source = BvwSource {
                    chunk_page: Arc::new(move |_, _| Some(stored.clone())),
                    // BVW exposes exact stored sample positions; other
                    // positions are Unset, not holes in the coarse cell.
                    voxels: Arc::new(move |p| {
                        let local = (p - origin).as_uvec3();
                        if (local % UVec3::splat(stride)) == UVec3::ZERO {
                            stored_voxel(&sparse, local)
                        } else {
                            WorldVoxel::Unset
                        }
                    }),
                    config: config.clone(),
                };
                assert_eq!(source.sample_page(key, false), expected);
                let resampled = source.sample_page(key, true);
                for z in 0..32 {
                    for y in 0..32 {
                        for x in 0..32 {
                            let local = UVec3::new(x, y, z);
                            assert_eq!(
                                resampled.material(local),
                                expected.material(local),
                                "stride {stride}, local {local}"
                            );
                        }
                    }
                }
            }
        }
        check(TestWorld);
        check(GeneratedWorld);
    }

    #[test]
    fn authoritative_resampling_applies_bvw_writes_without_erasing_coarse_neighbors() {
        let mut app = app();
        let key = IVec3::new(-1, 0, 1);
        let origin = key * 32;
        let removed = origin + IVec3::splat(3);
        let added = origin + IVec3::new(11, 3, 3);
        let voxels = app
            .world_mut()
            .run_system_once(move |mut world: VoxelWorld<TestWorld>| {
                world.set_voxel(removed, WorldVoxel::Air);
                world.set_voxel(added, WorldVoxel::Solid(0));
                world.get_voxel_fn()
            })
            .unwrap();
        let stored = coarse_page(key, 8, |p| (p.x - origin.x < 8).then_some(2));
        let expected = stored.clone();
        let mut source = BvwSource {
            chunk_page: Arc::new(move |_, _| Some(stored.clone())),
            voxels,
            config: TestWorld,
        };
        // Check both the current write buffer and the persisted edit map.
        for flushed in [false, true] {
            if flushed {
                app.update();
                source.voxels = app
                    .world_mut()
                    .run_system_once(|world: VoxelWorld<TestWorld>| world.get_voxel_fn())
                    .unwrap();
            }
            let resampled = source.sample_page(key, true);
            for z in 0..32 {
                for y in 0..32 {
                    for x in 0..32 {
                        let local = UVec3::new(x, y, z);
                        let p = origin + local.as_ivec3();
                        let actual = stored_voxel(&resampled, local);
                        let wanted = if p == removed {
                            WorldVoxel::Air
                        } else if p == added {
                            WorldVoxel::Solid(0)
                        } else {
                            stored_voxel(&expected, local)
                        };
                        assert_eq!(actual, wanted, "at {p}, flushed={flushed}");
                    }
                }
            }
        }
    }

    #[test]
    fn missing_pages_still_use_authoritative_values_then_the_generator() {
        let source = BvwSource {
            chunk_page: Arc::new(|_, _| None),
            voxels: Arc::new(|p| {
                if p == IVec3::ONE {
                    WorldVoxel::Air
                } else {
                    WorldVoxel::Unset
                }
            }),
            config: GeneratedWorld,
        };
        for resample in [false, true] {
            let page = source.sample_page(IVec3::ZERO, resample);
            assert_eq!(stored_voxel(&page, UVec3::ONE), WorldVoxel::Air);
            assert_eq!(stored_voxel(&page, UVec3::ZERO), WorldVoxel::Solid(7));
        }
    }
}
