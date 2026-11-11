#![doc = include_str!("../README.md")]
#![warn(missing_docs)]
mod material;

pub use material::{
    VoxelTransportExtension, VoxelTransportMaterial, material_template, prepare_texture_array,
    texture_array_ready,
};

use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use bevy_voxel_gi::{
    PageSample, VoxelGiInput, VoxelGiSource, VoxelGiSystems, VoxelSource, coarse_page, dense_page,
};
use bevy_voxel_world::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;

/// Positions no published chunk covers come from the world's own generator
/// when it has one. The default treats them as air. A coarse chunk's voxels
/// cover their whole block, even on resampling; the generator must not
/// replace the space between its samples.
pub trait GiFallbackVoxel {
    /// The voxel at `position` where no published chunk has one; `Unset` is air.
    fn fallback_voxel(&self, _position: IVec3) -> WorldVoxel<u8> {
        WorldVoxel::Unset
    }
}

/// The coarsest BVW level GI traces: a level-5 chunk spans 32³ GI pages,
/// one voxel sample each. Coarser chunks are left out of GI's world.
pub const MAX_GI_LOD: LodLevel = 5;

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
            bevy_voxel_gi::VOXEL_GI_PAGE_EDGE,
            "a level-0 BVW chunk must be one GI page"
        );
        material::install(app);
        app.init_resource::<PageCoverage<C>>().add_systems(
            Update,
            (forward_chunk_events::<C>, refresh_source::<C>)
                .chain()
                .before(VoxelGiSystems),
        );
    }
}

/// Edits through BVW that GI must see: the same value goes to both.
pub trait BvwVoxelEdits {
    /// Sets the voxel in BVW and reports the same edit to GI.
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

/// What the connector needs from a published chunk.
trait ChunkView {
    fn key(&self) -> ChunkKey;
    fn entity(&self) -> Entity;
    fn is_empty(&self) -> bool;
    fn data_shape(&self) -> UVec3;
    /// The voxel covering `position`, as the chunk's mesh shows it.
    fn voxel(&self, position: IVec3) -> Option<WorldVoxel<u8>>;
}

/// A published BVW chunk (BVW's `ChunkData` cannot be named outside it).
#[derive(Clone)]
struct Published {
    key: ChunkKey,
    entity: Entity,
    empty: bool,
    shape: UVec3,
    voxels: Arc<dyn Fn(IVec3) -> Option<WorldVoxel<u8>> + Send + Sync>,
}

type ChunkLookup = dyn Fn(ChunkKey) -> Option<Published> + Send + Sync;

/// The world's published chunks as `Published` views.
fn published_chunks<C: VoxelWorldConfig<MaterialIndex = u8>>(
    world: &VoxelWorld<C>,
) -> Arc<ChunkLookup> {
    let chunks = world.get_chunk_data_fn();
    Arc::new(move |key| {
        chunks(key).map(|data| Published {
            key: data.key(),
            entity: data.get_entity(),
            empty: data.is_empty(),
            shape: data.data_shape(),
            voxels: Arc::new(move |position| data.get_voxel_at_world_position(position)),
        })
    })
}

impl ChunkView for Published {
    fn key(&self) -> ChunkKey {
        self.key
    }
    fn entity(&self) -> Entity {
        self.entity
    }
    fn is_empty(&self) -> bool {
        self.empty
    }
    fn data_shape(&self) -> UVec3 {
        self.shape
    }
    fn voxel(&self, position: IVec3) -> Option<WorldVoxel<u8>> {
        (self.voxels)(position)
    }
}

/// The GI pages (level-0 chunk positions) a chunk spans; none above
/// `MAX_GI_LOD`.
fn chunk_pages(key: ChunkKey) -> impl Iterator<Item = IVec3> {
    let extent = if key.lod_level <= MAX_GI_LOD {
        key.extent()
    } else {
        0
    };
    (0..extent).flat_map(move |z| {
        (0..extent).flat_map(move |y| (0..extent).map(move |x| key.position + IVec3::new(x, y, z)))
    })
}

/// The finest published chunk covering `page`, as BVW's own voxel lookup
/// picks it. While a LoD change overlaps two levels, that is the finer.
fn finest_published<V>(page: IVec3, chunk: impl Fn(ChunkKey) -> Option<V>) -> Option<V> {
    (0..=MAX_GI_LOD).find_map(|lod| chunk(ChunkKey::containing_chunk(page, lod)))
}

/// Which chunk each GI page was last sampled from. GI hears that a page
/// loaded, changed or unloaded only when that changes (or the chunk's data
/// does), not for every chunk that comes and goes over it.
#[derive(Resource)]
struct PageCoverage<C> {
    pages: HashMap<IVec3, ChunkKey>,
    /// Chunks announced as despawning whose data BVW still publishes. Their
    /// pages update once it is gone, so GI never traces geometry no chunk
    /// shows, nor drops a page a finer chunk still covers.
    retiring: Vec<(ChunkKey, Entity)>,
    /// Spawned chunks' keys, for chunks a game system despawns without
    /// BVW announcing it.
    entities: HashMap<Entity, ChunkKey>,
    _world: PhantomData<C>,
}

impl<C> Default for PageCoverage<C> {
    fn default() -> Self {
        Self {
            pages: HashMap::default(),
            retiring: Vec::new(),
            entities: HashMap::default(),
            _world: PhantomData,
        }
    }
}

impl<C> PageCoverage<C> {
    /// Updates GI's pages for the chunks that spawned, published new data
    /// (`content`) or despawned (`retired`) this frame. Whether GI heard of
    /// any page.
    fn update<V: ChunkView>(
        &mut self,
        input: &mut VoxelGiInput,
        spawned: impl IntoIterator<Item = ChunkKey>,
        content: impl IntoIterator<Item = ChunkKey>,
        retired: impl IntoIterator<Item = (ChunkKey, Entity)>,
        chunk: impl Fn(ChunkKey) -> Option<V>,
    ) -> bool {
        let mut told = false;
        let mut dirty: HashSet<IVec3> = spawned.into_iter().flat_map(chunk_pages).collect();
        let content: HashSet<ChunkKey> = content.into_iter().collect();
        dirty.extend(content.iter().copied().flat_map(chunk_pages));
        self.retiring.extend(retired);
        self.retiring.retain(|&(key, entity)| {
            let gone = chunk(key).is_none_or(|data| data.entity() != entity);
            if gone {
                dirty.extend(chunk_pages(key));
            }
            !gone
        });
        // In a fixed order, so GI's input does not depend on hashing.
        let mut dirty: Vec<IVec3> = dirty.into_iter().collect();
        dirty.sort_unstable_by_key(|page| (page.z, page.y, page.x));
        for page in dirty {
            let now = finest_published(page, &chunk).map(|data| data.key());
            match (self.pages.get(&page).copied(), now) {
                (None, Some(key)) => {
                    input.page_loaded(page);
                    self.pages.insert(page, key);
                }
                (Some(before), Some(key)) if before != key || content.contains(&key) => {
                    input.page_changed(page);
                    self.pages.insert(page, key);
                }
                (Some(_), None) => {
                    input.page_unloaded(page);
                    self.pages.remove(&page);
                }
                _ => continue,
            }
            told = true;
        }
        told
    }
}

fn forward_chunk_events<C>(
    mut input: ResMut<VoxelGiInput>,
    mut coverage: ResMut<PageCoverage<C>>,
    world: VoxelWorld<C>,
    mut spawned: MessageReader<ChunkWillSpawn<C>>,
    mut published: MessageReader<ChunkPublished<C>>,
    mut updated: MessageReader<ChunkWillUpdate<C>>,
    mut despawned: MessageReader<ChunkWillDespawn<C>>,
    mut removed: RemovedComponents<Chunk<C>>,
) where
    C: VoxelWorldConfig<MaterialIndex = u8> + Resource,
{
    let spawned: Vec<_> = spawned
        .read()
        .map(|event| {
            coverage.entities.insert(event.entity, event.chunk_key);
            event.chunk_key
        })
        .collect();
    let mut content: Vec<_> = published.read().map(|event| event.chunk_key).collect();
    content.extend(updated.read().map(|event| event.chunk_key));
    let mut retired: Vec<_> = despawned
        .read()
        .map(|event| (event.chunk_key, event.entity))
        .collect();
    // Chunks despawned by a game system rather than BVW's lifecycle.
    for entity in removed.read() {
        if let Some(key) = coverage.entities.remove(&entity) {
            retired.push((key, entity));
        }
    }
    // Only frames that change a page refresh GI's source snapshot.
    if coverage.update(
        input.bypass_change_detection(),
        spawned,
        content,
        retired,
        &*published_chunks(&world),
    ) {
        input.set_changed();
    }
}

struct BvwSource<C: VoxelWorldConfig<MaterialIndex = u8>> {
    chunks: Arc<ChunkLookup>,
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
        let materials = finest_published(page, &*self.chunks).map(|chunk| {
            sample_chunk(page, &chunk, |position| {
                self.config.fallback_voxel(position)
            })
        });
        sample_page(page, resample, materials, &*self.voxels, |position| {
            self.config.fallback_voxel(position)
        })
    }
}

/// A page from its finest published chunk's `materials`, or from the world's
/// lookup and generator where no chunk covers it.
fn sample_page(
    page: IVec3,
    resample: bool,
    materials: Option<PageSample>,
    voxels: &dyn Fn(IVec3) -> WorldVoxel<u8>,
    fallback: impl Fn(IVec3) -> WorldVoxel<u8>,
) -> PageSample {
    let Some(materials) = materials else {
        return dense_page(page, |position| {
            solid(match voxels(position) {
                WorldVoxel::Unset => fallback(position),
                voxel => voxel,
            })
        });
    };
    if !resample {
        return materials;
    }
    let origin = page * CHUNK as i32;
    // The world's lookup includes uncommitted writes and BVW's edit map,
    // including edits inside a coarse chunk's blocks. Unset means nothing
    // covers the position there yet: keep the chunk's block. Only this
    // resampled page expands; ordinary ingestion stays strided.
    dense_page(page, |position| match voxels(position) {
        // Solid IDs fit u8; PageSample's air value does not.
        WorldVoxel::Unset => u8::try_from(materials.material((position - origin).as_uvec3())).ok(),
        voxel => solid(voxel),
    })
}

const CHUNK: u32 = bevy_voxel_gi::VOXEL_GI_PAGE_EDGE;

/// The voxels between a chunk's samples, when they are the same power of two
/// on every axis (BVW's default shapes: `1 << lod`).
fn sample_spacing(key: ChunkKey, shape: UVec3) -> Option<u32> {
    let samples = shape.x.checked_sub(2).filter(|samples| *samples > 0)?;
    let extent = key.voxel_extent() as u32;
    (shape == UVec3::splat(samples + 2) && extent.is_multiple_of(samples))
        .then_some(extent / samples)
        .filter(|spacing| spacing.is_power_of_two())
}

/// GI's page from a published chunk: what BVW draws. A level-L chunk's
/// sample covers a block `1 << L` voxels wide (with the default shapes), so
/// the page is strided by that, up to one cell per page.
fn sample_chunk(
    page: IVec3,
    chunk: &impl ChunkView,
    fallback: impl Fn(IVec3) -> WorldVoxel<u8>,
) -> PageSample {
    if chunk.is_empty() {
        return PageSample::air();
    }
    let at = |position| solid(chunk.voxel(position).unwrap_or_else(|| fallback(position)));
    match sample_spacing(chunk.key(), chunk.data_shape()) {
        Some(spacing) => coarse_page(page, spacing.min(CHUNK), at),
        None => dense_page(page, at),
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
    source.0 = Some(Arc::new(BvwSource {
        chunks: published_chunks(&world),
        voxels: world.get_voxel_fn(),
        config: config.clone(),
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::system::RunSystemOnce;
    use std::cell::RefCell;

    #[derive(Resource, Clone, Default)]
    struct TestWorld;
    impl VoxelWorldConfig for TestWorld {
        type MaterialIndex = u8;
        type ChunkUserBundle = ();
    }
    impl GiFallbackVoxel for TestWorld {}

    /// A published chunk whose voxel at a position is `voxel(position)`,
    /// sampled like BVW's: one value per block of its sample spacing.
    #[derive(Clone)]
    struct FakeChunk {
        key: ChunkKey,
        shape: UVec3,
        voxel: fn(IVec3) -> WorldVoxel<u8>,
    }

    impl FakeChunk {
        fn new(position: IVec3, lod: LodLevel) -> Self {
            Self {
                key: ChunkKey::new(position, lod),
                shape: UVec3::splat(34),
                voxel: |p| WorldVoxel::Solid((p.x.rem_euclid(200)) as u8),
            }
        }
    }

    impl ChunkView for FakeChunk {
        fn key(&self) -> ChunkKey {
            self.key
        }
        fn entity(&self) -> Entity {
            Entity::PLACEHOLDER
        }
        fn is_empty(&self) -> bool {
            false
        }
        fn data_shape(&self) -> UVec3 {
            self.shape
        }
        fn voxel(&self, position: IVec3) -> Option<WorldVoxel<u8>> {
            let spacing = self.key.voxel_extent() / (self.shape.x as i32 - 2);
            let offset = position - self.key.min_voxel();
            self.key
                .contains_voxel(position)
                .then(|| (self.voxel)(self.key.min_voxel() + offset / spacing * spacing))
        }
    }

    /// A chunk map and the GI input its coverage produced.
    #[derive(Default)]
    struct Coverage {
        map: RefCell<HashMap<ChunkKey, FakeChunk>>,
        coverage: PageCoverage<TestWorld>,
        input: VoxelGiInput,
    }

    impl Coverage {
        fn new() -> Self {
            Self {
                input: VoxelGiInput::tracked(),
                ..default()
            }
        }
        fn publish(&mut self, chunk: FakeChunk) {
            self.map.borrow_mut().insert(chunk.key, chunk);
        }
        fn remove(&mut self, key: ChunkKey) {
            self.map.borrow_mut().remove(&key);
        }
        fn step(
            &mut self,
            spawned: &[ChunkKey],
            content: &[ChunkKey],
            retired: &[ChunkKey],
        ) -> VoxelGiInput {
            self.input = VoxelGiInput::tracked();
            let map = &self.map;
            self.coverage.update(
                &mut self.input,
                spawned.iter().copied(),
                content.iter().copied(),
                retired.iter().map(|&key| (key, Entity::PLACEHOLDER)),
                |key| map.borrow().get(&key).cloned(),
            );
            std::mem::take(&mut self.input)
        }
    }

    fn expect(f: impl FnOnce(&mut VoxelGiInput)) -> VoxelGiInput {
        let mut input = VoxelGiInput::tracked();
        f(&mut input);
        input
    }

    fn pages(min: IVec3, extent: i32) -> Vec<IVec3> {
        chunk_pages(ChunkKey::new(min, extent.trailing_zeros() as LodLevel)).collect()
    }

    #[test]
    fn a_lod_split_moves_pages_to_the_finer_chunk_and_keeps_them_loaded() {
        let mut gi = Coverage::new();
        let coarse = FakeChunk::new(IVec3::ZERO, 1);
        gi.publish(coarse.clone());
        assert_eq!(
            gi.step(&[coarse.key], &[coarse.key], &[]),
            expect(|i| pages(IVec3::ZERO, 2)
                .into_iter()
                .for_each(|p| i.page_loaded(p))),
            "a level-1 chunk loads its eight pages"
        );
        // BVW spawns the finer chunk first, then retires the coarse one.
        let fine = FakeChunk::new(IVec3::X, 0);
        gi.publish(fine.clone());
        assert_eq!(
            gi.step(&[fine.key], &[fine.key], &[]),
            expect(|i| i.page_changed(IVec3::X)),
            "only the page the finer chunk covers is resampled"
        );
        assert_eq!(
            gi.step(&[], &[], &[coarse.key]),
            VoxelGiInput::tracked(),
            "nothing changes while the retiring chunk's data is published"
        );
        gi.remove(coarse.key);
        assert_eq!(
            gi.step(&[], &[], &[]),
            expect(|i| pages(IVec3::ZERO, 2)
                .into_iter()
                .filter(|p| *p != IVec3::X)
                .for_each(|p| i.page_unloaded(p))),
            "its other pages unload once BVW drops it; the finer page stays"
        );
    }

    #[test]
    fn a_lod_merge_keeps_tracing_the_fine_chunks_until_they_are_gone() {
        let mut gi = Coverage::new();
        let fine: Vec<_> = pages(IVec3::ZERO, 2)
            .into_iter()
            .map(|p| FakeChunk::new(p, 0))
            .collect();
        for chunk in &fine {
            gi.publish(chunk.clone());
        }
        let keys: Vec<_> = fine.iter().map(|chunk| chunk.key).collect();
        gi.step(&keys, &keys, &[]);
        let coarse = FakeChunk::new(IVec3::ZERO, 1);
        gi.publish(coarse.clone());
        assert_eq!(
            gi.step(&[coarse.key], &[coarse.key], &keys),
            VoxelGiInput::tracked(),
            "the fine chunks still show"
        );
        for key in &keys {
            gi.remove(*key);
        }
        assert_eq!(
            gi.step(&[], &[], &[]),
            expect(|i| pages(IVec3::ZERO, 2)
                .into_iter()
                .for_each(|p| i.page_changed(p))),
            "every page moves to the coarse chunk once the fine ones are gone"
        );
    }

    #[test]
    fn republished_data_resamples_the_pages_it_shows_on() {
        let mut gi = Coverage::new();
        let coarse = FakeChunk::new(IVec3::ZERO, 1);
        let fine = FakeChunk::new(IVec3::ZERO, 0);
        gi.publish(coarse.clone());
        gi.publish(fine.clone());
        gi.step(&[coarse.key, fine.key], &[coarse.key, fine.key], &[]);
        assert_eq!(
            gi.step(&[], &[fine.key], &[]),
            expect(|i| i.page_changed(IVec3::ZERO))
        );
        assert_eq!(
            gi.step(&[], &[coarse.key], &[]),
            expect(|i| pages(IVec3::ZERO, 2)
                .into_iter()
                .filter(|p| *p != IVec3::ZERO)
                .for_each(|p| i.page_changed(p))),
            "a coarse remesh leaves the page a finer chunk covers alone"
        );
    }

    #[test]
    fn chunks_above_the_gi_lod_limit_are_left_out() {
        assert_eq!(chunk_pages(ChunkKey::new(IVec3::ZERO, 0)).count(), 1);
        assert_eq!(
            chunk_pages(ChunkKey::new(IVec3::ZERO, MAX_GI_LOD)).count(),
            32 * 32 * 32
        );
        assert_eq!(
            chunk_pages(ChunkKey::new(IVec3::ZERO, MAX_GI_LOD + 1)).count(),
            0
        );
    }

    #[test]
    fn a_level_l_chunk_is_traced_at_its_sample_spacing() {
        // Material = the voxel's x, so each cell shows which block it took.
        for (lod, stride) in [(0, 1), (1, 2), (2, 4), (5, 32)] {
            let chunk = FakeChunk::new(IVec3::new(-32, 0, 0), lod);
            let page = IVec3::new(-32, 0, 0) + IVec3::X * (chunk.key.extent() - 1);
            let sample = sample_chunk(page, &chunk, |_| unreachable!());
            assert_eq!(sample.stride(), stride, "level {lod}");
            let origin = page * 32;
            for x in [0, 5, 31] {
                let block = (origin.x + x as i32).div_euclid(stride as i32) * stride as i32;
                assert_eq!(
                    sample.material(UVec3::new(x, 7, 3)),
                    block.rem_euclid(200) as u16,
                    "level {lod}, x {x}"
                );
            }
        }
        // Spacing that is not a power of two samples every voxel.
        let odd = FakeChunk {
            shape: UVec3::splat(14),
            ..FakeChunk::new(IVec3::ZERO, 0)
        };
        assert_eq!(sample_spacing(odd.key, odd.shape), None);
        assert_eq!(
            sample_chunk(IVec3::ZERO, &odd, |_| unreachable!()).stride(),
            1
        );
    }

    #[test]
    fn edits_reach_both_bvw_and_gi() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<TestWorld>::minimal()));
        app.update();
        let position = IVec3::new(-1, 31, 32);
        let input = app
            .world_mut()
            .run_system_once(move |mut world: VoxelWorld<TestWorld>| {
                let mut input = VoxelGiInput::tracked();
                input.set_voxel(&mut world, position, WorldVoxel::Solid(1));
                input.set_voxel(&mut world, position, WorldVoxel::Air);
                input.set_voxel(&mut world, position, WorldVoxel::Solid(2));
                input.set_voxel(&mut world, IVec3::ZERO, WorldVoxel::Unset);
                assert_eq!(world.get_voxel(position), WorldVoxel::Solid(2));
                input
            })
            .unwrap();
        let mut expected = VoxelGiInput::tracked();
        expected.edit_voxel(position, Some(2));
        expected.mark_voxel(IVec3::ZERO);
        assert_eq!(input, expected);
    }

    fn stored_voxel(page: &PageSample, local: UVec3) -> WorldVoxel<u8> {
        u8::try_from(page.material(local)).map_or(WorldVoxel::Air, WorldVoxel::Solid)
    }

    #[test]
    fn authoritative_resampling_preserves_coarse_coverage_with_or_without_a_generator() {
        for generator in [WorldVoxel::Unset, WorldVoxel::Solid(7)] {
            let key = IVec3::new(-2, 1, -1);
            let origin = key * 32;
            for stride in [1, 2, 4, 8, 16, 32] {
                let stored = coarse_page(key, stride, |p| {
                    ((p - origin).as_uvec3() / stride)
                        .element_sum()
                        .is_multiple_of(2)
                        .then_some(2)
                });
                // Positions no edit or chunk answers are Unset, not holes in
                // the coarse cell.
                let voxels = |_: IVec3| WorldVoxel::Unset;
                assert_eq!(
                    sample_page(key, false, Some(stored.clone()), &voxels, |_| generator),
                    stored
                );
                let resampled =
                    sample_page(key, true, Some(stored.clone()), &voxels, |_| generator);
                for i in 0..32u32 * 32 * 32 {
                    let local = UVec3::new(i % 32, i / 32 % 32, i / 1024);
                    assert_eq!(
                        resampled.material(local),
                        stored.material(local),
                        "stride {stride}, local {local}"
                    );
                }
            }
        }
    }

    #[test]
    fn authoritative_resampling_applies_bvw_writes_without_erasing_coarse_neighbors() {
        let mut app = App::new();
        app.add_plugins((MinimalPlugins, VoxelWorldPlugin::<TestWorld>::minimal()));
        app.update();
        let key = IVec3::new(-1, 0, 1);
        let origin = key * 32;
        let removed = origin + IVec3::splat(3);
        let added = origin + IVec3::new(11, 3, 3);
        app.world_mut()
            .run_system_once(move |mut world: VoxelWorld<TestWorld>| {
                world.set_voxel(removed, WorldVoxel::Air);
                world.set_voxel(added, WorldVoxel::Solid(0));
            })
            .unwrap();
        let stored = coarse_page(key, 8, |p| (p.x - origin.x < 8).then_some(2));
        // Check both the current write buffer and the persisted edit map.
        for flushed in [false, true] {
            if flushed {
                app.update();
            }
            let voxels = app
                .world_mut()
                .run_system_once(|world: VoxelWorld<TestWorld>| world.get_voxel_fn())
                .unwrap();
            let resampled = sample_page(key, true, Some(stored.clone()), &*voxels, |_| {
                WorldVoxel::Unset
            });
            for i in 0..32u32 * 32 * 32 {
                let local = UVec3::new(i % 32, i / 32 % 32, i / 1024);
                let p = origin + local.as_ivec3();
                let wanted = if p == removed {
                    WorldVoxel::Air
                } else if p == added {
                    WorldVoxel::Solid(0)
                } else {
                    stored_voxel(&stored, local)
                };
                assert_eq!(
                    stored_voxel(&resampled, local),
                    wanted,
                    "at {p}, flushed={flushed}"
                );
            }
        }
    }

    #[test]
    fn uncovered_pages_use_authoritative_values_then_the_generator() {
        let voxels = |p: IVec3| {
            if p == IVec3::ONE {
                WorldVoxel::Air
            } else {
                WorldVoxel::Unset
            }
        };
        for resample in [false, true] {
            let page = sample_page(IVec3::ZERO, resample, None, &voxels, |_| {
                WorldVoxel::Solid(7)
            });
            assert_eq!(stored_voxel(&page, UVec3::ONE), WorldVoxel::Air);
            assert_eq!(stored_voxel(&page, UVec3::ZERO), WorldVoxel::Solid(7));
        }
    }
}
