//! Host input: which 32³ pages are loaded, what changed in them, and how to
//! read their voxels. Storage engines (BVW through its connector, or any
//! other) report here; GI never reads a host's chunks or meshes directly.
use super::GiVoxel;
use super::world::{AIR, PAGE_EDGE, PAGE_SIZE};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use std::sync::Arc;

/// Queues page events and voxel edits for the next scene build.
///
/// A page spans 32 voxels on each axis. Page coordinates are the world voxel
/// coordinates divided by 32, rounded down on each axis. For example, voxel
/// `(-1, 0, 32)` belongs to page `(-1, 0, 1)`. Pass page coordinates to
/// [`Self::page_loaded`] and world voxel coordinates to [`Self::edit_voxel`].
#[derive(Resource, Default, Debug, PartialEq)]
pub struct VoxelGiInput {
    pub(super) pages: Vec<(IVec3, PageEvent)>,
    pub(super) voxels: HashSet<IVec3>,
    pub(super) edits: HashMap<IVec3, u16>,
    pub(super) complete_edit_tracking: bool,
    pub(super) all_loaded: bool,
}

impl VoxelGiInput {
    pub(super) fn is_idle(&self) -> bool {
        self.pages.is_empty() && self.voxels.is_empty() && self.edits.is_empty() && !self.all_loaded
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum PageEvent {
    Loaded,
    Changed,
    Resample,
    Unloaded,
}

impl VoxelGiInput {
    /// Opt in when the host reports every write.
    ///
    /// Use [`Self::edit_voxel`] for an exact value. Use [`Self::mark_voxel`]
    /// or [`Self::mark_all_loaded`] to request fresh samples. With default
    /// tracking, [`Self::page_changed`] resamples the page from the source.
    /// With tracked input, the edit notifications carry the changes.
    pub fn tracked() -> Self {
        Self {
            complete_edit_tracking: true,
            ..default()
        }
    }

    /// A page became available (or was regenerated): sample it.
    pub fn page_loaded(&mut self, page: IVec3) {
        self.pages.push((page, PageEvent::Loaded));
    }

    /// The host changed a loaded page's contents.
    ///
    /// Default tracking replaces the page's previous edit overlay with data
    /// from the source. With [`Self::tracked`], report the writes separately.
    pub fn page_changed(&mut self, page: IVec3) {
        self.pages.push((page, PageEvent::Changed));
    }

    /// A page was unloaded: its geometry and receivers leave GI.
    pub fn page_unloaded(&mut self, page: IVec3) {
        self.pages.push((page, PageEvent::Unloaded));
    }

    /// Resample a page from the source and discard its edit overlay, as a
    /// new source rather than an edit: nothing near it is relit urgently.
    ///
    /// GI keeps a page's exact edits (its overlay) across unloading, for
    /// hosts whose source lags their edits. When a page's source is genuinely
    /// new (the host regenerated or reloaded it), call this after
    /// [`Self::page_loaded`] so stale edits are not applied over it. Edits to
    /// the page queued earlier this frame are dropped too; include them in
    /// the source if they should survive. Unlike [`Self::mark_voxel`] it does
    /// not treat any voxel as edited.
    pub fn resample_page(&mut self, page: IVec3) {
        let edge = PAGE_EDGE as i32;
        self.edits
            .retain(|voxel, _| voxel.div_euclid(IVec3::splat(edge)) != page);
        self.pages.push((page, PageEvent::Resample));
    }

    /// An exact edit: the voxel is now `value` (a solid material or
    /// [`GiVoxel::AIR`]). Last write wins.
    pub fn edit_voxel(&mut self, voxel: IVec3, value: GiVoxel) {
        self.voxels.remove(&voxel);
        self.edits.insert(voxel, value.raw());
    }

    /// Requests an authoritative resample of the voxel's page.
    ///
    /// This discards the previous edit overlay for the page. Include earlier
    /// edits in the source if they should survive the resample.
    pub fn mark_voxel(&mut self, voxel: IVec3) {
        // A later untracked write supersedes an earlier exact value, including
        // when both notifications arrive before GI drains this frame.
        self.edits.remove(&voxel);
        self.voxels.insert(voxel);
    }

    /// Resamples all loaded pages from the source and clears old edit overlays.
    ///
    /// Use this after replacing the source with a regenerated world.
    pub fn mark_all_loaded(&mut self) {
        self.edits.clear();
        self.all_loaded = true;
    }
}

/// Reads a page's voxels on a worker thread.
///
/// Put a snapshot in [`VoxelGiSource`] and replace it when the underlying
/// data changes. GI applies exact edits reported through [`VoxelGiInput`]
/// on top of that snapshot. The page can reach the GPU after sampling and
/// scene construction finish.
///
/// A closure `Fn(IVec3) -> PageSample` is a source:
///
/// ```
/// # use bevy::prelude::*;
/// # use bevy_voxel_gi::{GiVoxel, VoxelGiSource, dense_page};
/// // A stone floor at y = 0 (material 0), air elsewhere.
/// let source = VoxelGiSource::new(|page: IVec3| {
///     dense_page(page, |voxel| (voxel.y == 0).then_some(GiVoxel::solid(0)))
/// });
/// # assert!(source.get().is_some());
/// ```
pub trait VoxelSource: Send + Sync + 'static {
    /// The page's voxels, at full resolution or at the stride the host
    /// stores it at (`PageSample`).
    fn sample_page(&self, page: IVec3) -> PageSample;

    /// The page's authoritative contents after a change GI was not told the
    /// details of (`page_changed`, `mark_voxel`, `resample_page`). Defaults to
    /// `sample_page`; override it only when `sample_page` reads a cache or
    /// a coarser copy that can lag the host's data.
    fn resample_page(&self, page: IVec3) -> PageSample {
        self.sample_page(page)
    }
}

impl<F> VoxelSource for F
where
    F: Fn(IVec3) -> PageSample + Send + Sync + 'static,
{
    fn sample_page(&self, page: IVec3) -> PageSample {
        self(page)
    }
}

/// A page's voxels as the host stores them: `(32 / stride)³` cells, x
/// fastest, where cell `c` covers the `stride³` voxels from `c * stride`.
/// Stride 1 is full resolution; an LOD host passes its coarse chunks as they
/// are (stride 2 to 32), and GI traces them as solid stride-sized blocks, the
/// shape a blocky LOD mesh rasterizes. Strides of 4 and up align with GI's
/// 4³ bricks, which are then stored as one material each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PageSample {
    stride: u32,
    cells: Vec<u16>,
}

impl PageSample {
    /// An all-air page (one stride-32 cell).
    pub fn air() -> Self {
        Self::uniform(GiVoxel::AIR)
    }

    /// A page of one value throughout (one stride-32 cell).
    pub fn uniform(voxel: GiVoxel) -> Self {
        Self::from_raw(PAGE_EDGE, vec![voxel.raw()])
    }

    /// A full-resolution page (`32³` voxels, x fastest).
    ///
    /// # Panics
    ///
    /// Panics if the input length differs from `32³`.
    pub fn full(voxels: Vec<GiVoxel>) -> Self {
        Self::coarse(1, voxels)
    }

    /// A page at `stride` (a power of two from 1 to 32) with `(32 / stride)³`
    /// cells, x fastest.
    ///
    /// # Panics
    ///
    /// Panics if the stride or cell count is invalid.
    pub fn coarse(stride: u32, cells: Vec<GiVoxel>) -> Self {
        // Same size and alignment: reinterpreted in place, no copy.
        Self::from_raw(stride, bytemuck::cast_vec(cells))
    }

    /// `coarse` over GI's internal encoding (`0..=255` solid, `AIR`).
    pub(crate) fn from_raw(stride: u32, cells: Vec<u16>) -> Self {
        assert!(
            stride.is_power_of_two() && stride <= PAGE_EDGE,
            "page stride {stride} must be a power of two up to {PAGE_EDGE}"
        );
        let edge = (PAGE_EDGE / stride) as usize;
        assert_eq!(
            cells.len(),
            edge * edge * edge,
            "a stride-{stride} page has {edge}³ cells"
        );
        debug_assert!(cells.iter().all(|&material| material <= AIR));
        Self { stride, cells }
    }

    /// Voxels per cell side: 1 at full resolution, up to 32.
    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// The voxel at a page-local position (each `0..32`).
    pub fn voxel(&self, local: UVec3) -> GiVoxel {
        GiVoxel::from_raw(self.raw(local))
    }

    /// `voxel` in GI's internal encoding.
    pub(crate) fn raw(&self, local: UVec3) -> u16 {
        let edge = PAGE_EDGE / self.stride;
        let cell = local / self.stride;
        self.cells[((cell.z * edge + cell.y) * edge + cell.x) as usize]
    }

    /// One material throughout, or the dense `32³` page.
    pub(super) fn into_voxels(self) -> super::world::PageVoxels {
        if self.cells.iter().all(|cell| *cell == self.cells[0]) {
            super::world::PageVoxels::Uniform(self.cells[0])
        } else {
            super::world::PageVoxels::Dense(self.into_dense())
        }
    }

    /// Full-resolution materials, each coarse cell repeated over its block.
    pub(super) fn into_dense(self) -> Vec<u16> {
        if self.stride == 1 {
            return self.cells;
        }
        let edge = PAGE_EDGE / self.stride;
        let mut materials = vec![AIR; PAGE_SIZE.element_product() as usize];
        for z in 0..PAGE_EDGE {
            for y in 0..PAGE_EDGE {
                let row = (z * PAGE_EDGE + y) * PAGE_EDGE;
                let cell_row = ((z / self.stride) * edge + y / self.stride) * edge;
                for x in 0..PAGE_EDGE {
                    materials[(row + x) as usize] =
                        self.cells[(cell_row + x / self.stride) as usize];
                }
            }
        }
        materials
    }
}

/// The host's current [`VoxelSource`]. Scene builds wait until one is present.
///
/// Insert one with [`Self::new`], or replace the source in place with
/// [`Self::set`] when the host's data is a new snapshot. Scene builds already
/// running keep the source they started with.
#[derive(Resource, Default, Clone)]
pub struct VoxelGiSource(Option<Arc<dyn VoxelSource>>);

impl VoxelGiSource {
    /// A resource holding `source`.
    pub fn new(source: impl VoxelSource) -> Self {
        Self(Some(Arc::new(source)))
    }

    /// A resource holding a source the host already shares.
    pub fn from_shared(source: Arc<dyn VoxelSource>) -> Self {
        Self(Some(source))
    }

    /// Replace the source; later scene builds sample this one.
    pub fn set(&mut self, source: impl VoxelSource) {
        self.0 = Some(Arc::new(source));
    }

    /// Replace the source with one the host already shares.
    pub fn set_shared(&mut self, source: Arc<dyn VoxelSource>) {
        self.0 = Some(source);
    }

    /// Remove the source: scene builds wait until another is set.
    pub fn clear(&mut self) {
        self.0 = None;
    }

    /// The current source, if any.
    pub fn get(&self) -> Option<&Arc<dyn VoxelSource>> {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for VoxelGiSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("VoxelGiSource")
            .field(&self.0.as_ref().map(|_| "dyn VoxelSource"))
            .finish()
    }
}

/// Build a full-resolution page by calling the lookup at each world voxel.
/// The lookup receives world coordinates and returns a [`GiVoxel`], or an
/// `Option<GiVoxel>` whose `None` is air.
pub fn dense_page<V: Into<GiVoxel>>(page: IVec3, voxel: impl Fn(IVec3) -> V) -> PageSample {
    coarse_page(page, 1, voxel)
}

/// Build a page by sampling the minimum world voxel of each coarse cell, as
/// [`dense_page`] does.
///
/// # Panics
///
/// Panics unless `stride` is a power of two from 1 to 32.
pub fn coarse_page<V: Into<GiVoxel>>(
    page: IVec3,
    stride: u32,
    voxel: impl Fn(IVec3) -> V,
) -> PageSample {
    let origin = page * PAGE_EDGE as i32;
    let edge = PAGE_EDGE / stride.max(1);
    let mut cells = vec![AIR; (edge * edge * edge) as usize];
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let cell = UVec3::new(x, y, z);
                cells[((z * edge + y) * edge + x) as usize] =
                    voxel(origin + (cell * stride).as_ivec3()).into().raw();
            }
        }
    }
    PageSample::from_raw(stride, cells)
}

/// Every page holding a voxel is this many voxels wide.
pub const VOXEL_GI_PAGE_EDGE: u32 = PAGE_EDGE;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coarse_pages_expand_each_cell_over_its_block() {
        // Stride 8: 4³ cells; the cell at (1, 0, 3) covers x 8..16, z 24..32.
        let page = coarse_page(IVec3::new(-1, 0, 2), 8, |p| {
            let local = p - IVec3::new(-32, 0, 64);
            (local == IVec3::new(8, 0, 24)).then_some(GiVoxel::solid(5))
        });
        assert_eq!(page.stride(), 8);
        let dense = page.clone().into_dense();
        let at = |x: u32, y: u32, z: u32| {
            dense[super::super::naadf::linear_index(UVec3::new(x, y, z), PAGE_SIZE)]
        };
        assert_eq!(at(8, 0, 24), 5);
        assert_eq!(at(15, 7, 31), 5);
        assert_eq!(at(16, 0, 24), AIR);
        assert_eq!(at(7, 0, 24), AIR);
        assert_eq!(dense.iter().filter(|&&m| m == 5).count(), 512);
        assert_eq!(page.voxel(UVec3::new(9, 3, 30)), GiVoxel::solid(5));
        assert_eq!(
            dense_page(IVec3::ZERO, |_| GiVoxel::solid(2)).into_dense(),
            vec![2; 32 * 32 * 32]
        );
        assert_eq!(
            PageSample::air(),
            coarse_page(IVec3::ZERO, 32, |_| None::<GiVoxel>)
        );
        assert_eq!(PageSample::air().voxel(UVec3::splat(31)), GiVoxel::AIR);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn page_strides_must_divide_the_page() {
        PageSample::coarse(3, vec![GiVoxel::AIR; 1000]);
    }

    #[test]
    fn unknown_notification_supersedes_an_exact_value_in_the_same_frame() {
        let mut input = VoxelGiInput::tracked();
        input.edit_voxel(IVec3::ONE, GiVoxel::solid(7));
        input.mark_voxel(IVec3::ONE);
        assert!(input.edits.is_empty());
        input.edit_voxel(IVec3::ZERO, GiVoxel::solid(4));
        input.mark_all_loaded();
        assert!(input.edits.is_empty());
    }

    #[test]
    fn a_page_resample_drops_only_that_pages_queued_edits() {
        let mut input = VoxelGiInput::tracked();
        input.edit_voxel(IVec3::new(5, 0, 0), GiVoxel::solid(1));
        input.edit_voxel(IVec3::new(-1, 0, 0), GiVoxel::solid(2));
        input.resample_page(IVec3::ZERO);
        assert_eq!(input.edits.len(), 1);
        assert!(input.edits.contains_key(&IVec3::new(-1, 0, 0)));
        assert!(input.voxels.is_empty());
        assert_eq!(input.pages, vec![(IVec3::ZERO, PageEvent::Resample)]);
    }
}
