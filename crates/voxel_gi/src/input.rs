//! Host input: which 32³ pages are loaded, what changed in them, and how to
//! read their voxels. Storage engines (BVW through its connector, or any
//! other) report here; GI never reads a host's chunks or meshes directly.
use super::world::{AIR, PAGE_EDGE, PAGE_SIZE};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use std::sync::Arc;

/// Page residency and voxel edits for the next scene build. A page is the
/// 32³ block containing `voxel.div_euclid(32)`.
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
    Unloaded,
}

impl VoxelGiInput {
    /// Opt in only when every host write is reported with `set_voxel`, or
    /// explicitly marked unknown with `mark_voxel`/`mark_all_loaded`. Without
    /// it, every `page_changed` resamples the whole page from the source.
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
    pub fn page_changed(&mut self, page: IVec3) {
        self.pages.push((page, PageEvent::Changed));
    }

    /// A page was unloaded: its geometry and receivers leave GI.
    pub fn page_unloaded(&mut self, page: IVec3) {
        self.pages.push((page, PageEvent::Unloaded));
    }

    /// An exact edit: `Some(material)` is solid, `None` air. Last write wins.
    pub fn edit_voxel(&mut self, voxel: IVec3, material: Option<u8>) {
        self.voxels.remove(&voxel);
        self.edits.insert(voxel, material.map_or(AIR, u16::from));
    }

    /// A change of unknown value: its page is resampled from the source.
    pub fn mark_voxel(&mut self, voxel: IVec3) {
        // A later untracked write supersedes an earlier exact value, including
        // when both notifications arrive before GI drains this frame.
        self.edits.remove(&voxel);
        self.voxels.insert(voxel);
    }

    /// Everything loaded may have changed (e.g. a regenerated world).
    pub fn mark_all_loaded(&mut self) {
        self.edits.clear();
        self.all_loaded = true;
    }
}

/// Reads a page's voxels on a worker thread. Rebuilt by the host whenever its
/// storage changes (a snapshot). Calls run off the main thread; a slow source
/// can still delay publication of its page.
pub trait VoxelSource: Send + Sync + 'static {
    /// The page's material ids, at full resolution or at the stride the host
    /// stores it at (`PageSample`). `resample` asks for authoritative
    /// contents after an untracked change, not a cached fast path.
    fn sample_page(&self, page: IVec3, resample: bool) -> PageSample;
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
    /// An all-air page (one stride-32 cell). Air is not material 0: GI's
    /// solids are 0-255 (`dense_page`/`coarse_page` map `None` to air).
    pub fn air() -> Self {
        Self::coarse(PAGE_EDGE, vec![AIR])
    }

    /// A full-resolution page (`32³` materials, x fastest).
    pub fn full(materials: Vec<u16>) -> Self {
        Self::coarse(1, materials)
    }

    /// A page at `stride` (a power of two from 1 to 32) with `(32 / stride)³`
    /// cell materials, x fastest.
    pub fn coarse(stride: u32, cells: Vec<u16>) -> Self {
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
        debug_assert!(
            cells.iter().all(|&material| material <= AIR),
            "transport materials are 0..=255 or AIR; map host block types with MaterialPalette"
        );
        Self { stride, cells }
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// The material at a page-local voxel (each `0..32`).
    pub fn material(&self, local: UVec3) -> u16 {
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

/// The host's current `VoxelSource`. Scene builds wait until one is present.
#[derive(Resource, Default, Clone)]
pub struct VoxelGiSource(pub Option<Arc<dyn VoxelSource>>);

/// Build a full-resolution page from a per-voxel lookup (`None` is air).
pub fn dense_page(page: IVec3, voxel: impl Fn(IVec3) -> Option<u8>) -> PageSample {
    coarse_page(page, 1, voxel)
}

/// Build a page at `stride` from a lookup sampled once per cell, at the
/// cell's minimum voxel (`None` is air).
pub fn coarse_page(page: IVec3, stride: u32, voxel: impl Fn(IVec3) -> Option<u8>) -> PageSample {
    let origin = page * PAGE_EDGE as i32;
    let edge = PAGE_EDGE / stride.max(1);
    let mut cells = vec![AIR; (edge * edge * edge) as usize];
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let cell = UVec3::new(x, y, z);
                if let Some(material) = voxel(origin + (cell * stride).as_ivec3()) {
                    cells[((z * edge + y) * edge + x) as usize] = u16::from(material);
                }
            }
        }
    }
    PageSample::coarse(stride, cells)
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
            (local == IVec3::new(8, 0, 24)).then_some(5)
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
        assert_eq!(page.material(UVec3::new(9, 3, 30)), 5);
        assert_eq!(
            dense_page(IVec3::ZERO, |_| Some(2)).into_dense(),
            vec![2; 32 * 32 * 32]
        );
        assert_eq!(PageSample::air(), coarse_page(IVec3::ZERO, 32, |_| None));
        assert_eq!(PageSample::air().material(UVec3::splat(31)), AIR);
    }

    #[test]
    #[should_panic(expected = "power of two")]
    fn page_strides_must_divide_the_page() {
        PageSample::coarse(3, vec![0; 1000]);
    }

    #[test]
    fn unknown_notification_supersedes_an_exact_value_in_the_same_frame() {
        let mut input = VoxelGiInput::tracked();
        input.edit_voxel(IVec3::ONE, Some(7));
        input.mark_voxel(IVec3::ONE);
        assert!(input.edits.is_empty());
        input.edit_voxel(IVec3::ZERO, Some(4));
        input.mark_all_loaded();
        assert!(input.edits.is_empty());
    }
}
