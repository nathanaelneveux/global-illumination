//! Host input: which 32³ pages are loaded, what changed in them, and how to
//! read their voxels. Storage engines (BVW through its connector, or any
//! other) report here; GI never reads a host's chunks or meshes directly.
use super::world::{AIR, PAGE_EDGE, PAGE_SIZE};
use bevy::platform::collections::{HashMap, HashSet};
use bevy::prelude::*;
use std::sync::Arc;

/// Page residency and voxel edits for the next scene build. A page is the
/// 32³ block containing `voxel.div_euclid(32)`.
#[derive(Resource, Default)]
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
/// storage changes (a snapshot), so it never blocks the main world.
pub trait VoxelSource: Send + Sync + 'static {
    /// Dense material ids for `page`, x fastest (`dense_page` builds one from
    /// a per-voxel lookup). `resample` asks for authoritative contents after
    /// an untracked change, not a cached fast path.
    fn sample_page(&self, page: IVec3, resample: bool) -> Vec<u16>;
}

/// The host's current `VoxelSource`. Scene builds wait until one is present.
#[derive(Resource, Default, Clone)]
pub struct VoxelGiSource(pub Option<Arc<dyn VoxelSource>>);

/// Build a dense page from a per-voxel lookup (`None` is air).
pub fn dense_page(page: IVec3, voxel: impl Fn(IVec3) -> Option<u8>) -> Vec<u16> {
    let origin = page * PAGE_EDGE as i32;
    let mut materials = vec![AIR; PAGE_SIZE.element_product() as usize];
    for z in 0..PAGE_EDGE {
        for y in 0..PAGE_EDGE {
            for x in 0..PAGE_EDGE {
                let local = UVec3::new(x, y, z);
                if let Some(material) = voxel(origin + local.as_ivec3()) {
                    materials[super::naadf::linear_index(local, PAGE_SIZE)] = u16::from(material);
                }
            }
        }
    }
    materials
}

/// Every page holding a voxel is this many voxels wide.
pub const VOXEL_GI_PAGE_EDGE: u32 = PAGE_EDGE;

#[cfg(test)]
mod tests {
    use super::*;

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
