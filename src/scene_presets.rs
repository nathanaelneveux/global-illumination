//! Host-only experiment selection. Transport code never branches on a scene.
use voxel_gi::VoxelMaterialTransportTable;
use bevy::prelude::*;
use bevy_voxel_world::prelude::WorldVoxel;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScenePreset {
    Terrain,
    Cave,
    Canyon,
}

impl ScenePreset {
    pub fn current() -> Self {
        if std::env::var("VOXEL_GI_SCENE").is_ok_and(|v| v == "canyon")
            || (cfg!(debug_assertions)
                && std::env::var_os("VOXEL_GI_VALIDATE").is_some()
                && std::env::var("VOXEL_GI_VALIDATE_PROFILE")
                    .is_ok_and(|v| v == "canyon" || v == "day-cycle"))
        {
            Self::Canyon
        } else if crate::cave_scene::enabled() {
            Self::Cave
        } else {
            Self::Terrain
        }
    }
    pub fn voxel(self, p: IVec3) -> Option<WorldVoxel<u8>> {
        match self {
            Self::Terrain => None,
            Self::Cave => Some(crate::cave_scene::voxel(p)),
            Self::Canyon => Some(crate::canyon_scene::voxel(p)),
        }
    }
    pub fn materials(self) -> Option<VoxelMaterialTransportTable> {
        match self {
            Self::Terrain => None,
            Self::Cave => Some(crate::cave_scene::materials()),
            Self::Canyon => Some(crate::canyon_scene::materials()),
        }
    }
    pub fn spawn(self) -> Option<Vec3> {
        match self {
            Self::Terrain => None,
            Self::Cave => Some(crate::cave_scene::SPAWN),
            Self::Canyon => Some(crate::canyon_scene::SPAWN),
        }
    }
}
