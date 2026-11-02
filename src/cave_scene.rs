//! Opt-in host scene shared by interactive rendering and transport studies.
//! Geometry/material IDs are demo policy, never special cases in the GI core.
use bevy::prelude::*;
use bevy_voxel_world::prelude::WorldVoxel;

use bevy_voxel_gi::{MaterialTransport, VoxelMaterialTransportTable};

pub fn enabled() -> bool {
    if background_terrain_requested() {
        return false;
    }
    std::env::var("VOXEL_GI_SCENE").is_ok_and(|v| v == "cave")
        || (cfg!(debug_assertions)
            && std::env::var_os("VOXEL_GI_VALIDATE").is_some()
            && std::env::var("VOXEL_GI_VALIDATE_PROFILE").is_ok_and(|v| v == "cave"))
}

/// Validation composition only: keep the real terrain generator and use the
/// nearby structure instead of replacing the world with the isolated chamber.
pub fn background_terrain_requested() -> bool {
    cfg!(debug_assertions)
        && std::env::var_os("VOXEL_GI_VALIDATE").is_some()
        && std::env::var("VOXEL_GI_VALIDATE_BACKGROUND").is_ok_and(|v| v == "1")
}

pub const SPAWN: Vec3 = Vec3::new(8.0, 6.5, 55.0);
#[cfg(debug_assertions)]
pub const EYE: Vec3 = Vec3::new(8.0, 21.0, 56.0);
#[cfg(debug_assertions)]
pub const TARGET: Vec3 = Vec3::new(38.0, 22.0, 32.0);
pub const ENVIRONMENT: Vec3 = Vec3::new(0.35, 0.45, 0.7);

pub fn sun_direction() -> Vec3 {
    Vec3::new(-0.2, 1.0, -0.15).normalize()
}

/// Artistic RGB emission and albedo, not calibrated lava/thorium physics.
pub fn materials() -> VoxelMaterialTransportTable {
    let mut table = VoxelMaterialTransportTable::default();
    for (id, diffuse, emission) in [
        (2, Vec3::splat(0.55), Vec3::ZERO),
        (4, Vec3::splat(0.15), Vec3::new(3.0, 0.45, 0.02)),
        (5, Vec3::splat(0.2), Vec3::new(0.03, 0.4, 12.0)),
        (6, Vec3::splat(0.2), Vec3::new(0.3, 6.0, 0.08)),
        (7, Vec3::splat(0.2), Vec3::new(14.0, 4.0, 0.8)),
    ] {
        table.set(id, MaterialTransport { diffuse, emission });
    }
    table
}

/// A finite 56×40×56 chamber with a broad lava floor, blue-lit spire,
/// green ceiling crystals, wall torches and a far-side roof opening.
pub fn voxel(p: IVec3) -> WorldVoxel<u8> {
    let IVec3 { x, y, z } = p;
    if !(4..=59).contains(&x) || !(4..=43).contains(&y) || !(4..=59).contains(&z) {
        return WorldVoxel::Air;
    }
    if y == 43 && (51..=55).contains(&x) && (51..=55).contains(&z) {
        return WorldVoxel::Air;
    }
    let id = if y == 4 {
        if (10..=54).contains(&x) && (10..=54).contains(&z) {
            4
        } else {
            2
        }
    } else if x == 4 || x == 59 || z == 4 || z == 59 || y == 43 {
        2
    } else if (30..=34).contains(&x) && (30..=34).contains(&z) && y <= 37 {
        if y % 4 == 0 && (x == 30 || x == 34 || z == 30 || z == 34) {
            5
        } else {
            2
        }
    } else if y >= 40 && x % 11 == 7 && z % 13 == 8 {
        6
    } else if y == 10
        && (((x == 5 || x == 58) && z % 8 == 0) || ((z == 5 || z == 58) && x % 8 == 0))
    {
        7
    } else {
        return WorldVoxel::Air;
    };
    WorldVoxel::Solid(id)
}
