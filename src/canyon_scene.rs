//! Deterministic, opt-in long-range daylight fixture. The host renderer and
//! offline transport studies share this exact geometry and optical palette.
use crate::voxel_gi::{EnergySpectrum, MaterialTransport, VoxelMaterialTransportTable};
use bevy::prelude::*;
use bevy_voxel_world::prelude::WorldVoxel;

pub const SIZE: IVec3 = IVec3::new(256, 256, 512);
pub const SPAWN: Vec3 = Vec3::new(128.0, 20.0, 80.0);
pub const EYE: Vec3 = Vec3::new(128.0, 22.0, 80.0);
pub const TARGET: Vec3 = Vec3::new(148.0, 125.0, 350.0);
pub const ENVIRONMENT: EnergySpectrum = EnergySpectrum::new(Vec3::new(0.25, 0.4, 0.7), 0.08, 0.0);

pub fn sun_direction() -> Vec3 {
    Vec3::new(-0.35, 1.0, -0.4).normalize()
}

fn wave(value: i32, period: i32) -> i32 {
    let phase = value.rem_euclid(period);
    phase.min(period - phase)
}

/// A column is solid through its top, with an irregular floor, stepped rims
/// and freestanding rock towers. Integer waves avoid noise-library/platform
/// differences and make arbitrary voxel/column queries agree exactly.
pub fn height(x: i32, z: i32) -> i32 {
    let floor = 8 + wave(x + 3 * z, 31) / 5 + wave(2 * x - z, 47) / 8;
    let bend = wave(z, 96) / 4 - 6;
    let left = 56 + bend + wave(z + 17, 43) / 5;
    let right = 200 + bend - wave(z + 3, 61) / 6;
    if x < left || x > right {
        return 220 + wave(z + x, 73) / 3 - wave(x - z, 41) / 4;
    }
    let mut top = floor;
    for (cx, cz, radius, peak) in [
        (106, 180, 11, 88),
        (150, 294, 15, 180),
        (98, 414, 12, 206),
        (173, 450, 8, 116),
    ] {
        let distance = (x - cx).abs().max((z - cz).abs());
        if distance <= radius {
            top = top.max(peak - (distance / 3) * 5 + wave(x + z, 13) / 4);
        }
    }
    // Fallen, terraced rock around the base of the high right wall.
    if x > right - 22 {
        top = top.max(floor + ((x - (right - 22)) / 4) * 9);
    }
    top
}

pub fn voxel(p: IVec3) -> WorldVoxel<u8> {
    if p.cmplt(IVec3::ZERO).any() || p.cmpge(SIZE).any() || p.y > height(p.x, p.z) {
        return WorldVoxel::Air;
    }
    WorldVoxel::Solid(if (p.y / 16) % 4 == 2 { 2 } else { 1 })
}

pub fn materials() -> VoxelMaterialTransportTable {
    let mut table = VoxelMaterialTransportTable::default();
    for (id, color) in [
        (1, Vec3::new(0.55, 0.20, 0.09)),
        (2, Vec3::new(0.65, 0.38, 0.23)),
    ] {
        table.set(
            id,
            MaterialTransport {
                diffuse: EnergySpectrum::new(color, 0.35, 0.04),
                emission: EnergySpectrum::ZERO,
            },
        );
    }
    table.set(4, crate::terrain::emissive_block_transport());
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn canyon_contains_rough_floor_high_rims_long_sightlines_and_towers() {
        assert!(height(128, 80) < SPAWN.y as i32 - 2);
        assert!(height(20, 300) > 200 && height(230, 300) > 200);
        assert!(height(150, 294) >= 180);
        assert!(height(98, 414) >= 200);
        let heights: std::collections::HashSet<_> = (120..140)
            .flat_map(|x| (70..90).map(move |z| height(x, z)))
            .collect();
        assert!(heights.len() > 1);
        for p in [IVec3::new(-1, 8, 0), SIZE, IVec3::new(128, 255, 200)] {
            assert_eq!(voxel(p), WorldVoxel::Air);
        }
        assert!(EYE.distance(TARGET) > 256.0);
        assert!(SIZE.max_element() > 256);
        for x in [32, 96, 128, 150, 224] {
            for z in [32, 180, 294, 414] {
                let y = height(x, z);
                assert!(matches!(voxel(IVec3::new(x, y, z)), WorldVoxel::Solid(_)));
                assert_eq!(voxel(IVec3::new(x, y + 1, z)), WorldVoxel::Air);
            }
        }
    }
}
