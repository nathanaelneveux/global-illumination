//! A controlled single-source hallway, shared by gameplay and diagnostics.
//! Geometry and optical material stay constant along the measurement floor;
//! distance markers belong in diagnostics, not in differently colored blocks.
use super::*;
use crate::terrain::TerrainMaterial;
use crate::terrain_noise::{BEDROCK_FLOOR_Y, MAX_SURFACE_Y};
#[cfg(debug_assertions)]
use bevy::math::Vec3;

pub const ORIGIN: IVec3 = IVec3::new(-128, -12, -24);
pub const SIZE: UVec3 = UVec3::new(96, 9, 9);
pub const SOURCE: IVec3 = IVec3::new(-124, -11, -20);
#[cfg(any(test, debug_assertions))]
pub const OFFSETS: [i32; 16] = [1, 2, 3, 4, 6, 8, 12, 14, 16, 18, 20, 24, 32, 40, 48, 64];

#[cfg(any(test, debug_assertions))]
pub fn floor_receiver(offset: i32) -> IVec3 {
    SOURCE + IVec3::new(offset, -1, 0)
}

/// Equal projected footprint at every distance, entirely inside one floor face.
/// A small tilt avoids the degenerate straight-down look-at basis.
#[cfg(debug_assertions)]
pub fn measurement_view(offset: i32) -> (Vec3, Vec3, f32) {
    let target = floor_receiver(offset).as_vec3() + Vec3::new(0.5, 1., 0.5);
    (target + Vec3::new(0., 2.5, 0.05), target, 12.)
}

#[cfg(debug_assertions)]
pub fn overview() -> (Vec3, Vec3, f32) {
    (
        Vec3::new(-111., -8.5, -19.5),
        SOURCE.as_vec3() + Vec3::splat(0.5),
        65.,
    )
}

pub fn instances() -> Vec<PlacedStructure> {
    let rock = VoxelOp::Solid(TerrainMaterial::CaveRock.index());
    let chamber = StructureTemplate::from_fn(SIZE, |p| {
        let entrance = p.x == 95 && (3..=5).contains(&p.z) && (1..=4).contains(&p.y);
        // Overlapping baffles remove direct sky visibility without sealing the
        // walking route. No extra lamps contaminate the uniform test region.
        let baffle = (p.x == 88 && p.z < 6) || (p.x == 92 && p.z > 2);
        if p + ORIGIN == SOURCE {
            VoxelOp::Solid(TerrainMaterial::Emissive.index())
        } else if entrance {
            VoxelOp::Air
        } else if p.x == 0 || p.x == 95 || p.y == 0 || p.y == 8 || p.z == 0 || p.z == 8 || baffle {
            rock
        } else {
            VoxelOp::Air
        }
    });
    let entry_min = IVec3::new(-32, -12, -24);
    let entry = StructureTemplate::from_fn(UVec3::new(22, (MAX_SURFACE_Y + 14) as u32, 9), |p| {
        let world = p + entry_min;
        let floor = world.x + 21;
        if world.y > floor + 6 {
            VoxelOp::Keep
        } else if world.y <= floor || world.y == floor + 6 || p.z == 0 || p.z == 8 {
            rock
        } else {
            VoxelOp::Air
        }
    });
    // Flat threshold from the house's western foundation to the stair mouth.
    let road_min = IVec3::new(-11, BEDROCK_FLOOR_Y + 1, -23);
    let road = StructureTemplate::from_fn(
        UVec3::new(1, (MAX_SURFACE_Y + 1 - BEDROCK_FLOOR_Y) as u32, 7),
        |p| {
            if p.y + road_min.y <= 10 {
                rock
            } else {
                VoxelOp::Air
            }
        },
    );
    vec![
        PlacedStructure::new(100, ORIGIN, chamber),
        PlacedStructure::new(110, entry_min, entry),
        PlacedStructure::new(120, road_min, road),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashSet, VecDeque};

    #[test]
    fn hallway_has_one_source_and_a_uniform_measurement_floor() {
        let layer = StructureLayer::new(instances(), 32);
        let mut sources = 0;
        for z in 0..9 {
            for y in 0..9 {
                for x in 0..96 {
                    sources += usize::from(
                        layer.sample(ORIGIN + IVec3::new(x, y, z))
                            == VoxelOp::Solid(TerrainMaterial::Emissive.index()),
                    );
                }
            }
        }
        assert_eq!(sources, 1);
        for offset in OFFSETS {
            let p = floor_receiver(offset);
            assert_eq!(
                layer.sample(p),
                VoxelOp::Solid(TerrainMaterial::CaveRock.index())
            );
            assert_eq!(layer.sample(p + IVec3::Y), VoxelOp::Air);
        }
    }

    #[test]
    fn hallway_route_connects_house_stairs_baffles_and_source() {
        let layer = StructureLayer::new(instances(), 32);
        let walkable = |p| {
            matches!(layer.sample(p - IVec3::Y), VoxelOp::Solid(_))
                && (0..3).all(|y| layer.sample(p + IVec3::Y * y) == VoxelOp::Air)
        };
        let start = IVec3::new(-11, 11, -20);
        let target = SOURCE + IVec3::X * 2;
        assert!(walkable(start) && walkable(target));
        let mut seen = HashSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(p) = queue.pop_front() {
            for dir in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z] {
                for step in -1..=1 {
                    let next = p + dir + IVec3::Y * step;
                    if walkable(next) && seen.insert(next) {
                        queue.push_back(next);
                    }
                }
            }
        }
        assert!(
            seen.contains(&target),
            "three-cell headroom route must reach the lamp"
        );
    }
}
