//! Host-authored structures. Placement, terrain adaptation and palette choices
//! are separate from the generic stamp/index implementation.
use super::*;
use crate::terrain::TerrainMaterial;
use crate::terrain_noise::{BEDROCK_FLOOR_Y, MAX_SURFACE_Y};
use bevy_voxel_world::prelude::WorldVoxel;
use std::sync::OnceLock;

/// Preserve the benchmark's local palette without changing normal lamps (4)
/// or snow-brick terrain (2). Material remapping happens only while authoring.
pub const CAVE_PALETTE: [(u8, TerrainMaterial); 5] = [
    (2, TerrainMaterial::CaveRock),
    (4, TerrainMaterial::Lava),
    (5, TerrainMaterial::SpireLight),
    (6, TerrainMaterial::ThoriumCrystal),
    (7, TerrainMaterial::CaveTorch),
];
pub const CAVE_ORIGIN: IVec3 = IVec3::new(40, -40, -64);

pub fn world_structures() -> Arc<StructureLayer> {
    static PLAN: OnceLock<Arc<StructureLayer>> = OnceLock::new();
    PLAN.get_or_init(|| {
        let mut instances = house();
        instances.extend(cave());
        Arc::new(StructureLayer::new(instances, 32))
    })
    .clone()
}

fn cave() -> Vec<PlacedStructure> {
    let rock = VoxelOp::Solid(TerrainMaterial::CaveRock.index());
    let chamber = StructureTemplate::from_fn(UVec3::new(56, 40, 56), |p| {
        match crate::cave_scene::voxel(p + IVec3::splat(4)) {
            WorldVoxel::Solid(local) => VoxelOp::Solid(
                CAVE_PALETTE
                    .iter()
                    .find(|(id, _)| *id == local)
                    .expect("cave author uses registered materials")
                    .1
                    .index(),
            ),
            WorldVoxel::Air => VoxelOp::Air,
            WorldVoxel::Unset => VoxelOp::Keep,
        }
    });
    let well = StructureTemplate::from_fn(UVec3::new(5, (MAX_SURFACE_Y - 2) as u32, 5), |_| {
        VoxelOp::Air
    });
    // Seven-wide passage, rock shell and a one-block descending staircase.
    // Keep above its roof leaves the natural hill intact instead of trenching it.
    let entry_min = IVec3::new(20, -16, -22);
    let entry = StructureTemplate::from_fn(UVec3::new(25, (MAX_SURFACE_Y + 18) as u32, 9), |p| {
        let y = p.y + entry_min.y;
        let floor = 10 - p.x;
        if y > floor + 6 {
            VoxelOp::Keep
        } else if y <= floor || y == floor + 6 {
            rock
        } else if p.z == 0 || p.z == 8 {
            if p.x % 8 == 0 && y == floor + 3 {
                VoxelOp::Solid(TerrainMaterial::CaveTorch.index())
            } else {
                rock
            }
        } else {
            VoxelOp::Air
        }
    });
    // Inside, turn toward -Z to descend along the safe west rim, outside lava.
    // Air above the steps stays transparent to the chamber's authored sources.
    let stairs_min = IVec3::new(45, -36, -43);
    let stairs = StructureTemplate::from_fn(UVec3::new(4, 23, 29), |p| {
        let world = p + stairs_min;
        let floor = if world.z >= -21 { -14 } else { world.z + 7 };
        if world.y <= floor {
            rock
        } else {
            VoxelOp::Keep
        }
    });
    let road = |size: UVec3| {
        StructureTemplate::from_fn(size, |p| {
            let world_y = p.y + BEDROCK_FLOOR_Y + 1;
            if world_y <= 10 { rock } else { VoxelOp::Air }
        })
    };
    let road_height = (MAX_SURFACE_Y + 1 - BEDROCK_FLOOR_Y) as u32;
    vec![
        PlacedStructure::new(30, CAVE_ORIGIN + IVec3::splat(4), chamber),
        PlacedStructure::new(40, CAVE_ORIGIN + IVec3::new(51, 44, 51), well),
        PlacedStructure::new(50, entry_min, entry),
        PlacedStructure::new(60, stairs_min, stairs),
        PlacedStructure::new(
            70,
            IVec3::new(-2, BEDROCK_FLOOR_Y + 1, -8),
            road(UVec3::new(22, road_height, 5)),
        ),
        PlacedStructure::new(
            80,
            IVec3::new(16, BEDROCK_FLOOR_Y + 1, -20),
            road(UVec3::new(4, road_height, 17)),
        ),
    ]
}

fn house() -> Vec<PlacedStructure> {
    let floor_min = IVec3::new(-10, BEDROCK_FLOOR_Y + 1, -28);
    let foundation = StructureTemplate::from_fn(
        UVec3::new(21, (MAX_SURFACE_Y + 1 - BEDROCK_FLOOR_Y) as u32, 21),
        |p| match p.y + floor_min.y {
            11.. => VoxelOp::Air,
            10 => VoxelOp::Solid(0),
            6..=9 => VoxelOp::Solid(2),
            _ => VoxelOp::Solid(1),
        },
    );
    let shell = StructureTemplate::from_fn(UVec3::new(13, 7, 13), |p| {
        let doorway = p.z == 12 && (5..=7).contains(&p.x) && p.y <= 3;
        let window = (p.z == 0 || p.z == 12) && (4..=8).contains(&p.x) && (2..=3).contains(&p.y);
        if p.y == 6 {
            VoxelOp::Solid(3)
        } else if p.x == 0 {
            VoxelOp::Solid(0)
        } else if p.x == 12 {
            VoxelOp::Solid(1)
        } else if (p.z == 0 || p.z == 12) && !doorway && !window {
            VoxelOp::Solid(2)
        } else {
            VoxelOp::Air
        }
    });
    vec![
        PlacedStructure::new(10, floor_min, foundation),
        PlacedStructure::new(20, IVec3::new(-6, 11, -24), shell),
    ]
}
