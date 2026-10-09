//! Host-authored structures. Placement, terrain adaptation and palette choices
//! are separate from the generic stamp/index implementation.
use super::*;
use crate::terrain_noise::{BEDROCK_FLOOR_Y, MAX_SURFACE_Y};
use std::sync::OnceLock;

pub fn world_structures() -> Arc<StructureLayer> {
    static PLAN: OnceLock<Arc<StructureLayer>> = OnceLock::new();
    PLAN.get_or_init(|| Arc::new(StructureLayer::new(house(), 32)))
        .clone()
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
