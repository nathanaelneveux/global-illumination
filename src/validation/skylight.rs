//! A high skylight in the isolated cave (`cave` profile with
//! VOXEL_GI_VALIDATE_CAVE_SKYLIGHT=open|pre). The ceiling edit is 38 voxels
//! above the floor its sunlight lands on, so the distance rule alone keeps
//! the floor's and the far wall's history: only sunlight-change detection
//! cuts it. `pre` opens the hole at startup and is the settled reference.
use super::*;
use voxel_gi_bvw::BvwVoxelEdits;

/// 7x7 ceiling opening, clear of the hanging crystals (x % 11 == 7,
/// z % 13 == 8); the cave sun lands about (+7.6, -38, +5.7) from it.
const HOLE_MIN: IVec3 = IVec3::new(42, 43, 23);
const HOLE_MAX: IVec3 = IVec3::new(48, 43, 29);
/// Light stone instead of dim lava where the patch lands, so its bounce
/// onto the east wall is what changes.
const FLOOR_MIN: IVec3 = IVec3::new(46, 4, 26);
const FLOOR_MAX: IVec3 = IVec3::new(58, 4, 38);
pub(super) const OPEN_AT: f32 = 15.0;

pub(super) fn mode() -> Option<&'static str> {
    static MODE: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VOXEL_GI_VALIDATE_CAVE_SKYLIGHT").as_deref() {
        Ok("open") => Some("open"),
        Ok("pre") => Some("pre"),
        _ => None,
    })
}

pub(super) const CAPTURES: &[(f32, &str)] = &[
    (14.5, "before"),
    (15.1, "open-0100ms"),
    (15.3, "open-0300ms"),
    (15.6, "open-0600ms"),
    (16.0, "open-1000ms"),
    (18.0, "open-3000ms"),
    (25.0, "open-10s"),
    (45.0, "open-30s"),
];

#[derive(Default)]
pub(super) struct SkylightRun {
    floor: bool,
    opened: bool,
}

impl SkylightRun {
    pub fn tick(&mut self, elapsed: f32, world: &mut VoxelWorld<PrototypeWorld>, dirty: &mut VoxelGiInput) {
        let Some(mode) = mode() else { return };
        let fill = |world: &mut VoxelWorld<PrototypeWorld>, dirty: &mut VoxelGiInput, min: IVec3, max: IVec3, voxel| {
            for z in min.z..=max.z {
                for x in min.x..=max.x {
                    for y in min.y..=max.y {
                        dirty.set_voxel(world, IVec3::new(x, y, z), voxel);
                    }
                }
            }
        };
        if !self.floor {
            fill(world, dirty, FLOOR_MIN, FLOOR_MAX, WorldVoxel::Solid(2));
            self.floor = true;
        }
        if !self.opened && (mode == "pre" || elapsed >= OPEN_AT) {
            fill(world, dirty, HOLE_MIN, HOLE_MAX, WorldVoxel::Air);
            self.opened = true;
        }
    }
}
