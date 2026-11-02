//! A high skylight in the isolated cave (`cave` profile with
//! VOXEL_GI_VALIDATE_CAVE_SKYLIGHT=open|pre). The ceiling edit is 38 voxels
//! above the floor its sunlight lands on, so the distance rule alone keeps
//! the floor's and the far wall's history: only sunlight-change detection
//! cuts it. `pre` opens the hole at startup and is the settled reference.
use super::*;
use bevy_voxel_gi_bvw::BvwVoxelEdits;

/// 7x7 ceiling opening, clear of the hanging crystals (x % 11 == 7,
/// z % 13 == 8); the cave sun lands about (+7.6, -38, +5.7) from it.
const HOLE_MIN: IVec3 = IVec3::new(42, 43, 23);
const HOLE_MAX: IVec3 = IVec3::new(48, 43, 29);
/// Light stone instead of dim lava where the patch lands, so its bounce
/// onto the east wall is what changes.
const FLOOR_MIN: IVec3 = IVec3::new(46, 4, 26);
const FLOOR_MAX: IVec3 = IVec3::new(58, 4, 38);
pub(super) const OPEN_AT: f32 = 15.0;

/// Moving-sun variant (with `=pre`): the sun turns about world Z at
/// `VOXEL_GI_VALIDATE_SKYLIGHT_SUN_RATE` degrees per second from 15 s (offset
/// `VOXEL_GI_VALIDATE_SUN_OFFSET`; a fixed reference is rate 0 at the offset
/// a capture reaches), sliding the floor patch, while
/// `VOXEL_GI_VALIDATE_SKYLIGHT_AWAY=1` keeps the camera on the far corner:
/// the patch's light reaches the screen only as bounce, through world-cache
/// faces off screen.
pub(super) fn sun() -> Option<(f32, f32)> {
    let env = |name: &str| std::env::var(name).ok().and_then(|v| v.parse::<f32>().ok());
    let rate = env("VOXEL_GI_VALIDATE_SKYLIGHT_SUN_RATE")?;
    Some((rate, env("VOXEL_GI_VALIDATE_SUN_OFFSET").unwrap_or(0.0)))
}

/// `VOXEL_GI_VALIDATE_SKYLIGHT_DARK=1`: no emitters (lava, crystals,
/// torches, spire bands emit nothing), so sunlight through the openings is
/// the cave's only light.
pub(super) fn dark() -> bool {
    std::env::var_os("VOXEL_GI_VALIDATE_SKYLIGHT_DARK").is_some()
}

pub(super) fn away() -> bool {
    std::env::var_os("VOXEL_GI_VALIDATE_SKYLIGHT_AWAY").is_some()
}

/// Eye and target facing the cave's -x/+z corner, the patch behind and right.
pub(super) const AWAY_VIEW: (Vec3, Vec3) =
    (Vec3::new(30.0, 22.0, 50.0), Vec3::new(5.0, 14.0, 58.0));

pub(super) const SUN_CAPTURES: &[(f32, &str)] =
    &[(14.5, "before"), (25.0, "sun-25s"), (40.0, "sun-40s")];

pub(super) fn sun_angle(elapsed: f32) -> Option<f32> {
    sun().map(|(rate, offset)| offset + rate * (elapsed - OPEN_AT).max(0.0))
}

pub(super) fn mode() -> Option<&'static str> {
    static MODE: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    *MODE.get_or_init(
        || match std::env::var("VOXEL_GI_VALIDATE_CAVE_SKYLIGHT").as_deref() {
            Ok("open") => Some("open"),
            Ok("pre") => Some("pre"),
            _ => None,
        },
    )
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
    darkened: bool,
}

impl SkylightRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        world: &mut VoxelWorld<PrototypeWorld>,
        dirty: &mut VoxelGiInput,
        materials: &mut ResMut<VoxelMaterialTransportTable>,
    ) {
        let Some(mode) = mode() else { return };
        if dark() && !self.darkened {
            self.darkened = true;
            for id in 0..=255u8 {
                let mut material = materials.get(id);
                if material.emission != Vec3::ZERO {
                    material.emission = Vec3::ZERO;
                    materials.set(id, material);
                }
            }
        }
        let fill = |world: &mut VoxelWorld<PrototypeWorld>,
                    dirty: &mut VoxelGiInput,
                    min: IVec3,
                    max: IVec3,
                    voxel| {
            for z in min.z..=max.z {
                for x in min.x..=max.x {
                    for y in min.y..=max.y {
                        dirty.set_voxel(world, IVec3::new(x, y, z), voxel);
                    }
                }
            }
        };
        if !self.floor {
            // A moving sun slides the patch toward -x (to x ~33 at -25 deg).
            let min = if sun().is_some() {
                IVec3::new(28, FLOOR_MIN.y, FLOOR_MIN.z)
            } else {
                FLOOR_MIN
            };
            fill(world, dirty, min, FLOOR_MAX, WorldVoxel::Solid(2));
            self.floor = true;
        }
        if !self.opened && (mode == "pre" || elapsed >= OPEN_AT) {
            fill(world, dirty, HOLE_MIN, HOLE_MAX, WorldVoxel::Air);
            self.opened = true;
        }
    }
}
