//! Repeated one-wide, two-high digging from daylight into a hillside. The
//! renderer and asynchronous mesh/transport publication remain unmodified.
use super::*;
use crate::voxel_gi::BvwVoxelEdits;

#[derive(Default)]
pub(super) struct TunnelRun {
    initialized: bool,
    blocks: usize,
    pub events: Vec<serde_json::Value>,
}

impl TunnelRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        world: &mut VoxelWorld<PrototypeWorld>,
        dirty: &mut VoxelGiInput,
        commands: &mut Commands,
        cameras: &Query<Entity, With<Camera3d>>,
        transforms: &mut Query<&mut Transform, Without<Player>>,
    ) {
        if !self.initialized {
            // A sloped, closed hill above ordinary terrain. No artificial
            // sources: the tunnel progressively loses its view of daylight.
            for z in -34..=-3 {
                for y in 38..=51 {
                    for x in -5i32..=5 {
                        let top = 42 + (-8 - z).max(0) / 4 - x.abs() / 3;
                        let solid = y <= 39 || (z <= -8 && y <= top);
                        dirty.set_voxel(world, IVec3::new(x, y, z),
                            if solid { WorldVoxel::Solid(2) } else { WorldVoxel::Air });
                    }
                }
            }
            for camera in cameras {
                commands.entity(camera).remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>();
            }
            self.initialized = true;
        }
        while self.blocks < 40 {
            let depth = self.blocks / 2;
            let lower = self.blocks % 2 == 1;
            let at = 15.0 + depth as f32 * 2.0 + if lower { 0.25 } else { 0.0 };
            if elapsed < at { break; }
            let voxel = IVec3::new(0, if lower { 40 } else { 41 }, -8 - depth as i32);
            let solid = matches!(world.get_voxel(voxel), WorldVoxel::Solid(_));
            dirty.set_voxel(world, voxel, WorldVoxel::Air);
            self.events.push(json!({"depth": depth + 1, "voxel": voxel.to_array(),
                "scheduled_s": at, "at_s": elapsed, "was_solid": solid}));
            self.blocks += 1;
        }
        let depth = ((elapsed - 15.0).max(0.0) / 2.0).floor().min(19.0);
        let eye = Vec3::new(0.5, 41.4, -6.5 - depth);
        for camera in cameras {
            if let Ok(mut transform) = transforms.get_mut(camera) {
                *transform = Transform::from_translation(eye)
                    .looking_at(eye + Vec3::new(0.0, -0.25, -2.0), Vec3::Y);
            }
        }
    }
}
