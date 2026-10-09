//! Three fixed views in the normal streamed world, not the isolated chamber.
//! Teleporting the camera exercises residency but does not test KCC locomotion.
use super::*;

#[derive(Default)]
pub(super) struct StructuresRun {
    initialized: bool,
}

impl StructuresRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        player: &mut Transform,
        commands: &mut Commands,
        cameras: &Query<Entity, With<Camera3d>>,
        transforms: &mut Query<&mut Transform, Without<Player>>,
    ) {
        if !self.initialized {
            for camera in cameras {
                commands
                    .entity(camera)
                    .remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>()
                    .insert(Projection::Perspective(PerspectiveProjection {
                        fov: 65.0_f32.to_radians(),
                        ..default()
                    }));
            }
            self.initialized = true;
        }
        let (eye, target) = match elapsed {
            t if t < 12.0 => (Vec3::new(6.0, 17.0, 5.0), Vec3::new(14.0, 9.0, -18.0)),
            t if t < 24.0 => (Vec3::new(18.0, 13.6, -18.0), Vec3::new(33.0, -2.0, -18.0)),
            _ => (Vec3::new(47.0, -11.4, -18.0), Vec3::new(76.0, -17.0, -34.0)),
        };
        // Keep near/far field diagnostics anchored to the view, unlike a
        // detached camera still measuring ages around the original spawn.
        player.translation = eye - Vec3::Y * 1.62;
        for camera in cameras {
            if let Ok(mut transform) = transforms.get_mut(camera) {
                *transform = Transform::from_translation(eye).looking_at(target, Vec3::Y);
            }
        }
    }
}
