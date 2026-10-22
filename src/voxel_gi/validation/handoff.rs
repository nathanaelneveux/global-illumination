//! Camera-only cache admission workload in the ordinary single-emitter hallway.
//! With a 32-voxel cache, the source/end-wall page is outside residency at the
//! far eye and enters at 30 s. It stays loaded and visible throughout. Returning
//! to the far eye at 40 s leaves it inside the normal retirement hysteresis.
use super::*;

pub(super) fn enabled(profile: Profile) -> bool {
    profile == Profile::Hallway && std::env::var_os("VOXEL_GI_VALIDATE_CACHE_HANDOFF").is_some()
}

pub(super) const CAPTURES: &[(f32, &str)] = &[
    (29., "far-before"),
    (30.1, "admit-0100ms"),
    (30.4, "admit-0400ms"),
    (31., "admit-1000ms"),
    (32., "admit-2000ms"),
    (39., "near-later"),
    (40.4, "return-0400ms"),
    (59., "far-later"),
];

pub(super) fn tick(
    elapsed: f32,
    player: &mut Transform,
    cameras: &Query<Entity, With<Camera3d>>,
    transforms: &mut Query<&mut Transform, Without<Player>>,
) {
    let x = if (30.0..40.0).contains(&elapsed) {
        -85.0
    } else {
        -55.0
    };
    let eye = Vec3::new(x, -8.5, -19.5);
    player.translation = eye - Vec3::Y * 1.62;
    for camera in cameras {
        if let Ok(mut transform) = transforms.get_mut(camera) {
            *transform = Transform::from_translation(eye)
                .looking_at(Vec3::new(-127.0, -8.5, -19.5), Vec3::Y);
        }
    }
}
