//! The composed cave's validation viewpoint.
use super::*;

/// The calibrated composed-cave viewpoint is deliberately different from the
/// isolated chamber camera. Share it with off-camera diagnostic edit guards.
pub(super) fn view() -> (Vec3, Vec3) {
    if crate::cave_scene::enabled() {
        (crate::cave_scene::EYE, crate::cave_scene::TARGET)
    } else {
        (Vec3::new(47.0, -11.4, -18.0), Vec3::new(76.0, -17.0, -34.0))
    }
}
