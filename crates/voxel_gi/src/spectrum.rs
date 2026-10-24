use bevy::prelude::*;

/// Sanitizing helpers for linear RGB transport energy.
pub(crate) trait Radiance {
    /// Non-finite or negative channels become zero.
    fn finite_or_zero(self) -> Self;
    /// `finite_or_zero`, then each channel capped at `maximum`.
    fn bounded(self, maximum: f32) -> Self;
}

impl Radiance for Vec3 {
    fn finite_or_zero(self) -> Self {
        if self.is_finite() { self.max(Vec3::ZERO) } else { Vec3::ZERO }
    }
    fn bounded(self, maximum: f32) -> Self {
        self.finite_or_zero().min(Vec3::splat(maximum))
    }
}

/// A visible source that does not map to a Bevy light. Point and directional
/// Bevy lights are imported automatically.
#[derive(Component, Clone, Copy, Debug)]
pub struct EnergyEmitter {
    pub spectrum: Vec3,
    pub range: f32,
}

impl Default for EnergyEmitter {
    fn default() -> Self {
        Self {
            spectrum: Vec3::ZERO,
            range: 32.0,
        }
    }
}
