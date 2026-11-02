//! Linear RGB sanitizing shared by sources, materials and GPU decoding.
use bevy::prelude::*;

/// Sanitizing helpers for linear RGB transport energy.
pub(crate) trait FiniteColor {
    /// Non-finite or negative channels become zero.
    fn finite_or_zero(self) -> Self;
    /// `finite_or_zero`, then each channel capped at `maximum`.
    fn bounded(self, maximum: f32) -> Self;
}

impl FiniteColor for Vec3 {
    fn finite_or_zero(self) -> Self {
        if self.is_finite() {
            self.max(Vec3::ZERO)
        } else {
            Vec3::ZERO
        }
    }
    fn bounded(self, maximum: f32) -> Self {
        self.finite_or_zero().min(Vec3::splat(maximum))
    }
}
