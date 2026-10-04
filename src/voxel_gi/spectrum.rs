use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};

use bevy::prelude::*;

/// Transport bands carried by the world-space solver.
///
/// Visible RGB is rendered. Infrared and ionizing energy are retained in the
/// same cache for gameplay queries, but deliberately have no visual mapping.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergySpectrum {
    pub visible: Vec3,
    pub infrared: f32,
    pub ionizing: f32,
}

impl EnergySpectrum {
    pub(crate) fn bounded(self, maximum: f32) -> Self {
        let value = self.finite_or_zero();
        Self::new(
            value.visible.min(Vec3::splat(maximum)),
            value.infrared.min(maximum),
            value.ionizing.min(maximum),
        )
    }
    pub const ZERO: Self = Self {
        visible: Vec3::ZERO,
        infrared: 0.0,
        ionizing: 0.0,
    };

    pub const ONE: Self = Self {
        visible: Vec3::ONE,
        infrared: 1.0,
        ionizing: 1.0,
    };

    pub const fn new(visible: Vec3, infrared: f32, ionizing: f32) -> Self {
        Self {
            visible,
            infrared,
            ionizing,
        }
    }

    pub fn max_component(self) -> f32 {
        self.visible
            .max_element()
            .max(self.infrared)
            .max(self.ionizing)
    }

    pub fn finite_or_zero(self) -> Self {
        Self {
            visible: if self.visible.is_finite() {
                self.visible.max(Vec3::ZERO)
            } else {
                Vec3::ZERO
            },
            infrared: if self.infrared.is_finite() {
                self.infrared.max(0.0)
            } else {
                0.0
            },
            ionizing: if self.ionizing.is_finite() {
                self.ionizing.max(0.0)
            } else {
                0.0
            },
        }
    }
}

impl Add for EnergySpectrum {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.visible + rhs.visible,
            self.infrared + rhs.infrared,
            self.ionizing + rhs.ionizing,
        )
    }
}

impl AddAssign for EnergySpectrum {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for EnergySpectrum {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.visible - rhs.visible,
            self.infrared - rhs.infrared,
            self.ionizing - rhs.ionizing,
        )
    }
}

impl Mul for EnergySpectrum {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(
            self.visible * rhs.visible,
            self.infrared * rhs.infrared,
            self.ionizing * rhs.ionizing,
        )
    }
}

impl Mul<f32> for EnergySpectrum {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.visible * rhs, self.infrared * rhs, self.ionizing * rhs)
    }
}

impl MulAssign<f32> for EnergySpectrum {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}

impl Div<f32> for EnergySpectrum {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.visible / rhs, self.infrared / rhs, self.ionizing / rhs)
    }
}

/// A source for non-visible bands, or a visible source that does not map to a
/// Bevy light. Point and directional Bevy lights are imported automatically.
#[derive(Component, Clone, Copy, Debug)]
pub struct EnergyEmitter {
    pub spectrum: EnergySpectrum,
    pub range: f32,
}

impl Default for EnergyEmitter {
    fn default() -> Self {
        Self {
            spectrum: EnergySpectrum::ZERO,
            range: 32.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrum_math_keeps_transport_bands_independent() {
        let a = EnergySpectrum::new(Vec3::new(1.0, 2.0, 3.0), 4.0, 5.0);
        let b = EnergySpectrum::new(Vec3::new(0.5, 0.25, 2.0), 3.0, 0.1);
        let product = a * b;

        assert_eq!(product.visible, Vec3::new(0.5, 0.5, 6.0));
        assert_eq!(product.infrared, 12.0);
        assert_eq!(product.ionizing, 0.5);
    }
}
