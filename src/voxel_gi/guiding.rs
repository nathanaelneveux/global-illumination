//! Source-informed directional proposals. These do not change the transport
//! model, retained history, or simulation energy. Every guided direction is
//! divided by its mixture PDF, also used by emission MIS. The cosine component
//! retains support when a source is hidden, removed, or an unhelpful guide.
//!
//! The spherical lobe uses the normalized Henyey–Greenstein distribution as a
//! sampling proposal, not a physical scattering phase function. Keep formulas
//! and source-selection bounds in sync with voxel_trace.wgsl.
use super::{
    emission::EmissionDistribution, sampling::hemisphere, tracer::TraceResult,
    world::TransportScene,
};
use bevy::prelude::*;

pub(crate) const RADIUS: f32 = 2.0;

fn weight(position: Vec3, voxel: IVec3, concentration: f32) -> f32 {
    let distance2 = (voxel.as_vec3() + Vec3::splat(0.5)).distance_squared(position);
    let focus = distance2 / (distance2 + 256.0 * RADIUS * RADIUS);
    0.5 * (6.0 * concentration).min(1.0) * focus
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GuideAnchor {
    position: Vec3,
    weight: f32,
}

impl GuideAnchor {
    pub fn select(
        scene: &TransportScene,
        emitters: &EmissionDistribution,
        position: Vec3,
        normal: Vec3,
    ) -> Self {
        let visible = |source: super::emission::GuideSource| {
            let voxel = source.voxel;
            if weight(position, voxel, source.concentration) < 0.05 {
                return None;
            }
            let delta = voxel.as_vec3() + Vec3::splat(0.5) - position;
            let distance = delta.length();
            Some(
                distance > 0.001
                    && matches!(scene.trace_ray_result(
                position + normal * 0.002, delta / distance, distance + 0.01),
                    TraceResult::Hit(hit) if hit.voxel == voxel),
            )
        };
        let Some(source) = emitters.guide_at(position, normal, visible) else {
            return Self::default();
        };
        let center = source.voxel.as_vec3() + Vec3::splat(0.5);
        let weight = weight(position, source.voxel, source.concentration);
        Self {
            position: center,
            weight: if weight < 0.05 { 0.0 } else { weight },
        }
    }

    pub fn at(self, position: Vec3) -> DirectionGuide {
        let delta = self.position - position;
        let distance = delta.length();
        if self.weight == 0.0 || distance <= 0.001 {
            return DirectionGuide::default();
        }
        DirectionGuide {
            axis: delta / distance,
            g: (distance / (distance + RADIUS)).min(0.999),
            weight: self.weight,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct DirectionGuide {
    axis: Vec3,
    g: f32,
    weight: f32,
}

impl DirectionGuide {
    pub fn pdf(self, normal: Vec3, direction: Vec3) -> f32 {
        let cosine = normal.dot(direction).max(0.0) / std::f32::consts::PI;
        if self.weight == 0.0 {
            return cosine;
        }
        let denominator = (1.0 - self.g).powi(2)
            + 2.0 * self.g * (1.0 - self.axis.dot(direction).clamp(-1.0, 1.0));
        let lobe = (1.0 - self.g * self.g)
            / (4.0 * std::f32::consts::PI * denominator * denominator.sqrt());
        (1.0 - self.weight) * cosine + self.weight * lobe
    }

    pub fn sample(self, normal: Vec3, u: f32, v: f32) -> Vec3 {
        if self.weight == 0.0 {
            return hemisphere(normal, u, v);
        }
        if u >= self.weight {
            return hemisphere(normal, (u - self.weight) / (1.0 - self.weight), v);
        }
        let ratio = (1.0 - self.g * self.g) / (1.0 - self.g + 2.0 * self.g * v);
        let cosine = if self.g < 0.001 {
            2.0 * v - 1.0
        } else {
            (1.0 - (ratio * ratio - (1.0 - self.g).powi(2)) / (2.0 * self.g)).clamp(-1.0, 1.0)
        };
        let radius = (1.0 - cosine * cosine).max(0.0).sqrt();
        let phi = std::f32::consts::TAU * u / self.weight;
        let tangent = self
            .axis
            .cross(if self.axis.z.abs() < 0.999 {
                Vec3::Z
            } else {
                Vec3::Y
            })
            .normalize();
        tangent * (phi.cos() * radius)
            + self.axis.cross(tangent) * (phi.sin() * radius)
            + self.axis * cosine
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixture_preserves_constant_hemisphere_integral_and_support() {
        for axis in [Vec3::Y, Vec3::X, -Vec3::Y] {
            for g in [0.0, 0.5, 0.95, 0.999] {
                let guide = DirectionGuide {
                    axis,
                    g,
                    weight: 0.5,
                };
                let mut integral = 0.0_f64;
                for i in 0..512 {
                    for j in 0..512 {
                        let direction = guide.sample(
                            Vec3::Y,
                            (i as f32 + 0.5) / 512.0,
                            (j as f32 + 0.5) / 512.0,
                        );
                        assert!((direction.length_squared() - 1.0).abs() < 1e-5);
                        let pdf = guide.pdf(Vec3::Y, direction);
                        assert!(pdf.is_finite() && pdf > 0.0);
                        let cosine = direction.y.max(0.0) / std::f32::consts::PI;
                        assert!(pdf >= 0.5 * cosine);
                        integral += f64::from(cosine / pdf) / (512.0 * 512.0);
                    }
                }
                assert!(
                    (integral - 1.0).abs() < 0.003,
                    "axis={axis:?} g={g} integral={integral}"
                );
            }
        }
    }
}
