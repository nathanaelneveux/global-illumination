//! Bounded, nonrecursive reconstruction on connected coplanar voxel surfaces.
//! Transport statistics and simulation remain untouched. The GPU implements
//! `evidence` in voxel_accumulate.wgsl; the CPU oracle qualifies captures.
use bevy::prelude::*;

pub(super) mod evidence;

pub(super) fn tangent_axes(face: usize) -> (IVec3, IVec3) {
    match face / 2 {
        0 => (IVec3::Y, IVec3::Z),
        1 => (IVec3::X, IVec3::Z),
        _ => (IVec3::X, IVec3::Y),
    }
}

#[cfg(any(test, debug_assertions))]
#[derive(Clone, Copy)]
pub(super) struct Sample {
    pub mean: Vec3,
    pub second: Vec3,
    pub count: u32,
    pub display: Vec3,
}

#[cfg(any(test, debug_assertions))]
impl Sample {
    fn error(self) -> Vec3 {
        ((self.second - self.mean * self.mean).max(Vec3::ZERO) / self.count.max(1) as f32)
            .map(f32::sqrt)
    }
}
