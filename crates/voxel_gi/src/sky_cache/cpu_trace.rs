//! Test-only CPU corner visibility, the reference the sky cache tests fill
//! corners with. Same rule as the GPU's display shadows (`sky_corner_visibility`):
//! only loaded geometry occludes; unloaded regions are skipped.
use super::super::world::{AIR, BRICK_EDGE, BRICKS_PER_AXIS, REGION_EDGE, TransportScene};
use super::SkyLight;
use crate::tracer::MAX_PATH_DISTANCE;
use bevy::prelude::*;

const RAY_EPSILON: f32 = 0.001;
const MAX_TRACE_STEPS: usize = 2_048;

enum Visibility {
    Escaped,
    Blocked,
    Unknown,
}

/// Which lights a corner sees: unknown (0) if traversal does not finish.
pub(super) fn sample_sky_corner(scene: &TransportScene, lights: &[SkyLight], position: IVec3) -> u32 {
    let mut word = super::super::sky::KNOWN;
    for (index, light) in lights.iter().enumerate() {
        // One canonical ray origin for every receiver sharing this vertex.
        let origin = position.as_vec3() + light.direction * (RAY_EPSILON * 2.0);
        match visibility(scene, origin, light.direction, MAX_PATH_DISTANCE) {
            Visibility::Escaped => word |= 1 << index,
            Visibility::Blocked => (),
            Visibility::Unknown => return 0,
        }
    }
    word
}

fn visibility(scene: &TransportScene, origin: Vec3, direction: Vec3, max_distance: f32) -> Visibility {
    let direction = direction.normalize_or_zero();
    if direction == Vec3::ZERO {
        return Visibility::Escaped;
    }
    let Some((minimum, maximum)) = scene.world_bounds() else {
        return Visibility::Escaped;
    };
    let Some((entry, _)) = ray_box_interval(origin, direction, minimum, maximum) else {
        return Visibility::Escaped;
    };
    let mut distance = entry.max(0.0) + RAY_EPSILON;
    let mut position = origin + direction * distance;
    for _ in 0..MAX_TRACE_STEPS {
        if distance > max_distance {
            return Visibility::Escaped;
        }
        let voxel = position.floor().as_ivec3();
        let coordinate = voxel.div_euclid(IVec3::splat(REGION_EDGE as i32));
        if !scene.top_level.contains(coordinate) {
            return Visibility::Escaped;
        }
        let (cell, negative, positive, size) = match scene.regions.get(&coordinate) {
            None => (coordinate, IVec3::ZERO, IVec3::ZERO, REGION_EDGE as f32),
            Some(region) if !region.any_solid => {
                let field = scene.top_level.field(coordinate);
                (coordinate, field.negative().as_ivec3(), field.positive().as_ivec3(), REGION_EDGE as f32)
            }
            Some(region) => {
                let local = (voxel - coordinate * REGION_EDGE as i32).as_uvec3();
                let brick = local / BRICK_EDGE;
                if !region.brick_occupied(brick) {
                    let field = region.brick_aadf(brick);
                    let global = coordinate * BRICKS_PER_AXIS as i32 + brick.as_ivec3();
                    (global, field.negative().as_ivec3(), field.positive().as_ivec3(), BRICK_EDGE as f32)
                } else if region.material(local) != AIR {
                    return Visibility::Blocked;
                } else {
                    let field = region.voxel_aadf(local);
                    (voxel, field.negative().as_ivec3(), field.positive().as_ivec3(), 1.0)
                }
            }
        };
        let travel = advance_empty_box(position, direction, cell, negative, positive, size);
        position += direction * travel;
        distance += travel;
    }
    Visibility::Unknown
}

fn advance_empty_box(position: Vec3, direction: Vec3, cell: IVec3, negative: IVec3, positive: IVec3, size: f32) -> f32 {
    let minimum = (cell - negative).as_vec3() * size;
    let maximum = (cell + positive + IVec3::ONE).as_vec3() * size;
    let mut best = f32::INFINITY;
    for axis in 0..3 {
        let component = direction[axis];
        if component.abs() <= 1e-8 {
            continue;
        }
        let plane = if component > 0.0 { maximum[axis] } else { minimum[axis] };
        let distance = (plane - position[axis]) / component;
        if distance >= -RAY_EPSILON && distance < best {
            best = distance.max(0.0);
        }
    }
    (best + RAY_EPSILON).max(RAY_EPSILON)
}

fn ray_box_interval(origin: Vec3, direction: Vec3, minimum: Vec3, maximum: Vec3) -> Option<(f32, f32)> {
    let mut entry = f32::NEG_INFINITY;
    let mut exit = f32::INFINITY;
    for axis in 0..3 {
        if direction[axis].abs() <= 1e-8 {
            if origin[axis] < minimum[axis] || origin[axis] > maximum[axis] {
                return None;
            }
            continue;
        }
        let inverse = direction[axis].recip();
        let first = (minimum[axis] - origin[axis]) * inverse;
        let second = (maximum[axis] - origin[axis]) * inverse;
        entry = entry.max(first.min(second));
        exit = exit.min(first.max(second));
    }
    (exit >= entry.max(0.0)).then_some((entry, exit))
}
