//! A punishing far-LOD workload modeled on `bevy_voxel_world`'s
//! `noise_terrain_lod` example: a large-scale noise heightfield streamed 200
//! chunks out with distance LOD strides 1-32 (the farthest chunks hold one
//! sample per axis), seen from a high, drifting, turning camera.
//! `VOXEL_GI_SCENE=lod`, or the `lod` validation profile.
use crate::terrain::TerrainMaterial;
use bevy::prelude::*;
use bevy_voxel_world::prelude::{LodLevel, WorldVoxel};

pub const SPAWNING_DISTANCE: u32 = 200;
/// The example's camera: start pose and drift (voxels and radians per second).
pub const EYE: Vec3 = Vec3::new(-200.0, 180.0, -200.0);
pub const TARGET: Vec3 = Vec3::new(0.0, 60.0, 0.0);
pub const DRIFT: Vec3 = Vec3::new(12.0, 0.0, 24.0);
pub const TURN_RATE: f32 = 0.1;

/// The example's bands: stride (voxels per sample) by chunk distance.
pub fn chunk_lod(chunk_position: IVec3, camera_position: Vec3) -> LodLevel {
    let camera_chunk = (camera_position / 32.0).floor();
    let distance = chunk_position.as_vec3().distance(camera_chunk);
    match distance {
        d if d < 10.0 => 1,
        d if d < 25.0 => 2,
        d if d < 50.0 => 4,
        d if d < 85.0 => 8,
        d if d < 125.0 => 16,
        _ => 32,
    }
}

/// Samples per chunk axis at a stride.
pub fn samples(lod: LodLevel) -> u32 {
    32 / u32::from(lod.max(1))
}

pub fn voxel(position: IVec3) -> WorldVoxel<u8> {
    if position.y < 0 {
        return TerrainMaterial::Brick.voxel();
    }
    if (position.y as f64) < height(position.x, position.z) {
        TerrainMaterial::Grass.voxel()
    } else {
        WorldVoxel::Air
    }
}

/// The example's fractal parameters (5 octaves, frequency 1.1, lacunarity
/// 2.8, persistence 0.4) over x/1000 and z/1000, scaled by 50. Plain fBm over
/// gradient noise rather than the `noise` crate's HybridMulti.
fn height(x: i32, z: i32) -> f64 {
    let (mut sum, mut amplitude, mut frequency) = (0.0, 1.0, 1.1);
    for octave in 0..5u32 {
        sum += amplitude * gradient_noise(x as f64 / 1000.0 * frequency, z as f64 / 1000.0 * frequency, octave);
        amplitude *= 0.4;
        frequency *= 2.8;
    }
    sum * 50.0
}

fn gradient_noise(x: f64, z: f64, seed: u32) -> f64 {
    let (x0, z0) = (x.floor(), z.floor());
    let (fx, fz) = (x - x0, z - z0);
    let corner = |dx: f64, dz: f64| {
        let mut hash = (x0 as i64 + dx as i64) as u32 ^ ((z0 as i64 + dz as i64) as u32).wrapping_mul(0x9E37_79B9);
        hash = (hash ^ seed.wrapping_mul(0x85EB_CA6B)).wrapping_mul(0x27D4_EB2D);
        hash ^= hash >> 15;
        let angle = f64::from(hash) / f64::from(u32::MAX) * std::f64::consts::TAU;
        angle.cos() * (fx - dx) + angle.sin() * (fz - dz)
    };
    let fade = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(fx), fade(fz));
    let a = corner(0.0, 0.0) + u * (corner(1.0, 0.0) - corner(0.0, 0.0));
    let b = corner(0.0, 1.0) + u * (corner(1.0, 1.0) - corner(0.0, 1.0));
    (a + v * (b - a)) * std::f64::consts::SQRT_2
}

/// The camera at `elapsed` seconds into the example's flight.
pub fn camera(elapsed: f32) -> Transform {
    let start = Transform::from_translation(EYE).looking_at(TARGET, Vec3::Y);
    Transform {
        translation: EYE + DRIFT * elapsed,
        rotation: Quat::from_rotation_y(TURN_RATE * elapsed) * start.rotation,
        ..start
    }
}
