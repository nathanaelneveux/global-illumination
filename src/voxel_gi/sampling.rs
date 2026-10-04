//! Stable world-space sample streams. History resets never repeat a sequence.
use bevy::prelude::*;

pub(crate) fn pcg(value: u32) -> u32 {
    let state = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

pub(crate) fn stream_id(voxel: IVec3, face: u32) -> u32 {
    pcg(pcg(pcg(voxel.x as u32) ^ voxel.y as u32) ^ voxel.z as u32) ^ pcg(face.wrapping_add(1))
}

/// Independently rotated Halton dimensions; the rotation is stable per face.
pub(crate) fn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {
    const PRIMES: [u32; 16] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53];
    let base = PRIMES[dimension as usize % PRIMES.len()];
    let mut index = (sequence & 0x00ff_ffff) + 1;
    let mut scale = 1.0;
    let mut value = 0.0;
    while index != 0 {
        scale /= base as f32;
        value += (index % base) as f32 * scale;
        index /= base;
    }
    let rotation = (pcg(stream ^ pcg(dimension)) >> 9) as f32 / (1_u32 << 23) as f32;
    (value + rotation).fract()
}

pub(crate) fn hemisphere(normal: Vec3, u: f32, v: f32) -> Vec3 {
    let phi = std::f32::consts::TAU * u;
    let radius = v.sqrt();
    let tangent = if normal.z.abs() < 0.999 {
        normal.cross(Vec3::Z).normalize()
    } else {
        normal.cross(Vec3::Y).normalize()
    };
    let bitangent = normal.cross(tangent);
    tangent * (phi.cos() * radius) + bitangent * (phi.sin() * radius) + normal * (1.0 - v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_has_uniform_coverage_and_does_not_repeat_at_history_limit() {
        let stream = stream_id(IVec3::new(-19, 7, 12), 2);
        let values: Vec<_> = (0..4096).map(|i| sample_1d(stream, i, 0)).collect();
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
        assert!((values.iter().sum::<f32>() / values.len() as f32 - 0.5).abs() < 0.001);
        assert_ne!(values[3], values[7]);
        assert_ne!(values[255], values[511]);
    }

    #[test]
    fn hemisphere_samples_are_normalized_and_cosine_weighted() {
        for normal in [Vec3::X, Vec3::NEG_Y, Vec3::Z] {
            let mut mean = 0.0;
            for i in 0..4096 {
                let ray = hemisphere(normal, sample_1d(21, i, 0), sample_1d(21, i, 1));
                assert!((ray.length() - 1.0).abs() < 1e-5);
                assert!(ray.dot(normal) >= 0.0);
                mean += ray.dot(normal);
            }
            assert!((mean / 4096.0 - 2.0 / 3.0).abs() < 0.002);
        }
    }
}
