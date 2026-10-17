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

/// Preserve rotated Halton path and branch dimensions; use independently
/// padded, hash-Owen-scrambled Sobol pairs for emitter selection and surface
/// coordinates. All streams continue independently of measurement resets.
pub(crate) fn sample_1d(stream: u32, sequence: u32, dimension: u32) -> f32 {
    if dimension < 12 {
        const PRIMES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        let base = PRIMES[dimension as usize];
        let mut index = (sequence & 0x00ff_ffff) + 1;
        let mut scale = 1.0;
        let mut value = 0.0;
        while index != 0 {
            scale /= base as f32;
            value += (index % base) as f32 * scale;
            index /= base;
        }
        let rotation = (pcg(stream ^ pcg(dimension)) >> 9) as f32 / 8_388_608.0;
        return (value + rotation).fract();
    }
    let seed = pcg(stream ^ pcg(dimension / 2));
    let mut index =
        scramble_reverse(sequence.reverse_bits(), pcg(seed ^ 0x79c6_8e4a)).reverse_bits();
    if dimension & 1 != 0 {
        // Pascal matrix over GF(2), the second Sobol dimension. These butterfly
        // stages equal XORing the 32 direction vectors, without a lookup table.
        for (shift, mask) in [
            (1, 0x5555_5555),
            (2, 0x3333_3333),
            (4, 0x0f0f_0f0f),
            (8, 0x00ff_00ff),
            (16, 0x0000_ffff),
        ] {
            index ^= (index >> shift) & mask;
        }
    }
    let value = scramble_reverse(index, pcg(seed ^ pcg(dimension))).reverse_bits();
    (value >> 9) as f32 / 8_388_608.0
}

// Hash-based reverse-bit Owen scramble adapted from Nathan Vegdahl's
// sobol_burley (MIT); see licenses/sobol_burley-MIT.txt.
fn scramble_reverse(mut value: u32, seed: u32) -> u32 {
    value ^= value.wrapping_mul(0x3d20_adea);
    value = value.wrapping_add(seed);
    value = value.wrapping_mul((seed >> 16) | 1);
    value ^= value.wrapping_mul(0x0552_6c56);
    value ^= value.wrapping_mul(0x53a2_2864);
    value
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
    fn path_and_conditional_emitter_dimensions_have_joint_support() {
        // Marginal uniformity is not sufficient: a light sample is evaluated
        // at a position determined by earlier direction dimensions. Each pair
        // must cover the product domain, including across bounce substreams.
        let stream = stream_id(IVec3::new(-54, -12, -20), 2);
        let n = 65536;
        for bounce in 1..=3u32 {
            let light_stream = stream ^ pcg(bounce.wrapping_add(0xa341_316c));
            for direction in 0..(2 * bounce).min(6) {
                for light in 12..=14 {
                    let mut bins = [0u32; 256];
                    for sequence in 0..n {
                        let x = (sample_1d(stream, sequence, direction) * 16.0) as usize;
                        let y = (sample_1d(light_stream, sequence, light) * 16.0) as usize;
                        bins[y * 16 + x] += 1;
                    }
                    let expected = f64::from(n) / 256.0;
                    let discrepancy: f64 = bins
                        .iter()
                        .map(|&count| (f64::from(count) - expected).powi(2) / expected)
                        .sum();
                    eprintln!(
                        "JOINT bounce={bounce} direction={direction} light={light} empty={} discrepancy/n={}",
                        bins.iter().filter(|&&n| n == 0).count(),
                        discrepancy / f64::from(n)
                    );
                    assert!(
                        discrepancy / f64::from(n) < 0.05,
                        "conditional dimensions lack joint coverage: bounce={bounce} direction={direction} light={light}"
                    );
                }
            }
        }
    }

    #[test]
    fn sequence_has_uniform_coverage_and_does_not_repeat_at_history_limit() {
        let stream = stream_id(IVec3::new(-19, 7, 12), 2);
        let values: Vec<_> = (0..4096).map(|i| sample_1d(stream, i, 0)).collect();
        assert!(values.iter().all(|v| (0.0..1.0).contains(v)));
        assert!((values.iter().sum::<f32>() / values.len() as f32 - 0.5).abs() < 0.001);
        assert_ne!(values[3], values[7]);
        assert_ne!(values[255], values[511]);
        assert_ne!(sample_1d(stream, 65_535, 0), sample_1d(stream, 65_536, 0));
    }

    #[test]
    fn pairs_are_dyadically_stratified_in_aligned_sequence_blocks() {
        for stream in [0, 123, stream_id(IVec3::new(-19, 7, 12), 2)] {
            for dimension in 0..16 {
                let mean = (0..4096)
                    .map(|i| {
                        let value = sample_1d(stream, i, dimension);
                        assert!((0.0..1.0).contains(&value));
                        value
                    })
                    .sum::<f32>()
                    / 4096.0;
                assert!((mean - 0.5).abs() < 0.004, "dimension {dimension}: {mean}");
            }
            for dimension in [12, 14] {
                for start in [0, 64, 4096, 65536, 16_777_216] {
                    for x_bits in 0..=6 {
                        let width = 1 << x_bits;
                        let height = 64 / width;
                        let mut counts = [0; 64];
                        for index in start..start + 64 {
                            let x = (sample_1d(stream, index, dimension) * width as f32) as usize;
                            let y =
                                (sample_1d(stream, index, dimension + 1) * height as f32) as usize;
                            counts[y * width + x] += 1;
                        }
                        assert!(
                            counts.into_iter().all(|count| count == 1),
                            "pair {dimension}, start {start}, grid {width}x{height}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn path_dimensions_preserve_the_previous_halton_sequence() {
        for stream in [0, 123, stream_id(IVec3::new(-19, 7, 12), 2)] {
            for (dimension, base) in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
                .into_iter()
                .enumerate()
            {
                for sequence in [0, 3, 7, 64, 2047, 65536, 0x00ff_ffff] {
                    let mut index = (sequence & 0x00ff_ffff) + 1;
                    let mut scale = 1.0;
                    let mut value = 0.0;
                    while index != 0 {
                        scale /= base as f32;
                        value += (index % base) as f32 * scale;
                        index /= base;
                    }
                    let rotation = (pcg(stream ^ pcg(dimension as u32)) >> 9) as f32 / 8_388_608.0;
                    assert_eq!(
                        sample_1d(stream, sequence, dimension as u32),
                        (value + rotation).fract()
                    );
                }
            }
        }
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
