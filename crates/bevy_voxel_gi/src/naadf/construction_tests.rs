//! Construction oracle. The full-sweep oracle freezes the pre-optimization
//! algorithm independently of the production loop.

use super::*;

fn full_sweep_reference(size: UVec3, blocked: &[bool]) -> Vec<Aadf> {
    assert_eq!(blocked.len(), volume_len(size));
    assert!(size.max_element() <= u8::MAX as u32);

    let mut source = vec![Aadf::default(); blocked.len()];
    let mut destination = source.clone();
    let rounds = size.max_element().saturating_sub(1);

    for _ in 0..rounds {
        for axis in 0..3 {
            // Every destination cell is assigned below, including blocked cells.
            // Copying the previous sweep first adds a redundant full-field pass.
            for z in 0..size.z {
                for y in 0..size.y {
                    for x in 0..size.x {
                        let position = UVec3::new(x, y, z);
                        let index = linear_index(position, size);
                        if blocked[index] {
                            destination[index] = Aadf::default();
                            continue;
                        }

                        let mut current = source[index];
                        let negative_direction = axis * 2;
                        let positive_direction = negative_direction + 1;
                        let axis_vector = match axis {
                            0 => IVec3::X,
                            1 => IVec3::Y,
                            _ => IVec3::Z,
                        };

                        reference_try_grow(
                            &mut current,
                            position.as_ivec3() - axis_vector,
                            negative_direction,
                            size,
                            blocked,
                            &source,
                        );
                        reference_try_grow(
                            &mut current,
                            position.as_ivec3() + axis_vector,
                            positive_direction,
                            size,
                            blocked,
                            &source,
                        );
                        destination[index] = current;
                    }
                }
            }
            std::mem::swap(&mut source, &mut destination);
        }
    }

    source
}

fn reference_try_grow(
    current: &mut Aadf,
    neighbour: IVec3,
    direction: usize,
    size: UVec3,
    blocked: &[bool],
    source: &[Aadf],
) {
    if neighbour.cmplt(IVec3::ZERO).any() || neighbour.cmpge(size.as_ivec3()).any() {
        return;
    }
    if current.bounds[direction] == u8::MAX {
        return;
    }

    let neighbour_index = linear_index(neighbour.as_uvec3(), size);
    if blocked[neighbour_index] {
        return;
    }

    let neighbour_aadf = source[neighbour_index];
    let opposite = direction ^ 1;
    for bound_direction in 0..6 {
        if bound_direction != opposite
            && neighbour_aadf.bounds[bound_direction] < current.bounds[bound_direction]
        {
            return;
        }
    }

    current.bounds[direction] += 1;
}

fn occupancy(size: UVec3, mut blocked: impl FnMut(UVec3) -> bool) -> Vec<bool> {
    (0..volume_len(size))
        .map(|index| {
            let index = index as u32;
            blocked(UVec3::new(
                index % size.x,
                (index / size.x) % size.y,
                index / (size.x * size.y),
            ))
        })
        .collect()
}

#[test]
fn construction_matches_full_sweep_oracle() {
    // Exhaust every 2³ topology, then larger seeded volumes, flat axes, uniform
    // fields and maximum-width bounds. Exact equality also preserves ray work.
    let size = UVec3::splat(2);
    for bits in 0..256_u32 {
        let blocked: Vec<_> = (0..8).map(|index| bits & (1 << index) != 0).collect();
        assert_eq!(
            build_aadf(size, &blocked),
            full_sweep_reference(size, &blocked)
        );
    }
    let mut random = 0x1234_5678_u32;
    for size in [
        UVec3::ZERO,
        UVec3::ONE,
        UVec3::splat(4),
        UVec3::new(9, 7, 8),
        UVec3::new(17, 3, 11),
        UVec3::new(255, 1, 1),
        UVec3::new(1, 255, 1),
        UVec3::new(1, 1, 255),
    ] {
        for divisor in [0, 1, 2, 7, 29] {
            let blocked = occupancy(size, |_| {
                random = random.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
                divisor != 0 && random.is_multiple_of(divisor)
            });
            assert_eq!(
                build_aadf(size, &blocked),
                full_sweep_reference(size, &blocked),
                "size={size:?}, divisor={divisor}"
            );
        }
    }
}
