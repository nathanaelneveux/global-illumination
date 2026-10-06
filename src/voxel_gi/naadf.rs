use bevy::prelude::*;

pub const NEG_X: usize = 0;
pub const POS_X: usize = 1;
pub const NEG_Y: usize = 2;
pub const POS_Y: usize = 3;
pub const NEG_Z: usize = 4;
pub const POS_Z: usize = 5;

/// Six directional extents of a guaranteed-empty axis-aligned box around a
/// cell. The current cell is not included in these distances.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Aadf {
    pub bounds: [u8; 6],
}

impl Aadf {
    pub fn negative(self) -> UVec3 {
        UVec3::new(
            self.bounds[NEG_X] as u32,
            self.bounds[NEG_Y] as u32,
            self.bounds[NEG_Z] as u32,
        )
    }

    pub fn positive(self) -> UVec3 {
        UVec3::new(
            self.bounds[POS_X] as u32,
            self.bounds[POS_Y] as u32,
            self.bounds[POS_Z] as u32,
        )
    }
}

/// Builds NAADF bounds with the same neighbor-merging rule used by the paper's
/// reference implementation and vxray. `blocked` cells receive zero bounds.
pub fn build_aadf(size: UVec3, blocked: &[bool]) -> Vec<Aadf> {
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

                        try_grow(
                            &mut current,
                            position.as_ivec3() - axis_vector,
                            negative_direction,
                            size,
                            blocked,
                            &source,
                        );
                        try_grow(
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

fn try_grow(
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

pub fn linear_index(position: UVec3, size: UVec3) -> usize {
    (position.x + size.x * (position.y + size.y * position.z)) as usize
}

pub fn volume_len(size: UVec3) -> usize {
    (size.x * size.y * size.z) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_generated_bound_contains_only_empty_cells() {
        let size = UVec3::new(9, 7, 8);
        let mut state = 0x1234_5678_u32;
        let mut blocked = vec![false; volume_len(size)];
        for value in &mut blocked {
            state = state.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
            *value = state.is_multiple_of(7);
        }

        let fields = build_aadf(size, &blocked);
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let position = UVec3::new(x, y, z);
                    let index = linear_index(position, size);
                    if blocked[index] {
                        assert_eq!(fields[index], Aadf::default());
                        continue;
                    }

                    let minimum = position - fields[index].negative();
                    let maximum = position + fields[index].positive();
                    for check_z in minimum.z..=maximum.z {
                        for check_y in minimum.y..=maximum.y {
                            for check_x in minimum.x..=maximum.x {
                                let check = UVec3::new(check_x, check_y, check_z);
                                assert!(!blocked[linear_index(check, size)]);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn empty_four_cubed_grid_expands_to_every_edge() {
        let size = UVec3::splat(4);
        let fields = build_aadf(size, &vec![false; volume_len(size)]);
        let at = fields[linear_index(UVec3::new(1, 2, 1), size)];
        assert_eq!(at.negative(), UVec3::new(1, 2, 1));
        assert_eq!(at.positive(), UVec3::new(2, 1, 2));
    }
}
