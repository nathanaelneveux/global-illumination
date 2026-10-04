//! CPU oracle for the shader's virtual voxel-corner reconstruction.
//! Kept independent of simulation estimates and of mesh tessellation.
#[cfg(test)]
mod tests {
    fn corner(center: f32, u: Option<f32>, v: Option<f32>, diagonal: Option<f32>) -> f32 {
        let mut sum = center;
        let mut count = 1.0;
        for value in [
            u,
            v,
            if u.is_some() || v.is_some() {
                diagonal
            } else {
                None
            },
        ]
        .into_iter()
        .flatten()
        {
            sum += value;
            count += 1.0;
        }
        sum / count
    }

    #[test]
    fn coplanar_neighbors_agree_at_shared_corners() {
        for mask in 0_u32..16 {
            // A 2x2 corner patch. Every connected component has one shared
            // corner value regardless of the face from which it is evaluated.
            let values = [1.0, 3.0, 5.0, 7.0];
            for a in 0..4 {
                if mask & (1 << a) == 0 {
                    continue;
                }
                let value = |i: usize| (mask & (1_u32 << i) != 0).then_some(values[i]);
                let ca = corner(values[a], value(a ^ 1), value(a ^ 2), value(a ^ 3));
                for b in [a ^ 1, a ^ 2] {
                    if mask & (1 << b) != 0 {
                        let cb = corner(values[b], value(b ^ 1), value(b ^ 2), value(b ^ 3));
                        assert_eq!(ca, cb, "mask {mask}");
                    }
                }
            }
        }
    }

    #[test]
    fn diagonal_light_does_not_cross_occluded_edges() {
        assert_eq!(corner(0.0, None, None, Some(100.0)), 0.0);
    }

    #[test]
    fn constant_field_is_preserved_at_missing_boundaries() {
        for mask in 0..8 {
            let value = |bit| (mask & bit != 0).then_some(2.5);
            assert_eq!(corner(2.5, value(1), value(2), value(4)), 2.5);
        }
    }
}
