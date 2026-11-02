//! CPU oracle for the shader's virtual voxel-corner reconstruction.
//! Kept independent of raw transport estimates and mesh tessellation.
#[cfg(test)]
mod tests {
    use bevy::prelude::*;
    use std::collections::{HashMap, HashSet};

    /// Count the sites required by moving the existing oriented hemisphere
    /// estimator to corners, preserving this reconstruction's connectivity.
    /// This is NOT a ray-throughput benchmark or a full-sphere estimator test.
    fn transport_sites(faces: &[(IVec3, usize)]) -> (usize, usize) {
        let mut corners = HashMap::<(IVec3, usize), Vec<IVec3>>::new();
        for &(voxel, face) in faces {
            for corner in 0..4 {
                corners
                    .entry((
                        super::super::sky::corner_position(voxel, face, corner),
                        face,
                    ))
                    .or_default()
                    .push(voxel);
            }
        }
        let positions: HashSet<_> = corners.keys().map(|(position, _)| *position).collect();
        let mut oriented_components = 0;
        for mut neighbors in corners.into_values() {
            while let Some(first) = neighbors.pop() {
                oriented_components += 1;
                let mut pending = vec![first];
                while let Some(voxel) = pending.pop() {
                    for index in (0..neighbors.len()).rev() {
                        if (neighbors[index] - voxel).abs().element_sum() == 1 {
                            pending.push(neighbors.swap_remove(index));
                        }
                    }
                }
            }
        }
        (positions.len(), oriented_components)
    }

    #[test]
    fn corner_transport_topology_cost_preserves_normals_and_disconnected_sides() {
        let plane: Vec<_> = (0..32)
            .flat_map(|z| (0..32).map(move |x| (IVec3::new(x, 0, z), 2)))
            .collect();
        assert_eq!(transport_sites(&plane), (1089, 1089));
        let cube: Vec<_> = (0..6).map(|face| (IVec3::ZERO, face)).collect();
        assert_eq!(transport_sites(&cube), (8, 24));
        let diagonal = [(IVec3::ZERO, 2), (IVec3::new(1, 0, 1), 2)];
        assert_eq!(
            transport_sites(&diagonal),
            (7, 8),
            "coincident corners on disconnected faces cannot share a display/transport state"
        );
    }

    fn weighted_corner(
        center: (f32, f32),
        u: Option<(f32, f32)>,
        v: Option<(f32, f32)>,
        diagonal: Option<(f32, f32)>,
    ) -> f32 {
        let mut sum = center.0 * center.1;
        let mut weight = center.1;
        for (value, w) in [
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
            sum += value * w;
            weight += w;
        }
        sum / weight.max(0.000001)
    }

    #[test]
    fn confidence_weighting_preserves_constants_and_reduces_unmeasured_neighbors() {
        assert_eq!(
            weighted_corner((2.5, 0.03125), Some((2.5, 0.125)), None, None),
            2.5
        );
        assert_eq!(weighted_corner((2.5, 0.03125), None, None, None), 2.5);
        let a = weighted_corner((0.0, 0.03125), Some((8.0, 1.0)), None, None);
        let b = weighted_corner((8.0, 1.0), Some((0.0, 0.03125)), None, None);
        assert_eq!(a, b, "shared corners agree even with different confidence");
        assert!(
            a > 7.7,
            "an unmeasured face should not halve measured lighting"
        );
        assert_eq!(
            weighted_corner((0.0, 0.125), None, None, Some((100.0, 1.0))),
            0.0
        );
    }

    fn corner(center: f32, u: Option<f32>, v: Option<f32>, diagonal: Option<f32>) -> f32 {
        weighted_corner(
            (center, 1.0),
            u.map(|value| (value, 1.0)),
            v.map(|value| (value, 1.0)),
            diagonal.map(|value| (value, 1.0)),
        )
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
                let weighted =
                    |i: usize| value(i).map(|value| (value, [0.03125, 0.125, 0.5, 1.0][i]));
                let wa = weighted_corner(
                    weighted(a).unwrap(),
                    weighted(a ^ 1),
                    weighted(a ^ 2),
                    weighted(a ^ 3),
                );
                for b in [a ^ 1, a ^ 2] {
                    if mask & (1 << b) != 0 {
                        let cb = corner(values[b], value(b ^ 1), value(b ^ 2), value(b ^ 3));
                        assert_eq!(ca, cb, "mask {mask}");
                        let wb = weighted_corner(
                            weighted(b).unwrap(),
                            weighted(b ^ 1),
                            weighted(b ^ 2),
                            weighted(b ^ 3),
                        );
                        assert!((wa - wb).abs() < 1e-6, "weighted corner seam: mask {mask}");
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
