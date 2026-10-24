//! Receiver-conditioned binary light hierarchy. The approximate importance
//! changes only the proposal, never visibility or energy. A power mixture at
//! every branch retains support for every emitting face in all five bands.
use super::*;

#[derive(Clone, Copy, Default)]
pub(crate) struct LightNode {
    pub minimum: Vec3,
    pub power: f32,
    pub maximum: Vec3,
    pub normals: u32,
    pub pivot: IVec3,
    pub pivot_face: u32,
    pub left: u32,
    pub right: u32,
    pub axis: u32,
}

fn key(voxel: IVec3, face: u32, axis: u32) -> (i32, i32, i32, u32) {
    (
        voxel[axis as usize],
        voxel[(axis as usize + 1) % 3],
        voxel[(axis as usize + 2) % 3],
        face,
    )
}

pub(super) fn build(faces: &[EmissiveFace]) -> Vec<LightNode> {
    fn split(nodes: &mut Vec<LightNode>, indices: &mut [usize], faces: &[EmissiveFace]) -> u32 {
        let index = nodes.len() as u32;
        let mut node = LightNode {
            minimum: Vec3::splat(f32::INFINITY),
            maximum: Vec3::splat(f32::NEG_INFINITY),
            ..default()
        };
        for &i in indices.iter() {
            let face = faces[i];
            let center = face_point(face, 0.5, 0.5);
            // Include the whole emitting square, not only its center.
            let extent = (Vec3::ONE - FACE_OFFSETS[face.face as usize].as_vec3().abs()) * 0.5;
            node.minimum = node.minimum.min(center - extent);
            node.maximum = node.maximum.max(center + extent);
            node.power += face.probability;
            node.normals |= 1 << face.face;
        }
        nodes.push(node);
        if indices.len() == 1 {
            node.left = u32::MAX;
            node.right = indices[0] as u32;
        } else {
            let extent = node.maximum - node.minimum;
            node.axis = if extent.x >= extent.y && extent.x >= extent.z {
                0
            } else if extent.y >= extent.z {
                1
            } else {
                2
            };
            indices.sort_unstable_by_key(|&i| key(faces[i].voxel, faces[i].face, node.axis));
            let mid = indices.len() / 2;
            let pivot = faces[indices[mid]];
            node.pivot = pivot.voxel;
            node.pivot_face = pivot.face;
            let (left, right) = indices.split_at_mut(mid);
            node.left = split(nodes, left, faces);
            node.right = split(nodes, right, faces);
        }
        nodes[index as usize] = node;
        index
    }
    let mut nodes = Vec::with_capacity(faces.len() * 2 - 1);
    split(&mut nodes, &mut (0..faces.len()).collect::<Vec<_>>(), faces);
    nodes
}

#[cfg(test)]
fn importance(node: &LightNode, position: Vec3, normal: Vec3) -> f32 {
    let delta = (node.minimum + node.maximum) * 0.5 - position;
    let radius = (node.maximum - node.minimum).length() * 0.5;
    let distance = delta.length().max(1e-5);
    let direction = delta / distance;
    let spread = (radius / distance).min(1.0);
    let receiver_cosine = (normal.dot(direction) + spread).clamp(0.0, 1.0);
    let mut emitter_cosine = 0.0_f32;
    for (face, n) in FACE_OFFSETS.iter().enumerate() {
        if node.normals & (1 << face) != 0 {
            emitter_cosine =
                emitter_cosine.max((-n.as_vec3().dot(direction) + spread).clamp(0.0, 1.0));
        }
    }
    node.power * receiver_cosine * emitter_cosine / (distance * distance + radius * radius + 0.25)
}

#[cfg(test)]
fn left_probability(nodes: &[LightNode], node: &LightNode, position: Vec3, normal: Vec3) -> f32 {
    let left = &nodes[node.left as usize];
    let right = &nodes[node.right as usize];
    let power = left.power / (left.power + right.power);
    let a = importance(left, position, normal);
    let b = importance(right, position, normal);
    let p = if a + b <= 1e-20 {
        power
    } else {
        0.05 * power + 0.95 * a / (a + b)
    };
    // Rounding must not turn an extreme HDR split into a zero-support branch.
    p.clamp(1e-6, 1.0 - 1e-6)
}

#[cfg(test)]
pub(super) fn select(
    nodes: &[LightNode],
    position: Vec3,
    normal: Vec3,
    mut uniform: f32,
) -> (usize, f32) {
    let mut index = 0;
    let mut probability = 1.0;
    loop {
        let node = &nodes[index];
        if node.left == u32::MAX {
            return (node.right as usize, probability);
        }
        let p = left_probability(nodes, node, position, normal);
        if uniform < p {
            probability *= p;
            uniform /= p;
            index = node.left as usize;
        } else {
            probability *= 1.0 - p;
            uniform = (uniform - p) / (1.0 - p);
            index = node.right as usize;
        }
        uniform = uniform.min(f32::from_bits(1.0_f32.to_bits() - 1));
    }
}

#[cfg(test)]
pub(super) fn probability(
    nodes: &[LightNode],
    faces: &[EmissiveFace],
    position: Vec3,
    normal: Vec3,
    voxel: IVec3,
    face: u32,
) -> f32 {
    let mut index = 0;
    let mut probability = 1.0;
    loop {
        let node = &nodes[index];
        if node.left == u32::MAX {
            let found = faces[node.right as usize];
            return if found.voxel == voxel && found.face == face {
                probability
            } else {
                0.0
            };
        }
        let p = left_probability(nodes, node, position, normal);
        if key(voxel, face, node.axis) < key(node.pivot, node.pivot_face, node.axis) {
            probability *= p;
            index = node.left as usize;
        } else {
            probability *= 1.0 - p;
            index = node.right as usize;
        }
    }
}

impl LightNode {
    /// Four vec4 words, mirrored exactly by the WGSL hierarchy loader. Nodes
    /// share the existing emitter binding so eight-storage-buffer devices work.
    pub fn words(self) -> [u32; 16] {
        [
            self.minimum.x.to_bits(),
            self.minimum.y.to_bits(),
            self.minimum.z.to_bits(),
            self.power.to_bits(),
            self.maximum.x.to_bits(),
            self.maximum.y.to_bits(),
            self.maximum.z.to_bits(),
            self.normals,
            self.pivot.x as u32,
            self.pivot.y as u32,
            self.pivot.z as u32,
            self.pivot_face,
            self.left,
            self.right,
            self.axis,
            0,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn light_selection_frequencies_match_reported_probabilities() {
        // Pearson goodness-of-fit with tiny expected bins pooled. Independent
        // uniforms isolate the light distribution from the production QMC map.
        for fixture in [
            crate::voxel_gi::research::hallway(),
            crate::voxel_gi::research::cave(),
        ] {
            let distribution = EmissionDistribution::with_importance(
                &fixture.scene,
                &fixture.materials,
                crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
            );
            for probe in fixture.receivers.iter().step_by(71).take(3) {
                let face = probe.face_mask.trailing_zeros() as usize;
                let normal = FACE_OFFSETS[face].as_vec3();
                let position = probe.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5;
                let mut expected = [0.0f64; 64];
                let mut observed = [0u32; 64];
                let lookup: bevy::platform::collections::HashMap<_, _> = distribution
                    .faces
                    .iter()
                    .enumerate()
                    .map(|(i, face)| ((face.voxel, face.face), i))
                    .collect();
                for (i, face) in distribution.faces.iter().enumerate() {
                    expected[i % 64] += f64::from(distribution.probability_at(
                        position,
                        normal,
                        face.voxel,
                        face.face,
                        face.material,
                        &fixture.materials,
                    ));
                }
                assert!((expected.iter().sum::<f64>() - 1.0).abs() < 2e-5);
                let mut state = 0x243f6a8885a308d3u64;
                const N: u32 = 262144;
                for _ in 0..N {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut word = state;
                    word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
                    word ^= word >> 31;
                    let uniform = (word >> 40) as f32 / 16777216.0;
                    let chosen = distribution.select_at(position, normal, uniform).unwrap();
                    observed[lookup[&(chosen.voxel, chosen.face)] % 64] += 1;
                }
                let mut bins = Vec::new();
                let (mut rare_observed, mut rare_expected) = (0.0, 0.0);
                for (observed, probability) in observed.into_iter().zip(expected) {
                    let expected = probability * f64::from(N);
                    if expected >= 8.0 {
                        bins.push((f64::from(observed), expected));
                    } else {
                        rare_observed += f64::from(observed);
                        rare_expected += expected;
                    }
                }
                if rare_expected >= 8.0 {
                    bins.push((rare_observed, rare_expected));
                } else {
                    bins[0].0 += rare_observed;
                    bins[0].1 += rare_expected;
                }
                let df = (bins.len() - 1) as f64;
                let chi2: f64 = bins.iter().map(|(o, e)| (o - e).powi(2) / e).sum();
                // Conservative engineering gate, not a reported exact p-value.
                assert!(
                    chi2 < df + 10.0 * (2.0 * df).sqrt() + 50.0,
                    "light distribution: chi2={chi2}, df={df}, receiver={position:?}"
                );
            }
        }
    }
    #[test]
    fn extreme_energy_ratios_do_not_erase_weak_emitters() {
        let faces: Vec<_> = (0..64)
            .map(|x| EmissiveFace {
                voxel: IVec3::new(x, 0, 0),
                face: 2,
                material: 4,
                cumulative: 0.0,
                probability: if x == 0 { 1.0 } else { 1e-18 },
            })
            .collect();
        let tree = build(&faces);
        let sum: f32 = faces
            .iter()
            .map(|f| {
                let p = probability(
                    &tree,
                    &faces,
                    Vec3::new(0.5, 2.0, 0.5),
                    -Vec3::Y,
                    f.voxel,
                    f.face,
                );
                assert!(p > 0.0 && p.is_finite());
                p
            })
            .sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }
    #[test]
    fn hierarchy_has_full_support_and_matching_selection_density() {
        let cave = crate::voxel_gi::research::cave();
        let distribution = EmissionDistribution::build(&cave.scene, &cave.materials);
        for (position, normal) in [
            (Vec3::new(6.0, 9.5, 20.5), Vec3::X),
            (Vec3::new(32.5, 42.0, 20.5), -Vec3::Y),
            (Vec3::new(31.0, 35.0, 29.0), -Vec3::Z),
        ] {
            let sum: f64 = distribution
                .faces
                .iter()
                .map(|f| {
                    let p = probability(
                        &distribution.tree,
                        &distribution.faces,
                        position,
                        normal,
                        f.voxel,
                        f.face,
                    );
                    assert!(p > 0.0 && p.is_finite());
                    f64::from(p)
                })
                .sum();
            assert!((sum - 1.0).abs() < 2e-5, "{sum}");
            for i in 0..4096 {
                let (index, p) = select(
                    &distribution.tree,
                    position,
                    normal,
                    (i as f32 + 0.5) / 4096.0,
                );
                let f = distribution.faces[index];
                assert_eq!(
                    p,
                    probability(
                        &distribution.tree,
                        &distribution.faces,
                        position,
                        normal,
                        f.voxel,
                        f.face
                    )
                );
            }
        }
    }
}
