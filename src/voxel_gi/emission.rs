//! Power-weighted surface-light sampling, independent of camera and raster mesh.
//! Paired with cosine path sampling using MIS, so tiny bright emitters do not
//! depend on accidentally being hit by a hemisphere ray.
use super::world::{AIR, FACE_OFFSETS, TransportScene, VoxelMaterialTransportTable};
use bevy::prelude::*;
use std::sync::Arc;

#[derive(Clone, Copy, Debug)]
pub(crate) struct EmissiveFace {
    pub voxel: IVec3,
    pub face: u32,
    pub material: u8,
    pub cumulative: f32,
    pub probability: f32,
}

#[derive(Default)]
pub(crate) struct EmissionDistribution {
    pub faces: Vec<EmissiveFace>,
    pub total_weight: f32,
}

pub(crate) struct EmissionCache {
    scene: Arc<TransportScene>,
    materials: Arc<VoxelMaterialTransportTable>,
    pub distribution: Arc<EmissionDistribution>,
}

impl EmissionCache {
    pub fn prepare(
        previous: Option<Self>,
        scene: Arc<TransportScene>,
        materials: Arc<VoxelMaterialTransportTable>,
    ) -> Self {
        if let Some(previous) = previous
            && Arc::ptr_eq(&scene, &previous.scene)
            && Arc::ptr_eq(&materials, &previous.materials)
        {
            return previous;
        }
        let distribution = Arc::new(EmissionDistribution::build(&scene, &materials));
        Self {
            scene,
            materials,
            distribution,
        }
    }
}

impl EmissionDistribution {
    pub fn build(scene: &TransportScene, materials: &VoxelMaterialTransportTable) -> Self {
        let weights: Vec<_> = (0..=255)
            .map(|material| materials.get(material).emission.max_component())
            .collect();
        if !weights.iter().any(|weight| *weight > 0.0) {
            return Self::default();
        }
        let mut faces = Vec::new();
        let mut total = 0.0_f64;
        let mut regions: Vec<_> = scene.regions.values().collect();
        regions.sort_unstable_by_key(|region| {
            (
                region.coordinate.z,
                region.coordinate.y,
                region.coordinate.x,
            )
        });
        for region in regions {
            if let Some(material) = region.uniform_material()
                && (material == AIR || weights[material as usize] == 0.0)
            {
                continue;
            }
            for z in 0..16 {
                for y in 0..16 {
                    for x in 0..16 {
                        let local = UVec3::new(x, y, z);
                        let material = region.material(local);
                        if material == AIR || weights[material as usize] <= 0.0 {
                            continue;
                        }
                        let voxel = region.coordinate * 16 + local.as_ivec3();
                        for (face, normal) in FACE_OFFSETS.iter().enumerate() {
                            if scene.material_at(voxel + *normal).is_some_and(|m| m != AIR) {
                                continue;
                            }
                            total += f64::from(weights[material as usize]);
                            faces.push(EmissiveFace {
                                voxel,
                                face: face as u32,
                                material: material as u8,
                                cumulative: total as f32,
                                probability: weights[material as usize],
                            });
                        }
                    }
                }
            }
        }
        if total == 0.0 {
            return Self::default();
        }
        let total_weight = total as f32;
        for face in &mut faces {
            face.cumulative /= total_weight;
            face.probability /= total_weight;
        }
        if let Some(last) = faces.last_mut() {
            last.cumulative = 1.0;
        }
        Self {
            faces,
            total_weight,
        }
    }

    pub fn select(&self, uniform: f32) -> Option<EmissiveFace> {
        if self.faces.is_empty() {
            return None;
        }
        let index = self
            .faces
            .partition_point(|face| face.cumulative <= uniform);
        self.faces.get(index.min(self.faces.len() - 1)).copied()
    }
}

pub(crate) fn face_point(face: EmissiveFace, u: f32, v: f32) -> Vec3 {
    let normal = FACE_OFFSETS[face.face as usize].as_vec3();
    let axis = face.face / 2;
    let tangent = if axis == 0 { Vec3::Y } else { Vec3::X };
    let bitangent = if axis == 2 { Vec3::Y } else { Vec3::Z };
    face.voxel.as_vec3()
        + Vec3::splat(0.5)
        + normal * 0.5
        + tangent * (u - 0.5)
        + bitangent * (v - 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::naadf::{linear_index, volume_len};
    use crate::voxel_gi::world::PAGE_SIZE;
    use crate::voxel_gi::{EnergySpectrum, MaterialTransport};
    #[test]
    fn distribution_contains_only_exposed_emitters_and_normalizes_power() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        page[linear_index(UVec3::new(4, 4, 4), PAGE_SIZE)] = 10;
        page[linear_index(UVec3::new(5, 4, 4), PAGE_SIZE)] = 10;
        let scene = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        );
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            10,
            MaterialTransport {
                emission: EnergySpectrum::ONE * 5.0,
                ..default()
            },
        );
        let distribution = EmissionDistribution::build(&scene, &materials);
        assert_eq!(distribution.faces.len(), 10);
        assert!(
            (distribution
                .faces
                .iter()
                .map(|f| f.probability)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-6
        );
        assert_eq!(distribution.select(0.0).unwrap().cumulative, 0.1);
        assert_eq!(distribution.select(0.999).unwrap().cumulative, 1.0);
    }
}
