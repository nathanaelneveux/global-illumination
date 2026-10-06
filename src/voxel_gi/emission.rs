//! Power-weighted surface-light sampling, independent of camera and raster mesh.
//! Paired with cosine path sampling using MIS, so tiny bright emitters do not
//! depend on accidentally being hit by a hemisphere ray.
use super::world::{AIR, FACE_OFFSETS, REGION_EDGE, TransportScene, VoxelMaterialTransportTable};
use bevy::prelude::*;
use std::sync::Arc;

/// Geometry-change metadata, computed on the scene worker. Keep material IDs
/// rather than an emission boolean: the live material table can change while
/// a scene build is in flight. Unknown/air are both non-emitting here.
pub(crate) fn changed_voxel_materials(old: &TransportScene, new: &TransportScene) -> [bool; 256] {
    let mut changed = [false; 256];
    for coordinate in new
        .regions
        .keys()
        .chain(old.regions.keys().filter(|p| !new.regions.contains_key(p)))
    {
        let before = old.regions.get(coordinate);
        let after = new.regions.get(coordinate);
        if before.zip(after).is_some_and(|(a, b)| Arc::ptr_eq(a, b)) {
            continue;
        }
        let uniform_before = before.map_or(Some(AIR), |r| r.uniform_material());
        let uniform_after = after.map_or(Some(AIR), |r| r.uniform_material());
        let mut record = |a: u16, b: u16| {
            if a != b {
                for material in [a, b] {
                    if material != AIR {
                        changed[material as usize] = true;
                    }
                }
            }
        };
        if let (Some(a), Some(b)) = (uniform_before, uniform_after) {
            record(a, b);
            continue;
        }
        for z in 0..REGION_EDGE {
            for y in 0..REGION_EDGE {
                for x in 0..REGION_EDGE {
                    let local = UVec3::new(x, y, z);
                    record(
                        before.map_or(AIR, |r| r.material(local)),
                        after.map_or(AIR, |r| r.material(local)),
                    );
                }
            }
        }
    }
    changed
}

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
        let weights: [f32; 256] =
            std::array::from_fn(|material| materials.get(material as u8).emission.max_component());
        let mut emitting = [0_u64; 4];
        for (material, &weight) in weights.iter().enumerate() {
            if weight > 0.0 {
                emitting[material / 64] |= 1 << (material % 64);
            }
        }
        if emitting == [0; 4] {
            return Self::default();
        }
        let mut faces = Vec::new();
        let mut total = 0.0_f64;
        let mut regions: Vec<_> = scene
            .regions
            .values()
            .filter(|region| region.contains_any_material(&emitting))
            .collect();
        regions.sort_unstable_by_key(|region| {
            (
                region.coordinate.z,
                region.coordinate.y,
                region.coordinate.x,
            )
        });
        for region in regions {
            for z in 0..REGION_EDGE {
                for y in 0..REGION_EDGE {
                    for x in 0..REGION_EDGE {
                        let local = UVec3::new(x, y, z);
                        let material = region.material(local);
                        if material == AIR || weights[material as usize] <= 0.0 {
                            continue;
                        }
                        let voxel = region.coordinate * REGION_EDGE as i32 + local.as_ivec3();
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
    fn region_filter_tracks_material_table_changes_and_neighbor_occlusion() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let emitter = UVec3::new(15, 8, 8);
        page[linear_index(emitter, PAGE_SIZE)] = 255;
        page[linear_index(emitter + UVec3::X, PAGE_SIZE)] = 0;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        // A registered but absent emitter must not scan any mixed region.
        materials.set(
            4,
            MaterialTransport {
                emission: EnergySpectrum::ONE,
                ..default()
            },
        );
        let mut mask = [0; 4];
        mask[0] = 1 << 4;
        assert!(
            scene
                .regions
                .values()
                .all(|region| !region.contains_any_material(&mask))
        );
        let cache = EmissionCache::prepare(None, scene.clone(), Arc::new(materials.clone()));
        assert!(cache.distribution.faces.is_empty());

        // Changing optical properties does not rebuild geometry or its mask.
        materials.set(
            255,
            MaterialTransport {
                emission: EnergySpectrum::ONE,
                ..default()
            },
        );
        let cache = EmissionCache::prepare(Some(cache), scene.clone(), Arc::new(materials));
        assert_eq!(cache.distribution.faces.len(), 5);
        assert!(!cache.distribution.faces.iter().any(|face| face.face == 0));

        // A changed non-emitter in a different region can expose the emitter.
        page[linear_index(emitter + UVec3::X, PAGE_SIZE)] = AIR;
        let edited = Arc::new(TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let materials = cache.materials.clone();
        let cache = EmissionCache::prepare(Some(cache), edited, materials);
        assert_eq!(cache.distribution.faces.len(), 6);
        assert!((cache.distribution.total_weight - 6.0).abs() < 1e-6);
    }

    #[test]
    fn material_delta_tracks_add_remove_unload_but_not_unchanged_emitters() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let empty = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        page[linear_index(UVec3::splat(8), PAGE_SIZE)] = 4;
        let lit = TransportScene::with_page_delta(&empty, vec![(IVec3::ZERO, page.clone())], &[]);
        for changed in [
            changed_voxel_materials(&empty, &lit),
            changed_voxel_materials(&lit, &empty),
            changed_voxel_materials(&lit, &TransportScene::default()),
        ] {
            assert!(changed[4]);
            assert_eq!(changed.iter().filter(|v| **v).count(), 1);
        }
        page[linear_index(UVec3::splat(9), PAGE_SIZE)] = 2;
        let edited = TransportScene::with_page_delta(&lit, vec![(IVec3::ZERO, page)], &[]);
        let changed = changed_voxel_materials(&lit, &edited);
        assert!(changed[2]);
        assert!(!changed[4], "ordinary nearby edits retain bounded history");
        assert!(!changed_voxel_materials(&lit, &lit).into_iter().any(|v| v));
    }
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
