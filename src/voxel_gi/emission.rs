//! Power-weighted surface-light sampling, independent of camera and raster mesh.
//! Paired with cosine path sampling using MIS, so tiny bright emitters do not
//! depend on accidentally being hit by a hemisphere ray.
use super::world::{AIR, FACE_OFFSETS, REGION_EDGE, TransportScene, VoxelMaterialTransportTable};
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

/// Deterministic proposal hint, not a sample from the light distribution.
/// Keep concentration separate from `EmissiveFace::probability`: it must never
/// enter a physical light PDF or change transported energy.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GuideSource {
    pub voxel: IVec3,
    pub concentration: f32,
}

#[derive(Default, Clone)]
pub(crate) struct EmissionDistribution {
    pub faces: Vec<EmissiveFace>,
    pub total_weight: f32,
    pub tree: Vec<tree::LightNode>,
    /// Sampling importance only; never scales transported energy.
    pub importance: super::EnergySpectrum,
}

mod tree;

pub(crate) struct EmissionCache {
    scene: Arc<TransportScene>,
    materials: Arc<VoxelMaterialTransportTable>,
    pub distribution: Arc<EmissionDistribution>,
}

impl EmissionCache {
    pub fn prepare_with_importance(
        previous: Option<Self>,
        scene: Arc<TransportScene>,
        materials: Arc<VoxelMaterialTransportTable>,
        importance: super::EnergySpectrum,
    ) -> Self {
        if let Some(mut previous) = previous
            && Arc::ptr_eq(&materials, &previous.materials)
            && previous.distribution.importance == importance
            && (Arc::ptr_eq(&scene, &previous.scene)
                || !emission_geometry_changed(&previous.scene, &scene, &materials))
        {
            // Visibility rays still use the new scene. Only the distribution
            // of exposed emitting faces (and its GPU allocation) is reused.
            previous.scene = scene;
            return previous;
        }
        let distribution = Arc::new(EmissionDistribution::with_importance(
            &scene, &materials, importance,
        ));
        Self {
            scene,
            materials,
            distribution,
        }
    }

    /// Historical equal-band control for the opt-in cache-cost studies.
    #[cfg(test)]
    pub fn prepare(
        previous: Option<Self>,
        scene: Arc<TransportScene>,
        materials: Arc<VoxelMaterialTransportTable>,
    ) -> Self {
        Self::prepare_with_importance(previous, scene, materials, super::EnergySpectrum::ONE)
    }
}

/// The sampling distribution depends on emitter materials and their six face
/// neighbors, not arbitrary geometry elsewhere. Region identities and material
/// masks give a conservative dependency check without scanning voxel payloads.
/// Checking old AND new neighbors covers removal/unload and newly exposed faces.
fn emission_geometry_changed(
    previous: &TransportScene,
    next: &TransportScene,
    materials: &VoxelMaterialTransportTable,
) -> bool {
    let mut emitting = [0_u64; 4];
    for material in 0..256 {
        if materials.get(material as u8).emission.max_component() > 0.0 {
            emitting[material / 64] |= 1 << (material % 64);
        }
    }
    if emitting == [0; 4] {
        return false;
    }
    next.changed_regions(previous).iter().any(|coordinate| {
        std::iter::once(IVec3::ZERO)
            .chain(FACE_OFFSETS)
            .any(|offset| {
                let neighbor = *coordinate + offset;
                [previous, next].into_iter().any(|scene| {
                    scene
                        .regions
                        .get(&neighbor)
                        .is_some_and(|region| region.contains_any_material(&emitting))
                })
            })
    })
}

impl EmissionDistribution {
    pub fn build(scene: &TransportScene, materials: &VoxelMaterialTransportTable) -> Self {
        Self::with_importance(scene, materials, super::EnergySpectrum::ONE)
    }

    pub fn with_importance(
        scene: &TransportScene,
        materials: &VoxelMaterialTransportTable,
        importance: super::EnergySpectrum,
    ) -> Self {
        assert!(
            importance.visible.min_element() > 0.0
                && importance.infrared > 0.0
                && importance.ionizing > 0.0
        );
        let weights: [f32; 256] = std::array::from_fn(|material| {
            (materials.get(material as u8).emission * importance).max_component()
        });
        let mut emitting = [0_u64; 4];
        for (material, &weight) in weights.iter().enumerate() {
            if weight > 0.0 {
                emitting[material / 64] |= 1 << (material % 64);
            }
        }
        if emitting == [0; 4] {
            return Self {
                importance,
                ..Self::default()
            };
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
            return Self {
                importance,
                ..Self::default()
            };
        }
        let total_weight = total as f32;
        for face in &mut faces {
            face.cumulative /= total_weight;
            face.probability /= total_weight;
        }
        if let Some(last) = faces.last_mut() {
            last.cumulative = 1.0;
        }
        let tree = if faces.len() >= 32 {
            tree::build(&faces)
        } else {
            Vec::new()
        };
        Self {
            faces,
            total_weight,
            tree,
            importance,
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

    pub fn select_at(&self, position: Vec3, normal: Vec3, uniform: f32) -> Option<EmissiveFace> {
        if self.tree.is_empty() {
            return self.select(uniform);
        }
        let (index, probability) = tree::select(&self.tree, position, normal, uniform);
        let mut face = self.faces[index];
        face.probability = probability;
        Some(face)
    }

    pub fn guide_at(
        &self,
        position: Vec3,
        normal: Vec3,
        visible: impl FnMut(IVec3) -> bool,
    ) -> Option<GuideSource> {
        if !self.tree.is_empty() {
            let root = &self.tree[0];
            let front = Vec3::select(normal.cmpge(Vec3::ZERO), root.maximum, root.minimum);
            if normal.dot(front - position) <= 0.0 {
                return None;
            }
            let (index, probability) =
                tree::dominant(&self.tree, &self.faces, position, normal, visible)?;
            return Some(GuideSource {
                voxel: self.faces[index].voxel,
                concentration: probability,
            });
        }
        let score = |face: EmissiveFace| {
            let delta = face_point(face, 0.5, 0.5) - position;
            let d2 = delta.length_squared().max(0.0001);
            let direction = delta / d2.sqrt();
            face.probability
                * normal.dot(direction).max(0.0)
                * FACE_OFFSETS[face.face as usize]
                    .as_vec3()
                    .dot(-direction)
                    .max(0.0)
                / d2
        };
        let mut best = -1.0;
        let mut source = None;
        for &face in &self.faces {
            let value = score(face);
            if value > best {
                best = value;
                source = Some(GuideSource {
                    voxel: face.voxel,
                    concentration: face.probability,
                });
            }
        }
        if best <= 0.0 { None } else { source }
    }

    pub fn probability_at(
        &self,
        position: Vec3,
        normal: Vec3,
        voxel: IVec3,
        face: u32,
        material: u8,
        materials: &VoxelMaterialTransportTable,
    ) -> f32 {
        if self.tree.is_empty() {
            return (materials.get(material).emission * self.importance).max_component()
                / self.total_weight.max(1e-20);
        }
        tree::probability(&self.tree, &self.faces, position, normal, voxel, face)
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
    fn importance_changes_pdf_not_energy_and_refreshes_cached_distribution() {
        let mut voxels = vec![AIR; volume_len(PAGE_SIZE)];
        voxels[0] = 4;
        voxels[3] = 5;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, voxels)],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            4,
            MaterialTransport {
                emission: EnergySpectrum::new(Vec3::ZERO, 8., 0.),
                ..default()
            },
        );
        materials.set(
            5,
            MaterialTransport {
                emission: EnergySpectrum::new(Vec3::ONE, 0., 0.),
                ..default()
            },
        );
        let materials = Arc::new(materials);
        let old = EmissionCache::prepare(None, scene.clone(), materials.clone());
        let original = old.distribution.clone();
        let importance = EnergySpectrum::new(Vec3::ONE, 0.125, 0.125);
        let next = EmissionCache::prepare_with_importance(
            Some(old),
            scene.clone(),
            materials.clone(),
            importance,
        );
        assert!(!Arc::ptr_eq(&original, &next.distribution));
        assert_eq!(next.distribution.faces.len(), 12);
        for face in &next.distribution.faces {
            assert!((face.probability - 1. / 12.).abs() < 1e-6);
            assert_eq!(
                face.probability,
                next.distribution.probability_at(
                    Vec3::ZERO,
                    Vec3::Y,
                    face.voxel,
                    face.face,
                    face.material,
                    &materials
                )
            );
        }
        assert_eq!(materials.get(4).emission.infrared, 8.);
        let unchanged = next.distribution.clone();
        let next = EmissionCache::prepare_with_importance(Some(next), scene, materials, importance);
        assert!(Arc::ptr_eq(&unchanged, &next.distribution));
    }

    #[test]
    fn distribution_reuses_unrelated_residency_but_tracks_emitter_changes() {
        let mut voxels = vec![AIR; volume_len(PAGE_SIZE)];
        voxels[0] = 4;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, voxels.clone())],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        // IR-only emitters participate even when their visible energy is zero.
        materials.set(
            4,
            MaterialTransport {
                emission: EnergySpectrum::new(Vec3::ZERO, 8.0, 0.0),
                ..default()
            },
        );
        let materials = Arc::new(materials);
        let mut cache = EmissionCache::prepare(None, scene.clone(), materials.clone());
        let original = cache.distribution.clone();
        assert_eq!(original.faces.len(), 6);
        for material in [2, 3, AIR] {
            let next = Arc::new(TransportScene::with_page_delta(
                &cache.scene,
                vec![(IVec3::X * 8, vec![material; volume_len(PAGE_SIZE)])],
                &[],
            ));
            cache = EmissionCache::prepare(Some(cache), next.clone(), materials.clone());
            assert!(Arc::ptr_eq(&cache.scene, &next));
            assert!(Arc::ptr_eq(&cache.distribution, &original));
        }
        let next = Arc::new(TransportScene::with_page_delta(
            &cache.scene,
            vec![],
            &[IVec3::X * 8],
        ));
        cache = EmissionCache::prepare(Some(cache), next, materials.clone());
        assert!(Arc::ptr_eq(&cache.distribution, &original));
        let next = Arc::new(TransportScene::with_page_delta(
            &cache.scene,
            vec![(IVec3::X * 8, voxels)],
            &[],
        ));
        cache = EmissionCache::prepare(Some(cache), next, materials.clone());
        assert_eq!(cache.distribution.faces.len(), 12);
        assert!(!Arc::ptr_eq(&cache.distribution, &original));
        let next = Arc::new(TransportScene::with_page_delta(
            &cache.scene,
            vec![],
            &[IVec3::X * 8],
        ));
        cache = EmissionCache::prepare(Some(cache), next, materials);
        assert_eq!(cache.distribution.faces.len(), 6);
        assert!(!Arc::ptr_eq(&cache.distribution, &original));
    }

    #[test]
    fn loading_and_unloading_an_adjacent_nonemitter_page_rebuilds_exposure() {
        let mut voxels = vec![AIR; volume_len(PAGE_SIZE)];
        voxels[linear_index(UVec3::new(31, 8, 8), PAGE_SIZE)] = 4;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, voxels)],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            4,
            MaterialTransport {
                emission: EnergySpectrum::ONE,
                ..default()
            },
        );
        let materials = Arc::new(materials);
        let mut cache = EmissionCache::prepare(None, scene, materials.clone());
        assert_eq!(cache.distribution.faces.len(), 6);
        let original = cache.distribution.clone();
        let next = Arc::new(TransportScene::with_page_delta(
            &cache.scene,
            vec![(IVec3::X, vec![2; volume_len(PAGE_SIZE)])],
            &[],
        ));
        cache = EmissionCache::prepare(Some(cache), next, materials.clone());
        assert_eq!(cache.distribution.faces.len(), 5);
        assert!(!Arc::ptr_eq(&cache.distribution, &original));
        let next = Arc::new(TransportScene::with_page_delta(
            &cache.scene,
            vec![],
            &[IVec3::X],
        ));
        cache = EmissionCache::prepare(Some(cache), next, materials);
        assert_eq!(cache.distribution.faces.len(), 6);
    }

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
    fn scene_delta_tracks_add_remove_unload_and_unchanged_geometry() {
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let empty = TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        );
        page[linear_index(UVec3::splat(8), PAGE_SIZE)] = 4;
        let lit = TransportScene::with_page_delta(&empty, vec![(IVec3::ZERO, page.clone())], &[]);
        for change in [
            super::super::geometry::SceneChanges::between(&empty, &lit),
            super::super::geometry::SceneChanges::between(&lit, &empty),
            super::super::geometry::SceneChanges::between(&lit, &TransportScene::default()),
        ] {
            assert!(change.changed && change.transport.is_some());
        }
        page[linear_index(UVec3::splat(9), PAGE_SIZE)] = 2;
        let edited = TransportScene::with_page_delta(&lit, vec![(IVec3::ZERO, page)], &[]);
        let changed = super::super::geometry::SceneChanges::between(&lit, &edited);
        assert!(changed.changed && changed.transport.is_some());
        let unchanged = super::super::geometry::SceneChanges::between(&lit, &lit);
        assert!(!unchanged.changed);
        assert!(unchanged.transport.is_none());
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
