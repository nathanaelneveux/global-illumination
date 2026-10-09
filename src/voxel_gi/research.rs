//! Deterministic transport fixtures shared by opt-in research studies. These
//! reuse the host cave's geometry/material policy and allocate only in tests.
use super::{
    EnergySpectrum, TransportSources, VoxelMaterialTransportTable,
    naadf::{linear_index, volume_len},
    sources::DirectionalTransportLight,
    world::{AIR, FACE_OFFSETS, PAGE_SIZE, SurfaceProbe, TransportScene},
};
use bevy::prelude::*;
use std::sync::Arc;

pub(super) struct Cave {
    pub scene: Arc<TransportScene>,
    pub materials: VoxelMaterialTransportTable,
    pub sources: TransportSources,
    pub receivers: Vec<SurfaceProbe>,
}

/// A 56×40×56 enclosed cavity: extensive lava, a blue-lit central spire,
/// green ceiling crystals, wall torches and a small far-side skylight.
/// IR and ionizing emissions are independent, normalized game transport bands,
/// not a calibrated physical model of lava or thorium.
pub(super) fn cave() -> Cave {
    let mut pages = Vec::new();
    for pz in 0..2 {
        for py in 0..2 {
            for px in 0..2 {
                let coordinate = IVec3::new(px, py, pz);
                let mut page = vec![AIR; volume_len(PAGE_SIZE)];
                for z in 0..32 {
                    for y in 0..32 {
                        for x in 0..32 {
                            let local = UVec3::new(x, y, z);
                            let p = coordinate * 32 + local.as_ivec3();
                            page[linear_index(local, PAGE_SIZE)] = match crate::cave_scene::voxel(p)
                            {
                                bevy_voxel_world::prelude::WorldVoxel::Solid(id) => id as u16,
                                _ => AIR,
                            };
                        }
                    }
                }
                pages.push((coordinate, page));
            }
        }
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        pages,
        &[],
    ));
    let materials = crate::cave_scene::materials();
    let mut receivers = Vec::new();
    for t in (8..58).step_by(7) {
        for y in [9, 22, 37] {
            for (voxel, face) in [
                (IVec3::new(4, y, t), 0),
                (IVec3::new(59, y, t), 1),
                (IVec3::new(t, y, 4), 4),
                (IVec3::new(t, y, 59), 5),
            ] {
                receivers.push(SurfaceProbe::new(voxel, 1 << face));
            }
        }
    }
    for x in [8, 20, 32, 44, 56] {
        for z in [8, 20, 32, 44, 56] {
            let p = IVec3::new(x, 43, z);
            if scene.material_at(p) == Some(2) && scene.material_at(p - IVec3::Y) == Some(AIR) {
                receivers.push(SurfaceProbe::new(p, 8));
            }
        }
    }
    for probe in &mut receivers {
        probe.material = 2;
    }
    Cave {
        scene,
        materials,
        receivers,
        sources: TransportSources {
            environment: crate::cave_scene::ENVIRONMENT,
            directional: vec![DirectionalTransportLight {
                direction: crate::cave_scene::sun_direction(),
                radiance: EnergySpectrum::new(Vec3::splat(4.0), 2.0, 0.0),
                custom_visible: Vec3::ZERO,
            }],
            points: vec![],
        },
    }
}

pub(super) fn channels(value: EnergySpectrum) -> [f32; 5] {
    [
        value.visible.x,
        value.visible.y,
        value.visible.z,
        value.infrared,
        value.ionizing,
    ]
}

#[test]
fn cave_has_exposed_receivers_many_sources_and_independent_hidden_bands() {
    let cave = cave();
    let emitters = super::emission::EmissionDistribution::build(&cave.scene, &cave.materials);
    assert!(emitters.faces.len() > 2_000);
    for id in [4, 5, 6, 7] {
        assert!(emitters.faces.iter().any(|f| f.material == id));
    }
    for probe in &cave.receivers {
        assert_eq!(cave.scene.material_at(probe.voxel), Some(2));
        for (face, normal) in FACE_OFFSETS.iter().enumerate() {
            if probe.face_mask & (1 << face) != 0 {
                assert_eq!(cave.scene.material_at(probe.voxel + *normal), Some(AIR));
            }
        }
    }
    assert_eq!(channels(cave.materials.get(4).emission)[4], 0.0);
    assert_eq!(channels(cave.materials.get(6).emission)[3], 0.0);
    assert!(cave.sources.environment.visible.max_element() > 0.0);
    eprintln!(
        "CAVE_FIXTURE receivers={} emissive_faces={}",
        cave.receivers.len(),
        emitters.faces.len()
    );
}
