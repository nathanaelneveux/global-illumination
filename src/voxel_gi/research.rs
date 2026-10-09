//! Deterministic transport fixtures shared by opt-in research studies. These
//! contain no gameplay/material-ID policies and allocate nothing in normal runs.
use super::{
    EnergySpectrum, MaterialTransport, TransportSources, VoxelMaterialTransportTable,
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
                            page[linear_index(local, PAGE_SIZE)] = cave_material(p);
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
    let mut materials = VoxelMaterialTransportTable::default();
    for (id, diffuse, emission) in [
        (
            2,
            EnergySpectrum::new(Vec3::splat(0.55), 0.35, 0.08),
            EnergySpectrum::ZERO,
        ),
        (
            4,
            EnergySpectrum::new(Vec3::splat(0.15), 0.1, 0.01),
            EnergySpectrum::new(Vec3::new(3.0, 0.45, 0.02), 9.0, 0.0),
        ),
        (
            5,
            EnergySpectrum::new(Vec3::splat(0.2), 0.1, 0.01),
            EnergySpectrum::new(Vec3::new(0.03, 0.4, 12.0), 0.05, 0.0),
        ),
        (
            6,
            EnergySpectrum::new(Vec3::splat(0.2), 0.1, 0.01),
            EnergySpectrum::new(Vec3::new(0.3, 6.0, 0.08), 0.0, 15.0),
        ),
        (
            7,
            EnergySpectrum::new(Vec3::splat(0.2), 0.1, 0.01),
            EnergySpectrum::new(Vec3::new(14.0, 4.0, 0.8), 5.0, 0.0),
        ),
    ] {
        materials.set(id, MaterialTransport { diffuse, emission });
    }
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
            environment: EnergySpectrum::new(Vec3::new(0.35, 0.45, 0.7), 0.08, 0.0),
            directional: vec![DirectionalTransportLight {
                direction: Vec3::new(-0.2, 1.0, -0.15).normalize(),
                radiance: EnergySpectrum::new(Vec3::splat(4.0), 2.0, 0.0),
                custom_visible: Vec3::ZERO,
            }],
            points: vec![],
        },
    }
}

fn cave_material(p: IVec3) -> u16 {
    let IVec3 { x, y, z } = p;
    if !(4..=59).contains(&x) || !(4..=43).contains(&y) || !(4..=59).contains(&z) {
        return AIR;
    }
    if y == 43 && (51..=55).contains(&x) && (51..=55).contains(&z) {
        return AIR;
    }
    if y == 4 {
        return if (10..=54).contains(&x) && (10..=54).contains(&z) {
            4
        } else {
            2
        };
    }
    if x == 4 || x == 59 || z == 4 || z == 59 || y == 43 {
        return 2;
    }
    if (30..=34).contains(&x) && (30..=34).contains(&z) && y <= 37 {
        return if y % 4 == 0 && (x == 30 || x == 34 || z == 30 || z == 34) {
            5
        } else {
            2
        };
    }
    if y >= 40 && x % 11 == 7 && z % 13 == 8 {
        return 6;
    }
    if y == 10 && (((x == 5 || x == 58) && z % 8 == 0) || ((z == 5 || z == 58) && x % 8 == 0)) {
        return 7;
    }
    AIR
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
