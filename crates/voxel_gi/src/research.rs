//! Deterministic transport fixtures shared by opt-in research studies. These
//! reuse the host scenes' geometry/material policy and allocate only in tests.
use super::{
    MaterialTransport, TransportSources, VoxelMaterialTransportTable,
    naadf::{linear_index, volume_len},
    sources::DirectionalTransportLight,
    world::{AIR, FACE_OFFSETS, PAGE_SIZE, SurfaceProbe, TransportScene},
};
use bevy::prelude::*;
use std::sync::Arc;

pub(super) struct ResearchFixture {
    pub scene: Arc<TransportScene>,
    pub materials: VoxelMaterialTransportTable,
    pub sources: TransportSources,
    pub receivers: Vec<SurfaceProbe>,
}

/// Enclosed diffuse/emissive furnace. Every direction from an inner surface
/// reaches the same material, so the finite-depth answer is E * sum(albedo^k),
/// k=0..=bounces, including the solver's terminal emission connection.
/// Sealed room with a 5x5 skylight, a red wall and a directional sun. Floor
/// receivers on a 2-voxel grid. Shared by resident/display native tests.
pub(super) fn skylight_room() -> ResearchFixture {
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 4..=27 {
        for y in 4..=20 {
            for x in 4..=27 {
                let shell = x == 4 || x == 27 || y == 4 || y == 20 || z == 4 || z == 27;
                let skylight = y == 20 && (13..=17).contains(&x) && (13..=17).contains(&z);
                if shell && !skylight {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = if x == 4 { 3 } else { 2 };
                }
            }
        }
    }
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(
        3,
        super::MaterialTransport {
            diffuse: Vec3::new(0.8, 0.06, 0.03),
            ..default()
        },
    );
    ResearchFixture {
        scene: Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        )),
        materials,
        sources: TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::new(0.3, 1.0, 0.2).normalize(),
                radiance: Vec3::splat(6.0),
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        },
        receivers: (6..26)
            .step_by(2)
            .flat_map(|z| (6..26).step_by(2).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect(),
    }
}

pub(super) fn furnace() -> ResearchFixture {
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 8..24 {
        for y in 8..24 {
            for x in 8..24 {
                if [x, y, z].iter().any(|&v| v == 8 || v == 23) {
                    page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 4;
                }
            }
        }
    }
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(
        4,
        super::MaterialTransport {
            diffuse: Vec3::new(0.3, 0.55, 0.7),
            emission: Vec3::new(0.7, 1.3, 0.2),
        },
    );
    ResearchFixture {
        scene: Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        )),
        materials,
        sources: TransportSources {
            environment: Vec3::ZERO,
            directional: vec![],
            points: vec![],
        },
        receivers: [
            (IVec3::new(16, 8, 16), 2),
            (IVec3::new(9, 8, 9), 2),
            (IVec3::new(23, 16, 16), 1),
        ]
        .into_iter()
        .map(|(voxel, face)| {
            let mut p = SurfaceProbe::new(voxel, 1 << face);
            p.material = 4;
            p
        })
        .collect(),
    }
}


/// A sealed 96x9x9 rock corridor with one emitter at one end and two baffles
/// near the other (the lab's hallway chamber, closed). No environment light.
pub(super) fn hallway() -> ResearchFixture {
    const ORIGIN: IVec3 = IVec3::new(-128, -12, -24);
    const SOURCE: IVec3 = IVec3::new(-124, -11, -20);
    let (rock, emitter) = (5_u16, 4_u16);
    let material = |p: IVec3| {
        let local = p - ORIGIN;
        let inside = (0..96).contains(&local.x) && (0..9).contains(&local.y) && (0..9).contains(&local.z);
        let shell = local.x == 0 || local.x == 95 || local.y == 0 || local.y == 8 || local.z == 0 || local.z == 8;
        let baffle = (local.x == 88 && local.z < 6) || (local.x == 92 && local.z > 2);
        if p == SOURCE {
            emitter
        } else if !inside || shell || baffle {
            rock
        } else {
            AIR
        }
    };
    let pages = (-4..0)
        .map(|px| {
            let coordinate = IVec3::new(px, -1, -1);
            let mut page = vec![rock; volume_len(PAGE_SIZE)];
            for z in 0..32 {
                for y in 0..32 {
                    for x in 0..32 {
                        let local = UVec3::new(x, y, z);
                        page[linear_index(local, PAGE_SIZE)] = material(coordinate * 32 + local.as_ivec3());
                    }
                }
            }
            (coordinate, page)
        })
        .collect();
    let scene = Arc::new(TransportScene::with_page_delta(&TransportScene::default(), pages, &[]));
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(rock as u8, MaterialTransport { diffuse: Vec3::splat(0.55), emission: Vec3::ZERO });
    materials.set(emitter as u8, MaterialTransport { diffuse: Vec3::splat(0.8), emission: Vec3::new(12.0, 5.0, 1.0) });
    let receivers = (1..80)
        .flat_map(|x| {
            [-2, 0, 2].map(move |z| {
                let mut probe = SurfaceProbe::new(SOURCE + IVec3::new(x, -1, z), 4);
                probe.material = rock as u8;
                probe
            })
        })
        .collect();
    ResearchFixture {
        scene,
        materials,
        receivers,
        sources: TransportSources {
            environment: Vec3::ZERO,
            directional: vec![],
            points: vec![],
        },
    }
}

/// The lab's composed-cave geometry and materials (cave_scene.rs), copied so
/// these fixtures do not depend on the lab: a 56x40x56 chamber with a lava
/// floor, a blue-lit spire, green ceiling crystals, wall torches and a roof
/// opening.
const CAVE_ENVIRONMENT: Vec3 = Vec3::new(0.35, 0.45, 0.7);

fn cave_sun() -> Vec3 {
    Vec3::new(-0.2, 1.0, -0.15).normalize()
}

fn cave_materials() -> VoxelMaterialTransportTable {
    let mut table = VoxelMaterialTransportTable::default();
    for (id, diffuse, emission) in [
        (2, 0.55, Vec3::ZERO),
        (4, 0.15, Vec3::new(3.0, 0.45, 0.02)),
        (5, 0.2, Vec3::new(0.03, 0.4, 12.0)),
        (6, 0.2, Vec3::new(0.3, 6.0, 0.08)),
        (7, 0.2, Vec3::new(14.0, 4.0, 0.8)),
    ] {
        table.set(id, MaterialTransport { diffuse: Vec3::splat(diffuse), emission });
    }
    table
}

fn cave_voxel(p: IVec3) -> u16 {
    let IVec3 { x, y, z } = p;
    if !(4..=59).contains(&x) || !(4..=43).contains(&y) || !(4..=59).contains(&z) {
        return AIR;
    }
    if y == 43 && (51..=55).contains(&x) && (51..=55).contains(&z) {
        return AIR;
    }
    if y == 4 {
        return if (10..=54).contains(&x) && (10..=54).contains(&z) { 4 } else { 2 };
    }
    if x == 4 || x == 59 || z == 4 || z == 59 || y == 43 {
        2
    } else if (30..=34).contains(&x) && (30..=34).contains(&z) && y <= 37 {
        if y % 4 == 0 && (x == 30 || x == 34 || z == 30 || z == 34) { 5 } else { 2 }
    } else if y >= 40 && x % 11 == 7 && z % 13 == 8 {
        6
    } else if y == 10 && (((x == 5 || x == 58) && z % 8 == 0) || ((z == 5 || z == 58) && x % 8 == 0)) {
        7
    } else {
        AIR
    }
}

/// A 56×40×56 enclosed cavity: extensive lava, a blue-lit central spire,
/// green ceiling crystals, wall torches and a small far-side skylight.
pub(super) fn cave() -> ResearchFixture {
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
                            page[linear_index(local, PAGE_SIZE)] = cave_voxel(p);
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
    let materials = cave_materials();
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
    ResearchFixture {
        scene,
        materials,
        receivers,
        sources: TransportSources {
            environment: CAVE_ENVIRONMENT,
            directional: vec![DirectionalTransportLight {
                direction: cave_sun(),
                radiance: Vec3::splat(4.0),
                custom_visible: Vec3::ZERO,
            }],
            points: vec![],
        },
    }
}

pub(super) fn channels(value: Vec3) -> [f32; 3] {
    value.to_array()
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
    assert!(cave.sources.environment.max_element() > 0.0);
    eprintln!(
        "CAVE_FIXTURE receivers={} emissive_faces={}",
        cave.receivers.len(),
        emitters.faces.len()
    );
}
