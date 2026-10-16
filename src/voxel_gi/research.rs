//! Deterministic transport fixtures shared by opt-in research studies. These
//! reuse the host scenes' geometry/material policy and allocate only in tests.
use super::{
    EnergySpectrum, TransportSources, VoxelMaterialTransportTable,
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
            diffuse: EnergySpectrum::new(Vec3::new(0.3, 0.55, 0.7), 0.2, 0.65),
            emission: EnergySpectrum::new(Vec3::new(0.7, 1.3, 0.2), 2.2, 0.9),
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
            environment: EnergySpectrum::ZERO,
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

#[test]
fn independent_oracle_matches_furnace_geometric_series() {
    let fixture = furnace();
    let material = fixture.materials.get(4);
    for bounces in 1..=3 {
        let mut expected = material.emission;
        let mut term = material.emission;
        for _ in 0..bounces {
            term = term * material.diffuse;
            expected += term;
        }
        for probe in &fixture.receivers {
            let actual = super::validation::reference_oracle::estimate_seeded(
                &fixture.scene,
                &fixture.materials,
                &fixture.sources,
                super::TransportSettings {
                    bounces,
                    ..default()
                },
                probe.voxel,
                probe.face_mask.trailing_zeros() as usize,
                2048,
                42,
            );
            for (a, e) in actual.mean.into_iter().zip(channels(expected)) {
                assert!(
                    (a - f64::from(e)).abs() < 1e-5,
                    "depth={bounces} actual={a} expected={e}"
                );
            }
        }
    }
}

/// Actual hallway chamber at its world coordinates, with opaque surroundings.
/// This isolates emitter paths; the entrance has no environment illumination.
pub(super) fn hallway() -> ResearchFixture {
    use crate::structures::{StructureLayer, VoxelOp, hallway};
    use crate::terrain::TerrainMaterial;
    let layer = StructureLayer::new(hallway::instances(), 32);
    let rock = TerrainMaterial::CaveRock.index();
    let mut pages = Vec::new();
    for px in -4..0 {
        let coordinate = IVec3::new(px, -1, -1);
        let mut page = vec![u16::from(rock); volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for y in 0..32 {
                for x in 0..32 {
                    let local = UVec3::new(x, y, z);
                    page[linear_index(local, PAGE_SIZE)] =
                        match layer.sample(coordinate * 32 + local.as_ivec3()) {
                            VoxelOp::Solid(id) => u16::from(id),
                            VoxelOp::Air => AIR,
                            VoxelOp::Keep => u16::from(rock),
                        };
                }
            }
        }
        pages.push((coordinate, page));
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        pages,
        &[],
    ));
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(rock, crate::cave_scene::materials().get(2));
    materials.set(
        TerrainMaterial::Emissive.index(),
        crate::terrain::emissive_block_transport(),
    );
    let receivers = (1..80)
        .flat_map(|x| {
            [-2, 0, 2].map(move |z| {
                let mut probe = SurfaceProbe::new(hallway::floor_receiver(x) + IVec3::Z * z, 4);
                probe.material = rock;
                probe
            })
        })
        .collect();
    ResearchFixture {
        scene,
        materials,
        receivers,
        sources: TransportSources {
            environment: EnergySpectrum::ZERO,
            directional: vec![],
            points: vec![],
        },
    }
}

/// The same sealed transport fixture with the normal world's nearby cave
/// emitters loaded. They cannot illuminate through its opaque surroundings,
/// but they compete in the global light hierarchy just as they do in gameplay.
pub(super) fn hallway_loaded() -> ResearchFixture {
    use crate::structures::{VoxelOp, showcase};
    let mut fixture = hallway();
    let layer = showcase::world_structures();
    let mut pages = Vec::new();
    for px in 1..=3 {
        for py in -2..=0 {
            for pz in -2..0 {
                let coordinate = IVec3::new(px, py, pz);
                let mut page = vec![5; volume_len(PAGE_SIZE)];
                for z in 0..32 {
                    for y in 0..32 {
                        for x in 0..32 {
                            let local = UVec3::new(x, y, z);
                            page[linear_index(local, PAGE_SIZE)] =
                                match layer.sample(coordinate * 32 + local.as_ivec3()) {
                                    VoxelOp::Solid(id) => u16::from(id),
                                    VoxelOp::Air => AIR,
                                    VoxelOp::Keep => 5,
                                };
                        }
                    }
                }
                pages.push((coordinate, page));
            }
        }
    }
    fixture.scene = Arc::new(TransportScene::with_page_delta(&fixture.scene, pages, &[]));
    let cave = crate::cave_scene::materials();
    for (local, global) in showcase::CAVE_PALETTE {
        fixture.materials.set(global.index(), cave.get(local));
    }
    fixture
}

/// A 56×40×56 enclosed cavity: extensive lava, a blue-lit central spire,
/// green ceiling crystals, wall torches and a small far-side skylight.
/// IR and ionizing emissions are independent, normalized game transport bands,
/// not a calibrated physical model of lava or thorium.
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
    ResearchFixture {
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

/// Fully resident canyon volume. Uniform air/rock regions compact naturally in
/// TransportScene; receiver sites cover floor, rims and every tower face. This
/// is a fixed numerical fixture, not a streaming/frame-pacing benchmark.
pub(super) fn canyon() -> ResearchFixture {
    let size = crate::canyon_scene::SIZE;
    let mut pages = Vec::new();
    for pz in 0..size.z / 32 {
        for py in 0..size.y / 32 {
            for px in 0..size.x / 32 {
                let coordinate = IVec3::new(px, py, pz);
                let mut page = vec![AIR; volume_len(PAGE_SIZE)];
                for z in 0..32 {
                    for x in 0..32 {
                        let wx = px * 32 + x;
                        let wz = pz * 32 + z;
                        let height = crate::canyon_scene::height(wx, wz);
                        for y in 0..32 {
                            let wy = py * 32 + y;
                            if wy <= height {
                                page[linear_index(
                                    UVec3::new(x as u32, y as u32, z as u32),
                                    PAGE_SIZE,
                                )] = if (wy / 16) % 4 == 2 { 2 } else { 1 };
                            }
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
    let mut sites = std::collections::BTreeSet::new();
    for z in (32..480).step_by(24) {
        for x in (48..208).step_by(16) {
            let y = crate::canyon_scene::height(x, z);
            sites.insert((x, y, z, 2));
        }
    }
    // Cross sections include vertical rims and any pillars crossing the slice.
    for z in (16..512).step_by(32).chain([180, 294, 414, 450]) {
        for y in [16, 48, 96, 144, 192, 224] {
            for x in 1..255 {
                let p = IVec3::new(x, y, z);
                if scene.material_at(p).is_none_or(|m| m == AIR) {
                    continue;
                }
                for (face, normal) in [(0, IVec3::X), (1, IVec3::NEG_X)] {
                    if scene.material_at(p + normal) == Some(AIR) {
                        sites.insert((x, y, z, face));
                    }
                }
            }
        }
    }
    let receivers = sites
        .into_iter()
        .map(|(x, y, z, face)| {
            let voxel = IVec3::new(x, y, z);
            let mut p = SurfaceProbe::new(voxel, 1 << face);
            p.material = scene.material_at(voxel).unwrap() as u8;
            p
        })
        .collect();
    ResearchFixture {
        scene,
        materials: crate::canyon_scene::materials(),
        receivers,
        sources: TransportSources {
            environment: crate::canyon_scene::ENVIRONMENT,
            directional: vec![DirectionalTransportLight {
                direction: crate::canyon_scene::sun_direction(),
                radiance: EnergySpectrum::new(Vec3::splat(2.0), 0.2, 0.0),
                custom_visible: Vec3::ZERO,
            }],
            points: vec![],
        },
    }
}

#[test]
fn canyon_transport_fixture_matches_host_geometry_and_has_exposed_receivers() {
    let canyon = canyon();
    assert!(canyon.receivers.len() > 300);
    for p in &canyon.receivers {
        assert!(matches!(
            crate::canyon_scene::voxel(p.voxel),
            bevy_voxel_world::prelude::WorldVoxel::Solid(_)
        ));
        for (face, normal) in FACE_OFFSETS.iter().enumerate() {
            if p.face_mask & (1 << face) != 0 {
                assert_eq!(canyon.scene.material_at(p.voxel + *normal), Some(AIR));
            }
        }
    }
    assert!(canyon.receivers.iter().any(|p| p.voxel.y > 200));
    assert!(canyon.receivers.iter().any(|p| p.voxel.z > 400));
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
