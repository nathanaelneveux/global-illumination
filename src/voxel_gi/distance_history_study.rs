//! Distance/history qualification uses real geometry changes and identical
//! production path streams. References are offline evidence, never policy input.
//! No runtime history, scene residency, source or transport defaults are changed.
use super::*;
use crate::voxel_gi::TransportSettings;

const BATCH: u32 = 16;
const STEPS: u32 = 512;
const WARM: u32 = 4096;
const WINDOWS: [u32; 2] = [574_117, 1_853_903];
const SEEDS: [u32; 2] = [17, 131_119];

fn distance(voxel: IVec3, min: IVec3, max: IVec3) -> f64 {
    let separation = (min.as_dvec3() - voxel.as_dvec3())
        .max(voxel.as_dvec3() - max.as_dvec3())
        .max(bevy::math::DVec3::ZERO);
    separation.length()
}

fn references(gpu: &mut GpuTracer, f: &Fixture, paths: u32) -> Vec<Vec<[f64; 5]>> {
    let emitters = Arc::new(EmissionDistribution::build(&f.scene, &f.materials));
    WINDOWS
        .into_iter()
        .map(|start| {
            super::horizon_study::integrate(
                gpu,
                f,
                &emitters,
                TransportSettings::default(),
                start,
                paths,
            )
            .0
        })
        .collect()
}

fn tape(gpu: &mut GpuTracer, f: &Fixture) -> Value {
    let mut runs = Vec::new();
    for seed in SEEDS {
        let (warm, _) = super::history_qualification::trace(gpu, f, f.scene.clone(), &[seed], WARM);
        let mut batches = Vec::new();
        // Bound both storage buffers and readback per dispatch independently of
        // the number of steps. The older small-room tape needs fewer receivers.
        for first in (0..STEPS).step_by(32) {
            let starts: Vec<_> = (first..(first + 32).min(STEPS))
                .map(|i| seed + WARM + i * BATCH)
                .collect();
            let (values, _) =
                super::history_qualification::trace(gpu, f, f.scene.clone(), &starts, BATCH);
            batches.extend(values);
        }
        runs.push(json!({"seed":seed,"warm":warm[0],"batches":batches}));
    }
    json!(runs)
}

fn record(
    gpu: &mut GpuTracer,
    mut f: Fixture,
    before: Arc<TransportScene>,
    changed_min: IVec3,
    changed_max: IVec3,
    paths: u32,
) -> Value {
    assert!(!f.probes.is_empty());
    let sites: Vec<_> = f
        .probes
        .iter()
        .map(|p| {
            let face = p.face_mask.trailing_zeros();
            assert_eq!(p.face_mask.count_ones(), 1);
            assert_eq!(before.material_at(p.voxel), f.scene.material_at(p.voxel));
            assert_eq!(
                f.scene
                    .material_at(p.voxel + crate::voxel_gi::world::FACE_OFFSETS[face as usize]),
                Some(AIR)
            );
            json!({"voxel":p.voxel.to_array(),"face":face,
            "distance":distance(p.voxel,changed_min,changed_max)})
        })
        .collect();
    let after_reference = references(gpu, &f, paths);
    let after = tape(gpu, &f);
    f.scene = before;
    let before_reference = references(gpu, &f, paths);
    let before = tape(gpu, &f);
    eprintln!(
        "DISTANCE_TAPE {} sites={} paths/window={paths}",
        f.name,
        sites.len()
    );
    json!({"name":f.name,"sites":sites,
        "changed_bounds":[changed_min.to_array(),changed_max.to_array()],
        "references_before":before_reference,"references_after":after_reference,
        "before":before,"after":after})
}

/// A 32² opaque roof above an open floor. The diffuse control measures angular
/// footprint decay. The sun control measures *indirect* light on a vertical
/// receiver, reflected from the floor that the distant roof shadows. Thus the
/// separate direct-sun display cache cannot hide the latter dependency.
fn roof_fixture(height: i32, sun: bool) -> (Fixture, Arc<TransportScene>) {
    let mut pages = Vec::new();
    for py in 0..=height / 32 {
        for px in 0..2 {
            let mut page = vec![AIR; volume_len(PAGE_SIZE)];
            if py == 0 {
                for z in 0..32 {
                    for y in 0..32 {
                        for x in 0..32 {
                            if (px == 0 && y == 0) || (px == 1 && x == 0) {
                                page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 2;
                            }
                        }
                    }
                }
            }
            pages.push((IVec3::new(px, py, 0), page));
        }
    }
    let before = Arc::new(TransportScene::with_page_delta(
        &TransportScene::default(),
        pages,
        &[],
    ));
    let mut roof = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for x in 0..32 {
            roof[linear_index(UVec3::new(x, height as u32 % 32, z), PAGE_SIZE)] = 3;
        }
    }
    let scene = Arc::new(TransportScene::with_page_delta(
        &before,
        vec![(IVec3::new(0, height / 32, 0), roof)],
        &[],
    ));
    let mut materials = VoxelMaterialTransportTable::default();
    materials.set(
        2,
        MaterialTransport {
            diffuse: EnergySpectrum::ONE * if sun { 0.5 } else { 0.0 },
            ..default()
        },
    );
    materials.set(
        3,
        MaterialTransport {
            diffuse: EnergySpectrum::ZERO,
            ..default()
        },
    );
    let probes = [8, 12, 20, 24]
        .into_iter()
        .flat_map(|z| {
            [8, 12, 20, 24].into_iter().map(move |t| {
                let (v, face) = if sun {
                    (IVec3::new(32, t, z), 2)
                } else {
                    (IVec3::new(t, 0, z), 4)
                };
                SurfaceProbe::new(v, face)
            })
        })
        .collect();
    let sources = TransportSources {
        environment: if sun {
            EnergySpectrum::ZERO
        } else {
            EnergySpectrum::ONE
        },
        directional: if sun {
            vec![DirectionalTransportLight {
                direction: Vec3::Y,
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::ZERO,
            }]
        } else {
            vec![]
        },
        points: vec![],
    };
    (
        Fixture {
            name: if sun {
                "roof_sun_bounce"
            } else {
                "roof_diffuse_sky"
            },
            scene,
            materials,
            sources,
            probes,
            analytic: None,
        },
        before,
    )
}

#[test]
fn distance_history_bounds_and_shadow_control_are_geometrically_valid() {
    assert_eq!(
        distance(IVec3::new(0, 0, 0), IVec3::splat(1), IVec3::splat(2)),
        3.0_f64.sqrt()
    );
    assert_eq!(distance(IVec3::splat(1), IVec3::ZERO, IVec3::splat(2)), 0.0);
    let (f, before) = roof_fixture(512, true);
    use crate::voxel_gi::tracer::TraceResult;
    assert!(matches!(
        before.trace_ray_result(Vec3::new(16.5, 1.001, 16.5), Vec3::Y, 768.0),
        TraceResult::Escaped
    ));
    assert!(
        matches!(f.scene.trace_ray_result(Vec3::new(16.5,1.001,16.5),Vec3::Y,768.0),TraceResult::Hit(hit) if hit.voxel.y == 512)
    );
    assert!(f.probes.iter().all(|p| distance(
        p.voxel,
        IVec3::new(0, 512, 0),
        IVec3::new(31, 512, 31)
    ) > 384.0));
}

#[test]
#[ignore = "native distance/history sample tape; no runtime or FPS change"]
fn record_distance_history_tape() {
    let output = std::env::var("VOXEL_GI_DISTANCE_OUTPUT")
        .unwrap_or_else(|_| "target/distance-history-tape.json".into());
    let paths = std::env::var("VOXEL_GI_DISTANCE_PATHS")
        .unwrap_or_else(|_| "65536".into())
        .parse::<u32>()
        .unwrap();
    assert!((4096..=262144).contains(&paths));
    let mut gpu = super::super::tests::native_tracer();
    let mut cases = Vec::new();
    for height in [64, 128, 256, 384, 512] {
        for sun in [false, true] {
            let (f, before) = roof_fixture(height, sun);
            cases.push(record(
                &mut gpu,
                f,
                before,
                IVec3::new(0, height, 0),
                IVec3::new(31, height, 31),
                paths,
            ));
        }
    }
    let f = crate::voxel_gi::research::canyon();
    let removed: Vec<_> = (14..16)
        .flat_map(|z| (0..8).flat_map(move |y| (0..8).map(move |x| IVec3::new(x, y, z))))
        .collect();
    let before = Arc::new(TransportScene::with_page_delta(&f.scene, vec![], &removed));
    let f = Fixture {
        name: "canyon_far_residency",
        scene: f.scene,
        sources: f.sources,
        materials: f.materials,
        probes: f
            .receivers
            .into_iter()
            .filter(|p| p.voxel.z < 448)
            .collect(),
        analytic: None,
    };
    cases.push(record(
        &mut gpu,
        f,
        before,
        IVec3::new(0, 0, 448),
        IVec3::new(255, 255, 511),
        paths,
    ));

    let f = crate::voxel_gi::research::cave();
    let before = f.scene.clone();
    let mut roof = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..32 {
        for x in 0..32 {
            roof[linear_index(UVec3::new(x, 0, z), PAGE_SIZE)] = 2;
        }
    }
    let added: Vec<_> = (0..2)
        .flat_map(|z| (0..2).map(move |x| IVec3::new(x, 16, z)))
        .map(|p| (p, roof.clone()))
        .collect();
    let scene = Arc::new(TransportScene::with_page_delta(&before, added, &[]));
    let f = Fixture {
        name: "cave_far_residency",
        scene,
        sources: f.sources,
        materials: f.materials,
        probes: f.receivers,
        analytic: None,
    };
    cases.push(record(
        &mut gpu,
        f,
        before,
        IVec3::new(0, 512, 0),
        IVec3::new(63, 543, 63),
        paths,
    ));
    let report = json!({"schema":1,"batch_paths":BATCH,"steps":STEPS,"warm_paths":WARM,
        "reference_paths_per_window":paths,"reference_windows":WINDOWS,"transport":TransportSettings::default(),"cases":cases,
        "note":"Two disjoint QMC windows, not independent random scrambles. Raw five-band indirect transport only, same paths per policy. No scheduling/display/FPS claim; runtime caps unchanged."});
    std::fs::write(output, serde_json::to_vec(&report).unwrap()).unwrap();
}
