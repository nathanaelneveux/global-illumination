//! Scene construction reference and opt-in edit benchmark. Keep the old
//! snapshot-identity invalidation rule here, not as a production runtime mode.

use super::*;
use std::hint::black_box;

// Frozen 28585a0 constructor: neither reuse bookkeeping nor its new branches
// are charged to the full-build control, including cold construction.
fn full_region_reference(coordinate: IVec3, materials: Vec<u16>) -> NaadfRegion {
    assert_eq!(materials.len(), volume_len(REGION_SIZE));

    if materials.iter().all(|material| *material == AIR) {
        return NaadfRegion {
            coordinate,
            storage: RegionStorage::Empty,
            any_solid: false,
            material_mask: [0; 4],
        };
    }
    if materials.iter().all(|material| *material == materials[0]) {
        let mut material_mask = [0; 4];
        material_mask[materials[0] as usize / 64] |= 1 << (materials[0] % 64);
        return NaadfRegion {
            coordinate,
            storage: RegionStorage::Uniform(materials[0]),
            any_solid: true,
            material_mask,
        };
    }

    let mut voxel_aadf = vec![Aadf::default(); materials.len()];
    let mut brick_occupied = vec![false; volume_len(BRICK_GRID_SIZE)];
    let mut material_mask = [0; 4];

    for brick_z in 0..BRICKS_PER_AXIS {
        for brick_y in 0..BRICKS_PER_AXIS {
            for brick_x in 0..BRICKS_PER_AXIS {
                let brick = UVec3::new(brick_x, brick_y, brick_z);
                let mut blocked = [false; (BRICK_EDGE * BRICK_EDGE * BRICK_EDGE) as usize];
                let brick_size = UVec3::splat(BRICK_EDGE);

                for z in 0..BRICK_EDGE {
                    for y in 0..BRICK_EDGE {
                        for x in 0..BRICK_EDGE {
                            let local = UVec3::new(x, y, z);
                            let region_local = brick * BRICK_EDGE + local;
                            let material = materials[linear_index(region_local, REGION_SIZE)];
                            blocked[linear_index(local, brick_size)] = material != AIR;
                            if material != AIR {
                                material_mask[material as usize / 64] |= 1 << (material % 64);
                            }
                        }
                    }
                }

                let brick_index = linear_index(brick, BRICK_GRID_SIZE);
                brick_occupied[brick_index] = blocked.iter().any(|occupied| *occupied);
                let local_aadf = build_aadf(brick_size, &blocked);
                for z in 0..BRICK_EDGE {
                    for y in 0..BRICK_EDGE {
                        for x in 0..BRICK_EDGE {
                            let local = UVec3::new(x, y, z);
                            let region_local = brick * BRICK_EDGE + local;
                            voxel_aadf[linear_index(region_local, REGION_SIZE)] =
                                local_aadf[linear_index(local, brick_size)];
                        }
                    }
                }
            }
        }
    }

    let brick_aadf = build_aadf(BRICK_GRID_SIZE, &brick_occupied);
    let any_solid = brick_occupied.iter().any(|occupied| *occupied);
    NaadfRegion {
        coordinate,
        storage: RegionStorage::Dense {
            materials: materials.into_boxed_slice(),
            voxel_aadf: voxel_aadf.into_boxed_slice(),
            brick_occupied: brick_occupied.into_boxed_slice(),
            brick_aadf: brick_aadf.into_boxed_slice(),
        },
        any_solid,
        material_mask,
    }
}

fn assert_same_region(actual: NaadfRegion, expected: NaadfRegion) {
    let a = TransportScene {
        regions: HashMap::from([(actual.coordinate, Arc::new(actual))]),
        ..default()
    };
    let b = TransportScene {
        regions: HashMap::from([(expected.coordinate, Arc::new(expected))]),
        ..default()
    };
    assert_same_scene(&a, &b);
}

#[test]
fn incremental_bricks_match_full_construction_for_edits_and_materials() {
    let coordinate = IVec3::new(-7, 2, -3);
    let mut materials = vec![AIR; volume_len(REGION_SIZE)];
    let mut previous = NaadfRegion::build(coordinate, materials.clone());
    let mut state = 19_u32;
    for step in 0..200 {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let index = state as usize % materials.len();
        materials[index] = [AIR, 0, 255, 2][step % 4];
        if step == 60 {
            materials.fill(255);
        }
        if step == 100 {
            materials.fill(AIR);
        }
        if step == 140 {
            materials.fill(0);
        }
        let next = previous.rebuild(materials.clone());
        assert_same_region(
            next.clone(),
            full_region_reference(coordinate, materials.clone()),
        );
        // Old immutable fields are still correct after their successor builds.
        let old: Vec<_> = (0..volume_len(REGION_SIZE))
            .map(|i| {
                previous.material(UVec3::new(
                    i as u32 % 16,
                    (i as u32 / 16) % 16,
                    i as u32 / 256,
                ))
            })
            .collect();
        assert_same_region(previous.clone(), NaadfRegion::build(coordinate, old));
        previous = next;
    }
    // Last solid and last air transitions, same-occupancy material replacement.
    for fill in [AIR, 0, 255] {
        let mut data = vec![fill; volume_len(REGION_SIZE)];
        let original = NaadfRegion::build(coordinate, data.clone());
        for value in [AIR, 0, 255] {
            data[123] = value;
            assert_same_region(
                original.rebuild(data.clone()),
                NaadfRegion::build(coordinate, data.clone()),
            );
        }
    }
}

#[test]
#[ignore = "paired CPU incremental brick construction benchmark"]
fn incremental_brick_construction_benchmark() {
    for (name, scattered) in [("terrain", false), ("scattered", true)] {
        let original: Vec<_> = (0..volume_len(REGION_SIZE))
            .map(|i| {
                let x = i % 16;
                let y = (i / 16) % 16;
                let z = i / 256;
                let solid = if scattered {
                    (x + 3 * y + 7 * z) % 5 == 0
                } else {
                    y < 6 + (x / 4 + z / 4) % 5
                };
                if solid { 2 } else { AIR }
            })
            .collect();
        let previous = NaadfRegion::build(IVec3::ZERO, original.clone());
        for material_only in [false, true] {
            let mut edited = original.clone();
            let index = edited.iter().position(|&v| v != AIR).unwrap();
            edited[index] = if material_only { 255 } else { AIR };
            assert_same_region(
                previous.rebuild(edited.clone()),
                NaadfRegion::build(IVec3::ZERO, edited.clone()),
            );
            let mut times = [Vec::new(), Vec::new(), Vec::new()];
            for pair in 0..44 {
                for mode in (0..3).map(|offset| (pair + offset) % 3) {
                    let start = Instant::now();
                    for _ in 0..64 {
                        let data = black_box(edited.clone());
                        black_box(match mode {
                            0 => full_region_reference(IVec3::ZERO, data),
                            1 => NaadfRegion::build(IVec3::ZERO, data),
                            _ => previous.rebuild(data),
                        });
                    }
                    if pair >= 4 {
                        times[mode].push(start.elapsed().as_secs_f64() * 1e6 / 64.0);
                    }
                }
            }
            for t in &mut times {
                t.sort_by(f64::total_cmp);
            }
            eprintln!(
                "BRICK_REPAIR fixture={name} material_only={material_only} variants=original/cold/repaired us_p50={:.3}/{:.3}/{:.3} us_p95={:.3}/{:.3}/{:.3} exact=true",
                times[0][20], times[1][20], times[2][20], times[0][37], times[1][37], times[2][37]
            );
        }
    }
}

fn rebuild_on_any_change(
    previous: &TransportScene,
    changed_pages: Vec<(IVec3, Vec<u16>)>,
    removed_pages: &[IVec3],
) -> TransportScene {
    let mut regions = previous.regions.clone();
    for page in removed_pages {
        for offset in page_region_offsets() {
            regions.remove(&(page * 2 + offset));
        }
    }
    let built: Vec<_> = changed_pages
        .into_iter()
        .flat_map(|(page, materials)| build_page_regions(page, &materials, previous))
        .collect();
    for region in built {
        regions.insert(region.coordinate, region);
    }
    let unchanged = regions.len() == previous.regions.len()
        && regions.iter().all(|(key, region)| {
            previous
                .regions
                .get(key)
                .is_some_and(|old| Arc::ptr_eq(old, region))
        });
    let top_level = if unchanged {
        previous.top_level.clone()
    } else {
        TopLevelAadf::build(&regions)
    };
    TransportScene { regions, top_level }
}

fn assert_same_scene(actual: &TransportScene, reference: &TransportScene) {
    assert_eq!(actual.top_level.minimum, reference.top_level.minimum);
    assert_eq!(actual.top_level.maximum, reference.top_level.maximum);
    assert_eq!(actual.top_level.size, reference.top_level.size);
    assert_eq!(actual.top_level.fields, reference.top_level.fields);
    assert_eq!(actual.regions.len(), reference.regions.len());
    for (key, region) in &reference.regions {
        let actual_region = &actual.regions[key];
        assert_eq!(actual_region.any_solid, region.any_solid);
        assert_eq!(actual_region.material_mask, region.material_mask);
        for z in 0..REGION_EDGE {
            for y in 0..REGION_EDGE {
                for x in 0..REGION_EDGE {
                    let local = UVec3::new(x, y, z);
                    assert_eq!(actual_region.material(local), region.material(local));
                    assert_eq!(actual_region.voxel_aadf(local), region.voxel_aadf(local));
                }
            }
        }
        for z in 0..BRICKS_PER_AXIS {
            for y in 0..BRICKS_PER_AXIS {
                for x in 0..BRICKS_PER_AXIS {
                    let brick = UVec3::new(x, y, z);
                    assert_eq!(
                        actual_region.brick_occupied(brick),
                        region.brick_occupied(brick)
                    );
                    assert_eq!(actual_region.brick_aadf(brick), region.brick_aadf(brick));
                }
            }
        }
    }
}

#[test]
fn scene_construction_matches_full_rebuild_through_edit_and_residency_changes() {
    let mut scene = TransportScene::default();
    let mut reference = TransportScene::default();
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    let point = linear_index(UVec3::new(4, 5, 6), PAGE_SIZE);
    let neighbour = point + 1;
    // Air -> occupied, material only, occupancy-preserving edit, last block
    // removed, negative coordinates, missing islands and replacement at the
    // same region count. Immutable older snapshots stay valid throughout.
    for stage in 0..9 {
        let (changed, removed) = match stage {
            0 => (vec![(IVec3::ZERO, page.clone())], vec![]),
            1 => {
                page[point] = 1;
                (vec![(IVec3::ZERO, page.clone())], vec![])
            }
            2 => {
                page[point] = 2;
                (vec![(IVec3::ZERO, page.clone())], vec![])
            }
            3 => {
                page[neighbour] = 1;
                (vec![(IVec3::ZERO, page.clone())], vec![])
            }
            4 => {
                page[point] = AIR;
                (vec![(IVec3::ZERO, page.clone())], vec![])
            }
            5 => {
                page[neighbour] = AIR;
                (vec![(IVec3::ZERO, page.clone())], vec![])
            }
            6 => (vec![(IVec3::new(-2, 0, -1), page.clone())], vec![]),
            7 => (
                vec![(IVec3::new(2, 0, 1), page.clone())],
                vec![IVec3::new(-2, 0, -1)],
            ),
            _ => (vec![], vec![IVec3::ZERO, IVec3::new(2, 0, 1)]),
        };
        let next = TransportScene::with_page_delta(&scene, changed.clone(), &removed);
        reference = rebuild_on_any_change(&reference, changed, &removed);
        assert_same_scene(&next, &reference);
        assert_eq!(
            scene.top_level.fields,
            TopLevelAadf::build(&scene.regions).fields
        );
        scene = next;
    }
}

#[test]
fn occupied_region_edits_share_only_top_level_acceleration() {
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    let point = linear_index(UVec3::new(4, 5, 6), PAGE_SIZE);
    page[point] = 1;
    let initial = TransportScene::with_page_delta(
        &TransportScene::default(),
        vec![(IVec3::ZERO, page.clone())],
        &[],
    );
    page[point + 1] = 2;
    let edited = TransportScene::with_page_delta(&initial, vec![(IVec3::ZERO, page.clone())], &[]);
    assert!(Arc::ptr_eq(
        &initial.top_level.fields,
        &edited.top_level.fields
    ));
    assert!(!Arc::ptr_eq(
        &initial.regions[&IVec3::ZERO],
        &edited.regions[&IVec3::ZERO]
    ));
    assert_eq!(initial.material_at(IVec3::new(5, 5, 6)), Some(AIR));
    assert_eq!(edited.material_at(IVec3::new(5, 5, 6)), Some(2));
    let changes = super::super::geometry::SceneChanges::between(&initial, &edited);
    assert!(changes.changed && changes.transport.is_some());
    assert!(changes.changed_materials[2]);

    page[point] = AIR;
    page[point + 1] = AIR;
    let emptied = TransportScene::with_page_delta(&edited, vec![(IVec3::ZERO, page)], &[]);
    assert!(!Arc::ptr_eq(
        &edited.top_level.fields,
        &emptied.top_level.fields
    ));
    assert_eq!(edited.material_at(IVec3::new(5, 5, 6)), Some(2));
    assert_eq!(
        emptied.top_level.fields,
        TopLevelAadf::build(&emptied.regions).fields
    );
}

fn terrain_scene() -> (TransportScene, Vec<u16>) {
    // 1,024 resident pages represented by 8,192 regions. Uniform strata keep
    // setup compact; the edited page still goes through normal brick building.
    let mut regions = HashMap::new();
    for z in -16..16 {
        for y in -2..6 {
            for x in -16..16 {
                let coordinate = IVec3::new(x, y, z);
                let any_solid = y <= 0;
                regions.insert(
                    coordinate,
                    Arc::new(NaadfRegion {
                        coordinate,
                        storage: if any_solid {
                            RegionStorage::Uniform(1)
                        } else {
                            RegionStorage::Empty
                        },
                        any_solid,
                        material_mask: if any_solid { [2, 0, 0, 0] } else { [0; 4] },
                    }),
                );
            }
        }
    }
    let top_level = TopLevelAadf::build(&regions);
    let mut page = vec![AIR; volume_len(PAGE_SIZE)];
    for z in 0..PAGE_EDGE {
        for y in 0..REGION_EDGE {
            for x in 0..PAGE_EDGE {
                page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 1;
            }
        }
    }
    (TransportScene { regions, top_level }, page)
}

#[test]
#[ignore = "paired CPU scene timings; run without concurrent workloads"]
fn scene_topology_reuse_benchmark() {
    let (scene, mut page) = terrain_scene();
    page[linear_index(UVec3::new(8, 15, 8), PAGE_SIZE)] = AIR;
    let actual = TransportScene::with_page_delta(&scene, vec![(IVec3::ZERO, page.clone())], &[]);
    let reference = rebuild_on_any_change(&scene, vec![(IVec3::ZERO, page.clone())], &[]);
    assert_same_scene(&actual, &reference);
    let mut timings = [Vec::new(), Vec::new()];
    for pair in 0..44 {
        for variant in [pair % 2, 1 - pair % 2] {
            let delta = vec![(IVec3::ZERO, page.clone())];
            let start = Instant::now();
            let built = if variant == 0 {
                rebuild_on_any_change(black_box(&scene), delta, &[])
            } else {
                TransportScene::with_page_delta(black_box(&scene), delta, &[])
            };
            let elapsed = start.elapsed().as_secs_f64() * 1e3;
            black_box(built);
            if pair >= 4 {
                timings[variant].push(elapsed);
            }
        }
    }
    for times in &mut timings {
        times.sort_by(f64::total_cmp);
    }
    println!(
        "SCENE_EDIT regions={} pairs=40 ms_p50={:.3}->{:.3} ms_p95={:.3}->{:.3} exact=true",
        scene.regions.len(),
        timings[0][20],
        timings[1][20],
        timings[0][37],
        timings[1][37]
    );
}
