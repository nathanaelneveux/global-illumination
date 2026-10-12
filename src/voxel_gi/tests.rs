//! Integration contracts for scheduling and asynchronous publication.
use super::*;
use super::{
    diagnostics::toggle_voxel_gi,
    dispatch::{append_sky_work, select_probes, select_transport},
    publication::{invalidate_near_voxel, publish_scene_page},
    render::{VoxelTransportAssets, VoxelTransportMaterial},
    sources::{SourceChange, invalidate_model, invalidate_source_change},
};
use bevy::platform::collections::HashMap;
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;

#[test]
fn settled_counter_follows_accepted_transport_not_sky_only_or_rejected_batches() {
    use bevy::ecs::system::RunSystemOnce;
    let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
    let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
    probe.samples = [256; 6];
    probe.stable_batches = [3; 6];
    let mut field = VoxelEnergyField::default();
    field.probes.insert(IVec3::ZERO, probe);
    field.update_indices(&[IVec3::ZERO], &HashMap::new());
    let mut world = World::new();
    world.insert_resource(field);
    world.insert_resource(VoxelGiRuntime {
        active_faces: 1,
        covered_faces: 1,
        completed_samples: 8,
        ..default()
    });
    world.init_resource::<VoxelGiStats>();
    world.init_resource::<VoxelGiSettings>();
    world.init_resource::<GiWorkTimings>();
    for (case, expected) in [(0, 1.0), (1, 1.0), (2, 1.0), (3, 0.0), (4, 0.0)] {
        let mut probe = world.resource::<VoxelEnergyField>().probes[&IVec3::ZERO].clone();
        probe.trace_mask = 4;
        if case == 1 {
            probe.trace_mask = 0;
            probe.stable_batches = [0; 6];
        } else if case == 2 {
            probe.revision += 1;
            probe.stable_batches = [0; 6];
        } else if case == 3 {
            probe.radiance[2] = EnergySpectrum::ONE;
            probe.second_moment[2] = EnergySpectrum::ONE;
        } else if case == 4 {
            // A batch dispatched before a tuning change must not publish into
            // the new model. Geometry and the last display remain usable.
            world.resource_mut::<VoxelGiSettings>().transport.reach = 16.0;
            world
                .run_system_once(settings::synchronize_transport_settings)
                .unwrap();
            probe.radiance[2] = EnergySpectrum::ONE * 100.0;
        }
        let task = pool.spawn(async move {
            TraceBatchResult {
                work: TraceWork::from_probes(&[probe], case == 1),
                paths: 4,
                trace_time: Duration::from_millis(1),
                gpu_tracer: None,
                emission_cache: None,
                cpu_sky_cache: default(),
                gpu_error: None,
                sky_only: case == 1,
                transport_count: usize::from(case != 1),
                sky_count: usize::from(case == 1),
                scene_generation: 0,
                sky_revision: 0,
            }
        });
        let deadline = Instant::now() + Duration::from_secs(5);
        while !task.is_finished() {
            assert!(Instant::now() < deadline, "test batch did not finish");
            std::thread::yield_now();
        }
        world.resource_mut::<VoxelGiRuntime>().trace_task = Some(task);
        world.run_system_once(apply_trace_task).unwrap();
        assert_eq!(
            world.resource::<VoxelGiStats>().settled_coverage,
            expected,
            "case {case}"
        );
        if case == 4 {
            assert_eq!(
                world.resource::<VoxelEnergyField>().probes[&IVec3::ZERO].radiance[2],
                EnergySpectrum::ONE
            );
        }
    }
}

#[test]
fn settled_counter_tracks_local_edits_global_revisions_and_page_removal() {
    let mut field = VoxelEnergyField::default();
    let mut runtime = VoxelGiRuntime::default();
    let mut stats = VoxelGiStats::default();
    let far = IVec3::new(128, 0, 0);
    for voxel in [IVec3::ZERO, far] {
        let mut probe = SurfaceProbe::new(voxel, 5);
        probe.samples = [128; 6];
        probe.stable_batches = [4; 6];
        let (faces, samples, covered, settled) = probe_metrics(&probe, 0);
        runtime.active_faces += faces;
        runtime.completed_samples += samples;
        runtime.covered_faces += covered;
        runtime.settled_faces += settled;
        field.probes.insert(voxel, probe);
        field.pages.insert(page_of(voxel), vec![voxel]);
    }
    field.update_indices(&[IVec3::ZERO, far], &HashMap::new());
    update_stats_from_counters(&runtime, &field, &mut stats);
    assert_eq!(stats.settled_coverage, 1.0);
    invalidate_near_voxel(
        &mut runtime,
        &mut field,
        IVec3::ZERO,
        48,
        &mut HashSet::new(),
    );
    update_stats_from_counters(&runtime, &field, &mut stats);
    assert_eq!(stats.settled_coverage, 0.5);
    assert_eq!(stats.interactive_coverage, 1.0);
    assert_eq!(stats.bootstrap_coverage, 1.0);
    assert_eq!(field.probes[&IVec3::ZERO].samples[0], 16);

    // O(1) invalidation must not scan/clear histories. Old contributions no
    // longer count, including late worker results from that old revision.
    invalidate_source_change(&mut runtime, &mut field, &mut stats, SourceChange::Global);
    assert_eq!(stats.settled_coverage, 0.0);
    assert_eq!(field.probes[&far].samples[0], 128);
    assert_eq!(
        probe_metrics(&field.probes[&far], field.transport_revision).3,
        0
    );
    let current = field.probes.get_mut(&IVec3::ZERO).unwrap();
    current.transport_revision = field.transport_revision;
    current.samples = [128; 6];
    current.fresh_mask = current.face_mask;
    current.stable_batches = [4; 6];
    runtime.settled_faces = probe_metrics(current, field.transport_revision).3;
    let mut result = SceneBuildResult {
        scene: runtime.scene.clone(),
        changed: true,
        geometry_change: Some(default()),
        pages: VecDeque::new(),
        exact_edits: vec![],
        build_time: Duration::ZERO,
    };
    // Removing old-revision terrain must not subtract freshly settled faces
    // elsewhere. Republishing an unchanged current page must count it once.
    publish_scene_page(
        &mut runtime,
        &mut field,
        &mut stats,
        &mut result,
        page_of(far),
        vec![],
    );
    assert_eq!(runtime.settled_faces, 2);
    assert_eq!(stats.settled_coverage, 1.0);
    publish_scene_page(
        &mut runtime,
        &mut field,
        &mut stats,
        &mut result,
        IVec3::ZERO,
        vec![SurfaceProbeDescriptor {
            voxel: IVec3::ZERO,
            material: 0,
            face_mask: 5,
        }],
    );
    assert_eq!(runtime.settled_faces, 2);
    publish_scene_page(
        &mut runtime,
        &mut field,
        &mut stats,
        &mut result,
        IVec3::ZERO,
        vec![],
    );
    assert_eq!(runtime.settled_faces, 0);
    assert_eq!(stats.settled_coverage, 0.0);
}

#[test]
fn scene_publication_preserves_other_pages_and_publishes_one_page_per_tick() {
    use bevy::ecs::system::RunSystemOnce;
    let mut field = VoxelEnergyField::default();
    let old = IVec3::ZERO;
    let retained = IVec3::new(32, 0, 0);
    for voxel in [old, retained] {
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        field.pages.insert(page_of(voxel), vec![voxel]);
    }
    let retained_address = field.probes.get(&retained).unwrap() as *const SurfaceProbe;
    field.update_indices(&[old, retained], &HashMap::new());
    field.publishing_pages.extend([IVec3::ZERO, IVec3::X]);
    let added = IVec3::Y;
    let result = SceneBuildResult {
        scene: Arc::new(TransportScene::default()),
        changed: true,
        geometry_change: Some(default()),
        pages: VecDeque::from([
            (
                IVec3::ZERO,
                vec![SurfaceProbeDescriptor {
                    voxel: added,
                    material: 0,
                    face_mask: 4,
                }],
            ),
            (
                IVec3::X,
                vec![SurfaceProbeDescriptor {
                    voxel: retained,
                    material: 0,
                    face_mask: 4,
                }],
            ),
        ]),
        exact_edits: vec![],
        build_time: Duration::ZERO,
    };
    let mut world = World::new();
    world.insert_resource(field);
    world.insert_resource(VoxelGiRuntime {
        scene_publication: Some(result),
        ..default()
    });
    world.insert_resource(VoxelGiStats::default());
    world.init_resource::<GiWorkTimings>();
    world.insert_resource(VoxelGiSettings {
        scene_publish_budget: Duration::ZERO,
        ..default()
    });
    world.run_system_once(apply_scene_task).unwrap();
    let field = world.resource::<VoxelEnergyField>();
    assert!(!field.probes.contains_key(&old));
    assert!(field.probes.contains_key(&added));
    assert!(field.probes.contains_key(&retained));
    assert_eq!(field.publishing_pages, HashSet::from([IVec3::X]));
    for (index, voxel) in field.order.iter().enumerate() {
        assert_eq!(field.probes[voxel].order_index, index);
    }
    world.run_system_once(apply_scene_task).unwrap();
    assert_eq!(
        world
            .resource::<VoxelEnergyField>()
            .probes
            .get(&retained)
            .unwrap() as *const SurfaceProbe,
        retained_address,
        "unchanged probes must keep their allocations across page publication",
    );
    assert!(
        !world
            .resource::<VoxelGiRuntime>()
            .gpu_value_dirty
            .contains(&retained),
        "unchanged page receivers do not need display uploads"
    );
    assert!(
        world
            .resource::<VoxelEnergyField>()
            .publishing_pages
            .is_empty()
    );
    assert!(
        world
            .resource::<VoxelGiRuntime>()
            .scene_publication
            .is_none()
    );
}

#[test]
fn new_receiver_display_clock_does_not_depend_on_material_id() {
    for material in [0, 2] {
        let mut runtime = VoxelGiRuntime::default();
        let mut field = VoxelEnergyField::default();
        let mut stats = VoxelGiStats::default();
        let mut result = SceneBuildResult {
            scene: runtime.scene.clone(),
            changed: true,
            geometry_change: None,
            pages: default(),
            exact_edits: vec![],
            build_time: Duration::ZERO,
        };
        publish_scene_page(
            &mut runtime,
            &mut field,
            &mut stats,
            &mut result,
            IVec3::ZERO,
            vec![SurfaceProbeDescriptor {
                voxel: IVec3::ZERO,
                face_mask: 4,
                material,
            }],
        );
        let probe = &field.probes[&IVec3::ZERO];
        assert!(probe.display_updated_at.is_some());
        assert!(probe.updated_at.is_none());
        assert_eq!(probe.samples, [0; 6]);
        assert!(field.estimate(IVec3::ZERO, Vec3::Y).is_none());
    }
}

#[test]
fn material_replacement_retains_incident_history_but_rejects_old_jobs() {
    use bevy::ecs::system::RunSystemOnce;
    use world::{AIR, MaterialTransport};
    let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
    let voxel = IVec3::splat(8);
    for material in [2, 4] {
        for face_mask in [4, 5] {
            let now = Instant::now();
            let mut probe = SurfaceProbe::new(voxel, 4);
            probe.material = 3;
            probe.samples = [1024; 6];
            probe.sequence = [2048; 6];
            probe.radiance = [EnergySpectrum::ONE; 6];
            probe.second_moment = [EnergySpectrum::ONE * 2.0; 6];
            probe.display[2].indirect = Vec3::splat(0.5);
            probe.display[2].confidence = 0.75;
            probe.updated_at = Some(now);
            probe.face_updated_at = [Some(now); 6];
            probe.display_updated_at = Some(now);
            probe.revision = 7;
            probe.stable_batches = [4; 6];
            let mut late = probe.clone();
            late.radiance = [EnergySpectrum::ONE * 100.0; 6];
            let mut field = VoxelEnergyField::default();
            field.probes.insert(voxel, probe);
            field.pages.insert(IVec3::ZERO, vec![voxel]);
            field.update_indices(&[voxel], &HashMap::new());
            let mut materials = VoxelMaterialTransportTable::default();
            materials.set(
                4,
                MaterialTransport {
                    emission: EnergySpectrum::ONE * 8.0,
                    ..default()
                },
            );
            let mut voxels = vec![AIR; 32 * 32 * 32];
            voxels[8 + 32 * (8 + 32 * 8)] = u16::from(material);
            let scene = Arc::new(TransportScene::with_page_delta(
                &default(),
                vec![(IVec3::ZERO, voxels)],
                &[],
            ));
            let (faces, samples, covered, settled) = probe_metrics(&field.probes[&voxel], 0);
            let mut runtime = VoxelGiRuntime {
                scene: scene.clone(),
                materials: Arc::new(materials),
                active_faces: faces,
                completed_samples: samples,
                covered_faces: covered,
                settled_faces: settled,
                ..default()
            };
            let mut stats = VoxelGiStats::default();
            let mut result = SceneBuildResult {
                scene,
                changed: true,
                geometry_change: None,
                pages: default(),
                exact_edits: vec![],
                build_time: Duration::ZERO,
            };
            publish_scene_page(
                &mut runtime,
                &mut field,
                &mut stats,
                &mut result,
                IVec3::ZERO,
                vec![SurfaceProbeDescriptor {
                    voxel,
                    face_mask,
                    material,
                }],
            );
            let p = &field.probes[&voxel];
            assert!(p.revision > late.revision);
            assert_eq!(p.samples[2], geometry::NEAR_HISTORY);
            assert_eq!(p.sequence[2], 2048);
            assert_eq!(p.radiance[2], EnergySpectrum::ONE);
            assert_eq!(p.second_moment[2], EnergySpectrum::ONE * 2.0);
            assert_eq!(p.face_updated_at[2], Some(now));
            assert_eq!(p.display_updated_at, Some(now));
            assert_eq!(p.display[2].indirect, Vec3::splat(0.5));
            assert_eq!(p.display[2].confidence, 0.75);
            assert_eq!(p.emission, runtime.materials.get(material).emission);
            assert_eq!(p.fresh_mask, 0);
            assert_eq!(p.stable_batches, [0; 6]);
            let estimate = field.estimate(voxel, Vec3::Y).unwrap();
            assert!(!estimate.transport_current && !estimate.sources_current);
            assert_eq!(
                estimate.incident,
                EnergySpectrum::ONE,
                "self-emission is not incoming energy"
            );
            if face_mask == 5 {
                assert_eq!(p.samples[0], 0);
                assert_eq!(p.radiance[0], EnergySpectrum::ZERO);
                assert!(field.estimate(voxel, Vec3::X).is_none());
            }
            let task = pool.spawn(async move {
                TraceBatchResult {
                    work: TraceWork::from_probes(&[late], false),
                    paths: 4,
                    trace_time: Duration::from_millis(1),
                    gpu_tracer: None,
                    emission_cache: None,
                    cpu_sky_cache: default(),
                    gpu_error: None,
                    sky_only: false,
                    transport_count: 1,
                    sky_count: 0,
                    scene_generation: 0,
                    sky_revision: 0,
                }
            });
            let deadline = Instant::now() + Duration::from_secs(5);
            while !task.is_finished() {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
            runtime.trace_task = Some(task);
            let mut world = World::new();
            world.insert_resource(runtime);
            world.insert_resource(field);
            world.insert_resource(stats);
            world.init_resource::<VoxelGiSettings>();
            world.init_resource::<GiWorkTimings>();
            world.run_system_once(apply_trace_task).unwrap();
            assert_eq!(
                world.resource::<VoxelEnergyField>().probes[&voxel].radiance[2],
                EnergySpectrum::ONE,
                "pre-replacement jobs must not overwrite retained incident energy"
            );
        }
    }
}

#[test]
fn scene_publication_retains_the_same_spatial_history_for_emitters_and_occluders() {
    use bevy::ecs::system::RunSystemOnce;
    use world::{AIR, MaterialTransport};
    let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
    let near = IVec3::new(8, 4, 8);
    let far = IVec3::new(136, 4, 8);
    let edit = IVec3::new(8, 8, 8);
    for material in [2, 4] {
        let mut voxels = vec![AIR; 32 * 32 * 32];
        let index = |p: IVec3| (p.x + 32 * (p.y + 32 * p.z)) as usize;
        voxels[index(near)] = 2;
        voxels[index(edit)] = material;
        let mut remote = vec![AIR; voxels.len()];
        remote[index(far - IVec3::X * 128)] = 2;
        let previous = TransportScene::with_page_delta(
            &default(),
            vec![(IVec3::ZERO, voxels.clone()), (IVec3::X * 4, remote)],
            &[],
        );
        voxels[index(edit)] = AIR;
        let scene = Arc::new(TransportScene::with_page_delta(
            &previous,
            vec![(IVec3::ZERO, voxels)],
            &[],
        ));
        let changes = geometry::SceneChanges::between(&previous, &scene);
        let result = SceneBuildResult {
            scene,
            changed: changes.changed,
            geometry_change: changes.transport,
            // No receiver topology changed. This exercises scene acceptance,
            // not the separate page-merge/seeding tests above.
            pages: default(),
            exact_edits: vec![],
            build_time: Duration::ZERO,
        };
        let task = pool.spawn(async move { result });
        let deadline = Instant::now() + Duration::from_secs(5);
        while !task.is_finished() {
            assert!(Instant::now() < deadline, "test scene did not finish");
            std::thread::yield_now();
        }
        let mut field = VoxelEnergyField::default();
        for voxel in [near, far] {
            let mut probe = SurfaceProbe::new(voxel, 4);
            probe.samples = [4096; 6];
            probe.radiance = [EnergySpectrum::ONE; 6];
            probe.second_moment = [EnergySpectrum::ONE; 6];
            probe.sequence = [4096; 6];
            probe.display[2].indirect = Vec3::ONE;
            probe.updated_at = Some(Instant::now());
            probe.face_updated_at[2] = probe.updated_at;
            field.probes.insert(voxel, probe);
            field.pages.insert(page_of(voxel), vec![voxel]);
        }
        field.update_indices(&[near, far], &HashMap::new());
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            4,
            MaterialTransport {
                emission: EnergySpectrum::ONE,
                ..default()
            },
        );
        let mut world = World::new();
        world.insert_resource(field);
        world.insert_resource(VoxelGiRuntime {
            scene: Arc::new(previous),
            scene_task: Some(task),
            materials: Arc::new(materials),
            ..default()
        });
        world.init_resource::<VoxelGiStats>();
        world.init_resource::<VoxelGiSettings>();
        world.init_resource::<GiWorkTimings>();
        world.run_system_once(apply_scene_task).unwrap();
        world.resource_scope(|world, mut runtime: Mut<VoxelGiRuntime>| {
            let field = world.resource::<VoxelEnergyField>();
            assert_eq!(runtime.source_revision, 0);
            assert_eq!(runtime.model_reset_source_revision, 0);
            assert_eq!(field.transport_revision, 1);
            for voxel in [near, far] {
                let estimate = field.estimate(voxel, Vec3::Y).unwrap();
                assert!(estimate.sources_current);
                assert!(!estimate.transport_current);
                assert_eq!(estimate.effective_samples, 4096);
            }
            let selected = select_probes(&mut runtime, field);
            assert_eq!(selected.len(), 2);
            for probe in selected {
                let retained = if probe.voxel == near { 16 } else { 256 };
                assert_eq!(probe.samples[2], retained, "material {material}");
                assert_eq!(probe.radiance[2], EnergySpectrum::ONE);
                assert_eq!(probe.sequence[2], 4096);
                assert_eq!(probe.display[2].indirect, Vec3::ONE);
            }
        });
    }
}

#[test]
fn publication_batches_cheap_pages_without_unbounded_display_work() {
    use bevy::ecs::system::RunSystemOnce;
    let pages: VecDeque<_> = (0..6)
        .map(|x| {
            let page = IVec3::new(x, 0, 0);
            (
                page,
                vec![SurfaceProbeDescriptor {
                    voxel: page * PAGE_EDGE as i32,
                    material: 0,
                    face_mask: 4,
                }],
            )
        })
        .collect();
    let mut field = VoxelEnergyField::default();
    field
        .publishing_pages
        .extend(pages.iter().map(|(page, _)| *page));
    let mut world = World::new();
    world.insert_resource(field);
    world.insert_resource(VoxelGiRuntime {
        scene_publication: Some(SceneBuildResult {
            scene: Arc::new(TransportScene::default()),
            changed: true,
            geometry_change: Some(default()),
            pages,
            exact_edits: vec![],
            build_time: Duration::ZERO,
        }),
        ..default()
    });
    world.insert_resource(VoxelGiStats::default());
    world.init_resource::<GiWorkTimings>();
    world.insert_resource(VoxelGiSettings {
        scene_publish_budget: Duration::MAX,
        ..default()
    });
    world.run_system_once(apply_scene_task).unwrap();
    let published = world.resource::<VoxelEnergyField>().probe_count();
    assert!(
        published <= 4,
        "downstream topology patches have a page-count cap"
    );
    // A delayed test runner may exhaust the wall-clock budget sooner; every
    // invocation still advances, and the queue must preserve all six pages.
    assert!(published > 0);
    for _ in 0..6 {
        world.run_system_once(apply_scene_task).unwrap();
    }
    let field = world.resource::<VoxelEnergyField>();
    assert_eq!(field.probe_count(), 6);
    assert!(field.publishing_pages.is_empty());
    assert!(
        world
            .resource::<VoxelGiRuntime>()
            .scene_publication
            .is_none()
    );
}

#[test]
fn urgent_work_cannot_starve_loaded_world_sweep() {
    let mut field = VoxelEnergyField::default();
    for x in 0..64 {
        let voxel = IVec3::new(x, 0, 0);
        field.order.push(voxel);
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
    }
    let mut runtime = VoxelGiRuntime {
        batch_size: 8,
        nearby: vec![IVec3::ZERO],
        ..default()
    };
    let mut visited = HashSet::new();
    for _ in 0..32 {
        request_global_refresh(&mut runtime, field.order.len());
        enqueue_urgent(&mut runtime, (0..6).map(|x| IVec3::new(x, 0, 0)));
        visited.extend(
            select_probes(&mut runtime, &field)
                .into_iter()
                .map(|probe| probe.voxel),
        );
    }
    assert_eq!(visited.len(), 64);
}

#[test]
fn refinement_regions_keep_close_window_and_global_sweep() {
    let mut field = VoxelEnergyField::default();
    let focus = IVec3::new(1, -1, -1);
    for page in [focus, focus + IVec3::X * 2, focus + IVec3::X * 3] {
        let voxel = page * 32;
        field.order.push(voxel);
        field.pages.insert(page, vec![voxel]);
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
    }
    let mut runtime = VoxelGiRuntime::default();
    dispatch::update_focus(&mut runtime, &field, focus);
    assert_eq!(runtime.nearby, [focus * 32]);
    assert_eq!(runtime.regional, [(focus + IVec3::X * 2) * 32]);
    let mut seen = HashSet::new();
    for _ in 0..12 {
        seen.extend(
            select_probes(&mut runtime, &field)
                .into_iter()
                .map(|p| p.voxel),
        );
    }
    assert_eq!(seen.len(), field.order.len());
}

#[test]
fn distant_streaming_preserves_focus_indices_and_sweep_progress() {
    let mut field = VoxelEnergyField::default();
    let mut runtime = VoxelGiRuntime::default();
    let focus = IVec3::new(-2, 1, 3);
    field.pages.insert(focus, vec![focus * 32]);
    dispatch::update_focus(&mut runtime, &field, focus);
    runtime.nearby_cursor = 127;
    for page in [focus + IVec3::X * 3, focus - IVec3::Z * 50] {
        field.pages.insert(page, vec![page * 32]);
        dispatch::invalidate_focus_page(&mut runtime, page);
        assert!(!runtime.focus_dirty);
        dispatch::update_focus(&mut runtime, &field, focus);
        assert_eq!(runtime.nearby, [focus * 32]);
        assert_eq!(runtime.nearby_cursor, 127);
    }
    let ring = focus + IVec3::X * 2;
    field.pages.insert(ring, vec![ring * 32]);
    dispatch::invalidate_focus_page(&mut runtime, ring);
    assert!(runtime.focus_dirty);
    dispatch::update_focus(&mut runtime, &field, focus);
    assert_eq!(runtime.regional, [ring * 32]);
    assert_eq!(runtime.nearby_cursor, 127);
    field.pages.remove(&focus);
    dispatch::invalidate_focus_page(&mut runtime, focus);
    dispatch::update_focus(&mut runtime, &field, focus);
    assert!(runtime.nearby.is_empty());
    dispatch::update_focus(&mut runtime, &field, ring);
    assert_eq!(runtime.nearby_cursor, 0);
    assert!(runtime.nearby.contains(&(ring * 32)));
}

#[test]
fn in_flight_lighting_cannot_restore_indices_moved_by_streaming() {
    let mut field = VoxelEnergyField::default();
    let voxels = [IVec3::ZERO, IVec3::X, IVec3::Y, IVec3::Z];
    for voxel in voxels {
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
    }
    field.update_indices(&voxels, &HashMap::new());
    let pending_result =
        TraceWork::from_probes(std::slice::from_ref(&field.probes[&IVec3::Z]), false);
    let mut removed = HashMap::new();
    removed.insert(IVec3::ZERO, field.probes.remove(&IVec3::ZERO).unwrap());
    field.update_indices(&[], &removed);
    assert_eq!(field.probes[&IVec3::Z].order_index, 0);
    pending_result.probes[0].publish(
        &pending_result.faces,
        field.probes.get_mut(&IVec3::Z).unwrap(),
        Instant::now(),
        &VoxelGiSettings::default(),
    );
    removed.clear();
    removed.insert(IVec3::Z, field.probes.remove(&IVec3::Z).unwrap());
    field.update_indices(&[], &removed);
    assert_eq!(field.order.len(), field.probes.len());
    for (index, voxel) in field.order.iter().enumerate() {
        assert_eq!(field.probes[voxel].order_index, index);
    }
}

#[test]
fn moving_sources_preserve_history_and_work_queues() {
    let voxel = IVec3::ZERO;
    let mut probe = SurfaceProbe::new(voxel, 4);
    probe.samples[2] = 64;
    probe.sequence[2] = 100;
    let mut field = VoxelEnergyField::default();
    field.probes.insert(voxel, probe);
    field.order.push(voxel);
    let mut runtime = VoxelGiRuntime::default();
    enqueue_many(&mut runtime, [voxel]);
    invalidate_source_change(
        &mut runtime,
        &mut field,
        &mut VoxelGiStats::default(),
        SourceChange::Global,
    );
    assert_eq!(field.probes[&voxel].samples[2], 64);
    assert_eq!(runtime.pending.len(), 1);
    let selected = select_probes(&mut runtime, &field);
    assert_eq!(selected[0].samples[2], 4);
    assert_eq!(selected[0].sequence[2], 100);
}

#[test]
fn global_model_replacement_restarts_measurements_not_presentation() {
    let mut field = VoxelEnergyField::default();
    for voxel in [IVec3::ZERO, IVec3::new(10_000, 0, 0)] {
        let mut probe = SurfaceProbe::new(voxel, 5);
        probe.samples = [128; 6];
        probe.radiance = [EnergySpectrum::ONE; 6];
        probe.second_moment = [EnergySpectrum::ONE; 6];
        probe.sequence = [1000; 6];
        probe.display[0].indirect = Vec3::ONE;
        field.probes.insert(voxel, probe);
        field.order.push(voxel);
    }
    let mut runtime = VoxelGiRuntime::default();
    invalidate_model(&mut runtime, &mut field, &mut VoxelGiStats::default());
    // A later moving-light revision must not hide the pending model reset.
    invalidate_source_change(
        &mut runtime,
        &mut field,
        &mut VoxelGiStats::default(),
        SourceChange::Global,
    );
    let selected = select_probes(&mut runtime, &field);
    assert_eq!(selected.len(), 2);
    for probe in selected {
        assert_eq!(probe.samples, [0; 6]);
        assert_eq!(probe.radiance, [EnergySpectrum::ZERO; 6]);
        assert_eq!(probe.second_moment, [EnergySpectrum::ZERO; 6]);
        assert_eq!(probe.display[0].indirect, Vec3::ONE);
        assert_eq!(probe.sequence, [1000; 6]);
        assert_eq!(probe.revision, 0, "worker result must remain mergeable");
        assert_eq!(probe.source_revision, runtime.source_revision);
    }
}

#[test]
fn remote_local_light_motion_wakes_without_destroying_transport_history() {
    let voxel = IVec3::new(10_000, 0, 0);
    let mut probe = SurfaceProbe::new(voxel, 4);
    probe.samples[2] = 128;
    probe.sequence[2] = 1000;
    probe.updated_at = Some(Instant::now());
    probe.face_updated_at[2] = probe.updated_at;
    let mut field = VoxelEnergyField::default();
    field.probes.insert(voxel, probe);
    field.order.push(voxel);
    let mut runtime = VoxelGiRuntime::default();
    sources::invalidate_source_change(
        &mut runtime,
        &mut field,
        &mut VoxelGiStats::default(),
        sources::SourceChange::Local(vec![(Vec3::ZERO, 24.0)]),
    );
    let selected = select_probes(&mut runtime, &field);
    assert_eq!(selected[0].samples[2], 64);
    assert_eq!(selected[0].sequence[2], 1000);
    assert_eq!(selected[0].fresh_mask, 0);
    assert_eq!(selected[0].trace_mask, 4);
    assert!(!field.estimate(voxel, Vec3::Y).unwrap().transport_current);
    assert_eq!(field.probes[&voxel].samples[2], 128);
}

#[test]
fn sky_queue_piggybacks_without_duplicating_rays_or_changing_history() {
    let initial = SurfaceProbe::new(IVec3::ZERO, 4);
    let mut transport = TraceWork::from_probes(std::slice::from_ref(&initial), false);
    transport.probes[0].sky_requested = false;
    let mut sky_only = SurfaceProbe::new(IVec3::X, 5);
    sky_only.samples = [128; 6];
    sky_only.sequence = [1000; 6];
    let duplicate = initial;
    let mut field = VoxelEnergyField::default();
    for probe in [duplicate, sky_only] {
        field.probes.insert(probe.voxel, probe);
    }
    let mut runtime = VoxelGiRuntime::default();
    // Includes an already-selected transport receiver and a repeated sky
    // candidate. Neither should allocate or overwrite another probe clone.
    runtime
        .sky_pending
        .extend([IVec3::ZERO, IVec3::X, IVec3::X]);
    assert_eq!(append_sky_work(&mut runtime, &field, &mut transport), 1);
    assert_eq!(transport.probes.len(), 2);
    assert_eq!(transport.probes[0].trace_mask, 4);
    assert!(transport.probes.iter().all(|job| job.sky_requested));
    assert_eq!(transport.probes[1].face_mask, 5);
    assert_eq!(transport.probes[1].trace_mask, 0);
    assert_eq!(
        transport.faces.len(),
        1,
        "sky-only work carries no transport history"
    );
    assert_eq!(field.probes[&IVec3::X].samples, [128; 6]);
    assert_eq!(field.probes[&IVec3::X].sequence, [1000; 6]);
}

#[test]
fn indirect_work_requests_only_unknown_sky_without_the_explicit_queue() {
    let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
    let push = |probe: &SurfaceProbe| {
        let mut work = TraceWork::default();
        work.push(probe, 4, 0, 0, false, 64, u32::MAX);
        work
    };
    assert!(push(&probe).probes[0].sky_requested);
    probe.display[2].sky = [1; 4]; // measured black is valid, not missing
    assert!(!push(&probe).probes[0].sky_requested);
    let work = push(&probe);
    work.copy_measurements(std::slice::from_mut(&mut probe));
    assert_eq!(
        probe.display[2].sky, [1; 4],
        "unrequested zero output must never replace valid sky"
    );
    probe.display[2].sky[3] = 0; // incomplete/unknown ray must retry
    assert!(push(&probe).probes[0].sky_requested);
}

#[test]
fn dispatch_reuses_returned_storage_without_retaining_candidates() {
    let mut field = VoxelEnergyField::default();
    field.order.push(IVec3::ZERO);
    field
        .probes
        .insert(IVec3::ZERO, SurfaceProbe::new(IVec3::ZERO, 4));
    let mut runtime = VoxelGiRuntime {
        trace_scratch: TraceWork {
            probes: Vec::with_capacity(1024),
            faces: Vec::with_capacity(1024),
            ..default()
        },
        batch_size: 64,
        ..default()
    };
    let allocation = runtime.trace_scratch.probes.as_ptr();
    let face_allocation = runtime.trace_scratch.faces.as_ptr();
    for _ in 0..2 {
        let mut selected = select_transport(&mut runtime, &field, &VoxelGiSettings::default());
        assert_eq!(selected.probes.as_ptr(), allocation);
        assert_eq!(selected.faces.as_ptr(), face_allocation);
        assert_eq!(selected.probes.len(), 1);
        selected.clear();
        runtime.trace_scratch = selected;
    }
}

#[test]
fn resting_candidates_do_not_consume_the_dispatch_allowance() {
    let now = Instant::now();
    let mut field = VoxelEnergyField::default();
    for x in 0..4096 {
        let voxel = IVec3::new(x, 0, 0);
        let mut probe = SurfaceProbe::new(voxel, 4);
        probe.samples[2] = 256;
        probe.stable_batches[2] = 4;
        probe.face_updated_at[2] = Some(now);
        field.order.push(voxel);
        field.probes.insert(voxel, probe);
    }
    let noisy = IVec3::new(3000, 0, 0);
    field.probes.get_mut(&noisy).unwrap().stable_batches[2] = 0;
    let mut runtime = VoxelGiRuntime {
        batch_size: 64,
        ..default()
    };
    // A previously requested bootstrap queue can contain newly settled
    // probes; this also places the distant target beyond the ranked window.
    enqueue_many(&mut runtime, field.order.iter().take(64).copied());
    let selected = select_probes(&mut runtime, &field);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].voxel, noisy);
    assert_eq!(selected[0].samples[2], 256);
}

#[test]
fn sky_comparison_is_display_only_and_independent_of_gi_toggle() {
    let mut app = App::new();
    let mut materials = Assets::<VoxelTransportMaterial>::default();
    let handle = materials.add(render::material_template(&VoxelTransportAssets {
        texture: default(),
        hash_entries: default(),
        probes: default(),
    }));
    let mut runtime = VoxelGiRuntime {
        source_revision: 7,
        generation: 9,
        ..default()
    };
    enqueue_urgent(&mut runtime, [IVec3::ZERO]);
    app.insert_resource(runtime)
        .insert_resource(materials)
        .insert_resource(VoxelWorldMaterialHandle {
            handle: handle.clone(),
        })
        .init_resource::<ButtonInput<KeyCode>>()
        .add_systems(Update, toggle_voxel_gi);
    let window = app
        .world_mut()
        .spawn((Window::default(), bevy::window::PrimaryWindow))
        .id();
    for (key, sky, gi) in [
        (KeyCode::F7, 0, 1),
        (KeyCode::F4, 0, 0),
        (KeyCode::F7, 1, 0),
        (KeyCode::F4, 1, 1),
    ] {
        let mut keyboard = app.world_mut().resource_mut::<ButtonInput<KeyCode>>();
        keyboard.reset_all();
        keyboard.press(key);
        app.update();
        let settings = app
            .world()
            .resource::<Assets<VoxelTransportMaterial>>()
            .get(&handle)
            .unwrap()
            .extension
            .settings;
        assert_eq!(settings.voxel_sky, sky);
        assert_eq!(settings.enabled, gi);
        assert_eq!(settings.reconstruction, 1);
        let runtime = app.world().resource::<VoxelGiRuntime>();
        assert_eq!(runtime.source_revision, 7);
        assert_eq!(runtime.generation, 9);
        assert_eq!(runtime.urgent_pending.len(), 1);
    }
    assert!(
        app.world()
            .get::<Window>(window)
            .unwrap()
            .title
            .contains("voxel corners")
    );
}

#[test]
fn simulation_rejects_unmeasured_or_buried_faces_and_reports_stale_sources() {
    let voxel = IVec3::ZERO;
    let mut field = VoxelEnergyField::default();
    field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
    assert!(field.estimate(voxel, Vec3::Y).is_none());
    let probe = field.probes.get_mut(&voxel).unwrap();
    probe.samples[2] = 8;
    probe.updated_at = Some(Instant::now());
    assert!(field.estimate(voxel, Vec3::Y).unwrap().sources_current);
    assert!(field.estimate(voxel, Vec3::X).is_none());
    field.source_revision += 1;
    assert!(!field.estimate(voxel, Vec3::Y).unwrap().sources_current);
    let probe = field.probes.get_mut(&voxel).unwrap();
    probe.sequence[2] = 100;
    probe.restart_measurements();
    assert_eq!(probe.sequence[2], 100);
    assert!(field.estimate(voxel, Vec3::Y).is_none());
}

#[test]
fn partial_face_queries_keep_independent_age_and_revision_freshness() {
    let now = Instant::now();
    let voxel = IVec3::ZERO;
    let mut probe = SurfaceProbe::new(voxel, 5);
    probe.samples = [64; 6];
    probe.updated_at = Some(now);
    probe.face_updated_at[0] = Some(now - Duration::from_secs(10));
    probe.face_updated_at[2] = Some(now);
    probe.fresh_mask = 4;
    probe.transport_revision = 1;
    let mut field = VoxelEnergyField {
        transport_revision: 1,
        ..default()
    };
    field.probes.insert(voxel, probe);
    assert!(field.estimate(voxel, Vec3::Y).unwrap().transport_current);
    let old = field.estimate(voxel, Vec3::X).unwrap();
    assert!(!old.transport_current);
    assert!(!old.sources_current);
    assert!(old.age >= Duration::from_secs(10));
    field.transport_revision += 1;
    assert!(!field.estimate(voxel, Vec3::Y).unwrap().transport_current);
}

#[test]
fn settled_scheduler_can_be_idle_and_far_geometry_invalidation_wakes_it() {
    let now = Instant::now();
    let mut field = VoxelEnergyField::default();
    for x in 0..64 {
        let voxel = IVec3::new(10_000 + x, 0, 0);
        let mut probe = SurfaceProbe::new(voxel, 5);
        probe.samples = [256; 6];
        probe.stable_batches = [4; 6];
        probe.updated_at = Some(now);
        probe.face_updated_at = [Some(now); 6];
        field.order.push(voxel);
        field.probes.insert(voxel, probe);
    }
    let mut runtime = VoxelGiRuntime::default();
    assert!(select_probes(&mut runtime, &field).is_empty());
    let requested = field.order[runtime.refresh_cursor % field.order.len()];
    enqueue_urgent(&mut runtime, [requested]);
    let forced = select_probes(&mut runtime, &field);
    assert_eq!(
        forced.len(),
        1,
        "the reserved sweep must not swallow a simulation refresh request"
    );
    assert_eq!(forced[0].voxel, requested);
    assert_eq!(forced[0].trace_mask, 5);
    field.transport_revision += 1;
    let selected = select_probes(&mut runtime, &field);
    assert!(!selected.is_empty());
    let mut visited: HashSet<_> = selected.iter().map(|probe| probe.voxel).collect();
    for _ in 0..8 {
        visited.extend(
            select_probes(&mut runtime, &field)
                .iter()
                .map(|probe| probe.voxel),
        );
    }
    assert_eq!(visited.len(), 64);
    assert!(
        selected
            .iter()
            .all(|probe| probe.trace_mask == 5 && probe.fresh_mask == 0 && probe.samples[2] == 64)
    );
    assert!(
        field.probes.values().all(|probe| probe.samples[2] == 256),
        "scheduling must not mutate published histories"
    );
    enqueue_sky(&mut runtime, field.order.iter().copied());
    let mut sky = TraceWork::default();
    append_sky_work(&mut runtime, &field, &mut sky);
    assert_eq!(sky.probes.len(), 64);
    assert!(sky.faces.is_empty());
    assert!(field.probes.values().all(|probe| probe.samples[2] == 256));
}

#[test]
fn energy_field_uses_directional_ambient_cube_weights() {
    let voxel = IVec3::new(2, 3, 4);
    let mut field = VoxelEnergyField::default();
    field.probes.insert(
        voxel,
        SurfaceProbe {
            voxel,
            face_mask: 0b11_1111,
            radiance: [
                EnergySpectrum::new(Vec3::X, 1.0, 0.0),
                EnergySpectrum::ZERO,
                EnergySpectrum::new(Vec3::Y, 0.0, 2.0),
                EnergySpectrum::ZERO,
                EnergySpectrum::new(Vec3::Z, 0.0, 0.0),
                EnergySpectrum::ZERO,
            ],
            direct: [EnergySpectrum::ZERO; 6],
            samples: [1; 6],
            updated_at: Some(Instant::now()),
            ..SurfaceProbe::new(voxel, 0b11_1111)
        },
    );
    let energy = field.sample(voxel, Vec3::new(1.0, 1.0, 0.0)).unwrap();
    assert!(energy.visible.abs_diff_eq(Vec3::new(0.5, 0.5, 0.0), 1e-6));
    assert!((energy.infrared - 0.5).abs() < 1e-6);
    assert!((energy.ionizing - 1.0).abs() < 1e-6);
}
