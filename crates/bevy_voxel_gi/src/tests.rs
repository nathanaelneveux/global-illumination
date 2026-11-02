//! Integration contracts for scheduling and asynchronous publication.
use super::*;
use super::{
    dispatch::{select_probes, select_transport},
    publication::publish_scene_page,
    sources::{SourceChange, invalidate_source_change},
};
use bevy::platform::collections::HashMap;

#[test]
fn transport_failure_missing_device_does_not_select_cpu_implicitly() {
    // No device: the world cache idles (with a warning). There is no CPU
    // tracer to fall back to, and the selected work stays queued.
    use bevy::ecs::system::RunSystemOnce;
    let mut world = World::new();
    world.init_resource::<ReceiverField>();
    world
        .resource_mut::<ReceiverField>()
        .order
        .push(IVec3::ZERO);
    world.init_resource::<VoxelGiRuntime>();
    world.init_resource::<VoxelGiSettings>();
    world.init_resource::<VoxelGiStats>();
    world.init_resource::<GiWorkTimings>();
    world.run_system_once(start_trace_task).unwrap();
    assert!(
        world.resource::<VoxelGiRuntime>().trace_task.is_none(),
        "no trace started"
    );
    assert_eq!(
        world.resource::<ReceiverField>().order,
        vec![IVec3::ZERO],
        "work kept for a device"
    );
}

#[test]
fn transport_build_failure_idles_like_a_missing_device() {
    // A tracer whose pipelines fail to build (FXC on DX12) means this device
    // cannot run GI: warn once and idle, no panic and no retry every frame.
    use bevy::ecs::system::RunSystemOnce;
    let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
    let task = pool.spawn(async {
        TraceBatchResult {
            geometry: None,
            batch: None,
            pending: default(),
            bindings: default(),
            trace_time: Duration::ZERO,
            gpu_tracer: None,
            emission_cache: None,
            gpu_error: None,
            build_error: Some("injected pipeline build failure".into()),
        }
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while !task.is_finished() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let mut world = World::new();
    world.init_resource::<ReceiverField>();
    world
        .resource_mut::<ReceiverField>()
        .order
        .push(IVec3::ZERO);
    world.init_resource::<VoxelGiRuntime>();
    world.resource_mut::<VoxelGiRuntime>().trace_task = Some(task);
    world.init_resource::<VoxelGiSettings>();
    world.init_resource::<VoxelGiStats>();
    world.init_resource::<GiWorkTimings>();
    world.run_system_once(apply_trace_task).unwrap();
    assert!(world.resource::<VoxelGiRuntime>().transport_unavailable);
    world.run_system_once(start_trace_task).unwrap();
    assert!(
        world.resource::<VoxelGiRuntime>().trace_task.is_none(),
        "no rebuild attempt"
    );
    assert_eq!(
        world.resource::<ReceiverField>().order,
        vec![IVec3::ZERO],
        "work kept, as with no device"
    );
}

#[test]
#[should_panic(expected = "Voxel GI GPU transport failed: injected readback failure")]
fn transport_failure_worker_error_is_fatal_at_publication() {
    use bevy::ecs::system::RunSystemOnce;
    let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
    let task = pool.spawn(async {
        TraceBatchResult {
            geometry: None,
            batch: None,
            pending: default(),
            bindings: default(),
            trace_time: Duration::ZERO,
            gpu_tracer: None,
            emission_cache: None,
            gpu_error: Some("injected readback failure".into()),
            build_error: None,
        }
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while !task.is_finished() {
        assert!(Instant::now() < deadline);
        std::thread::yield_now();
    }
    let mut world = World::new();
    world.init_resource::<ReceiverField>();
    world.init_resource::<VoxelGiRuntime>();
    world.resource_mut::<VoxelGiRuntime>().trace_task = Some(task);
    world.init_resource::<VoxelGiSettings>();
    world.init_resource::<VoxelGiStats>();
    world.init_resource::<GiWorkTimings>();
    world.run_system_once(apply_trace_task).unwrap();
}

#[test]
fn settled_counter_follows_accepted_transport_not_untraced_or_rejected_batches() {
    use bevy::ecs::system::RunSystemOnce;
    let pool = bevy::tasks::TaskPoolBuilder::new().num_threads(1).build();
    // Settled is the GPU's verdict, carried by each merged face.
    let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
    probe.stats_mut().samples = [256; 6];
    let mut field = ReceiverField::default();
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
        let mut probe = world.resource::<ReceiverField>().probes[&IVec3::ZERO].clone();
        probe.trace_mask = 4;
        if case == 1 {
            probe.trace_mask = 0;
        } else if case == 2 {
            probe.revision += 1;
        } else if case == 3 {
            probe.stats_mut().radiance[2] = Vec3::ONE;
            probe.stats_mut().second_moment[2] = Vec3::ONE;
        } else if case == 4 {
            // A batch dispatched before a tuning change must not publish into
            // the new model. Geometry and the last display remain usable.
            world.resource_mut::<VoxelGiSettings>().transport.reach = 16.0;
            world
                .run_system_once(settings::synchronize_transport_settings)
                .unwrap();
            probe.stats_mut().radiance[2] = Vec3::ONE * 100.0;
        }
        let mut work = TraceWork::from_probes(&[probe]);
        for face in &mut work.faces {
            face.gpu_settled = Some(case != 3);
        }
        let task = pool.spawn(async move {
            TraceBatchResult {
                geometry: None,
                batch: Some(TransportBatch {
                    work,
                    paths: 4,
                    ..default()
                }),
                pending: default(),
                bindings: default(),
                trace_time: Duration::from_millis(1),
                gpu_tracer: None,
                emission_cache: None,
                gpu_error: None,
                build_error: None,
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
                world.resource::<ReceiverField>().probes[&IVec3::ZERO]
                    .stats()
                    .radiance[2],
                Vec3::ONE
            );
        }
    }
}

#[test]
fn settled_counter_tracks_global_revisions_and_page_removal() {
    let mut field = ReceiverField::default();
    let mut runtime = VoxelGiRuntime::default();
    let mut stats = VoxelGiStats::default();
    let far = IVec3::new(128, 0, 0);
    for voxel in [IVec3::ZERO, far] {
        let mut probe = SurfaceProbe::new(voxel, 5);
        probe.stats_mut().samples = [128; 6];
        probe.progress.settled = 63;
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
    // O(1) invalidation must not scan/clear histories. Old contributions no
    // longer count, including late worker results from that old revision.
    invalidate_source_change(&mut runtime, &mut field, &mut stats, SourceChange::Global);
    assert_eq!(stats.settled_coverage, 0.0);
    assert_eq!(field.probes[&far].stats().samples[0], 128);
    assert_eq!(
        probe_metrics(&field.probes[&far], field.transport_revision).3,
        0
    );
    let current = field.probes.get_mut(&IVec3::ZERO).unwrap();
    current.transport_revision = field.transport_revision;
    current.stats_mut().samples = [128; 6];
    current.fresh_mask = current.face_mask;
    current.progress.settled = 63;
    runtime.settled_faces = probe_metrics(current, field.transport_revision).3;
    let mut result = SceneBuildResult {
        scene: runtime.scene.clone(),
        changed: true,
        transport_changed: true,
        retention: None,
        pages: VecDeque::new(),
        receivers_changed: false,
        exact_edits: vec![],
        deferred: Vec::new(),
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
    let mut field = ReceiverField::default();
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
        transport_changed: true,
        retention: None,
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
        receivers_changed: false,
        exact_edits: vec![],
        deferred: Vec::new(),
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
    let mut settings = VoxelGiSettings::default();
    settings.scheduler.scene_publish_budget = Duration::ZERO;
    world.insert_resource(settings);
    world.run_system_once(apply_scene_task).unwrap();
    let field = world.resource::<ReceiverField>();
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
            .resource::<ReceiverField>()
            .probes
            .get(&retained)
            .unwrap() as *const SurfaceProbe,
        retained_address,
        "unchanged probes must keep their allocations across page publication",
    );
    assert!(
        world
            .resource::<ReceiverField>()
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
        let mut field = ReceiverField::default();
        let mut stats = VoxelGiStats::default();
        let mut result = SceneBuildResult {
            scene: runtime.scene.clone(),
            changed: true,
            transport_changed: false,
            retention: None,
            pages: default(),
            receivers_changed: false,
            exact_edits: vec![],
            deferred: Vec::new(),
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
        assert!(probe.updated_at.is_none());
        assert_eq!(probe.stats().samples, [0; 6]);
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
            probe.stats_mut().samples = [1024; 6];
            probe.stats_mut().sequence = [2048; 6];
            probe.stats_mut().radiance = [Vec3::ONE; 6];
            probe.stats_mut().second_moment = [Vec3::ONE * 2.0; 6];
            probe.display[2].indirect = Vec3::splat(0.5);
            probe.display[2].confidence = 0.75;
            probe.updated_at = Some(now);
            probe.stats_mut().face_updated_at = [Some(now); 6];
            probe.revision = 7;
            probe.stats_mut().stable_batches = [4; 6];
            let mut late = probe.clone();
            late.stats_mut().radiance = [Vec3::ONE * 100.0; 6];
            let mut field = ReceiverField::default();
            field.probes.insert(voxel, probe);
            field.pages.insert(IVec3::ZERO, vec![voxel]);
            field.update_indices(&[voxel], &HashMap::new());
            let mut materials = VoxelMaterialTransportTable::default();
            materials.set(
                4,
                MaterialTransport {
                    emission: Vec3::ONE * 8.0,
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
                transport_changed: false,
                retention: None,
                pages: default(),
                receivers_changed: false,
                exact_edits: vec![],
                deferred: Vec::new(),
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
            assert!(
                runtime.urgent_queued.contains(&voxel),
                "a relit probe is CPU event work: only its job carries the cap"
            );
            assert_eq!(p.stats().samples[2], geometry::NEAR_HISTORY);
            assert_eq!(p.stats().sequence[2], 2048);
            assert_eq!(p.stats().radiance[2], Vec3::ONE);
            assert_eq!(p.stats().second_moment[2], Vec3::ONE * 2.0);
            assert_eq!(p.stats().face_updated_at[2], Some(now));
            assert_eq!(p.display[2].indirect, Vec3::splat(0.5));
            assert_eq!(p.display[2].confidence, 0.75);
            assert_eq!(p.emission, runtime.materials.get(material).emission);
            assert_eq!(p.fresh_mask, 0);
            assert_eq!(p.stats().stable_batches, [0; 6]);
            if face_mask == 5 {
                assert_eq!(p.stats().samples[0], 0);
                assert_eq!(p.stats().radiance[0], Vec3::ZERO);
            }
            let task = pool.spawn(async move {
                TraceBatchResult {
                    geometry: None,
                    batch: Some(TransportBatch {
                        work: TraceWork::from_probes(&[late]),
                        paths: 4,
                        ..default()
                    }),
                    pending: default(),
                    bindings: default(),
                    trace_time: Duration::from_millis(1),
                    gpu_tracer: None,
                    emission_cache: None,
                    gpu_error: None,
                    build_error: None,
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
                world.resource::<ReceiverField>().probes[&voxel]
                    .stats()
                    .radiance[2],
                Vec3::ONE,
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
            transport_changed: changes.transport.is_some(),
            retention: changes
                .transport
                .and_then(|change| retention::SpatialHistory::event(change, &[])),
            // No receiver topology changed. This exercises scene acceptance,
            // not the separate page-merge/seeding tests above.
            pages: default(),
            receivers_changed: false,
            exact_edits: vec![],
            deferred: Vec::new(),
            build_time: Duration::ZERO,
        };
        let task = pool.spawn(async move { result });
        let deadline = Instant::now() + Duration::from_secs(5);
        while !task.is_finished() {
            assert!(Instant::now() < deadline, "test scene did not finish");
            std::thread::yield_now();
        }
        let mut field = ReceiverField::default();
        for voxel in [near, far] {
            let mut probe = SurfaceProbe::new(voxel, 4);
            probe.stats_mut().samples = [4096; 6];
            probe.stats_mut().radiance = [Vec3::ONE; 6];
            probe.stats_mut().second_moment = [Vec3::ONE; 6];
            probe.stats_mut().sequence = [4096; 6];
            probe.display[2].indirect = Vec3::ONE;
            probe.updated_at = Some(Instant::now());
            probe.stats_mut().face_updated_at[2] = probe.updated_at;
            field.probes.insert(voxel, probe);
            field.pages.insert(page_of(voxel), vec![voxel]);
        }
        field.update_indices(&[near, far], &HashMap::new());
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            4,
            MaterialTransport {
                emission: Vec3::ONE,
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
            let field = world.resource::<ReceiverField>();
            assert_eq!(runtime.source_revision, 0);
            assert_eq!(runtime.model_reset_source_revision, 0);
            assert_eq!(field.transport_revision, 1);
            for voxel in [near, far] {
                let probe = &field.probes[&voxel];
                assert_eq!(probe.source_revision, field.source_revision);
                assert_ne!(probe.transport_revision, field.transport_revision);
                assert_eq!(probe.stats().samples[2], 4096);
            }
            // One GPU event for both. Whole-range curve: 4 voxels from the
            // edit keeps nothing; 128 voxels away the cap (12,971) exceeds
            // the 4,096 retained samples. CPU counts wait for the GPU's.
            let events = runtime.spatial_history.take_events();
            assert_eq!(events.len(), 1);
            for voxel in [near, far] {
                let retained = if voxel == near { 0 } else { 4096 };
                let (cap, _) = retention::resolve(&events, voxel);
                assert_eq!(cap.min(4096), retained, "material {material}");
                let probe = &field.probes[&voxel];
                assert_eq!(probe.stats().samples[2], 4096);
                assert_eq!(probe.stats().radiance[2], Vec3::ONE);
                assert_eq!(probe.stats().sequence[2], 4096);
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
    let mut field = ReceiverField::default();
    field
        .publishing_pages
        .extend(pages.iter().map(|(page, _)| *page));
    let mut world = World::new();
    world.insert_resource(field);
    world.insert_resource(VoxelGiRuntime {
        scene_publication: Some(SceneBuildResult {
            scene: Arc::new(TransportScene::default()),
            changed: true,
            transport_changed: true,
            retention: None,
            pages,
            receivers_changed: false,
            exact_edits: vec![],
            deferred: Vec::new(),
            build_time: Duration::ZERO,
        }),
        ..default()
    });
    world.insert_resource(VoxelGiStats::default());
    world.init_resource::<GiWorkTimings>();
    let mut settings = VoxelGiSettings::default();
    settings.scheduler.scene_publish_budget = Duration::MAX;
    world.insert_resource(settings);
    world.run_system_once(apply_scene_task).unwrap();
    let published = world.resource::<ReceiverField>().probe_count();
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
    let field = world.resource::<ReceiverField>();
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
fn in_flight_lighting_cannot_restore_indices_moved_by_streaming() {
    let mut field = ReceiverField::default();
    let voxels = [IVec3::ZERO, IVec3::X, IVec3::Y, IVec3::Z];
    for voxel in voxels {
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
    }
    field.update_indices(&voxels, &HashMap::new());
    let pending_result = TraceWork::from_probes(std::slice::from_ref(&field.probes[&IVec3::Z]));
    let mut removed = HashMap::new();
    removed.insert(IVec3::ZERO, field.probes.remove(&IVec3::ZERO).unwrap());
    field.update_indices(&[], &removed);
    assert_eq!(field.probes[&IVec3::Z].order_index, 0);
    pending_result.probes[0].publish(
        &pending_result.faces,
        field.probes.get_mut(&IVec3::Z).unwrap(),
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
fn dispatch_reuses_returned_storage_without_retaining_candidates() {
    let mut field = ReceiverField::default();
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
        enqueue_urgent(&mut runtime, [IVec3::ZERO]);
        let mut selected = select_transport(&mut runtime, &field);
        assert_eq!(selected.probes.as_ptr(), allocation);
        assert_eq!(selected.faces.as_ptr(), face_allocation);
        assert_eq!(selected.probes.len(), 1);
        selected.clear();
        runtime.trace_scratch = selected;
    }
}

#[test]
fn cpu_selects_only_event_work() {
    let now = Instant::now();
    let mut field = ReceiverField::default();
    for x in 0..64 {
        let voxel = IVec3::new(10_000 + x, 0, 0);
        let mut probe = SurfaceProbe::new(voxel, 5);
        probe.stats_mut().samples = [256; 6];
        probe.stats_mut().stable_batches = [4; 6];
        probe.updated_at = Some(now);
        probe.stats_mut().face_updated_at = [Some(now); 6];
        field.order.push(voxel);
        field.probes.insert(voxel, probe);
    }
    let mut runtime = VoxelGiRuntime::default();
    assert!(select_probes(&mut runtime, &field).is_empty());
    let requested = field.order[17];
    enqueue_urgent(&mut runtime, [requested]);
    let forced = select_probes(&mut runtime, &field);
    assert_eq!(
        forced.len(),
        1,
        "a simulation refresh request is CPU event work"
    );
    assert_eq!(forced[0].voxel, requested);
    assert_eq!(forced[0].trace_mask, 5);
    // Far invalidation makes every face due, but bulk work is the GPU's pick.
    field.transport_revision += 1;
    assert!(select_probes(&mut runtime, &field).is_empty());
    assert!(
        field
            .probes
            .values()
            .all(|probe| probe.stats().samples[2] == 256),
        "scheduling must not mutate published histories"
    );
}

#[test]
fn cache_radius_admits_near_pages_and_retires_far_ones_with_hysteresis() {
    use bevy::ecs::system::RunSystemOnce;
    let mut world = World::new();
    let mut runtime = VoxelGiRuntime::default();
    runtime.loaded_pages.extend([
        IVec3::ZERO,
        IVec3::new(2, 0, 0),
        IVec3::new(4, 0, 0),
        IVec3::new(9, 0, 0),
    ]);
    world.insert_resource(runtime);
    world.insert_resource(VoxelGiSettings {
        cache_radius: 64.0,
        ..default()
    });
    let camera = world
        .spawn((
            Camera3d::default(),
            GlobalTransform::from_translation(Vec3::splat(16.0)),
        ))
        .id();
    // Inactive streaming cameras load geometry, not lighting receivers.
    world.spawn((
        Camera3d::default(),
        Camera {
            is_active: false,
            ..default()
        },
        GlobalTransform::from_translation(Vec3::new(304.0, 16.0, 16.0)),
    ));
    world
        .run_system_once(pages::update_receiver_residency)
        .unwrap();
    let runtime = world.resource::<VoxelGiRuntime>();
    let allowed = runtime.receiver_pages.as_ref().unwrap();
    // Page 2 starts 48 voxels away (within 64); page 4 is 112 away.
    assert!(allowed.contains(&IVec3::ZERO) && allowed.contains(&IVec3::new(2, 0, 0)));
    assert!(!allowed.contains(&IVec3::new(4, 0, 0)));
    // Startup retires the far pages' existing receivers.
    assert!(runtime.receiver_retired.contains(&IVec3::new(9, 0, 0)));
    assert!(runtime.receiver_dirty.is_empty());
    // At 88 voxels page 0 stays (radius 64 + 32 hysteresis); at 108 it retires.
    world
        .entity_mut(camera)
        .insert(GlobalTransform::from_translation(Vec3::new(
            120.0, 16.0, 16.0,
        )));
    world
        .run_system_once(pages::update_receiver_residency)
        .unwrap();
    assert!(
        world
            .resource::<VoxelGiRuntime>()
            .receiver_pages
            .as_ref()
            .unwrap()
            .contains(&IVec3::ZERO)
    );
    world
        .entity_mut(camera)
        .insert(GlobalTransform::from_translation(Vec3::new(
            140.0, 16.0, 16.0,
        )));
    world
        .resource_mut::<VoxelGiRuntime>()
        .receiver_retired
        .clear();
    world
        .run_system_once(pages::update_receiver_residency)
        .unwrap();
    let runtime = world.resource::<VoxelGiRuntime>();
    assert!(
        runtime.receiver_dirty.contains(&IVec3::new(4, 0, 0)),
        "entered"
    );
    assert!(
        runtime.receiver_retired.contains(&IVec3::ZERO),
        "beyond radius + hysteresis"
    );
}

#[test]
fn cacheless_residency_retires_every_page_without_a_camera_or_geometry_changes() {
    use bevy::ecs::system::RunSystemOnce;
    let mut world = World::new();
    let mut runtime = VoxelGiRuntime::default();
    runtime
        .loaded_pages
        .extend([IVec3::ZERO, IVec3::new(1, 0, 0)]);
    world.insert_resource(runtime);
    world.insert_resource(VoxelGiSettings {
        cache_radius: 0.0,
        ..default()
    });
    world
        .run_system_once(pages::update_receiver_residency)
        .unwrap();
    let runtime = world.resource::<VoxelGiRuntime>();
    assert!(runtime.receiver_pages.as_ref().unwrap().is_empty());
    assert_eq!(runtime.receiver_retired, runtime.loaded_pages);
    assert!(runtime.dirty_pages.is_empty());
    assert_eq!(runtime.loaded_pages.len(), 2);
}
