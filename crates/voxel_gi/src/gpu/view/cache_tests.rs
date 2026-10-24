//! Native ownership tests operate on raw moments, never reconstructed evidence.
use super::*;
use crate::{emission::EmissionDistribution, work::TraceWork};

#[derive(Clone, Copy, ShaderType)]
struct Receiver {
    surface: Vec3,
    face: u32,
    voxel: IVec3,
    sequence: u32,
}

struct Fixture {
    gpu: GpuTracer,
    work: TraceWork,
    scene: Arc<TransportScene>,
    sources: TransportSources,
    materials: VoxelMaterialTransportTable,
    emitters: Arc<EmissionDistribution>,
    renderer: ViewRenderer,
    camera: CameraState,
}

impl Fixture {
    fn new() -> Self {
        Self::make(false)
    }
    fn make(furnace: bool) -> Self {
        let mut gpu = crate::gpu::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, materials, mut work) = if furnace {
            let f = crate::research::furnace();
            (
                f.scene,
                f.sources,
                f.materials,
                TraceWork::from_probes(&f.receivers[..1], false),
            )
        } else {
            let (scene, sources, probes) = crate::gpu::tests::open_floor_fixture();
            (
                scene,
                sources,
                VoxelMaterialTransportTable::default(),
                TraceWork::from_probes(&probes[264..265], false),
            )
        };
        let emitters = Arc::new(EmissionDistribution::build(&scene,
            &materials));
        work.shared_view_cache = true;
        gpu.trace_compact(
            scene.clone(),
            &sources,
            &materials,
            &mut work,
            16,
            &emitters,
            false,
        )
        .unwrap();
        let device = &gpu.context.device;
        let queue = &gpu.context.queue;
        let mut world = World::new();
        world.insert_resource(device.clone());
        world.insert_resource(queue.clone());
        let renderer = ViewRenderer::from_world(&mut world);
        let mut camera = CameraState::new(device, UVec2::ONE);
        let f = &work.faces[0];
        let normal = crate::world::FACE_OFFSETS[f.face].as_vec3();
        camera.history[0]
            .receivers
            .upload(
                device,
                queue,
                &ShaderBuffer::from(vec![Receiver {
                    surface: f.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5,
                    voxel: f.voxel,
                    face: f.face as u32,
                    sequence: 0,
                }])
                .data
                .unwrap(),
            )
            .unwrap();
        let mut fixture = Self {
            gpu,
            work,
            scene,
            sources,
            materials,
            emitters,
            renderer,
            camera,
        };
        fixture.control();
        fixture.pass("index", None);
        fixture
    }

    fn control(&mut self) {
        let scene = self.gpu.view_scene().unwrap();
        let cache = scene.cache.unwrap();
        self.camera
            .control
            .upload(
                &self.gpu.context.device,
                &self.gpu.context.queue,
                &ShaderBuffer::from(vec![Control {
                    grid: UVec4::new(1, 1, 1, self.camera.history[0].hash.capacity as u32 / 4 - 1),
                    cache: UVec4::new(cache.epoch, 100, 1, 0),
                    ..default()
                }])
                .data
                .unwrap(),
            )
            .unwrap();
        let mut params = scene.params;
        params.counts.x = 1;
        self.camera
            .params
            .upload(
                &self.gpu.context.device,
                &self.gpu.context.queue,
                &ShaderBuffer::from(vec![params]).data.unwrap(),
            )
            .unwrap();
        self.allowance(0);
        let mut queue_words = [0u32; 13];
        queue_words[0] = 1;
        self.camera
            .queue
            .upload(
                &self.gpu.context.device,
                &self.gpu.context.queue,
                &queue_words
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
    }

    fn allowance(&mut self, visits: u32) {
        self.wave(visits, 3);
    }
    fn wave(&mut self, visits: u32, hits: u32) {
        self.camera
            .wave
            .upload(
                &self.gpu.context.device,
                &self.gpu.context.queue,
                &ShaderBuffer::from(vec![Wave {
                    range: UVec4::new(0, visits, hits, 0),
                    ticket: UVec4::ZERO,
                }])
                .data
                .unwrap(),
            )
            .unwrap();
        self.gpu
            .context
            .queue
            .write_buffer(&self.camera.queue.buffer, 52, &0u32.to_le_bytes());
    }

    fn pass(&self, name: &str, held: Option<&SceneBindings>) {
        self.passes(&[name], 1, held);
    }
    fn passes(&self, names: &[&str], repeats: usize, held: Option<&SceneBindings>) {
        let current = self.gpu.view_scene().unwrap();
        let scene = held.unwrap_or(&current);
        let cache = scene.cache.as_ref().unwrap();
        let device = &self.gpu.context.device;
        let operations: Vec<_> = names
            .iter()
            .map(|name| {
                let kernel = match *name {
                    "index" => &self.renderer.index,
                    "load" => &self.renderer.load_cache,
                    "store" => &self.renderer.store_cache,
                    "trace" => &self.renderer.trace[0],
                    _ => panic!("unknown test pass"),
                };
                let history = &self.camera.history[0];
                let groups: Vec<_> = kernel
                    .bindings
                    .iter()
                    .enumerate()
                    .map(|(group, bindings)| {
                        let entries: Vec<_> = bindings
                            .iter()
                            .map(|&binding| {
                                let buffer = match (group, binding) {
                                    (0, 0) => &self.camera.params.buffer,
                                    (0, 1..=5) => &scene.buffers[binding as usize - 1],
                                    (0, 6) => &history.receivers.buffer,
                                    (0, 7) => &history.paths.buffer,
                                    (0, 8) => &scene.buffers[5],
                                    (0, 9) => &scene.buffers[6],
                                    (0, 10) => &scene.buffers[7],
                                    (1, 0) => &history.tails.buffer,
                                    (1, 1) => &self.camera.wave.buffer,
                                    (2, 0) => &self.camera.control.buffer,
                                    (2, 1) => &self.camera.history[1].receivers.buffer,
                                    (2, 4) => &self.camera.history[1].estimates.buffer,
                                    (2, 5) => &history.estimates.buffer,
                                    (2, 6) => &self.camera.history[1].hash.buffer,
                                    (2, 7) => &history.hash.buffer,
                                    (2, 10) => &self.camera.queue.buffer,
                                    (2, 14) => &scene.buffers[6],
                                    (2, 15) => &cache.meta,
                                    (2, 16) => &cache.guard,
                                    _ => panic!("unexpected bridge binding {group}:{binding}"),
                                };
                                BindGroupEntry {
                                    binding,
                                    resource: buffer.as_entire_binding(),
                                }
                            })
                            .collect();
                        device.create_bind_group(
                            "Face handoff test",
                            &kernel.pipeline.get_bind_group_layout(group as u32).into(),
                            &entries,
                        )
                    })
                    .collect();
                (kernel, groups)
            })
            .collect();
        let mut encoder = device.create_command_encoder(&default());
        for _ in 0..repeats {
            for (kernel, groups) in &operations {
                let mut pass = encoder.begin_compute_pass(&default());
                pass.set_pipeline(&kernel.pipeline);
                for (i, group) in groups.iter().enumerate() {
                    pass.set_bind_group(i as u32, &**group, &[]);
                }
                pass.dispatch_workgroups(1, 1, 1);
            }
        }
        self.gpu.context.queue.submit([encoder.finish()]);
    }

    fn estimate(&self) -> Vec<u32> {
        crate::gpu::maintenance_tests::read_words(
            &self.gpu.context,
            &self.camera.history[0].estimates.buffer,
            64,
        )
    }
    fn set_estimate(&self, words: &[u32]) {
        self.gpu.context.queue.write_buffer(
            &self.camera.history[0].estimates.buffer,
            0,
            &words
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        );
    }
    fn resident(&self) -> Vec<u32> {
        let scene = self.gpu.view_scene().unwrap();
        let base = self.work.faces[0].resident.x as usize * 36;
        let words = crate::gpu::maintenance_tests::read_words(
            &self.gpu.context,
            &scene.buffers[6],
            (base + 36) as u64 * 4,
        );
        words[base..base + 36].to_vec()
    }
    fn background(&mut self) {
        self.gpu
            .trace_compact(
                self.scene.clone(),
                &self.sources,
                &self.materials,
                &mut self.work,
                16,
                &self.emitters,
                false,
            )
            .unwrap();
        self.control();
    }
}

#[test]
#[ignore = "native raw topology prior, zero-ray initialization and cache handoff"]
fn native_new_face_bootstrap_is_weak_and_does_not_restart_sampling() {
    use crate::{sampling::pcg, world::FACE_OFFSETS};
    let mut f = Fixture::new();
    let voxel = f.work.faces[0].voxel;
    let face = f.work.faces[0].face;
    let normal = FACE_OFFSETS[face];
    let material = f.work.faces[0].material as u32;
    let rgb = [2.0f32, 4.0, 8.0];
    let scene = f.gpu.view_scene().unwrap();
    let cache = scene.cache.as_ref().unwrap();
    let device = &f.gpu.context.device;
    let queue = &f.gpu.context.queue;
    let mask = f.camera.history[0].hash.capacity as u32 / 4 - 1;
    let mut control = Control {
        grid: UVec4::new(1, 1, 1, mask),
        history: UVec4::new(1, mask, 1, 1),
        cache: UVec4::new(cache.epoch, 100, 0, 0),
        ..default()
    };
    let upload_control = |f: &mut Fixture, c: Control| {
        f.camera.control.upload(&f.gpu.context.device, &f.gpu.context.queue,
            &ShaderBuffer::from(vec![c]).data.unwrap()).unwrap();
    };
    // The old exposed block has been removed. Only its last-frame raw face
    // survives, not a displayed/reconstructed RGB value or a current slot.
    let donor = voxel + normal;
    f.camera.history[1].receivers.upload(device, queue,
        &ShaderBuffer::from(vec![Receiver { voxel: donor, face: face as u32,
            surface: donor.as_vec3() + Vec3::splat(0.5) + normal.as_vec3() * 0.5,
            sequence: 64 }]).data.unwrap()).unwrap();
    let hash = pcg(pcg(pcg(donor.x as u32) ^ donor.y as u32) ^ donor.z as u32)
        ^ pcg(face as u32 + 1);
    queue.write_buffer(&f.camera.history[1].hash.buffer, u64::from(hash & mask) * 4, &1u32.to_le_bytes());
    let mut old = [0u32; 16];
    old[..3].copy_from_slice(&rgb.map(f32::to_bits));
    old[4..7].copy_from_slice(&rgb.map(|v| (v * v + 1.).to_bits()));
    old[8..11].copy_from_slice(&[900f32.to_bits(); 3]); // presentation MUST NOT seed raw
    let mut blank = [0u32; 16];
    blank[13] = material;
    f.set_estimate(&blank);
    upload_control(&mut f, control);
    for support in [1.0f32, 64.] {
        old[3] = support.to_bits();
        f.gpu.context.queue.write_buffer(&f.camera.history[1].estimates.buffer, 0,
            &old.into_iter().flat_map(u32::to_le_bytes).collect::<Vec<_>>());
        f.pass("load", None);
        let e = f.estimate();
        if support == 1. {
            assert_eq!(e[3], 0, "a pure prior cannot recursively seed another face");
        } else {
            assert_eq!(&e[..3], &old[..3]);
            assert_eq!(&e[4..7], &old[4..7]);
            assert_eq!(f32::from_bits(e[3]), 1., "zero-ray prior has one effective weight");
            assert_eq!(e[12], 0, "a prior does not consume a QMC path");
            assert_eq!(&e[8..11], &old[..3], "first display starts at raw prior, not black");
        }
    }
    // Repeated admission does not add pseudo-counts. An empty resident face
    // accepts the view prior, then becomes the single authority for both.
    let side = (face + 2) % 6;
    for (position, orientation, expected) in [
        (donor, side, 1.0), // newly exposed perpendicular wall
        (voxel - normal, face, 1.0), // placement covers an aligned face
        (voxel - FACE_OFFSETS[side], side, 1.0), // covered perpendicular floor
        (voxel + normal * 4, face, 0.0), // not a local topology neighbor
        (donor, face, 1.0),
    ] {
        let queue = &f.gpu.context.queue;
        let mut encoder = f.gpu.context.device.create_command_encoder(&default());
        encoder.clear_buffer(&f.camera.history[1].hash.buffer, 0, None);
        queue.submit([encoder.finish()]);
        f.camera.history[1].receivers.upload(&f.gpu.context.device, queue,
            &ShaderBuffer::from(vec![Receiver { voxel: position, face: orientation as u32,
                surface: position.as_vec3(), sequence: 64 }]).data.unwrap()).unwrap();
        let hash = pcg(pcg(pcg(position.x as u32) ^ position.y as u32) ^ position.z as u32)
            ^ pcg(orientation as u32 + 1);
        queue.write_buffer(&f.camera.history[1].hash.buffer, u64::from(hash & mask) * 4, &1u32.to_le_bytes());
        f.set_estimate(&blank);
        f.pass("load", None);
        let e = f.estimate();
        assert_eq!(f32::from_bits(e[3]), expected, "donor {position:?} face {orientation}");
        if expected > 0. { assert_eq!(&e[..3], &old[..3]); }
    }
    let mut resident = f.resident();
    resident[..28].fill(0);
    let offset = u64::from(f.work.faces[0].resident.x) * 144;
    f.gpu.context.queue.write_buffer(&scene.buffers[6], offset,
        &resident.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    control.cache.z = 1;
    upload_control(&mut f, control);
    f.passes(&["load", "store"], 4, None);
    assert_eq!(f.resident()[5], 1);
    assert_eq!(f.resident()[6], 0);
    let mut sum = rgb;
    for n in 1..=16 {
        f.allowance(1);
        f.passes(&["load", "trace", "store"], 1, None);
        let path = crate::gpu::maintenance_tests::read_words(
            &f.gpu.context, &f.camera.history[0].paths.buffer, 128);
        assert_eq!(path[3], n - 1, "prior must not skip or repeat paths");
        assert_eq!(path[23], 1, "fixture path must be known");
        for i in 0..3 { sum[i] += f32::from_bits(path[20 + i]) + f32::from_bits(path[24 + i]); }
        let e = f.estimate();
        assert_eq!(f32::from_bits(e[3]), (n + 1) as f32);
        for i in 0..3 {
            assert!((f32::from_bits(e[i]) - sum[i] / (n + 1) as f32).abs() < 1e-5,
                "prior influence must dilute as 1/(N+1), including dark samples");
        }
    }
    // A previously sampled identity capped to zero must not be bootstrapped
    // again; its next ray, not a neighbor, decides the new raw measurement.
    control.cache.z = 0;
    upload_control(&mut f, control);
    blank[12] = 77;
    f.set_estimate(&blank);
    f.pass("load", None);
    assert_eq!(f.estimate()[3], 0);
    assert_eq!(f.estimate()[12], 77);
    // A raster-only face can hold a prior but must not claim an existing
    // world's cache slot while the transport geometry is incompatible.
    let authority = f.resident();
    control.cache.z = 1;
    upload_control(&mut f, control);
    blank[12] = 0;
    blank[13] = 256 | 0x80000000;
    f.set_estimate(&blank);
    f.passes(&["load", "store"], 1, None);
    assert_eq!(f32::from_bits(f.estimate()[3]), 1.);
    assert_eq!(f.estimate()[15], 0, "pending geometry has no resident authority");
    assert_eq!(f.resident(), authority, "pending geometry cannot modify resident transport");
}

#[test]
#[ignore = "native analytic RGB furnace with shared-cache sequence reservations"]
fn native_two_way_cache_reserved_paths_match_furnace_energy() {
    let mut f = Fixture::make(true);
    let mut e = vec![0u32; 16];
    e[13] = 4;
    f.set_estimate(&e);
    f.gpu
        .context
        .queue
        .write_buffer(&f.camera.control.buffer, super::CONTROL_VIEW_FIRST_OFFSET, &1u32.to_le_bytes());
    f.wave(32, 3);
    // Bound command-buffer size, as the runtime does with per-frame submits.
    // Thousands of Metal compute encoders in one submit can lose the device.
    for _ in 0..128 {
        f.passes(&["load", "trace", "store"], 64, None);
        f.resident();
    }
    let result = f.resident();
    let mean = Vec3::from_array(std::array::from_fn(|i| f32::from_bits(result[i])));
    let material = f.materials.get(4);
    let expected = material.emission / (Vec3::ONE - material.diffuse);
    eprintln!(
        "shared furnace: count={} sequence={} mean={mean:?} expected={expected:?}",
        result[5], result[6]
    );
    assert!(result[5] > 16384);
    let counters = crate::gpu::maintenance_tests::read_words(
        &f.gpu.context,
        &f.camera.queue.buffer,
        48,
    );
    let tail = crate::gpu::maintenance_tests::read_words(
        &f.gpu.context,
        &f.camera.history[0].tails.buffer,
        128,
    );
    assert_eq!(
        result[6],
        16 + counters[2] + u32::from(tail[19] == 1),
        "unused reservations must not create path-length-dependent QMC gaps"
    );
    assert!(((mean - expected).abs() / expected).max_element() < 0.015);
}

#[test]
#[ignore = "native shared actual path sampling across alternating schedulers"]
fn native_two_way_cache_traces_each_sequence_once_across_owners() {
    let mut f = Fixture::new();
    let mut e = vec![0u32; 16];
    e[13] = f.work.faces[0].material as u32;
    f.set_estimate(&e);
    let before = f.resident();
    f.allowance(1);
    f.pass("load", None);
    f.pass("trace", None);
    f.pass("store", None);
    let after = f.resident();
    assert_eq!(after[5], before[5] + 1);
    assert_eq!(after[6], before[6] + 1);
    let path = crate::gpu::maintenance_tests::read_words(
        &f.gpu.context,
        &f.camera.history[0].paths.buffer,
        128,
    );
    assert_eq!(path[3], before[6]);
    f.background();
    let worker = f.resident();
    assert_eq!(worker[5], after[5] + 16);
    assert_eq!(worker[6], after[6] + 16);
    f.allowance(1);
    f.pass("load", None);
    f.pass("trace", None);
    f.pass("store", None);
    let path = crate::gpu::maintenance_tests::read_words(
        &f.gpu.context,
        &f.camera.history[0].paths.buffer,
        128,
    );
    assert_eq!(path[3], worker[6]);
    assert_eq!(f.resident()[5], before[5] + 18);
    // A different material must neither import nor overwrite this face.
    let authority = f.resident();
    e = f.estimate();
    e[13] ^= 1;
    e[3] = 65000.0f32.to_bits();
    f.set_estimate(&e);
    f.allowance(1);
    f.pass("load", None);
    f.pass("store", None);
    assert_eq!(f.resident(), authority);
}

#[test]
#[ignore = "native shared RGB ownership: import, transfer, no double count and sequences"]
fn native_two_way_cache_moves_history_without_counting_it_twice() {
    let mut f = Fixture::new();
    let material = f.work.faces[0].material as u32;
    let initial = f.resident();
    let mut e = vec![0u32; 16];
    e[13] = material;
    f.set_estimate(&e);
    f.allowance(1);
    f.pass("load", None);
    e = f.estimate();
    assert_eq!(&e[..3], &initial[..3]);
    assert_eq!(f32::from_bits(e[3]), initial[5] as f32);
    assert_eq!(e[12], initial[6]);
    assert!(e[15] > 0);
    f.pass("trace", None);
    e = f.estimate();
    // A completed visible sample updates the shared record. Presentation is
    // intentionally unrelated: it must not contaminate the raw statistics.
    e[..4].copy_from_slice(&[
        2.0f32.to_bits(),
        3.0f32.to_bits(),
        4.0f32.to_bits(),
        17.0f32.to_bits(),
    ]);
    e[8..12].copy_from_slice(&[
        90.0f32.to_bits(),
        80.0f32.to_bits(),
        70.0f32.to_bits(),
        1.0f32.to_bits(),
    ]);
    f.set_estimate(&e);
    f.pass("store", None);
    let updated = f.resident();
    assert_eq!(&updated[..3], &e[..3]);
    assert_eq!(updated[5], 17);
    assert_eq!(updated[6], e[12]);
    f.allowance(0);
    f.pass("load", None);
    f.pass("store", None);
    assert_eq!(f.resident(), updated, "copying history is not a new sample");
    f.background();
    let after = f.resident();
    assert_eq!(after[5], 33);
    assert_eq!(
        after[6],
        e[12] + 16,
        "background resumes the shared sequence"
    );
    f.pass("load", None);
    assert_eq!(f32::from_bits(f.estimate()[3]), 33.0);
    // First cache admission transfers a better-supported view estimate; it
    // cannot sum overlapping prefixes with the existing resident estimate.
    e[3] = 512.0f32.to_bits();
    e[12] = 600;
    e[15] = 0;
    f.set_estimate(&e);
    f.pass("load", None);
    f.pass("store", None);
    assert_eq!(f.resident()[5], 512, "admission adopts, never adds 33+512");
    assert_eq!(f.resident()[6], 600);
}

#[test]
#[ignore = "native reservations preserve long view paths across background sampling"]
fn native_two_way_cache_does_not_discard_long_paths_when_background_advances() {
    let mut f = Fixture::make(true);
    let mut e = vec![0u32; 16];
    e[13] = f.work.faces[0].material as u32;
    f.set_estimate(&e);
    f.wave(1, 1);
    // A cache-independent pending prefix must not be merged into an authority
    // which already contains that sequence. Only admitted tails own a lease.
    f.pass("trace", None);
    let independent = crate::gpu::maintenance_tests::read_words(
        &f.gpu.context,
        &f.camera.history[0].tails.buffer,
        128,
    );
    assert_eq!(independent[19], 1);
    let first = f.resident()[6];
    f.pass("load", None);
    assert_eq!(
        f.resident()[6],
        first + 1,
        "view reserves before background can consume its sequence"
    );
    f.pass("trace", None);
    f.pass("store", None);
    let read_tail = |f: &Fixture| {
        crate::gpu::maintenance_tests::read_words(
            &f.gpu.context,
            &f.camera.history[0].tails.buffer,
            128,
        )
    };
    assert_eq!(
        read_tail(&f)[19],
        1,
        "fixture must exercise an unfinished path"
    );
    let mut finished = false;
    for _ in 0..64 {
        let old_bounce = read_tail(&f)[15];
        f.background();
        let before = f.resident();
        f.wave(1, 1);
        f.pass("load", None);
        assert_eq!(
            f.resident()[6],
            before[6],
            "resuming needs no new reservation"
        );
        f.pass("trace", None);
        f.pass("store", None);
        let path = crate::gpu::maintenance_tests::read_words(
            &f.gpu.context,
            &f.camera.history[0].paths.buffer,
            128,
        );
        assert_eq!(
            path[3], first,
            "background work must not replace the unfinished path"
        );
        let tail = read_tail(&f);
        if tail[19] == 2 {
            assert_eq!(f.resident()[5], before[5] + 1);
            finished = true;
            break;
        }
        assert_eq!(
            tail[15],
            old_bounce + 1,
            "the path resumes rather than restarting"
        );
        assert_eq!(
            f.resident()[5],
            before[5],
            "partial energy is never accumulated"
        );
    }
    assert!(finished, "the reserved path eventually completes");
}

#[test]
#[ignore = "native old render generations cannot write after edit maintenance"]
fn native_two_way_cache_rejects_stale_writes_and_obeys_retention() {
    let mut f = Fixture::new();
    let mut e = vec![0u32; 16];
    e[3] = 1024.0f32.to_bits();
    e[12] = 1024;
    e[13] = f.work.faces[0].material as u32;
    f.set_estimate(&e);
    f.pass("load", None);
    f.pass("store", None);
    let held = f.gpu.view_scene().unwrap();
    // Force a model restart/cap and an otherwise dangerous CPU rollback.
    for job in &mut f.work.probes {
        job.revision += 1;
        job.restart = true;
    }
    for face in &mut f.work.faces {
        face.history_cap = 4;
        face.rollback = true;
    }
    f.background();
    let capped = f.resident();
    assert_eq!(capped[5], 20);
    // The old encoder descriptor has the old token even if its buffers still
    // happen to be the current allocation.
    let cache = held.cache.as_ref().unwrap();
    f.gpu
        .context
        .queue
        .write_buffer(&f.camera.control.buffer, super::CONTROL_EPOCH_OFFSET, &cache.epoch.to_le_bytes());
    f.pass("store", Some(&held));
    assert_eq!(f.resident(), capped);
    f.control();
    f.pass("load", None);
    assert_eq!(
        f32::from_bits(f.estimate()[3]),
        20.0,
        "a retained 1024 view copy cannot undo the cap"
    );
    let mut pending = f.estimate();
    pending[7] = 1.0f32.to_bits();
    f.set_estimate(&pending);
    f.pass("store", None);
    assert_eq!(
        f.resident()[5],
        0x80000000 | 20,
        "an unresolved edit must not serve stale light to secondary rays"
    );
    f.pass("load", None);
    assert_eq!(
        f32::from_bits(f.estimate()[3]),
        20.0,
        "flags are not sample weight"
    );
}

#[test]
fn control_offsets_used_by_native_tests_match_the_encoded_layout() {
    let bytes = ShaderBuffer::from(vec![Control {
        history: UVec4::new(0, 0, 0, 0xdead),
        cache: UVec4::new(0xbeef, 0, 0, 0),
        ..default()
    }])
    .data
    .unwrap();
    let word = |offset: u64| {
        let at = offset as usize;
        u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    };
    assert_eq!(word(super::CONTROL_VIEW_FIRST_OFFSET), 0xdead);
    assert_eq!(word(super::CONTROL_EPOCH_OFFSET), 0xbeef);
}
