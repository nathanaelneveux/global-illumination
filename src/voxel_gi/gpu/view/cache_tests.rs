//! Native ownership tests operate on raw moments, never reconstructed evidence.
use super::*;
use crate::voxel_gi::{emission::EmissionDistribution, work::TraceWork};

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
        let mut gpu = crate::voxel_gi::gpu::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = crate::voxel_gi::gpu::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut work = TraceWork::from_probes(&probes[264..265], false);
        work.shared_view_cache = true;
        gpu.trace_compact(scene.clone(), &sources, &materials, &mut work, 16, &emitters, false).unwrap();
        let device = &gpu.context.device;
        let queue = &gpu.context.queue;
        let mut world = World::new();
        world.insert_resource(device.clone());
        world.insert_resource(queue.clone());
        let renderer = ViewRenderer::from_world(&mut world);
        let mut camera = CameraState::new(device, 1);
        let f = &work.faces[0];
        let normal = crate::voxel_gi::world::FACE_OFFSETS[f.face].as_vec3();
        camera.history[0].receivers.upload(device, queue, &ShaderBuffer::from(vec![Receiver {
            surface: f.voxel.as_vec3() + Vec3::splat(0.5) + normal * 0.5,
            voxel: f.voxel, face: f.face as u32, sequence: 0,
        }]).data.unwrap()).unwrap();
        let mut fixture = Self { gpu, work, scene, sources, materials, emitters, renderer, camera };
        fixture.control();
        fixture.pass("index", None);
        fixture
    }

    fn control(&mut self) {
        let scene = self.gpu.view_scene().unwrap();
        let cache = scene.cache.unwrap();
        self.camera.control.upload(&self.gpu.context.device, &self.gpu.context.queue,
            &ShaderBuffer::from(vec![Control {
                grid: UVec4::new(1, 1, 1, self.camera.history[0].hash.capacity as u32 / 4 - 1),
                cache: UVec4::new(cache.epoch, 100, 1, 0),
                ..default()
            }]).data.unwrap()).unwrap();
        let mut params = scene.params;
        params.counts.x = 1;
        self.camera.params.upload(&self.gpu.context.device, &self.gpu.context.queue,
            &ShaderBuffer::from(vec![params]).data.unwrap()).unwrap();
        self.camera.wave.upload(&self.gpu.context.device, &self.gpu.context.queue,
            &ShaderBuffer::from(vec![Wave { range: UVec4::new(0, 1, 3, 0), ticket: UVec4::ZERO }]).data.unwrap()).unwrap();
        let mut queue_words = [0u32; 13];
        queue_words[0] = 1;
        self.camera.queue.upload(&self.gpu.context.device, &self.gpu.context.queue,
            &queue_words.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    }

    fn pass(&self, name: &str, held: Option<&SceneBindings>) {
        let current = self.gpu.view_scene().unwrap();
        let scene = held.unwrap_or(&current);
        let cache = scene.cache.as_ref().unwrap();
        let kernel = match name {
            "index" => &self.renderer.index,
            "load" => &self.renderer.load_cache,
            "store" => &self.renderer.store_cache,
            "trace" => &self.renderer.trace[0],
            _ => panic!("unknown test pass"),
        };
        let history = &self.camera.history[0];
        let device = &self.gpu.context.device;
        let groups: Vec<_> = kernel.bindings.iter().enumerate().map(|(group, bindings)| {
            let entries: Vec<_> = bindings.iter().map(|&binding| {
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
                    (2, 5) => &history.estimates.buffer,
                    (2, 7) => &history.hash.buffer,
                    (2, 10) => &self.camera.queue.buffer,
                    (2, 14) => &scene.buffers[6],
                    (2, 15) => &cache.meta,
                    (2, 16) => &cache.guard,
                    _ => panic!("unexpected bridge binding {group}:{binding}"),
                };
                BindGroupEntry { binding, resource: buffer.as_entire_binding() }
            }).collect();
            device.create_bind_group("Face handoff test", &kernel.pipeline.get_bind_group_layout(group as u32).into(), &entries)
        }).collect();
        let mut encoder = device.create_command_encoder(&default());
        {
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&kernel.pipeline);
            for (i, group) in groups.iter().enumerate() { pass.set_bind_group(i as u32, &**group, &[]); }
            pass.dispatch_workgroups(1, 1, 1);
        }
        self.gpu.context.queue.submit([encoder.finish()]);
    }

    fn estimate(&self) -> Vec<u32> {
        crate::voxel_gi::gpu::maintenance_tests::read_words(&self.gpu.context,
            &self.camera.history[0].estimates.buffer, 64)
    }
    fn set_estimate(&self, words: &[u32]) {
        self.gpu.context.queue.write_buffer(&self.camera.history[0].estimates.buffer, 0,
            &words.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
    }
    fn resident(&self) -> Vec<u32> {
        let scene = self.gpu.view_scene().unwrap();
        let base = self.work.faces[0].resident.x as usize * 36;
        let words = crate::voxel_gi::gpu::maintenance_tests::read_words(&self.gpu.context,
            &scene.buffers[6], (base + 36) as u64 * 4);
        words[base..base + 36].to_vec()
    }
    fn background(&mut self) {
        self.gpu.trace_compact(self.scene.clone(), &self.sources, &self.materials,
            &mut self.work, 16, &self.emitters, false).unwrap();
        self.control();
    }
}

#[test]
#[ignore = "native shared actual path sampling across alternating schedulers"]
fn native_two_way_cache_traces_each_sequence_once_across_owners() {
    let mut f = Fixture::new();
    let mut e = vec![0u32; 16];
    e[13] = f.work.faces[0].material as u32;
    f.set_estimate(&e);
    let before = f.resident();
    f.pass("load", None);
    f.pass("trace", None);
    f.pass("store", None);
    let after = f.resident();
    assert_eq!(after[5], before[5] + 1);
    assert_eq!(after[6], before[6] + 1);
    let path = crate::voxel_gi::gpu::maintenance_tests::read_words(&f.gpu.context,
        &f.camera.history[0].paths.buffer, 128);
    assert_eq!(path[3], before[6]);
    f.background();
    let worker = f.resident();
    assert_eq!(worker[5], after[5] + 16);
    assert_eq!(worker[6], after[6] + 16);
    f.pass("load", None);
    f.pass("trace", None);
    f.pass("store", None);
    let path = crate::voxel_gi::gpu::maintenance_tests::read_words(&f.gpu.context,
        &f.camera.history[0].paths.buffer, 128);
    assert_eq!(path[3], worker[6]);
    assert_eq!(f.resident()[5], before[5] + 18);
    // A different material must neither import nor overwrite this face.
    let authority = f.resident();
    e = f.estimate();
    e[13] ^= 1;
    e[3] = 65000.0f32.to_bits();
    f.set_estimate(&e);
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
    f.pass("load", None);
    e = f.estimate();
    assert_eq!(&e[..3], &initial[..3]);
    assert_eq!(f32::from_bits(e[3]), initial[5] as f32);
    assert_eq!(e[12], initial[6]);
    assert!(e[15] > 0);
    // A completed visible sample updates the shared record. Presentation is
    // intentionally unrelated: it must not contaminate the raw statistics.
    e[..4].copy_from_slice(&[2.0f32.to_bits(), 3.0f32.to_bits(), 4.0f32.to_bits(), 17.0f32.to_bits()]);
    e[12] += 1;
    e[8..12].copy_from_slice(&[90.0f32.to_bits(), 80.0f32.to_bits(), 70.0f32.to_bits(), 1.0f32.to_bits()]);
    f.set_estimate(&e);
    f.pass("store", None);
    let updated = f.resident();
    assert_eq!(&updated[..3], &e[..3]);
    assert_eq!(updated[5], 17);
    assert_eq!(updated[6], e[12]);
    f.pass("load", None);
    f.pass("store", None);
    assert_eq!(f.resident(), updated, "copying history is not a new sample");
    f.background();
    let after = f.resident();
    assert_eq!(after[5], 33);
    assert_eq!(after[6], e[12] + 16, "background resumes the shared sequence");
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
    for job in &mut f.work.probes { job.revision += 1; job.restart = true; }
    for face in &mut f.work.faces { face.history_cap = 4; face.rollback = true; }
    f.background();
    let capped = f.resident();
    assert_eq!(capped[5], 20);
    // The old encoder descriptor has the old token even if its buffers still
    // happen to be the current allocation.
    let cache = held.cache.as_ref().unwrap();
    f.gpu.context.queue.write_buffer(&f.camera.control.buffer, 144,
        &cache.epoch.to_le_bytes());
    f.pass("store", Some(&held));
    assert_eq!(f.resident(), capped);
    f.control();
    f.pass("load", None);
    assert_eq!(f32::from_bits(f.estimate()[3]), 20.0, "a retained 1024 view copy cannot undo the cap");
}
