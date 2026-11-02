//! Dispatch boundaries must change latency, never the represented lighting.
//! These blocking readbacks are native qualification only, not a runtime queue.
use super::*;

#[derive(Clone, Copy, Default, ShaderType)]
struct WaveControl {
    range: UVec4,
    ticket: UVec4,
}

struct WaveHarness {
    context: GpuTransportDevice,
    pipeline: ComputePipeline,
    inputs: UploadBuffer,
    paths: UploadBuffer,
    states: UploadBuffer,
    control: UploadBuffer,
    groups: [BindGroup; 2],
    count: usize,
}

impl WaveHarness {
    fn new(gpu: &GpuTracer, receivers: &[ViewReceiver], samples: u32) -> Self {
        let context = gpu.context.clone();
        let device = &context.device;
        let scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("Resumable view paths"),
            source: ShaderSource::Wgsl(shader().into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Resumable view paths"),
            layout: None,
            module: &module,
            entry_point: Some("advance_view_paths"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[("LIMITED_REACH", f64::from(gpu.pipeline_key.1))],
                ..default()
            },
            cache: None,
        });
        let count = receivers.len() * samples as usize;
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mut inputs = UploadBuffer::new(device, "Wave receivers", usage);
        inputs
            .upload(
                device,
                &context.queue,
                &ShaderBuffer::from(receivers.to_vec()).data.unwrap(),
            )
            .unwrap();
        let mut paths = UploadBuffer::new(device, "Wave records", usage);
        paths.ensure(device, count * PATH_BYTES).unwrap();
        let mut states = UploadBuffer::new(device, "Wave continuation state", usage);
        states.ensure(device, count * 128).unwrap();
        let control = UploadBuffer::new(
            device,
            "Wave control",
            BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        );
        let (state, index) = gpu
            .resident
            .as_ref()
            .map(|r| r.cache_buffers())
            .unwrap_or((&gpu.external_cache.0, &gpu.external_cache.1));
        let buffers = [
            &gpu.params,
            &gpu.directory,
            &gpu.regions,
            &gpu.cells,
            &gpu.materials,
            &gpu.lights,
            &inputs,
            &paths,
            &gpu.emitters,
            state,
            index,
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: binding as u32,
                resource: buffer.buffer.as_entire_binding(),
            })
            .collect();
        let groups = [
            device.create_bind_group(
                "Wave scene",
                &pipeline.get_bind_group_layout(0).into(),
                &entries,
            ),
            device.create_bind_group(
                "Wave state",
                &pipeline.get_bind_group_layout(1).into(),
                &[
                    BindGroupEntry {
                        binding: 0,
                        resource: states.buffer.as_entire_binding(),
                    },
                    BindGroupEntry {
                        binding: 1,
                        resource: control.buffer.as_entire_binding(),
                    },
                ],
            ),
        ];
        if let Some(error) = bevy::tasks::block_on(scope.pop()) {
            panic!("wave pipeline: {error}");
        }
        Self {
            context,
            pipeline,
            inputs,
            paths,
            states,
            control,
            groups,
            count,
        }
    }

    fn advance(&mut self, steps: u32, finite_depth: u32, ticket: UVec4) {
        self.control
            .upload(
                &self.context.device,
                &self.context.queue,
                &ShaderBuffer::from(vec![WaveControl {
                    range: UVec4::new(0, self.count as u32, steps, finite_depth),
                    ticket,
                }])
                .data
                .unwrap(),
            )
            .unwrap();
        let mut encoder = self.context.device.create_command_encoder(&default());
        {
            let mut pass = encoder.begin_compute_pass(&default());
            pass.set_pipeline(&self.pipeline);
            for (i, group) in self.groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups((self.count as u32).div_ceil(64), 1, 1);
        }
        self.context.queue.submit([encoder.finish()]);
    }

    fn read(&self) -> (Vec<u32>, Vec<u32>) {
        let records = super::super::maintenance_tests::read_words(
            &self.context,
            &self.paths.buffer,
            (self.count * PATH_BYTES) as u64,
        );
        let states = super::super::maintenance_tests::read_words(
            &self.context,
            &self.states.buffer,
            (self.count * 128) as u64,
        );
        (records, states)
    }

    fn replace_receivers(&mut self, receivers: &[ViewReceiver]) {
        self.inputs
            .upload(
                &self.context.device,
                &self.context.queue,
                &ShaderBuffer::from(receivers.to_vec()).data.unwrap(),
            )
            .unwrap();
    }
}

fn prepare(
    gpu: &mut GpuTracer,
    fixture: &research::ResearchFixture,
    first: u32,
    samples: u32,
) -> TraceWork {
    let emitters = Arc::new(EmissionDistribution::build(
        &fixture.scene,
        &fixture.materials,
    ));
    let mut work = TraceWork::from_probes(&fixture.receivers);
    for face in &mut work.faces {
        face.sequence = first;
    }
    gpu.trace_compact(
        fixture.scene.clone(),
        &fixture.sources,
        &fixture.materials,
        &mut work,
        samples,
        &emitters,
    )
    .unwrap();
    work
}

fn completed(records: &[u32], states: &[u32]) -> bool {
    let mut done = true;
    for (record, state) in records
        .chunks_exact(PATH_WORDS)
        .zip(states.chunks_exact(32))
    {
        let status = state[19];
        assert!((1..=4).contains(&status));
        assert_eq!(
            record[23] == 1,
            status == 2,
            "only complete known paths carry evidence"
        );
        if status == 1 {
            assert_eq!(
                &record[24..27],
                &[0; 3],
                "paused energy must not leak into display"
            );
            done = false;
        }
    }
    done
}

fn compare_records(reference: &[u32], actual: &[u32]) {
    assert_eq!(reference.len(), actual.len());
    // All integer identity/status fields must be exact. Allow only roundoff in
    // floats across the storage boundary (Metal may fuse arithmetic differently).
    for (a, b) in reference
        .chunks_exact(PATH_WORDS)
        .zip(actual.chunks_exact(PATH_WORDS))
    {
        for i in 0..PATH_WORDS {
            if [3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 23, 27, 31].contains(&i) {
                assert_eq!(a[i], b[i], "word {i}");
            } else {
                let (a, b) = (f32::from_bits(a[i]), f32::from_bits(b[i]));
                assert!(
                    (a - b).abs() <= 2e-5 * (1.0 + a.abs()),
                    "word {i}: {a} != {b}"
                );
            }
        }
    }
}

#[test]
#[ignore = "native path pause/resume and model-epoch regression"]
fn view_wave_boundaries_preserve_paths_and_reject_stale_state() {
    let fixture = research::furnace();
    let mut gpu =
        GpuTracer::with_kernel(super::super::tests::native_tracer().context, false).unwrap();
    let work = prepare(&mut gpu, &fixture, 257, 65);
    let receivers = receivers(&work, 257);
    let reference = view_paths_with_depth(&gpu, &receivers, 65, 16);
    let ticket = UVec4::new(1, 2, 3, 4);
    for steps in [1u32, 3, 8] {
        let mut wave = WaveHarness::new(&gpu, &receivers, 65);
        for batch in 0..16u32.div_ceil(steps) {
            wave.advance(steps, 16, ticket);
            let (records, states) = wave.read();
            let done = completed(&records, &states);
            if steps == 1 && batch == 0 {
                assert!(!done, "fixture exercises pausing");
            }
        }
        let (records, states) = wave.read();
        assert!(completed(&records, &states));
        compare_records(&reference, &records);
        wave.advance(steps, 16, ticket);
        assert_eq!(
            wave.read(),
            (records, states),
            "completed paths are idempotent"
        );
    }
    let mut wave = WaveHarness::new(&gpu, &receivers, 65);
    wave.advance(1, 0, ticket);
    let initial = wave.read();
    wave.advance(0, 0, ticket);
    assert_eq!(
        wave.read(),
        initial,
        "zero allowance is a pause, not an observation"
    );
    for component in 0..4 {
        let mut changed = ticket;
        changed[component] += 1;
        wave.advance(1, 0, changed);
        let (records, _) = wave.read();
        compare_records(&initial.0, &records);
    }
    let mut moved = receivers.clone();
    for receiver in &mut moved {
        receiver.surface.x += 0.125;
        receiver.sequence += 73;
    }
    wave.replace_receivers(&moved);
    wave.advance(1, 0, ticket);
    let mut fresh = WaveHarness::new(&gpu, &moved, 65);
    fresh.advance(1, 0, ticket);
    assert_eq!(
        wave.read(),
        fresh.read(),
        "a new receiver cannot resume the old ray"
    );
    for receiver in &mut moved {
        receiver.face = u32::MAX;
    }
    wave.replace_receivers(&moved);
    wave.advance(1, 0, ticket);
    let (records, states) = wave.read();
    assert!(records.iter().all(|&word| word == 0));
    assert!(states.chunks_exact(32).all(|state| state[19] == 4));
}

#[test]
#[ignore = "native analytic unlimited paths completed over bounded GPU waves"]
fn view_wave_uncached_tails_reach_the_multibounce_solution() {
    let fixture = research::furnace();
    let mut gpu =
        GpuTracer::with_kernel(super::super::tests::native_tracer().context, false).unwrap();
    let material = fixture.materials.get(4);
    let expected = material.emission / (Vec3::ONE - material.diffuse);
    let mut sum = [0.0f64; 3];
    let mut count = 0;
    let mut longest = 0;
    for batch in 0..16 {
        let first = batch * 1024;
        let work = prepare(&mut gpu, &fixture, first, 1024);
        let receivers = receivers(&work, first);
        let mut wave = WaveHarness::new(&gpu, &receivers, 1024);
        let mut result = None;
        // No shader depth cutoff. This host bound fails the test rather than
        // publishing partial light if a path is still alive after 192 hits.
        for turn in 1..=64 {
            wave.advance(3, 0, UVec4::ONE);
            let (records, states) = wave.read();
            if completed(&records, &states) {
                longest = longest.max(turn);
                result = Some(records);
                break;
            }
        }
        let records = result.expect("all paths must finish by roulette/cache/escape");
        // The matched prefix at 64 hits is only a reference for this fixture:
        // all these paths finished earlier, so its terminal mixture is unused.
        let reference = view_paths_with_depth(&gpu, &receivers, 1024, 64);
        compare_records(&reference, &records);
        for record in records.chunks_exact(PATH_WORDS) {
            assert_eq!(record[23], 1);
            let value = vector(&record[20..]) + vector(&record[24..]);
            for band in 0..3 {
                sum[band] += f64::from(value[band]);
            }
            count += 1;
        }
    }
    let mean = Vec3::from_array(sum.map(|value| (value / f64::from(count)) as f32));
    eprintln!(
        "VIEW_WAVE paths={count} steps_per_wave=3 maximum_waves={longest} mean={mean:?} expected={expected:?} fraction={:?}",
        mean / expected
    );
    assert!(((mean - expected).abs() / expected).max_element() < 0.02);
}
