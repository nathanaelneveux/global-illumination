//! Qualification of the view receiver/path layout before render-graph wiring.
//! Raw matched prefixes: no display filter, temporal blend or borrowed evidence.
use super::*;
use crate::voxel_gi::{research, sampling, world::FACE_OFFSETS};

const PATH_BYTES: usize = 128;
const PATH_WORDS: usize = PATH_BYTES / 4;

#[derive(Clone, Copy, Default, ShaderType)]
struct ViewReceiver {
    surface: Vec3,
    face: u32,
    voxel: IVec3,
    sequence: u32,
}

fn shader() -> String {
    include_paths(include_str!(
        "../../../assets/shaders/voxel_view_trace.wgsl"
    ))
}

#[test]
fn view_path_shader_and_receiver_layout_validate() {
    let source = shader();
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(
        ShaderBuffer::from(vec![ViewReceiver::default()])
            .data
            .unwrap()
            .len(),
        32
    );
    let path = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("ViewPath"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { span, .. } = &path.inner else {
        panic!("path record struct")
    };
    assert_eq!(*span as usize, PATH_BYTES);
}

/// Runs the actual far consumer with the near tracer's uploaded model and
/// frozen raw-cache bindings. Test readback only; the eventual view owner keeps
/// these records GPU-resident and supplies receivers from its first-hit pass.
fn view_paths(gpu: &GpuTracer, receivers: &[ViewReceiver], samples: u32) -> Vec<u32> {
    let context = &gpu.context;
    let device = &context.device;
    let queue = &context.queue;
    let scope = device
        .wgpu_device()
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
        label: Some("Matched view transport"),
        source: ShaderSource::Wgsl(shader().into()),
    });
    let constants = [
        ("DIFFUSE_HITS", f64::from(gpu.pipeline_key.0)),
        ("LIMITED_REACH", f64::from(gpu.pipeline_key.1)),
    ];
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Matched view paths"),
        layout: None,
        module: &module,
        entry_point: Some("trace_view_paths"),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants: &constants,
            ..default()
        },
        cache: None,
    });
    let mut inputs = UploadBuffer::new(
        device,
        "View receivers",
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    inputs
        .upload(
            device,
            queue,
            &ShaderBuffer::from(receivers.to_vec()).data.unwrap(),
        )
        .unwrap();
    let count = receivers.len() * samples as usize;
    let size = count * PATH_BYTES;
    let mut output = UploadBuffer::new(
        device,
        "View path records",
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    // A dirty output and trailing guard check inactive lanes and partial workgroups.
    output
        .upload(device, queue, &vec![0xa5; size + PATH_BYTES])
        .unwrap();
    let (state, index) = gpu
        .resident
        .as_ref()
        .map(|r| r.cache_buffers())
        .unwrap_or((&gpu.external_cache.0, &gpu.external_cache.1));
    let bindings = [
        &gpu.params,
        &gpu.directory,
        &gpu.regions,
        &gpu.cells,
        &gpu.materials,
        &gpu.lights,
        &inputs,
        &output,
        &gpu.emitters,
        state,
        index,
    ];
    let entries: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(binding, buffer)| BindGroupEntry {
            binding: binding as u32,
            resource: buffer.buffer.as_entire_binding(),
        })
        .collect();
    let group = device.create_bind_group(
        "View path inputs",
        &pipeline.get_bind_group_layout(0).into(),
        &entries,
    );
    let mut encoder = device.create_command_encoder(&default());
    {
        let mut pass = encoder.begin_compute_pass(&default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let words =
        super::maintenance_tests::read_words(context, &output.buffer, (size + PATH_BYTES) as u64);
    assert!(
        words[count * PATH_WORDS..]
            .iter()
            .all(|&word| word == 0xa5a5a5a5),
        "partial workgroup overwrote guard"
    );
    if let Some(error) = bevy::tasks::block_on(scope.pop()) {
        panic!("view transport validation: {error}");
    }
    words[..count * PATH_WORDS].to_vec()
}

fn vector(words: &[u32]) -> Vec3 {
    Vec3::new(
        f32::from_bits(words[0]),
        f32::from_bits(words[1]),
        f32::from_bits(words[2]),
    )
}

fn receivers(work: &TraceWork, first: u32) -> Vec<ViewReceiver> {
    work.faces
        .iter()
        .map(|face| ViewReceiver {
            surface: face.voxel.as_vec3()
                + Vec3::splat(0.5)
                + FACE_OFFSETS[face.face].as_vec3() * 0.5,
            face: face.face as u32,
            voxel: face.voxel,
            sequence: first,
        })
        .collect()
}

fn compare(gpu: &GpuTracer, work: &TraceWork, first: u32, samples: u32) -> (usize, usize) {
    let receivers = receivers(work, first);
    let words = view_paths(gpu, &receivers, samples);
    let mut hits = 0;
    let mut unknown = 0;
    for ((receiver, near), records) in receivers
        .iter()
        .zip(&work.faces)
        .zip(words.chunks_exact(samples as usize * PATH_WORDS))
    {
        let mut sum = Vec3::ZERO;
        let mut count = 0;
        for (sample, record) in records.chunks_exact(PATH_WORDS).enumerate() {
            assert_eq!(vector(record), receiver.surface);
            assert_eq!(record[3], first + sample as u32);
            assert_eq!(&record[4..7], &receiver.voxel.to_array().map(|v| v as u32));
            assert_eq!(record[7], receiver.face);
            assert_eq!(
                record[31],
                sampling::stream_id(receiver.voxel, receiver.face)
            );
            let direction = vector(&record[8..]);
            let pdf = f32::from_bits(record[11]);
            let normal = FACE_OFFSETS[receiver.face as usize].as_vec3();
            assert!((direction.length() - 1.0).abs() < 1e-5);
            assert!((pdf - normal.dot(direction).max(0.0) / std::f32::consts::PI).abs() < 1e-6);
            assert!(pdf > 0.0);
            assert!(record[15] <= 3);
            if record[15] == 1 {
                hits += 1;
                assert!(
                    record[19] < 6 && record[27] > 0,
                    "hit geometry/material retained"
                );
                assert!(vector(&record[12..]).is_finite());
            }
            assert!(record[23] <= 1);
            if record[23] == 0 {
                unknown += 1;
                continue;
            }
            count += 1;
            let energy = vector(&record[20..]) + vector(&record[24..]);
            assert!(energy.is_finite() && energy.min_element() >= 0.0);
            sum += energy;
            assert!(
                (vector(&record[28..]) - near.direct.visible)
                    .abs()
                    .max_element()
                    < 2e-5
            );
        }
        assert_eq!(
            count, near.samples,
            "known black paths still count; unknown paths do not"
        );
        let far = sum / count.max(1) as f32;
        let error = (far - near.radiance.visible).abs().max_element();
        assert!(
            error <= 2e-5 * (1.0 + near.radiance.visible.max_element()),
            "{:?}/{} prefix {first}+{samples}: near={:?} far={far:?}",
            receiver.voxel,
            receiver.face,
            near.radiance.visible
        );
    }
    (hits, unknown)
}

#[test]
#[ignore = "native raw near/view matched-prefix qualification"]
fn view_paths_match_near_prefixes_across_sources_and_reach() {
    let native = super::tests::native_tracer();
    for (name, fixture) in [
        ("hallway", research::hallway()),
        ("skylight", research::skylight_room()),
        ("cave", research::cave()),
    ] {
        let emitters = Arc::new(EmissionDistribution::with_importance(
            &fixture.scene,
            &fixture.materials,
            crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
        ));
        let probes: Vec<_> = fixture
            .receivers
            .iter()
            .step_by((fixture.receivers.len() / 12).max(1))
            .take(12)
            .cloned()
            .collect();
        for (first, samples, bounces, reach) in
            [(0, 16, 1, 768.0), (257, 64, 3, 768.0), (2048, 17, 3, 32.0)]
        {
            let mut gpu = GpuTracer::with_kernel(native.context.clone(), false).unwrap();
            let mut work = TraceWork::from_probes(&probes, false);
            work.transport = super::super::TransportSettings { bounces, reach };
            for face in &mut work.faces {
                face.sequence = first;
            }
            for probe in &mut work.probes {
                probe.sky_requested = false;
            }
            gpu.trace_compact(
                fixture.scene.clone(),
                &fixture.sources,
                &fixture.materials,
                &mut work,
                samples,
                &emitters,
                false,
            )
            .unwrap();
            let (hits, unknown) = compare(&gpu, &work, first, samples);
            eprintln!(
                "VIEW_PARITY {name}: {} receivers x {samples}, first={first}, bounces={bounces}, reach={reach}, hits={hits}, unknown={unknown}",
                work.faces.len()
            );
        }
    }
}

#[test]
#[ignore = "native dark, missing geometry and inactive-lane accounting"]
fn view_paths_preserve_dark_unknown_and_inactive_measurements() {
    use crate::voxel_gi::{naadf::volume_len, world::PAGE_SIZE};
    let mut gpu = GpuTracer::with_kernel(super::tests::native_tracer().context, false).unwrap();
    let (scene, _, probes) = super::tests::open_floor_fixture();
    let sources = TransportSources {
        environment: EnergySpectrum::ZERO,
        directional: vec![],
        points: vec![],
    };
    let emitters = Arc::new(EmissionDistribution::default());
    for gap in [false, true] {
        let scene = if gap {
            Arc::new(TransportScene::with_page_delta(
                &scene,
                vec![(IVec3::new(0, 2, 0), vec![2; volume_len(PAGE_SIZE)])],
                &[],
            ))
        } else {
            scene.clone()
        };
        let mut work = TraceWork::from_probes(&probes[400..405], false);
        for probe in &mut work.probes {
            probe.sky_requested = false;
        }
        gpu.trace_compact(scene, &sources, &default(), &mut work, 17, &emitters, false)
            .unwrap();
        let (_, unknown) = compare(&gpu, &work, 0, 17);
        assert_eq!(unknown > 0, gap);
        assert!(
            work.faces
                .iter()
                .all(|face| face.radiance.visible == Vec3::ZERO)
        );
        if !gap {
            assert!(work.faces.iter().all(|face| face.samples == 17));
        }
        let mut inactive = receivers(&work, 0);
        for receiver in &mut inactive {
            receiver.face = u32::MAX;
        }
        assert!(
            view_paths(&gpu, &inactive, 17)
                .iter()
                .all(|&word| word == 0),
            "inactive lanes erase old data"
        );
    }
}

#[test]
#[ignore = "native near/view termination on the same frozen raw cache"]
fn view_paths_match_near_at_the_surface_cache_boundary() {
    let fixture = research::furnace();
    let emitters = Arc::new(EmissionDistribution::with_importance(
        &fixture.scene,
        &fixture.materials,
        crate::voxel_gi::VoxelGiSettings::default().sampling_importance(),
    ));
    let mut warm = super::tests::native_tracer();
    warm.enable_resident().unwrap();
    let all_faces = super::cache_tests::all_interior_faces(&fixture);
    let mut history = TraceWork::from_probes(&all_faces, false);
    for _ in 0..4 {
        warm.trace_compact(
            fixture.scene.clone(),
            &fixture.sources,
            &fixture.materials,
            &mut history,
            16,
            &emitters,
            false,
        )
        .unwrap();
    }
    // Freeze the raw cache. No resident accumulator runs on the readers.
    let mut gpu = GpuTracer::with_kernel(warm.context.clone(), false).unwrap();
    gpu.share_cache_from(&warm).unwrap();
    let mut work = TraceWork::from_probes(&fixture.receivers, false);
    for face in &mut work.faces {
        face.sequence = 4096;
    }
    for probe in &mut work.probes {
        probe.sky_requested = false;
    }
    gpu.trace_compact(
        fixture.scene.clone(),
        &fixture.sources,
        &fixture.materials,
        &mut work,
        17,
        &emitters,
        false,
    )
    .unwrap();
    let (hits, unknown) = compare(&gpu, &work, 4096, 17);
    assert_eq!(hits, work.faces.len() * 17);
    assert_eq!(unknown, 0);
}
