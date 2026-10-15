//! Worker-owned stable face slots. Allocation identity is distinct from scene
//! freshness: unrelated streaming never resets a measurement. Geometry removal
//! retires only affected regions; recycled slots are initialized before use.
use super::super::world::{AIR, FACE_OFFSETS, REGION_EDGE};
use super::*;
use bevy::platform::collections::{HashMap, HashSet};

const RECORD_BYTES: usize = 128;
const SHADER: &str = include_str!("../../../assets/shaders/voxel_accumulate.wgsl");
type Key = (IVec3, usize);

#[derive(Clone, Copy)]
struct Slot {
    index: u32,
    incarnation: u32,
    revision: u64,
    model_epoch: u64,
    task_epoch: u64,
    task_index: u32,
}

pub(super) struct ResidentAccumulator {
    pub bindings: Vec<ResidentBinding>,
    slots: HashMap<Key, Slot>,
    regions: HashMap<IVec3, HashSet<Key>>,
    free: Vec<u32>,
    high_water: u32,
    incarnation: u32,
    task_epoch: u64,
    pub reuse: bool,
    scene: Option<Arc<TransportScene>>,
    origin: Instant,
    #[cfg(test)]
    clock_ms: Option<u32>,
    state: UploadBuffer,
    inputs: UploadBuffer,
    seeds: UploadBuffer,
    control: UploadBuffer,
    input_bytes: Vec<u8>,
    cap_bytes: Vec<u8>,
    seed_bytes: Vec<u8>,
    prepare: ComputePipeline,
    accumulate: ComputePipeline,
    prepare_bindings: Option<CachedBindings<5>>,
    accumulate_bindings: Option<CachedBindings<4>>,
}

impl ResidentAccumulator {
    pub(super) fn new(context: &GpuTransportDevice) -> Result<Self, String> {
        let device = &context.device;
        let scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("GI persistent measurements"),
            source: ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: default(),
                cache: None,
            })
        };
        let prepare = pipeline("prepare");
        let accumulate = pipeline("accumulate");
        if let Some(error) = bevy::tasks::block_on(scope.pop()) {
            return Err(error.to_string());
        }
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        Ok(Self {
            bindings: Vec::new(),
            slots: default(),
            regions: default(),
            free: Vec::new(),
            high_water: 0,
            incarnation: 0,
            task_epoch: 0,
            reuse: false,
            scene: None,
            origin: Instant::now(),
            #[cfg(test)]
            clock_ms: None,
            state: UploadBuffer::new(
                device,
                "GI resident measurements",
                storage | BufferUsages::COPY_SRC,
            ),
            inputs: UploadBuffer::new(device, "GI resident jobs", storage),
            seeds: UploadBuffer::new(device, "GI new resident measurements", storage),
            control: UploadBuffer::new(
                device,
                "GI resident controls",
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            input_bytes: Vec::new(),
            cap_bytes: Vec::new(),
            seed_bytes: Vec::new(),
            prepare,
            accumulate,
            prepare_bindings: None,
            accumulate_bindings: None,
        })
    }

    pub(super) fn prepare_inputs(
        &mut self,
        context: &GpuTransportDevice,
        scene: &Arc<TransportScene>,
        work: &mut TraceWork,
        samples: u32,
    ) -> Result<(), String> {
        if let Some(previous) = &self.scene {
            let mut affected = HashSet::new();
            for region in scene.changed_regions(previous) {
                affected.insert(region);
                affected.extend(FACE_OFFSETS.map(|normal| region + normal));
            }
            for region in affected {
                let Some(keys) = self.regions.get_mut(&region) else {
                    continue;
                };
                keys.retain(|key| {
                    let present = scene.material_at(key.0).is_some_and(|m| m != AIR)
                        && scene
                            .material_at(key.0 + FACE_OFFSETS[key.1])
                            .is_none_or(|m| m == AIR);
                    if !present && let Some(slot) = self.slots.remove(key) {
                        self.free.push(slot.index);
                    }
                    present
                });
                if keys.is_empty() {
                    self.regions.remove(&region);
                }
            }
        }
        self.scene = Some(Arc::clone(scene));
        self.input_bytes.clear();
        self.cap_bytes.clear();
        self.seed_bytes.clear();
        self.task_epoch = self
            .task_epoch
            .checked_add(1)
            .ok_or("resident task epoch exhausted")?;
        for job in &work.probes {
            let cap = work.faces[job.faces()]
                .first()
                .map_or(u32::MAX, |face| face.history_cap);
            // Probe-level evidence is consumed atomically by CPU publication.
            // Apply it to resident faces that were not selected too, otherwise
            // their next job would incorrectly resurrect pre-change weight.
            if cap != u32::MAX {
                for face in 0..6 {
                    if job.trace_mask & (1 << face) == 0
                        && let Some(slot) = self.slots.get(&(job.voxel, face))
                    {
                        for word in [slot.index, u32::MAX, cap, u32::MAX] {
                            self.cap_bytes.extend_from_slice(&word.to_le_bytes());
                        }
                    }
                }
            }
            for face in &mut work.faces[job.faces()] {
                let key = (face.voxel, face.face);
                let slot = self.slots.entry(key).or_insert_with(|| {
                    self.regions
                        .entry(face.voxel.div_euclid(IVec3::splat(REGION_EDGE as i32)))
                        .or_default()
                        .insert(key);
                    let index = self.free.pop().unwrap_or_else(|| {
                        let index = self.high_water;
                        self.high_water += 1;
                        index
                    });
                    Slot {
                        index,
                        incarnation: 0,
                        revision: u64::MAX,
                        model_epoch: u64::MAX,
                        task_epoch: 0,
                        task_index: 0,
                    }
                });
                let mut seed = u32::MAX;
                if slot.revision != job.revision
                    || (job.restart && slot.model_epoch != job.source_revision)
                {
                    self.incarnation = self
                        .incarnation
                        .checked_add(1)
                        .ok_or("resident allocation incarnation exhausted")?;
                    slot.incarnation = self.incarnation;
                    slot.revision = job.revision;
                    slot.model_epoch = job.source_revision;
                    seed = (self.seed_bytes.len() / RECORD_BYTES) as u32;
                    encode_seed(&mut self.seed_bytes, face, slot.incarnation, self.origin);
                }
                face.resident = UVec2::new(slot.index, slot.incarnation);
                if seed != u32::MAX {
                    self.bindings.push(ResidentBinding {
                        voxel: face.voxel,
                        face: face.face,
                        revision: job.revision,
                        source_revision: job.source_revision,
                        address: face.resident,
                    });
                }
                let task_index = self.input_bytes.len() / 16;
                slot.task_epoch = self.task_epoch;
                slot.task_index = task_index as u32;
                for word in [slot.index, seed, face.history_cap, task_index as u32] {
                    self.input_bytes.extend_from_slice(&word.to_le_bytes());
                }
            }
        }
        if self.reuse {
            // Use the existing stable ownership index, not a per-frame sort or
            // another hash table. Selection is geometric, never radiometric.
            for face in &mut work.faces {
                face.neighbor = 0;
            }
            for index in 0..work.faces.len() {
                if work.faces[index].neighbor != 0 {
                    continue;
                }
                let voxel = work.faces[index].voxel;
                let orientation = work.faces[index].face;
                let normal = FACE_OFFSETS[orientation];
                // Prefer donors outside the immediate corner reconstruction
                // footprint, so sharing contributes new paths to that footprint.
                for offset in [3, 1]
                    .into_iter()
                    .flat_map(|distance| FACE_OFFSETS.map(|v| v * distance))
                {
                    if offset.dot(normal) != 0 {
                        continue;
                    }
                    if let Some(slot) = self.slots.get(&(voxel + offset, orientation))
                        && slot.task_epoch == self.task_epoch
                        && work.faces[slot.task_index as usize].neighbor == 0
                    {
                        work.faces[index].neighbor = slot.task_index + 1;
                        work.faces[slot.task_index as usize].neighbor = index as u32 + 1;
                        break;
                    }
                }
            }
            debug!(target: "voxel_gi_reuse", faces=work.faces.len(), paired=work.faces.iter().filter(|f| f.neighbor != 0).count(), "fresh path neighbors");
        }
        self.input_bytes.extend_from_slice(&self.cap_bytes);
        let device = &context.device;
        let queue = &context.queue;
        let previous = self.state.buffer.clone();
        let capacity = self.state.capacity;
        if self
            .state
            .ensure(device, self.high_water as usize * RECORD_BYTES)?
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Preserve GI resident measurements during growth"),
            });
            encoder.copy_buffer_to_buffer(&previous, 0, &self.state.buffer, 0, capacity);
            queue.submit([encoder.finish()]);
        }
        self.inputs.upload(device, queue, &self.input_bytes)?;
        self.seeds.upload(device, queue, &self.seed_bytes)?;
        let milliseconds = self.origin.elapsed().as_millis() as u32;
        #[cfg(test)]
        let milliseconds = self.clock_ms.unwrap_or(milliseconds);
        let control = [
            work.faces.len() as u32,
            samples,
            milliseconds,
            HISTORY_LIMIT,
            work.response_seconds.max(0.001).to_bits(),
            (self.input_bytes.len() / 16) as u32,
            0,
            0,
        ];
        self.control.upload(
            device,
            queue,
            &control
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )?;
        Ok(())
    }

    pub(super) fn buffer(&self) -> Buffer {
        self.state.buffer.clone()
    }

    pub(super) fn dispatch_prepare(
        &mut self,
        device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        tasks: &UploadBuffer,
        count: usize,
        timestamps: Option<&wgpu::QuerySet>,
    ) {
        let bindings = cached_bindings(
            &mut self.prepare_bindings,
            device,
            || self.prepare.get_bind_group_layout(0).into(),
            [
                (0, &self.control),
                (1, tasks),
                (3, &self.state),
                (4, &self.inputs),
                (5, &self.seeds),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI resident task preparation"),
            timestamp_writes: timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        pass.set_pipeline(&self.prepare);
        pass.set_bind_group(0, &**bindings, &[]);
        debug_assert!(count <= self.input_bytes.len() / 16);
        pass.dispatch_workgroups((self.input_bytes.len() as u32 / 16).div_ceil(64), 1, 1);
    }

    pub(super) fn dispatch_accumulate(
        &mut self,
        device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        output: &UploadBuffer,
        count: usize,
        timestamps: Option<&wgpu::QuerySet>,
    ) {
        let bindings = cached_bindings(
            &mut self.accumulate_bindings,
            device,
            || self.accumulate.get_bind_group_layout(0).into(),
            [
                (0, &self.control),
                (2, output),
                (3, &self.state),
                (4, &self.inputs),
            ],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI resident accumulation"),
            timestamp_writes: timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(4),
                end_of_pass_write_index: Some(5),
            }),
        });
        pass.set_pipeline(&self.accumulate);
        pass.set_bind_group(0, &**bindings, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
}

fn encode_seed(
    bytes: &mut Vec<u8>,
    face: &super::super::work::FaceWork,
    incarnation: u32,
    origin: Instant,
) {
    let mut words = [0_u32; RECORD_BYTES / 4];
    for (base, energy) in [
        (0, face.radiance),
        (8, face.second_moment),
        (16, face.direct),
    ] {
        for (i, value) in spectrum_vec(energy)
            .to_array()
            .into_iter()
            .chain([energy.ionizing])
            .enumerate()
        {
            words[base + i] = value.to_bits();
        }
    }
    words[5] = face.samples;
    words[6] = face.sequence;
    for (i, value) in face.custom_direct.to_array().into_iter().enumerate() {
        words[21 + i] = value.to_bits();
    }
    for (i, value) in face.presentation.to_array().into_iter().enumerate() {
        words[24 + i] = value.to_bits();
    }
    words[28] = incarnation;
    if let Some(time) = face.updated_at {
        words[29] = if time >= origin {
            time.duration_since(origin).as_millis() as u32
        } else {
            0_u32.wrapping_sub(origin.duration_since(time).as_millis() as u32)
        };
        words[30] = 1;
    }
    bytes.extend(words.into_iter().flat_map(u32::to_le_bytes));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resident_shader_validates() {
        let module = naga::front::wgsl::parse_str(SHADER).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    #[ignore = "native persistent storage parity, growth, retention and recycling"]
    fn resident_accumulation_preserves_measurements() {
        let mut control = super::super::tests::native_tracer();
        let mut candidate = GpuTracer::new(control.context.clone()).unwrap();
        candidate.enable_resident().unwrap();
        let (mut scene, mut sources, initial) = super::super::tests::open_floor_fixture();
        sources.environment = EnergySpectrum::new(Vec3::new(0.3, 0.1, 0.8), 0.4, 0.2);
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut expected = initial.clone();
        let mut actual = initial;
        for step in 0..20 {
            let milliseconds = 1000 + step * 50;
            candidate.resident.as_mut().unwrap().clock_ms = Some(milliseconds);
            let measured_at = candidate.resident.as_ref().unwrap().origin
                + Duration::from_millis(u64::from(milliseconds));
            let mut indices: Vec<usize> = (0..if step < 3 { 17 } else { 1024 }).collect();
            if step % 2 == 1 {
                indices.reverse();
            }
            if step % 3 == 0 {
                indices.retain(|i| i % 2 == 0);
            }
            if step == 10 {
                // Retire all geometry, then reuse its freed resident allocations.
                let empty = Arc::new(TransportScene::default());
                let mut blank = TraceWork::default();
                blank.probes.clear();
                candidate
                    .resident
                    .as_mut()
                    .unwrap()
                    .prepare_inputs(&candidate.context, &empty, &mut blank, 16)
                    .unwrap();
                assert!(candidate.resident.as_ref().unwrap().slots.is_empty());
                scene = Arc::new((*scene).clone());
                for probes in [&mut expected, &mut actual] {
                    for p in probes {
                        p.revision += 1;
                    }
                }
            }
            let mut a = TraceWork::from_probes(
                &indices
                    .iter()
                    .map(|&i| expected[i].clone())
                    .collect::<Vec<_>>(),
                false,
            );
            let mut b = TraceWork::from_probes(
                &indices
                    .iter()
                    .map(|&i| actual[i].clone())
                    .collect::<Vec<_>>(),
                false,
            );
            for work in [&mut a, &mut b] {
                work.response_seconds = 0.075;
                for job in &mut work.probes {
                    job.sky_requested = false;
                }
                if step == 6 {
                    for face in &mut work.faces {
                        face.samples = face.samples.min(3);
                        face.history_cap = 3;
                    }
                }
                if step == 14 {
                    for job in &mut work.probes {
                        job.restart = true;
                        job.source_revision = 100;
                    }
                    for face in &mut work.faces {
                        face.samples = 0;
                        face.history_cap = 0;
                        face.radiance = EnergySpectrum::ZERO;
                        face.second_moment = EnergySpectrum::ZERO;
                    }
                }
            }
            for (tracer, work) in [(&mut control, &mut a), (&mut candidate, &mut b)] {
                tracer
                    .trace_compact(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        work,
                        16,
                        &emitters,
                        false,
                    )
                    .unwrap();
                for face in &mut work.faces {
                    face.updated_at = Some(measured_at);
                }
            }
            for (left, right) in a.faces.iter().zip(&b.faces) {
                assert_eq!(
                    (left.voxel, left.face, left.samples, left.sequence),
                    (right.voxel, right.face, right.samples, right.sequence)
                );
                for (a, b) in [
                    (left.radiance, right.radiance),
                    (left.second_moment, right.second_moment),
                    (left.direct, right.direct),
                ] {
                    let delta = a - b;
                    assert!(
                        delta
                            .visible
                            .abs()
                            .max_element()
                            .max(delta.infrared.abs())
                            .max(delta.ionizing.abs())
                            < 1e-5 * (1.0 + a.max_component()),
                        "step={step} {a:?} vs {b:?}"
                    );
                }
            }
            for (resident, work, probes) in [(false, &a, &mut expected), (true, &b, &mut actual)] {
                for (job, &i) in work.probes.iter().zip(&indices) {
                    for face in &work.faces[job.faces()] {
                        let previous_time = probes[i].face_updated_at[face.face];
                        face.write(&mut probes[i]);
                        if resident {
                            probes[i].display[face.face].indirect = face.presentation.truncate();
                            probes[i].display[face.face].confidence = face.presentation.w;
                        } else {
                            probes[i].resolve_face(face.face, previous_time, measured_at, 0.075);
                        }
                    }
                }
            }
            for &i in &indices {
                for face in 0..6 {
                    let a = expected[i].display[face];
                    let b = actual[i].display[face];
                    assert!(
                        (a.indirect - b.indirect).abs().max_element() < 1e-5,
                        "filter step={step} face={face} expected={a:?} resident={b:?}"
                    );
                    assert!((a.confidence - b.confidence).abs() < 1e-5);
                }
            }
        }
        eprintln!(
            "RESIDENT_PARITY steps=20 five_bands=true growth=true subsets=true reorder=true retirement=true history_cap=true model_restart=true"
        );
    }

    #[test]
    #[ignore = "native partial-face retention and model reset ownership"]
    fn resident_caps_unselected_faces_before_evidence_is_consumed() {
        let mut control = super::super::tests::native_tracer();
        let mut resident = GpuTracer::new(control.context.clone()).unwrap();
        resident.enable_resident().unwrap();
        let (scene, sources, _) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut a = SurfaceProbe::new(IVec3::new(0, 4, 0), 12);
        let mut b = a.clone();
        for (step, mask) in [12, 4, 8, 4, 8].into_iter().enumerate() {
            for (tracer, probe) in [(&mut control, &mut a), (&mut resident, &mut b)] {
                let mut work = TraceWork::default();
                let cap = match step {
                    1 => 3,
                    3 => 0,
                    _ => u32::MAX,
                };
                work.push(
                    probe,
                    mask,
                    0,
                    u64::from(step >= 3),
                    step == 3,
                    u32::MAX,
                    cap,
                );
                work.probes[0].sky_requested = false;
                work.probes[0].retention = Some((step as u64, cap));
                work.response_seconds = 0.075;
                tracer
                    .trace_compact(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut work,
                        16,
                        &emitters,
                        false,
                    )
                    .unwrap();
                work.probes[0].publish(&work.faces, probe, Instant::now(), &default());
            }
            assert_eq!(a.samples, b.samples, "partial cap step={step}");
            assert_eq!(a.sequence, b.sequence);
            for face in [2, 3] {
                assert!(
                    (a.radiance[face] - b.radiance[face])
                        .visible
                        .abs()
                        .max_element()
                        < 1e-5
                );
            }
        }
        assert_eq!(b.samples[3], 16);
    }

    #[test]
    #[ignore = "paired resident compute cost, no rendering or gameplay claim"]
    fn resident_compute_cost() {
        let control = super::super::tests::native_tracer();
        let mut candidate = GpuTracer::new(control.context.clone()).unwrap();
        candidate.enable_resident().unwrap();
        let mut tracers = [control, candidate];
        let (scene, sources, initial) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut probes = [initial.clone(), initial];
        let mut times: [Vec<f32>; 2] = default();
        for round in 0..40 {
            for mode in (0..2).map(|offset| (round + offset) % 2) {
                let mut work = TraceWork::from_probes(&probes[mode], false);
                for job in &mut work.probes {
                    job.sky_requested = false;
                }
                work.response_seconds = 0.075;
                tracers[mode]
                    .trace_compact(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut work,
                        16,
                        &emitters,
                        false,
                    )
                    .unwrap();
                if round >= 10 {
                    times[mode].push(tracers[mode].last_gpu_ms.unwrap());
                }
                for (job, probe) in work.probes.iter().zip(&mut probes[mode]) {
                    for face in &work.faces[job.faces()] {
                        face.write(probe);
                    }
                }
            }
        }
        for (mode, times) in times.iter_mut().enumerate() {
            times.sort_by(f32::total_cmp);
            eprintln!(
                "RESIDENT_COST mode={mode} GPU_p50={:.4} GPU_p95={:.4}",
                times[15], times[28]
            );
        }
    }

    #[test]
    #[ignore = "native overlapping readbacks with growth and recycled resident slots"]
    fn resident_queued_snapshots_keep_their_own_measurements() {
        let mut control = super::super::tests::native_tracer();
        let mut candidate = GpuTracer::new(control.context.clone()).unwrap();
        candidate.enable_resident().unwrap();
        let (scene, mut sources, initial) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut pending = Vec::new();
        for (step, count) in [17, 1024, 128].into_iter().enumerate() {
            sources.environment = EnergySpectrum::new(Vec3::splat(step as f32 + 0.25), 0.7, 0.3);
            // Explicit retirement before reuse, while older snapshots are still
            // mapped/pending. Their staging allocations must remain independent.
            if step == 2 {
                candidate
                    .resident
                    .as_mut()
                    .unwrap()
                    .prepare_inputs(
                        &candidate.context,
                        &Arc::new(TransportScene::default()),
                        &mut TraceWork::default(),
                        16,
                    )
                    .unwrap();
            }
            let mut probes = initial[..count].to_vec();
            for probe in &mut probes {
                probe.revision = step as u64;
            }
            let mut expected = TraceWork::from_probes(&probes, false);
            let mut actual = TraceWork::from_probes(&probes, false);
            for work in [&mut expected, &mut actual] {
                work.response_seconds = 0.075;
                for job in &mut work.probes {
                    job.sky_requested = false;
                }
            }
            let snapshot = candidate
                .submit_compact(
                    Arc::clone(&scene),
                    &sources,
                    &materials,
                    &mut actual,
                    16,
                    &emitters,
                    false,
                )
                .unwrap()
                .unwrap();
            control
                .trace_compact(
                    Arc::clone(&scene),
                    &sources,
                    &materials,
                    &mut expected,
                    16,
                    &emitters,
                    false,
                )
                .unwrap();
            pending.push((snapshot, actual, expected));
        }
        for (mut snapshot, mut actual, expected) in pending {
            assert!(candidate.snapshot_ready(&mut snapshot).unwrap());
            candidate.complete_readback(snapshot, &mut actual).unwrap();
            for (a, b) in actual.faces.iter().zip(&expected.faces) {
                assert_eq!((a.samples, a.sequence), (b.samples, b.sequence));
                for (x, y) in [
                    (a.radiance, b.radiance),
                    (a.second_moment, b.second_moment),
                    (a.direct, b.direct),
                ] {
                    let d = x - y;
                    assert!(
                        d.visible
                            .abs()
                            .max_element()
                            .max(d.infrared.abs())
                            .max(d.ionizing.abs())
                            < 1e-5
                    );
                }
            }
        }
        assert_eq!(candidate.readbacks.len(), 3);
    }
}
