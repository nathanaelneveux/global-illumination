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
    revision: u64,
    model_epoch: u64,
}

pub(super) struct ResidentAccumulator {
    slots: HashMap<Key, Slot>,
    regions: HashMap<IVec3, HashSet<Key>>,
    free: Vec<u32>,
    high_water: u32,
    scene: Option<Arc<TransportScene>>,
    origin: Instant,
    state: UploadBuffer,
    inputs: UploadBuffer,
    seeds: UploadBuffer,
    control: UploadBuffer,
    input_bytes: Vec<u8>,
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
            slots: default(),
            regions: default(),
            free: Vec::new(),
            high_water: 0,
            scene: None,
            origin: Instant::now(),
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
        work: &TraceWork,
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
        self.seed_bytes.clear();
        for job in &work.probes {
            for face in &work.faces[job.faces()] {
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
                        revision: u64::MAX,
                        model_epoch: u64::MAX,
                    }
                });
                let mut seed = u32::MAX;
                if slot.revision != job.revision
                    || (job.restart && slot.model_epoch != job.source_revision)
                {
                    slot.revision = job.revision;
                    slot.model_epoch = job.source_revision;
                    seed = (self.seed_bytes.len() / RECORD_BYTES) as u32;
                    encode_seed(&mut self.seed_bytes, face);
                }
                for word in [slot.index, seed, face.history_cap, 0] {
                    self.input_bytes.extend_from_slice(&word.to_le_bytes());
                }
            }
        }
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
        let control = [
            work.faces.len() as u32,
            samples,
            self.origin.elapsed().as_millis() as u32,
            HISTORY_LIMIT,
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

    pub(super) fn dispatch_prepare(
        &mut self,
        device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        tasks: &UploadBuffer,
        count: usize,
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
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.prepare);
        pass.set_bind_group(0, &**bindings, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }

    pub(super) fn dispatch_accumulate(
        &mut self,
        device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        output: &UploadBuffer,
        count: usize,
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
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.accumulate);
        pass.set_bind_group(0, &**bindings, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
    }
}

fn encode_seed(bytes: &mut Vec<u8>, face: &super::super::work::FaceWork) {
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
                    .prepare_inputs(&candidate.context, &empty, &blank, 16)
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
            for (work, probes) in [(&a, &mut expected), (&b, &mut actual)] {
                for (job, &i) in work.probes.iter().zip(&indices) {
                    for face in &work.faces[job.faces()] {
                        face.write(&mut probes[i]);
                    }
                }
            }
        }
        eprintln!(
            "RESIDENT_PARITY steps=20 five_bands=true growth=true subsets=true reorder=true retirement=true history_cap=true model_restart=true"
        );
    }
}
