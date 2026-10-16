//! Worker-owned stable face slots. Allocation identity is distinct from scene
//! freshness: unrelated streaming never resets a measurement. Geometry removal
//! retires only affected regions; recycled slots are initialized before use.
use super::super::world::{AIR, FACE_OFFSETS, REGION_EDGE};
use super::*;
use bevy::platform::collections::{HashMap, HashSet};

const RECORD_BYTES: usize = 144;
const DONOR_BYTES: usize = 8;
const SHADER: &str = include_str!("../../../assets/shaders/voxel_accumulate.wgsl");
type Key = (IVec3, usize);

#[path = "resident_topology.rs"]
mod topology;

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
    pub adaptive_reuse: bool,
    scene: Option<Arc<TransportScene>>,
    origin: Instant,
    #[cfg(test)]
    clock_ms: Option<u32>,
    state: UploadBuffer,
    inputs: UploadBuffer,
    seeds: UploadBuffer,
    control: UploadBuffer,
    spatial_jobs: UploadBuffer,
    spatial_bytes: Vec<u8>,
    spatial_donors: UploadBuffer,
    donor_bytes: Vec<u8>,
    evidence_gather: super::super::spatial::evidence::IndexedGather<UVec2>,
    evidence_topology: Option<topology::SurfaceTopology>,
    topology_new: Vec<Key>,
    evidence_target: u32,
    input_bytes: Vec<u8>,
    cap_bytes: Vec<u8>,
    seed_bytes: Vec<u8>,
    prepare: ComputePipeline,
    accumulate: ComputePipeline,
    spatial: ComputePipeline,
    evidence: ComputePipeline,
    prepare_bindings: Option<CachedBindings<5>>,
    accumulate_bindings: Option<CachedBindings<4>>,
    spatial_bindings: Option<CachedBindings<3>>,
    evidence_bindings: Option<CachedBindings<4>>,
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
        let spatial = pipeline("reconstruct_spatial");
        let evidence = pipeline("reconstruct_evidence");
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
            adaptive_reuse: false,
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
            spatial_jobs: UploadBuffer::new(device, "GI connected spatial neighborhoods", storage),
            spatial_bytes: Vec::new(),
            spatial_donors: UploadBuffer::new(device, "GI adaptive spatial donors", storage),
            donor_bytes: Vec::new(),
            evidence_gather: default(),
            evidence_topology: None,
            topology_new: Vec::new(),
            evidence_target: 0,
            input_bytes: Vec::new(),
            cap_bytes: Vec::new(),
            seed_bytes: Vec::new(),
            prepare,
            accumulate,
            spatial,
            evidence,
            prepare_bindings: None,
            accumulate_bindings: None,
            spatial_bindings: None,
            evidence_bindings: None,
        })
    }

    pub(super) fn prepare_inputs(
        &mut self,
        context: &GpuTransportDevice,
        scene: &Arc<TransportScene>,
        work: &mut TraceWork,
        samples: u32,
    ) -> Result<(), String> {
        let indexed_spatial = work.spatial_evidence_target > 0
            && work.spatial_reconstruction.is_finite()
            && work.spatial_reconstruction > 0.0;
        if !indexed_spatial {
            // No persistent adjacency overhead for the default/unfiltered or
            // fixed-stencil modes. Re-enabling rebuilds from current ownership.
            self.evidence_topology = None;
        }
        self.topology_new.clear();
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
                        if let Some(graph) = &mut self.evidence_topology {
                            graph.remove(slot.index);
                        }
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
                    if indexed_spatial {
                        self.topology_new.push(key);
                    }
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
                    if let Some(graph) = &mut self.evidence_topology {
                        graph.set_incarnation(slot.index, slot.incarnation);
                    }
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
        if indexed_spatial {
            if let Some(graph) = &mut self.evidence_topology {
                // All new nodes exist before linking; either insertion order
                // produces identical bidirectional coplanar adjacency.
                for &key in &self.topology_new {
                    graph.connect(key, &self.slots);
                }
            } else {
                self.evidence_topology = Some(topology::SurfaceTopology::build(
                    &self.slots,
                    self.high_water,
                ));
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
                if self.adaptive_reuse && work.faces[index].samples >= 1024 {
                    continue;
                }
                let voxel = work.faces[index].voxel;
                let orientation = work.faces[index].face;
                let normal = FACE_OFFSETS[orientation];
                // Prefer donors outside the immediate corner reconstruction
                // footprint, so sharing contributes new paths to that footprint.
                if self.adaptive_reuse {
                    let limit = if work.faces[index].samples < 256 {
                        4
                    } else {
                        2
                    };
                    let mut members = [index; 4];
                    let mut member_count = 1;
                    for distance in [3, 6, 1] {
                        for offset in FACE_OFFSETS.map(|v| v * distance) {
                            if offset.dot(normal) != 0 {
                                continue;
                            }
                            if let Some(slot) = self.slots.get(&(voxel + offset, orientation))
                                && slot.task_epoch == self.task_epoch
                            {
                                let other = slot.task_index as usize;
                                if other != index
                                    && work.faces[other].neighbor == 0
                                    && work.faces[other].samples
                                        < if limit == 4 { 256 } else { 1024 }
                                    && !members[..member_count].contains(&other)
                                {
                                    members[member_count] = other;
                                    member_count += 1;
                                }
                            }
                            if member_count == limit {
                                break;
                            }
                        }
                        if member_count == limit {
                            break;
                        }
                    }
                    if member_count > 1 {
                        for (i, &member) in members[..member_count].iter().enumerate() {
                            work.faces[member].neighbor =
                                members[(i + 1) % member_count] as u32 + 1;
                        }
                    }
                    continue;
                }
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
        self.spatial_bytes.clear();
        self.donor_bytes.clear();
        self.evidence_target = work.spatial_evidence_target.min(4096);
        let spatial_strength = if work.spatial_reconstruction.is_finite() {
            work.spatial_reconstruction.clamp(0.0, 1.0)
        } else {
            0.0
        };
        if spatial_strength > 0.0 {
            for (index, face) in work.faces.iter().enumerate() {
                if self.evidence_target > 0 {
                    let start = self.donor_bytes.len() / DONOR_BYTES;
                    // CPU feedback only bounds candidate work; current GPU
                    // histories decide support. Caps are applied before rays.
                    let predicted = face.samples.min(face.history_cap).saturating_add(samples);
                    if predicted < self.evidence_target {
                        let graph = self.evidence_topology.as_ref().expect("adaptive topology");
                        self.evidence_gather.collect(
                            face.resident,
                            super::super::spatial::evidence::candidate_limit(
                                self.evidence_target,
                                predicted,
                                work.spatial_max_donors,
                            ),
                            |address, edge| graph.neighbor(address.x, edge),
                        );
                        for address in self.evidence_gather.values().skip(1) {
                            let bytes = address.to_array().map(u32::to_le_bytes);
                            self.donor_bytes.extend_from_slice(bytes.as_flattened());
                        }
                    }
                    for word in [
                        face.resident.x,
                        face.resident.y,
                        start as u32,
                        (self.donor_bytes.len() / DONOR_BYTES - start) as u32,
                    ] {
                        self.spatial_bytes.extend_from_slice(&word.to_le_bytes());
                    }
                    continue;
                }
                for word in [face.resident.x, face.resident.y, index as u32, 0] {
                    self.spatial_bytes.extend_from_slice(&word.to_le_bytes());
                }
                for address in super::super::spatial::neighbors(face.voxel, face.face, |p, f| {
                    self.slots
                        .get(&(p, f))
                        .map(|slot| UVec2::new(slot.index, slot.incarnation))
                }) {
                    self.spatial_bytes.extend(
                        address
                            .unwrap_or(UVec2::ZERO)
                            .to_array()
                            .into_iter()
                            .flat_map(u32::to_le_bytes),
                    );
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
        if !self.spatial_bytes.is_empty() {
            self.spatial_jobs
                .upload(device, queue, &self.spatial_bytes)?;
        }
        if !self.donor_bytes.is_empty() {
            self.spatial_donors
                .upload(device, queue, &self.donor_bytes)?;
        }
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
            spatial_strength.to_bits(),
            self.evidence_target,
            work.display_frame,
            work.display_transition_frames,
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
        let spatial_bindings = if self.spatial_bytes.is_empty() || self.evidence_target > 0 {
            None
        } else {
            Some(cached_bindings(
                &mut self.spatial_bindings,
                device,
                || self.spatial.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (3, &self.state),
                    (6, &self.spatial_jobs),
                ],
            ))
        };
        let evidence_bindings = if self.spatial_bytes.is_empty() || self.evidence_target == 0 {
            None
        } else {
            Some(cached_bindings(
                &mut self.evidence_bindings,
                device,
                || self.evidence.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (3, &self.state),
                    (6, &self.spatial_jobs),
                    (7, &self.spatial_donors),
                ],
            ))
        };
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
        if let Some(bindings) = spatial_bindings {
            pass.set_pipeline(&self.spatial);
            pass.set_bind_group(0, &**bindings, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
        if let Some(bindings) = evidence_bindings {
            pass.set_pipeline(&self.evidence);
            pass.set_bind_group(0, &**bindings, &[]);
            pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
        }
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
    // The last metadata word was spare. Bounded periodic coordinates recover
    // exact spatial distances without uploading one distance per donor edge.
    words[31] = super::super::spatial::evidence::position_tag(face.voxel);
    // Initialize final presentation even before the first accumulation pass.
    words[7] = words[24];
    words[13] = words[25];
    words[14] = words[26];
    words[15] = words[27];
    words.copy_within(24..27, 32);
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
    #[ignore = "native frame-continuous RGB targets; unchanged physical and temporal feedback"]
    fn resident_display_transition_retargets_continuously_without_feedback() {
        let mut control = super::super::tests::native_tracer();
        control.enable_resident().unwrap();
        let mut gpu = GpuTracer::new(control.context.clone()).unwrap();
        gpu.enable_resident().unwrap();
        let fixture = super::super::refinement::research_fixture(2);
        let emitters = Arc::new(EmissionDistribution::build(&fixture.scene, &fixture.materials));
        let probe = SurfaceProbe::new(IVec3::new(8, 4, 8), 4);
        let mut a = TraceWork::from_probes(std::slice::from_ref(&probe), false);
        let mut b = TraceWork::from_probes(&[probe], false);
        b.display_transition_frames = 8;
        let rgb = |words: &[u32], offset: usize| {
            Vec3::new(f32::from_bits(words[offset]), f32::from_bits(words[offset + 1]), f32::from_bits(words[offset + 2]))
        };
        let mut previous: Option<Vec<u32>> = None;
        // Includes interrupted transitions, completion, counter wrap and a
        // submission one frame ahead of the visible renderer's counter.
        for (step, frame) in [u32::MAX - 5, u32::MAX - 1, 2, 20, 19].into_iter().enumerate() {
            a.display_frame = frame;
            b.display_frame = frame;
            for (tracer, work) in [(&mut control, &mut a), (&mut gpu, &mut b)] {
                tracer.resident.as_mut().unwrap().clock_ms = Some(1000 + step as u32 * 50);
                tracer.trace_compact(fixture.scene.clone(), &fixture.sources, &fixture.materials,
                    work, 16, &emitters, false).unwrap();
            }
            assert_eq!(a.faces[0].radiance, b.faces[0].radiance);
            assert_eq!(a.faces[0].second_moment, b.faces[0].second_moment);
            assert_eq!(a.faces[0].presentation, b.faces[0].presentation);
            assert_eq!(a.faces[0].samples, b.faces[0].samples);
            assert_eq!(a.faces[0].sequence, b.faces[0].sequence);
            let words = read_state(&gpu);
            assert_eq!(words[35], frame);
            assert_eq!(words[30], 8);
            if let Some(old) = previous {
                let age = frame.wrapping_sub(old[35]);
                let blend = if age > i32::MAX as u32 { 0.0 } else { (age as f32 / 8.0).min(1.0) };
                let target = Vec3::new(f32::from_bits(old[7]), f32::from_bits(old[13]), f32::from_bits(old[14]));
                let expected = rgb(&old, 32).lerp(target, blend);
                assert!((rgb(&words, 32) - expected).abs().max_element() < 0.000001);
            } else {
                assert_eq!(rgb(&words, 32), Vec3::ZERO);
            }
            previous = Some(words);
        }
    }

    fn read_state(gpu: &GpuTracer) -> Vec<u32> {
        let resident = gpu.resident.as_ref().unwrap();
        let device = &gpu.context.device;
        let size = resident.high_water as u64 * RECORD_BYTES as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial reconstruction test snapshot"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&default());
        encoder.copy_buffer_to_buffer(&resident.state.buffer, 0, &buffer, 0, size);
        let submission = gpu.context.queue.submit([encoder.finish()]);
        let (tx, rx) = mpsc::sync_channel(1);
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(5)),
            })
            .unwrap();
        rx.recv_timeout(Duration::from_secs(1)).unwrap().unwrap();
        let mapped = buffer.slice(..).get_mapped_range();
        let words = mapped
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        drop(mapped);
        buffer.unmap();
        words
    }

    #[test]
    #[ignore = "native spatial CPU/GPU oracle, unchanged transport/temporal snapshots, partial jobs and disable"]
    fn spatial_reconstruction_matches_oracle_without_feedback() {
        check_spatial_oracle(0, IVec3::ZERO, 0);
    }

    #[test]
    #[ignore = "native adaptive support oracle with heterogeneous edit histories and unchanged physical statistics"]
    fn evidence_reconstruction_matches_oracle_without_feedback() {
        for target in [512, 1024, 1536, 2048, 4096] {
            for shift in [IVec3::ZERO, IVec3::new(-16, 0, -16)] {
                check_spatial_oracle(target, shift, 0);
            }
        }
    }

    #[test]
    #[ignore = "native capped adaptive support oracle, uneven histories, edit caps and unchanged statistics"]
    fn capped_evidence_matches_oracle_without_feedback() {
        for cap in [32, 64, 128] {
            for shift in [IVec3::ZERO, IVec3::new(-16, 0, -16)] {
                check_spatial_oracle(2048, shift, cap);
            }
        }
    }

    fn check_spatial_oracle(target: u32, shift: IVec3, cap: u32) {
        use super::super::super::spatial::{self, Sample};
        let evidence = target > 0;
        let mut control = super::super::tests::native_tracer();
        control.enable_resident().unwrap();
        let mut gpu = GpuTracer::new(control.context.clone()).unwrap();
        gpu.enable_resident().unwrap();
        let fixture = super::super::refinement::research_fixture(2);
        let emitters = Arc::new(EmissionDistribution::build(
            &fixture.scene,
            &fixture.materials,
        ));
        let probes: Vec<_> = (6..if evidence { 26 } else { 13 })
            .flat_map(|z| {
                (6..if evidence { 26 } else { 15 })
                    .map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z) + shift, 4))
            })
            .collect();
        let mut a = TraceWork::from_probes(&probes, false);
        let mut b = TraceWork::from_probes(&probes, false);
        b.spatial_evidence_target = target;
        b.spatial_max_donors = cap;
        let mut changed = false;
        for step in 0..5 {
            let spp = if evidence {
                [1, 3, 16, 16, 16][step as usize]
            } else {
                16
            };
            if evidence && step == 2 {
                // Retained histories vary with distance from an edit. Seed
                // uneven weights once, then apply different caps without
                // discarding the means on the following step.
                for work in [&mut a, &mut b] {
                    for job in &mut work.probes {
                        job.revision += 1;
                    }
                    for (i, face) in work.faces.iter_mut().enumerate() {
                        face.samples = [1, 16, 128, 511, 2048][i % 5];
                    }
                }
            }
            if evidence && step == 3 {
                for work in [&mut a, &mut b] {
                    for (i, face) in work.faces.iter_mut().enumerate() {
                        face.history_cap = [1, 16, 128, 256, u32::MAX][i % 5];
                    }
                }
            }
            b.spatial_reconstruction = if step == 4 { 0.0 } else { 1.0 };
            for (tracer, work) in [(&mut control, &mut a), (&mut gpu, &mut b)] {
                tracer.resident.as_mut().unwrap().clock_ms = Some(1000 + step * 50);
                tracer
                    .trace_compact(
                        fixture.scene.clone(),
                        &fixture.sources,
                        &fixture.materials,
                        work,
                        spp,
                        &emitters,
                        false,
                    )
                    .unwrap();
            }
            for (a, b) in a.faces.iter().zip(&b.faces) {
                assert_eq!(a.radiance, b.radiance);
                assert_eq!(a.second_moment, b.second_moment);
                assert_eq!(a.samples, b.samples);
                assert_eq!(a.sequence, b.sequence);
                assert_eq!(
                    a.presentation, b.presentation,
                    "spatial filtering must not feed CPU scheduling or temporal history"
                );
            }
            let words = read_state(&gpu);
            let sample = |slot: &Slot| {
                let base = slot.index as usize * (RECORD_BYTES / 4);
                let vec = |offset| {
                    Vec3::new(
                        f32::from_bits(words[base + offset]),
                        f32::from_bits(words[base + offset + 1]),
                        f32::from_bits(words[base + offset + 2]),
                    )
                };
                Sample {
                    mean: vec(0),
                    second: vec(8),
                    count: words[base + 5],
                    display: vec(24),
                }
            };
            let slots = &gpu.resident.as_ref().unwrap().slots;
            let by_index: HashMap<_, _> = slots.iter().map(|(k, s)| (s.index, (k, s))).collect();
            let mut gather = spatial::evidence::Gather::default();
            let mut position_gather = spatial::evidence::Gather::default();
            for (key, slot) in slots {
                let center = sample(slot);
                let expected = if evidence && b.spatial_reconstruction > 0.0 {
                    // Replay the exact uploaded candidates; feedback may lag
                    // the current GPU count, so CPU work bounds need not match
                    // the resulting count. The weighting oracle still reads
                    // current independent moments/counts from native storage.
                    let resident = gpu.resident.as_ref().unwrap();
                    let header = &resident.spatial_bytes[slot.task_index as usize * 16..];
                    let word = |offset| {
                        u32::from_le_bytes(header[offset..offset + 4].try_into().unwrap()) as usize
                    };
                    assert!(cap == 0 || word(12) <= cap as usize);
                    position_gather.collect(
                        key.0,
                        key.1,
                        UVec2::new(slot.index, slot.incarnation),
                        word(12) + 1,
                        |p, f| {
                            slots
                                .get(&(p, f))
                                .map(|s| UVec2::new(s.index, s.incarnation))
                        },
                    );
                    gather.sites.clear();
                    gather.sites.push((IVec2::ZERO, center));
                    if b.spatial_reconstruction > 0.0 {
                        assert_eq!(position_gather.sites.len(), word(12) + 1);
                        for (expected, record) in position_gather.sites[1..].iter().zip(
                            resident.donor_bytes
                                [word(8) * DONOR_BYTES..(word(8) + word(12)) * DONOR_BYTES]
                                .chunks_exact(DONOR_BYTES),
                        ) {
                            let index = u32::from_le_bytes(record[..4].try_into().unwrap());
                            let incarnation = u32::from_le_bytes(record[4..8].try_into().unwrap());
                            assert_eq!(expected.1, UVec2::new(index, incarnation));
                            // Recover the real offset from the stable keys.
                            let (donor_key, donor_slot) = by_index[&index];
                            let delta = donor_key.0 - key.0;
                            let offset = match key.1 / 2 {
                                0 => IVec2::new(delta.y, delta.z),
                                1 => IVec2::new(delta.x, delta.z),
                                _ => IVec2::new(delta.x, delta.y),
                            };
                            assert_eq!(expected.0, offset);
                            gather.sites.push((offset, sample(donor_slot)));
                        }
                    }
                    spatial::evidence::resolve(&gather.sites, target, b.spatial_reconstruction).rgb
                } else {
                    spatial::reconstruct(
                        center,
                        spatial::neighbors(key.0, key.1, |p, f| slots.get(&(p, f)).map(sample)),
                        b.spatial_reconstruction,
                    )
                };
                let base = slot.index as usize * (RECORD_BYTES / 4);
                let actual = Vec3::new(
                    f32::from_bits(words[base + 7]),
                    f32::from_bits(words[base + 13]),
                    f32::from_bits(words[base + 14]),
                );
                assert!(
                    (expected - actual).abs().max_element() < 1e-5,
                    "{expected:?} vs {actual:?}"
                );
                changed |= (actual - center.display).length() > 1e-5;
            }
        }
        assert!(changed, "exercise a nontrivial spatial resolve");
    }

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
        check_resident_lifecycle(false);
    }

    #[test]
    #[ignore = "native indexed spatial lifecycle: growth, recycling, disable/re-enable and unchanged statistics"]
    fn indexed_spatial_preserves_measurements_through_lifecycle() {
        check_resident_lifecycle(true);
    }

    fn check_resident_lifecycle(spatial: bool) {
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
                if spatial {
                    blank.spatial_reconstruction = 1.0;
                    blank.spatial_evidence_target = 2048;
                }
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
            if spatial {
                b.spatial_reconstruction = if step == 8 { 0.0 } else { 1.0 };
                b.spatial_evidence_target = 2048;
            }
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
