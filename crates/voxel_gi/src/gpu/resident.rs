//! Worker-owned stable face slots. Allocation identity is distinct from scene
//! freshness: unrelated streaming never resets a measurement. Geometry removal
//! retires only affected regions; recycled slots are initialized before use.
use super::super::world::{AIR, FACE_OFFSETS, REGION_EDGE};
use super::*;
use bevy::platform::collections::{HashMap, HashSet};

const RECORD_BYTES: usize = 144;
/// Per-slot pre-merge statistics and merge revision (three uint4s).
const CHECKPOINT_BYTES: usize = 48;
/// Input word y: a seed index below this bit. Otherwise bit 30 requests a
/// rollback of the last merge and the low 30 bits are a history cap.
const NO_SEED: u32 = 0x8000_0000;
const ROLLBACK: u32 = 0x4000_0000;
const NO_CAP: u32 = 0x3fff_ffff;
/// Per-slot GPU metadata, three uint4s: identity `(voxel.xyz, face |
/// META_LIVE)` so GPU passes can address faces by slot alone, then
/// scheduling and ranking records owned by GPU passes (zeroed at allocation).
const META_BYTES: usize = 48;
const META_WORDS: u32 = (META_BYTES / 4) as u32;
const META_LIVE: u32 = 8;
/// GPU selection scratch: three 256-bucket histograms, thresholds,
/// remainders, tie counters and the pick total.
const SELECTION_WORDS: usize = 4 * 256 + 4 * 3 + 2;
/// A statistics-mirror record: a header and six uint4s. The GPU emits at most
/// `TraceWork::mirror_budget` per batch; the CPU merges only those.
pub(super) const MIRROR_RECORD_BYTES: usize = 7 * 16;
const SEL_MIRRORS: u64 = 1037;
static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_retention(include_str!(
        "../../shaders/voxel_accumulate.wgsl"
    ))
});
type Key = (IVec3, usize);

#[path = "scatter.rs"]
mod scatter;

#[derive(Clone, Copy)]
struct Slot {
    index: u32,
    incarnation: u32,
    /// Material id in the GPU identity (bits 16+): edit measurement's albedo.
    material: u8,
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
    scene: Option<Arc<TransportScene>>,
    origin: Instant,
    shared_view_cache: bool,
    /// Revokes writable render snapshots before ownership/scene maintenance.
    /// Ordinary sampling leaves the epoch alone: both command buffers advance
    /// the same statistics in GPU queue order.
    cache_epoch: u32,
    cache_guard: UploadBuffer,
    #[cfg(test)]
    clock_ms: Option<u32>,
    state: UploadBuffer,
    face_index: super::face_index::FaceIndex,
    face_index_buffer: UploadBuffer,
    index_scatter: UploadBuffer,
    index_scatter_bytes: Vec<u8>,
    index_scatter_count: u32,
    inputs: UploadBuffer,
    checkpoint: UploadBuffer,
    seeds: UploadBuffer,
    control: UploadBuffer,
    meta: UploadBuffer,
    meta_scatter: UploadBuffer,
    meta_writes: scatter::Scatter,
    meta_scatter_bindings: Option<CachedBindings<2>>,
    /// Published changes not yet applied on the GPU, oldest first.
    pub(super) events: Vec<super::super::retention::RetentionEvent>,
    event_bytes: Vec<u8>,
    event_count: u32,
    publish_view_changes: bool,
    pub(super) view_changes: Vec<super::view::ViewChange>,
    measure_bytes: Vec<u8>,
    measure_entries: u32,
    measure_positions: u32,
    measures: UploadBuffer,
    measure_max: UploadBuffer,
    measure_edits: ComputePipeline,
    finish_edits: ComputePipeline,
    measure_bindings: Option<CachedBindings<7>>,
    finish_bindings: Option<CachedBindings<4>>,
    /// Shadow footprints of this batch's changes (`plan_footprints`), traced
    /// by the transport module before preparation applies the events.
    footprint_bytes: Vec<u8>,
    /// Tests of the CPU-planned events alone turn footprints off.
    pub(super) trace_footprints: bool,
    footprint_threads: u32,
    footprint_count: u32,
    pub(super) footprint_tasks: UploadBuffer,
    pub(super) footprint_sums: UploadBuffer,
    /// Luminance (albedo, emission) per material id, for edit measurement.
    material_light: UploadBuffer,
    material_light_values: Vec<[f32; 2]>,
    events_buffer: UploadBuffer,
    apply_events: ComputePipeline,
    mark_changed: ComputePipeline,
    events_bindings: Option<CachedBindings<5>>,
    mark_bindings: Option<CachedBindings<4>>,
    /// Slot index -> face, for decoding GPU picks.
    by_index: Vec<Option<Key>>,
    /// GPU-picked faces (address, count after that merge) whose donor lists
    /// the next batch builds: the CPU learns of a pick only at readback.
    /// GPU pick budget and CPU traced faces of the prepared batch.
    picks: u32,
    cpu_faces: u32,
    selection: UploadBuffer,
    slot_keys: UploadBuffer,
    mirror_out: UploadBuffer,
    mirror_budget: u32,
    /// Bytes of `slot_keys` initialized (to none, or by classification).
    slot_keys_valid: u64,
    select_classify: ComputePipeline,
    select_threshold: ComputePipeline,
    select_emit: ComputePipeline,
    select_histogram: ComputePipeline,
    histogram_bindings: Option<CachedBindings<3>>,
    /// Slots classified this batch: one quarter of the range, rotating.
    classify_range: (u32, u32),
    classify_bindings: Option<CachedBindings<6>>,
    threshold_bindings: Option<CachedBindings<2>>,
    emit_bindings: Option<CachedBindings<7>>,
    input_bytes: Vec<u8>,
    cap_bytes: Vec<u8>,
    ensure_bytes: Vec<u8>,
    seed_bytes: Vec<u8>,
    prepare: ComputePipeline,
    accumulate: ComputePipeline,
    scatter: ComputePipeline,
    scatter_bindings: Option<CachedBindings<2>>,
    prepare_bindings: Option<CachedBindings<8>>,
    accumulate_bindings: Option<CachedBindings<6>>,
    words: ComputePipeline,
}

impl ResidentAccumulator {
    pub(super) fn new(context: &GpuTransportDevice) -> Result<Self, String> {
        let device = &context.device;
        let scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("GI persistent measurements"),
            source: ShaderSource::Wgsl(Cow::Borrowed(&SHADER)),
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
        let scatter = pipeline("scatter_face_index");
        let words = pipeline("scatter_words");
        let apply_events = pipeline("apply_events");
        let mark_changed = pipeline("mark_changed");
        let measure_edits = pipeline("measure_edits");
        let finish_edits = pipeline("finish_edits");
        let select_classify = pipeline("select_classify");
        let select_threshold = pipeline("select_threshold");
        let select_emit = pipeline("select_emit");
        let select_histogram = pipeline("select_histogram");
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
            scene: None,
            origin: Instant::now(),
            shared_view_cache: false,
            cache_epoch: 0,
            cache_guard: UploadBuffer::new(device, "GI cache write epoch", storage),
            #[cfg(test)]
            clock_ms: None,
            state: UploadBuffer::new(
                device,
                "GI resident measurements",
                storage | BufferUsages::COPY_SRC,
            ),
            face_index: super::face_index::FaceIndex::new(),
            face_index_buffer: UploadBuffer::new(
                device,
                "GI resident face index",
                storage | BufferUsages::COPY_SRC,
            ),
            index_scatter: UploadBuffer::new(device, "GI face index changes", storage),
            index_scatter_bytes: Vec::new(),
            index_scatter_count: 0,
            inputs: UploadBuffer::new(device, "GI resident jobs", storage | BufferUsages::COPY_SRC),
            checkpoint: UploadBuffer::new(
                device,
                "GI resident merge checkpoints",
                storage | BufferUsages::COPY_SRC,
            ),
            seeds: UploadBuffer::new(device, "GI new resident measurements", storage),
            meta: UploadBuffer::new(
                device,
                "GI resident slot metadata",
                storage | BufferUsages::COPY_SRC,
            ),
            meta_scatter: UploadBuffer::new(device, "GI slot metadata changes", storage),
            meta_writes: default(),
            meta_scatter_bindings: None,
            events: Vec::new(),
            event_bytes: Vec::new(),
            event_count: 0,
            publish_view_changes: false,
            view_changes: Vec::new(),
            measure_bytes: Vec::new(),
            measure_entries: 0,
            measure_positions: 0,
            measures: UploadBuffer::new(device, "GI edit measurements", storage),
            measure_max: UploadBuffer::new(device, "GI edit brightest light", storage),
            measure_edits,
            finish_edits,
            measure_bindings: None,
            finish_bindings: None,
            footprint_bytes: Vec::new(),
            trace_footprints: true,
            footprint_threads: 0,
            footprint_count: 0,
            footprint_tasks: UploadBuffer::new(device, "GI change footprints", storage),
            footprint_sums: UploadBuffer::new(device, "GI change footprint sums", storage),
            material_light: UploadBuffer::new(device, "GI material light", storage),
            material_light_values: Vec::new(),
            events_buffer: UploadBuffer::new(
                device,
                "GI retention events",
                storage | BufferUsages::COPY_SRC,
            ),
            apply_events,
            mark_changed,
            events_bindings: None,
            mark_bindings: None,
            by_index: Vec::new(),
            picks: 0,
            cpu_faces: 0,
            selection: UploadBuffer::new(
                device,
                "GI selection scratch",
                storage | BufferUsages::COPY_SRC,
            ),
            slot_keys: UploadBuffer::new(
                device,
                "GI selection keys",
                storage | BufferUsages::COPY_SRC,
            ),
            slot_keys_valid: 0,
            mirror_budget: 1024,
            mirror_out: UploadBuffer::new(
                device,
                "GI statistics mirror records",
                storage | BufferUsages::COPY_SRC,
            ),
            select_classify,
            select_threshold,
            select_emit,
            select_histogram,
            histogram_bindings: None,
            classify_range: (0, 0),
            classify_bindings: None,
            threshold_bindings: None,
            emit_bindings: None,
            control: UploadBuffer::new(
                device,
                "GI resident controls",
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            input_bytes: Vec::new(),
            cap_bytes: Vec::new(),
            ensure_bytes: Vec::new(),
            seed_bytes: Vec::new(),
            prepare,
            accumulate,
            scatter,
            scatter_bindings: None,
            prepare_bindings: None,
            accumulate_bindings: None,
            words,
        })
    }

    pub(super) fn prepare_inputs(
        &mut self,
        context: &GpuTransportDevice,
        scene: &Arc<TransportScene>,
        work: &mut TraceWork,
        samples: u32,
    ) -> Result<(), String> {
        let _span = info_span!("gi_prepare_inputs").entered();
        let changed_scene = self.scene.as_ref().is_none_or(|old| !Arc::ptr_eq(old, scene));
        if changed_scene || self.shared_view_cache != work.shared_view_cache
            || !work.ensure.is_empty() || !work.retire_pages.is_empty()
            || !work.retention_events.is_empty() || !self.events.is_empty()
            || work.probes.iter().any(|probe| probe.restart || work.faces[probe.faces()].iter().any(|face|
                self.slots.get(&(face.voxel, face.face)).is_none_or(|slot|
                    slot.revision != probe.revision || slot.material != face.material)))
        {
            self.cache_epoch = self.cache_epoch.checked_add(1).ok_or("cache write epoch exhausted")?;
            self.cache_guard.upload(&context.device, &context.queue, &self.cache_epoch.to_le_bytes())?;
        }
        self.shared_view_cache = work.shared_view_cache;
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
                        self.face_index.remove(key.0, key.1);
                        self.free.push(slot.index);
                        self.meta_writes.write(slot.index * META_WORDS + 3, 0);
                        self.by_index[slot.index as usize] = None;
                    }
                    present
                });
                if keys.is_empty() {
                    self.regions.remove(&region);
                }
            }
        }
        // Pages whose receivers were retired (receiver radius): release every
        // slot in them, as geometry removal does above.
        let per_page = (super::super::world::PAGE_EDGE / REGION_EDGE) as i32;
        for &page in &work.retire_pages {
            for z in 0..per_page {
                for y in 0..per_page {
                    for x in 0..per_page {
                        let region = page * per_page + IVec3::new(x, y, z);
                        let Some(keys) = self.regions.remove(&region) else {
                            continue;
                        };
                        for key in keys {
                            let Some(slot) = self.slots.remove(&key) else {
                                continue;
                            };
                            self.face_index.remove(key.0, key.1);
                            self.free.push(slot.index);
                            self.meta_writes.write(slot.index * META_WORDS + 3, 0);
                            self.by_index[slot.index as usize] = None;
                        }
                    }
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
                        // No seed, no rollback; the cap rides in word z.
                        for word in [slot.index, NO_SEED | NO_CAP, cap, u32::MAX] {
                            self.cap_bytes.extend_from_slice(&word.to_le_bytes());
                        }
                    }
                }
            }
            for face in &mut work.faces[job.faces()] {
                let key = (face.voxel, face.face);
                if !self.slots.contains_key(&key) {
                    let slot = self.allocate_slot(key, face.material);
                    self.slots.insert(key, slot);
                }
                let slot = self.slots.get_mut(&key).unwrap();
                if slot.material != face.material {
                    // Material replacement (the probe was relit).
                    slot.material = face.material;
                    self.meta_writes.write(
                        slot.index * META_WORDS + 3,
                        key.1 as u32 | META_LIVE | (u32::from(face.material) << 16),
                    );
                }
                // GPU state is authoritative: only a new slot is seeded. A
                // changed probe revision caps the GPU history (zero on a model
                // reset). If the CPU rejected this face's last merged batch
                // (selected before the change, in flight then), the GPU first
                // undoes that merge, exactly as the CPU snapshot excludes it.
                let mut seed = NO_SEED | NO_CAP;
                if slot.incarnation == 0 {
                    self.incarnation = self
                        .incarnation
                        .checked_add(1)
                        .ok_or("resident allocation incarnation exhausted")?;
                    slot.incarnation = self.incarnation;
                    slot.revision = job.revision;
                    slot.model_epoch = job.source_revision;
                    self.meta_writes
                        .write(slot.index * META_WORDS + 6, job.revision as u32);
                    seed = (self.seed_bytes.len() / RECORD_BYTES) as u32;
                    encode_seed(&mut self.seed_bytes, face, slot.incarnation, self.origin);
                } else if slot.revision != job.revision
                    || (job.restart && slot.model_epoch != job.source_revision)
                {
                    slot.revision = job.revision;
                    slot.model_epoch = job.source_revision;
                    self.meta_writes
                        .write(slot.index * META_WORDS + 6, job.revision as u32);
                    // The explicit relight/stale/source cap (zero on restart),
                    // never the CPU's count: that mirror can lag the GPU.
                    seed = NO_SEED | face.revision_cap.min(NO_CAP);
                }
                face.resident = UVec2::new(slot.index, slot.incarnation);
                if seed >= NO_SEED && face.rollback && !work.shared_view_cache {
                    // Shared RGB history has additional render-queue merges.
                    // Its GPU edit events, not a lagging CPU snapshot, own
                    // retention; rolling back a worker checkpoint would erase
                    // newer visible samples as well as the rejected batch.
                    seed |= ROLLBACK;
                }
                if seed < NO_SEED {
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
        // Faces published but not traced yet get their slot and seed now, so
        // every face has GPU state early. After the traced faces, so a face in
        // both lists is seeded once, by its trace input.
        self.ensure_bytes.clear();
        for ensure in &work.ensure {
            let key = (ensure.face.voxel, ensure.face.face);
            // The scene may have dropped the face since publication; only the
            // current surface gets a slot, or retirement would never see it.
            let present = scene.material_at(key.0).is_some_and(|m| m != AIR)
                && scene
                    .material_at(key.0 + FACE_OFFSETS[key.1])
                    .is_none_or(|m| m == AIR);
            if !present {
                continue;
            }
            // Already resident (a probe recreated on the CPU, or traced
            // first): the request still learns the slot's address.
            if let Some(slot) = self.slots.get(&key) {
                if slot.incarnation != 0 {
                    self.bindings.push(ResidentBinding {
                        voxel: key.0,
                        face: key.1,
                        revision: ensure.revision,
                        source_revision: ensure.source_revision,
                        address: UVec2::new(slot.index, slot.incarnation),
                    });
                }
                continue;
            }
            let mut slot = self.allocate_slot(key, ensure.face.material);
            self.incarnation = self
                .incarnation
                .checked_add(1)
                .ok_or("resident allocation incarnation exhausted")?;
            slot.incarnation = self.incarnation;
            slot.revision = ensure.revision;
            slot.model_epoch = ensure.source_revision;
            self.meta_writes
                .write(slot.index * META_WORDS + 6, ensure.revision as u32);
            let seed = (self.seed_bytes.len() / RECORD_BYTES) as u32;
            encode_seed(
                &mut self.seed_bytes,
                &ensure.face,
                slot.incarnation,
                self.origin,
            );
            self.bindings.push(ResidentBinding {
                voxel: key.0,
                face: key.1,
                revision: ensure.revision,
                source_revision: ensure.source_revision,
                address: UVec2::new(slot.index, slot.incarnation),
            });
            // Seed-only record: no trace task, no cap.
            for word in [slot.index, seed, u32::MAX, u32::MAX] {
                self.ensure_bytes.extend_from_slice(&word.to_le_bytes());
            }
            self.slots.insert(key, slot);
        }
        // GPU picks append after the CPU's traced faces: unused records stay
        // 0xffffffff, which every pass skips.
        self.cpu_faces = (self.input_bytes.len() / 16) as u32;
        debug_assert_eq!(self.cpu_faces as usize, work.faces.len());
        self.picks = work.picks;
        self.input_bytes
            .resize(self.input_bytes.len() + work.picks as usize * 16, 0xff);
        self.input_bytes.extend_from_slice(&self.cap_bytes);
        self.input_bytes.extend_from_slice(&self.ensure_bytes);
        self.meta_writes.finish();
        let device = &context.device;
        let queue = &context.queue;
        self.ensure_cache_generation(context)?;
        self.index_scatter_count = info_span!("gi_face_index_upload").in_scope(|| {
            self.face_index.upload(
                &mut self.face_index_buffer,
                device,
                queue,
                &mut self.index_scatter_bytes,
            )
        })?;
        if self.index_scatter_count > 0 {
            self.index_scatter
                .upload(device, queue, &self.index_scatter_bytes)?;
        }
        self.inputs.upload(device, queue, &self.input_bytes)?;
        {
            let previous = self.checkpoint.buffer.clone();
            let capacity = self.checkpoint.capacity;
            if self.checkpoint.ensure(
                device,
                (self.high_water as usize * CHECKPOINT_BYTES).max(16),
            )? {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Preserve GI merge checkpoints during growth"),
                });
                encoder.copy_buffer_to_buffer(&previous, 0, &self.checkpoint.buffer, 0, capacity);
                queue.submit([encoder.finish()]);
            }
        }
        self.seeds.upload(device, queue, &self.seed_bytes)?;
        self.events.append(&mut work.retention_events);
        self.publish_view_changes = work.view_transport;
        let footprints = if self.trace_footprints {
            plan_footprints(&mut self.events)
        } else {
            Vec::new()
        };
        let encoded = encode_events(&mut self.event_bytes, &mut self.measure_bytes, &self.events);
        self.footprint_count = footprints.len() as u32;
        self.footprint_threads =
            encode_footprints(&mut self.footprint_bytes, &footprints, &self.events);
        if self.footprint_count > 0 {
            self.footprint_tasks.upload(device, queue, &self.footprint_bytes)?;
            let mut sums = Vec::with_capacity(footprints.len() * 32);
            for _ in &footprints {
                for word in [u32::MAX, u32::MAX, u32::MAX, 0, 0, 0, 0, 0] {
                    sums.extend_from_slice(&word.to_le_bytes());
                }
            }
            self.footprint_sums.upload(device, queue, &sums)?;
        }
        if self.high_water == 0 {
            // With no world receivers there is no outgoing-light measurement.
            // Zero observations are not evidence of a dark edit: advertise
            // unknown flux so all consumers use the shared distance rule.
            for bound in self.event_bytes[encoded.events as usize * 48..].chunks_exact_mut(32) {
                let kind = u32::from_le_bytes(bound[28..32].try_into().unwrap());
                if kind & 7 == 6 {
                    bound[28..32].copy_from_slice(&(kind & !7).to_le_bytes());
                }
            }
        }
        self.event_count = encoded.events;
        self.measure_entries = if self.high_water == 0 { 0 } else { encoded.entries };
        self.measure_positions = if self.high_water == 0 { 0 } else { encoded.positions };
        if self.event_count > 0 {
            self.events_buffer
                .upload(device, queue, &self.event_bytes)?;
        }
        if self.measure_entries > 0 {
            self.measures.upload(device, queue, &self.measure_bytes)?;
            self.measure_max
                .ensure(device, self.measure_entries as usize * 4)?;
        }
        self.selection.ensure(device, SELECTION_WORDS * 4)?;
        self.mirror_budget = work.mirror_budget.max(1);
        self.mirror_out
            .ensure(device, self.mirror_budget as usize * MIRROR_RECORD_BYTES)?;
        {
            // Keys persist across batches (rotating classification): keep
            // them through growth; new slots have none until classified.
            let previous = self.slot_keys.buffer.clone();
            let capacity = self.slot_keys.capacity;
            if self
                .slot_keys
                .ensure(device, (self.high_water as usize * 4).max(16))?
            {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Preserve GI selection keys during growth"),
                });
                encoder.copy_buffer_to_buffer(&previous, 0, &self.slot_keys.buffer, 0, capacity);
                queue.submit([encoder.finish()]);
                // Keys never written (including the initial allocation's,
                // which read zero: nearest coverage) start as none.
                let valid = self.slot_keys_valid.min(capacity);
                let fresh = vec![0xff; (self.slot_keys.capacity - valid) as usize];
                queue.write_buffer(&self.slot_keys.buffer, valid, &fresh);
                self.slot_keys_valid = self.slot_keys.capacity;
            }
        }
        let chunk = self.high_water.div_ceil(4);
        let first = (work.batch_id % 4) * chunk;
        self.classify_range = (
            first.min(self.high_water),
            (first + chunk).min(self.high_water),
        );
        {
            let previous = self.meta.buffer.clone();
            let capacity = self.meta.capacity;
            if self
                .meta
                .ensure(device, (self.high_water as usize * META_BYTES).max(16))?
            {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Preserve GI slot metadata during growth"),
                });
                encoder.copy_buffer_to_buffer(&previous, 0, &self.meta.buffer, 0, capacity);
                queue.submit([encoder.finish()]);
            }
        }
        if self.meta_writes.count > 0 {
            self.meta_scatter
                .upload(device, queue, &self.meta_writes.bytes)?;
        }
        let milliseconds = self.origin.elapsed().as_millis() as u32;
        #[cfg(test)]
        let milliseconds = self.clock_ms.unwrap_or(milliseconds);
        let control = [
            self.cpu_faces + self.picks,
            samples,
            milliseconds,
            if work.history_limit == 0 {
                HISTORY_LIMIT
            } else {
                work.history_limit.min(HISTORY_LIMIT)
            },
            work.response_seconds.max(0.001).to_bits(),
            (self.input_bytes.len() / 16) as u32,
            0, // reserved (was world display reconstruction)
            0,
            0, // reserved (was the render transition clock)
            0,
            if work.history_recency >= 1.0 {
                work.history_recency.to_bits()
            } else {
                1.0f32.to_bits()
            },
            work.measured_retention.max(0.0).to_bits(),
            0,
            self.event_count,
            self.high_water,
            (super::super::retention::INFLUENCE_RADIUS as u32).pow(2),
            work.transport_revision as u32,
            work.stable_error.to_bits(),
            work.stable_samples,
            work.audit_ms,
            work.focus.x as u32,
            work.focus.y as u32,
            work.focus.z as u32,
            self.picks,
            self.cpu_faces,
            work.batch_id,
            self.classify_range.0,
            self.classify_range.1,
            self.measure_positions,
            self.measure_entries,
            (super::super::geometry::MEASURED_RELIGHT_RADIUS as u32).pow(2),
            (super::super::geometry::RELIGHT_RADIUS as u32).pow(2),
            self.mirror_budget,
            0,
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

    /// A free slot for `key`, registered in the region map and face index.
    /// Seeding is the caller's.
    fn allocate_slot(&mut self, key: Key, material: u8) -> Slot {
        let region = key.0.div_euclid(IVec3::splat(REGION_EDGE as i32));
        self.regions.entry(region).or_default().insert(key);
        let index = self.free.pop().unwrap_or_else(|| {
            let index = self.high_water;
            self.high_water += 1;
            index
        });
        self.face_index.insert(key.0, key.1, index);
        if self.by_index.len() <= index as usize {
            self.by_index.resize(index as usize + 1, None);
        }
        self.by_index[index as usize] = Some(key);
        let base = index * META_WORDS;
        for (word, value) in [
            key.0.x as u32,
            key.0.y as u32,
            key.0.z as u32,
            key.1 as u32 | META_LIVE | (u32::from(material) << 16),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        .into_iter()
        .enumerate()
        {
            self.meta_writes.write(base + word as u32, value);
        }
        Slot {
            index,
            incarnation: 0,
            material,
            revision: u64::MAX,
            model_epoch: u64::MAX,
            task_epoch: 0,
            task_index: 0,
        }
    }

    /// The material table's luminance albedo and emission, for measuring
    /// edits; uploaded when it changes.
    pub(super) fn set_materials(
        &mut self,
        context: &GpuTransportDevice,
        materials: &VoxelMaterialTransportTable,
    ) -> Result<(), String> {
        let w = Vec3::new(0.2126, 0.7152, 0.0722);
        let values: Vec<[f32; 2]> = (0..=u8::MAX)
            .map(|i| {
                let m = materials.get(i);
                [w.dot(m.diffuse), w.dot(m.emission)]
            })
            .collect();
        if values != self.material_light_values {
            let bytes: Vec<u8> = values
                .iter()
                .flatten()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            self.material_light
                .upload(&context.device, &context.queue, &bytes)?;
            self.material_light_values = values;
        }
        Ok(())
    }

    /// The prepared batch's GPU pick records: buffer, byte offset and count.
    pub(super) fn pick_records(&self) -> (&Buffer, u64, u32) {
        (
            &self.inputs.buffer,
            u64::from(self.cpu_faces) * 16,
            self.picks,
        )
    }

    /// The statistics mirror of the prepared batch: the buffer and byte
    /// offset of its record count, the records, and their budget.
    pub(super) fn mirror_buffers(&self) -> (&Buffer, u64, &Buffer, u32) {
        (
            &self.selection.buffer,
            SEL_MIRRORS * 4,
            &self.mirror_out.buffer,
            self.mirror_budget,
        )
    }

    /// The face a pick record names, if its slot still holds that allocation.
    pub(super) fn picked_face(&self, slot: u32, incarnation: u32) -> Option<Key> {
        let key = (*self.by_index.get(slot as usize)?)?;
        (self.slots.get(&key)?.incarnation == incarnation).then_some(key)
    }

    /// Measurements and face ownership read by path termination.
    /// Keep the index and its addressed measurements in one allocation
    /// generation. A render-view consumer may retain both across worker
    /// batches: growing just one would let an old index address recycled slots
    /// or a new index address beyond an old measurement allocation.
    fn ensure_cache_generation(&mut self, context: &GpuTransportDevice) -> Result<(), String> {
        let state_bytes = self.high_water as usize * RECORD_BYTES;
        let index_bytes = self.face_index.gpu_bytes();
        if state_bytes as u64 <= self.state.capacity
            && index_bytes as u64 <= self.face_index_buffer.capacity
        {
            return Ok(());
        }
        let device = &context.device;
        let mut state = UploadBuffer::new(device, self.state.label, self.state.usage);
        let mut index = UploadBuffer::new(
            device,
            self.face_index_buffer.label,
            self.face_index_buffer.usage,
        );
        state.ensure(device, state_bytes.max(self.state.capacity as usize))?;
        index.ensure(
            device,
            index_bytes.max(self.face_index_buffer.capacity as usize),
        )?;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Preserve GI transport cache generation"),
        });
        encoder.copy_buffer_to_buffer(&self.state.buffer, 0, &state.buffer, 0, self.state.capacity);
        context.queue.submit([encoder.finish()]);
        self.state = state;
        self.face_index_buffer = index;
        self.face_index.require_full_upload();
        Ok(())
    }

    /// Ordinary updates remain queue-ordered, in-place transactions. Only
    /// allocation changes freeze an old pair; this is not a per-batch snapshot.
    /// Threads and tasks of the prepared batch's shadow footprints.
    pub(super) fn footprints(&self) -> Option<(u32, u32)> {
        (self.footprint_count > 0).then_some((self.footprint_threads, self.footprint_count))
    }

    pub(super) fn events_buffer(&self) -> &UploadBuffer {
        &self.events_buffer
    }

    pub(super) fn cache_buffers(&self) -> (&UploadBuffer, &UploadBuffer) {
        (&self.state, &self.face_index_buffer)
    }

    pub(super) fn view_cache(&self) -> super::view::CacheBinding {
        super::view::CacheBinding {
            meta: self.meta.buffer.clone(),
            guard: self.cache_guard.buffer.clone(),
            epoch: self.cache_epoch,
            enabled: self.shared_view_cache,
            origin: self.origin,
        }
    }

    pub(super) fn dispatch_prepare(
        &mut self,
        device: &RenderDevice,
        encoder: &mut wgpu::CommandEncoder,
        tasks: &UploadBuffer,
        count: usize,
        timestamps: Option<&wgpu::QuerySet>,
    ) {
        let view_event_count = if self.publish_view_changes {
            self.event_count
        } else {
            0
        };
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
                (12, &self.checkpoint),
                (13, &self.meta),
                (16, &self.slot_keys),
            ],
        );
        if self.picks > 0 {
            encoder.clear_buffer(&self.selection.buffer, 0, None);
        }
        if self.measure_entries > 0 {
            encoder.clear_buffer(&self.measure_max.buffer, 0, None);
        }
        let scatter_bindings = (self.index_scatter_count > 0).then(|| {
            cached_bindings(
                &mut self.scatter_bindings,
                device,
                || self.scatter.get_bind_group_layout(0).into(),
                [(8, &self.face_index_buffer), (9, &self.index_scatter)],
            )
            .clone()
        });
        let words = &self.words;
        let meta_scatter = (self.meta_writes.count > 0).then(|| {
            cached_bindings(
                &mut self.meta_scatter_bindings,
                device,
                || words.get_bind_group_layout(0).into(),
                [(10, &self.meta), (11, &self.meta_scatter)],
            )
            .clone()
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI resident task preparation"),
            timestamp_writes: timestamps.map(|query_set| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(0),
                end_of_pass_write_index: Some(1),
            }),
        });
        // Face-index changes land before the trace pass reads the index.
        if let Some(scatter) = &scatter_bindings {
            pass.set_pipeline(&self.scatter);
            pass.set_bind_group(0, &**scatter, &[]);
            pass.dispatch_workgroups(self.index_scatter_count.div_ceil(64), 1, 1);
        }
        if let Some(scatter) = &meta_scatter {
            pass.set_pipeline(&self.words);
            pass.set_bind_group(0, &**scatter, &[]);
            pass.dispatch_workgroups(self.meta_writes.count.div_ceil(64), 1, 1);
        }
        // Consumed only once dispatched: a prepare without a dispatch (a
        // retirement-only update) keeps its writes for the next upload.
        self.meta_writes.clear();
        pass.set_pipeline(&self.prepare);
        pass.set_bind_group(0, &**bindings, &[]);
        debug_assert!(count <= self.input_bytes.len() / 16);
        pass.dispatch_workgroups((self.input_bytes.len() as u32 / 16).div_ceil(64), 1, 1);
        // Edits' flux, from the GPU's statistics, before retention uses it.
        if self.measure_entries > 0 {
            let measure = cached_bindings(
                &mut self.measure_bindings,
                device,
                || self.measure_edits.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (3, &self.state),
                    (8, &self.face_index_buffer),
                    (13, &self.meta),
                    (19, &self.material_light),
                    (20, &self.measures),
                    (21, &self.measure_max),
                ],
            );
            pass.set_pipeline(&self.measure_edits);
            pass.set_bind_group(0, &**measure, &[]);
            pass.dispatch_workgroups(self.measure_positions.div_ceil(64), 1, 1);
            let finish = cached_bindings(
                &mut self.finish_bindings,
                device,
                || self.finish_edits.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (14, &self.events_buffer),
                    (20, &self.measures),
                    (21, &self.measure_max),
                ],
            );
            pass.set_pipeline(&self.finish_edits);
            pass.set_bind_group(0, &**finish, &[]);
            pass.dispatch_workgroups(self.measure_entries.div_ceil(64), 1, 1);
            self.measure_entries = 0;
        }
        // Retention after seeds and source caps, before the traced faces'
        // changes are marked and their rays read the cache.
        if self.event_count > 0 {
            let events = cached_bindings(
                &mut self.events_bindings,
                device,
                || self.apply_events.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (3, &self.state),
                    (12, &self.checkpoint),
                    (13, &self.meta),
                    (14, &self.events_buffer),
                ],
            );
            pass.set_pipeline(&self.apply_events);
            pass.set_bind_group(0, &**events, &[]);
            pass.dispatch_workgroups(self.high_water.div_ceil(64), 1, 1);
            self.events.clear();
            self.event_count = 0;
        }
        drop(pass);
        if view_event_count > 0 {
            // Copy AFTER edit-flux measurement, not its CPU prior, and before
            // a later worker batch reuses the event buffer. All view domains
            // consume the same event and distance law, without a CPU readback.
            let events = device.create_buffer(&BufferDescriptor {
                label: Some("GI immutable measured view changes"),
                size: self.event_bytes.len() as u64,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(
                &self.events_buffer.buffer,
                0,
                &events,
                0,
                self.event_bytes.len() as u64,
            );
            let reach = super::super::retention::INFLUENCE_RADIUS as u32;
            let bytes = [view_event_count, reach * reach, 0, 0].map(u32::to_le_bytes);
            let control = device.create_buffer_with_data(
                &bevy::render::render_resource::BufferInitDescriptor {
                    label: Some("GI view event descriptor"),
                    contents: bytes.as_flattened(),
                    usage: BufferUsages::UNIFORM,
                },
            );
            self.view_changes
                .push(super::view::ViewChange { events, control });
        }
        // GPU picks, in their own pass (own timing): after retention (which
        // may revoke stability), before the traced faces' changes are marked.
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GI resident selection"),
            timestamp_writes: timestamps.filter(|_| self.picks > 0).map(|query_set| {
                wgpu::ComputePassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(6),
                    end_of_pass_write_index: Some(7),
                }
            }),
        });
        if self.picks > 0 {
            let classify = cached_bindings(
                &mut self.classify_bindings,
                device,
                || self.select_classify.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (3, &self.state),
                    (13, &self.meta),
                    (15, &self.selection),
                    (16, &self.slot_keys),
                    (17, &self.mirror_out),
                ],
            );
            let (first, end) = self.classify_range;
            if end > first {
                pass.set_pipeline(&self.select_classify);
                pass.set_bind_group(0, &**classify, &[]);
                pass.dispatch_workgroups((end - first).div_ceil(256), 1, 1);
            }
            let histogram = cached_bindings(
                &mut self.histogram_bindings,
                device,
                || self.select_histogram.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (15, &self.selection),
                    (16, &self.slot_keys),
                ],
            );
            pass.set_pipeline(&self.select_histogram);
            pass.set_bind_group(0, &**histogram, &[]);
            pass.dispatch_workgroups(self.high_water.div_ceil(256), 1, 1);
            let threshold = cached_bindings(
                &mut self.threshold_bindings,
                device,
                || self.select_threshold.get_bind_group_layout(0).into(),
                [(0, &self.control), (15, &self.selection)],
            );
            pass.set_pipeline(&self.select_threshold);
            pass.set_bind_group(0, &**threshold, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            let emit = cached_bindings(
                &mut self.emit_bindings,
                device,
                || self.select_emit.get_bind_group_layout(0).into(),
                [
                    (0, &self.control),
                    (1, tasks),
                    (3, &self.state),
                    (4, &self.inputs),
                    (13, &self.meta),
                    (15, &self.selection),
                    (16, &self.slot_keys),
                ],
            );
            pass.set_pipeline(&self.select_emit);
            pass.set_bind_group(0, &**emit, &[]);
            pass.dispatch_workgroups(self.high_water.div_ceil(256), 1, 1);
        }
        let mark = cached_bindings(
            &mut self.mark_bindings,
            device,
            || self.mark_changed.get_bind_group_layout(0).into(),
            [
                (0, &self.control),
                (3, &self.state),
                (4, &self.inputs),
                (13, &self.meta),
            ],
        );
        pass.set_pipeline(&self.mark_changed);
        pass.set_bind_group(0, &**mark, &[]);
        pass.dispatch_workgroups((count as u32).div_ceil(64), 1, 1);
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
                (12, &self.checkpoint),
                (13, &self.meta),
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

/// Changed boxes per batch whose shadows the GPU traces; the rest keep only
/// their own retention.
const MAX_FOOTPRINTS: usize = 64;
/// Tubes per footprint side: one per voxel of the box's diagonal, at most this.
const FOOTPRINT_GRID: f32 = 64.0;

/// Reserve one Light event per edited box (player edits first, then unknown
/// changes), with an empty bound the GPU fills with where the box's shadow
/// lands (`trace_footprints`). Returns each box and its event. Loads get
/// none: streaming loads arrive continuously, and their conservative
/// footprints (a loaded region's shadow lands on ground that is often new
/// or no receiver) cut history all around the player (house reveal FLIP
/// .012 -> .10). A load's real sunlight changes on visible faces are their
/// observed corner flips (`report_sun_flips`).
fn plan_footprints(
    events: &mut Vec<super::super::retention::RetentionEvent>,
) -> Vec<(super::super::geometry::Bounds, usize)> {
    use super::super::geometry::{Bounds, ChangeKind, ChangeWeight};
    let mut boxes = Vec::new();
    for edits in [true, false] {
        for event in events.iter().filter(|event| event.source_cap.is_none()) {
            for (bounds, weight) in event.bounds.iter().zip(&event.weights) {
                let wanted = match weight.kind {
                    ChangeKind::Edit { .. } => edits,
                    ChangeKind::Unknown => !edits,
                    ChangeKind::Load { .. } | ChangeKind::Unload | ChangeKind::Light => false,
                };
                if wanted && boxes.len() < MAX_FOOTPRINTS {
                    boxes.push(*bounds);
                }
            }
        }
    }
    let empty = Bounds {
        min: IVec3::splat(1 << 29),
        max: IVec3::splat(-(1 << 29)),
    };
    boxes
        .into_iter()
        .map(|bounds| {
            events.push(super::super::retention::RetentionEvent {
                bounds: vec![empty],
                weights: vec![ChangeWeight { kind: ChangeKind::Light, flux: 0.0 }],
                union: empty,
                urgent: vec![false],
                source_cap: None,
                spheres: Vec::new(),
            });
            (bounds, events.len() - 1)
        })
        .collect()
}

/// Footprint tasks for `trace_footprints`: a count, then per task `(min,
/// header record) (max, bound record) (first thread, grid, 0, 0)`, records
/// as `encode_events` lays them out. Returns the thread count.
fn encode_footprints(
    bytes: &mut Vec<u8>,
    footprints: &[(super::super::geometry::Bounds, usize)],
    events: &[super::super::retention::RetentionEvent],
) -> u32 {
    bytes.clear();
    // Bounds follow every header, each event's after the previous event's.
    let mut first = 0u32;
    let mut first_bound = Vec::with_capacity(events.len());
    for event in events {
        first_bound.push(first);
        first += if event.source_cap.is_some() {
            event.spheres.len() as u32
        } else {
            event.bounds.len() as u32
        };
    }
    let records = events.len() as u32 * 3;
    let push = |bytes: &mut Vec<u8>, words: [i32; 4]| {
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    };
    push(bytes, [footprints.len() as i32, 0, 0, 0]);
    let mut threads = 0u32;
    for &(bounds, event) in footprints {
        let (min, max) = (bounds.min, bounds.max);
        let grid = ((max - min + IVec3::ONE).as_vec3().length().ceil()).clamp(1.0, FOOTPRINT_GRID) as u32;
        push(bytes, [min.x, min.y, min.z, event as i32 * 3]);
        push(bytes, [max.x, max.y, max.z, (records + first_bound[event] * 2) as i32]);
        push(bytes, [threads as i32, grid as i32, 0, 0]);
        threads += grid * grid;
    }
    threads
}

/// Event headers `(union.min, flags) (union.max, 0) (cap, 0, first bound,
/// bounds)`, then bounds `(min, flux) (max, kind | top << 3)`. Kinds: 0
/// unknown, 1 unload, 2 load, 3 edit, 4 light. Tops saturate to 28 bits (an
/// air-only load's i32::MIN replaces no sky either way). Flag bit 1: a source
/// change; `cap` applies everywhere and its bounds are spheres
/// `(center, radius^2)` (f32), within which the cap is 4.
fn encode_events(
    bytes: &mut Vec<u8>,
    measures: &mut Vec<u8>,
    events: &[super::super::retention::RetentionEvent],
) -> EncodedEvents {
    use super::super::geometry::ChangeKind;
    bytes.clear();
    measures.clear();
    let event_count = events.len() as u32;
    let mut entries = 0u32;
    let mut positions = 0u32;
    let saturate = |top: i32| top.clamp(-(1 << 27), (1 << 27) - 1);
    let mut first = 0u32;
    let push = |bytes: &mut Vec<u8>, words: [u32; 4]| {
        for word in words {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    };
    for event in events {
        if let Some(cap) = event.source_cap {
            let count = event.spheres.len() as u32;
            push(bytes, [0, 0, 0, 2]);
            push(bytes, [0; 4]);
            push(bytes, [cap, 0, first, count]);
            first += count;
            continue;
        }
        let u = event.union;
        push(bytes, [u.min.x as u32, u.min.y as u32, u.min.z as u32, 0]);
        push(bytes, [u.max.x as u32, u.max.y as u32, u.max.z as u32, 0]);
        let count = event.bounds.len() as u32;
        push(bytes, [0, 0, first, count]);
        first += count;
    }
    let mut record = event_count * 3;
    for event in events {
        for &(center, radius) in &event.spheres {
            let c = center.to_array().map(f32::to_bits);
            push(bytes, [c[0], c[1], c[2], (radius * radius).to_bits()]);
            push(bytes, [0, 0, 0, 5]);
            record += 2;
        }
        for (i, (b, w)) in event.bounds.iter().zip(&event.weights).enumerate() {
            let urgent = event.urgent.get(i).copied().unwrap_or(false);
            // Edits and unknown changes carry the player-edit flag in bit 3
            // (loads use those bits for their top instead).
            let (kind, high) = match w.kind {
                ChangeKind::Unknown => (0, u32::from(urgent) << 3),
                ChangeKind::Unload => (1, 0),
                ChangeKind::Load { top } => (2, (saturate(top) as u32) << 3),
                ChangeKind::Edit { measure, voxels } => {
                    if measure {
                        // One-voxel margin scan, measured by `measure_edits`.
                        let min = b.min - IVec3::ONE;
                        let size = (b.max - b.min + IVec3::splat(3)).as_uvec3();
                        let count = size.x * size.y * size.z;
                        for words in [
                            [record, positions, count, voxels],
                            [min.x as u32, min.y as u32, min.z as u32, w.flux.to_bits()],
                            [size.x, size.y, size.z, 0],
                        ] {
                            push(measures, words);
                        }
                        entries += 1;
                        positions += count;
                    }
                    (if measure { 6 } else { 3 }, u32::from(urgent) << 3)
                }
                ChangeKind::Light => (4, 0),
            };
            push(
                bytes,
                [
                    b.min.x as u32,
                    b.min.y as u32,
                    b.min.z as u32,
                    w.flux.to_bits(),
                ],
            );
            push(
                bytes,
                [b.max.x as u32, b.max.y as u32, b.max.z as u32, kind | high],
            );
            record += 2;
        }
    }
    EncodedEvents {
        events: event_count,
        entries,
        positions,
    }
}

struct EncodedEvents {
    events: u32,
    /// Edit bounds whose flux the GPU measures, and their scan positions.
    entries: u32,
    positions: u32,
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
            .chain([0.0])
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
    // Initialize final presentation even before the first accumulation pass.
    words[7] = words[24];
    words[13] = words[25];
    words[14] = words[26];
    words[15] = words[27];
    // Record 8: the corner sunlight word the view tier last saw (none yet).
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

    fn assert_slot_metadata(gpu: &GpuTracer) {
        let resident = gpu.resident.as_ref().unwrap();
        let words = read_words(
            gpu,
            &resident.meta.buffer,
            (resident.high_water as u64 * META_BYTES as u64).max(16),
        );
        let mut live = vec![false; resident.high_water as usize];
        for (key, slot) in &resident.slots {
            let base = (slot.index * META_WORDS) as usize;
            assert_eq!(
                &words[base..base + 4],
                &[
                    key.0.x as u32,
                    key.0.y as u32,
                    key.0.z as u32,
                    key.1 as u32 | META_LIVE
                ],
                "slot {}",
                slot.index
            );
            live[slot.index as usize] = true;
        }
        for (index, live) in live.into_iter().enumerate() {
            if !live {
                let flags = words[index * META_WORDS as usize + 3];
                assert_eq!(flags & META_LIVE, 0, "freed slot {index} still live");
            }
        }
    }

    fn read_state(gpu: &GpuTracer) -> Vec<u32> {
        let resident = gpu.resident.as_ref().unwrap();
        read_words(
            gpu,
            &resident.state.buffer,
            resident.high_water as u64 * RECORD_BYTES as u64,
        )
    }

    fn read_words(gpu: &GpuTracer, source: &Buffer, size: u64) -> Vec<u32> {
        let device = &gpu.context.device;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spatial reconstruction test snapshot"),
            size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&default());
        encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, size);
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
    fn resident_shader_validates() {
        let module = naga::front::wgsl::parse_str(&SHADER).unwrap();
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
        check_resident_lifecycle();
    }

    #[test]
    #[ignore = "native: a rejected in-flight merge is undone on the GPU, an accepted one is kept"]
    fn rejected_merge_rolls_back_on_the_gpu() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, mut sources, probes) = super::super::tests::open_floor_fixture();
        sources.environment = Vec3::new(0.3, 0.1, 0.8);
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut work = TraceWork::from_probes(&probes[..8]);
        let trace = |gpu: &mut GpuTracer, work: &mut TraceWork| {
            gpu.trace_compact(
                Arc::clone(&scene),
                &sources,
                &materials,
                work,
                16,
                &emitters,
            )
            .unwrap();
            let words = read_state(gpu);
            let slots = &gpu.resident.as_ref().unwrap().slots;
            work.faces
                .iter()
                .map(|face| {
                    words[slots[&(face.voxel, face.face)].index as usize * (RECORD_BYTES / 4) + 5]
                })
                .collect::<Vec<_>>()
        };
        let after_a = trace(&mut gpu, &mut work);
        let after_b = trace(&mut gpu, &mut work);
        assert!(after_b.iter().zip(&after_a).all(|(b, a)| b > a));
        // An edit relit these probes while batch B was in flight, so the CPU
        // rejected B. The next selection carries the new revision and asks
        // for a rollback: B is undone, then C merges.
        for job in &mut work.probes {
            job.revision += 1;
        }
        for face in &mut work.faces {
            face.rollback = true;
        }
        let after_c = trace(&mut gpu, &mut work);
        for ((a, b), c) in after_a.iter().zip(&after_b).zip(&after_c) {
            assert_eq!(c - a, b - a, "C replaced B rather than adding to it");
        }
        // A revision change whose last batch the CPU accepted keeps it (a
        // young face's only merge must survive), and so does a normal batch.
        for job in &mut work.probes {
            job.revision += 1;
        }
        for face in &mut work.faces {
            face.rollback = false;
        }
        let after_d = trace(&mut gpu, &mut work);
        assert!(after_d.iter().zip(&after_c).all(|(d, c)| d > c));
        let after_e = trace(&mut gpu, &mut work);
        assert!(after_e.iter().zip(&after_d).all(|(e, d)| e > d));
    }

    #[test]
    #[ignore = "native: a rolled-back merge keeps the retention its change applied"]
    fn rollback_keeps_history_removed_by_a_change() {
        use crate::geometry::{Bounds, ChangeKind, ChangeWeight};
        use crate::retention::RetentionEvent;
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut work = TraceWork::from_probes(&probes[264..265]);
        work.faces[0].samples = 4096;
        work.faces[0].radiance = Vec3::splat(0.4);
        work.faces[0].second_moment = Vec3::splat(0.17);
        work.measured_retention = 1.0;
        work.retention_events.push(RetentionEvent::single(
            Bounds::point(work.faces[0].voxel + IVec3::Y),
            ChangeWeight {
                kind: ChangeKind::Edit {
                    voxels: 1,
                    measure: false,
                },
                flux: 100.0,
            },
        ));
        gpu.trace_compact(
            scene.clone(),
            &sources,
            &materials,
            &mut work,
            16,
            &emitters,
        )
        .unwrap();
        let retained = work.faces[0].samples;
        assert!(retained <= 16);
        // A second edit rejects that merge (not the first edit's change):
        // the rollback must not restore the pre-change history (was 4,112).
        work.retention_events.clear();
        work.faces[0].rollback = true;
        work.probes[0].revision += 1;
        gpu.trace_compact(scene, &sources, &materials, &mut work, 16, &emitters)
            .unwrap();
        assert!(
            work.faces[0].samples <= retained + 16,
            "{}",
            work.faces[0].samples
        );
    }

    #[test]
    #[ignore = "native: a batch with no known paths defers a pending change"]
    fn all_unknown_batch_defers_a_pending_change() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut work = TraceWork::from_probes(&probes[..1]);
        work.measured_retention = 1.0;
        gpu.trace_compact(scene, &sources, &materials, &mut work, 16, &emitters)
            .unwrap();
        let slot =
            gpu.resident.as_ref().unwrap().slots[&(work.faces[0].voxel, work.faces[0].face)].index;
        // The state `mark_changed` leaves for a binding change, then an
        // all-unknown result: nothing to resolve the change with yet.
        let resident = gpu.resident.as_ref().unwrap();
        gpu.context.queue.write_buffer(
            &resident.state.buffer,
            u64::from(slot) * RECORD_BYTES as u64 + 20,
            &(4096u32 | 0x8000_0000).to_le_bytes(),
        );
        gpu.context.queue.write_buffer(
            &resident.meta.buffer,
            u64::from(slot) * META_BYTES as u64 + 16,
            &1.0f32.to_le_bytes(),
        );
        let mut unknown = UploadBuffer::new(
            &gpu.context.device,
            "All-unknown result",
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        unknown
            .upload(
                &gpu.context.device,
                &gpu.context.queue,
                &[0u8; RESULT_BYTES],
            )
            .unwrap();
        let mut encoder = gpu.context.device.create_command_encoder(&default());
        gpu.resident.as_mut().unwrap().dispatch_accumulate(
            &gpu.context.device,
            &mut encoder,
            &unknown,
            1,
            None,
        );
        gpu.context.queue.submit([encoder.finish()]);
        let resident = gpu.resident.as_ref().unwrap();
        let meta = read_words(
            &gpu,
            &resident.meta.buffer,
            resident.high_water as u64 * META_BYTES as u64,
        );
        let pending = f32::from_bits(meta[slot as usize * META_WORDS as usize + 4]);
        assert!(pending > 0.0, "an all-unknown batch consumed the change");
    }

    #[test]
    #[ignore = "native: a mirror record classified before a same-batch CPU job is older than it"]
    fn mirror_record_never_replaces_a_newer_cpu_result() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let run = |gpu: &mut GpuTracer, work: &mut TraceWork| {
            work.picks = 64;
            work.mirror_budget = 1024;
            gpu.trace_compact(
                scene.clone(),
                &sources,
                &materials,
                work,
                16,
                &emitters,
            )
            .unwrap();
        };
        let mut first = TraceWork::default();
        first.batch_id = 1;
        for probe in &probes {
            first.ensure_slots(probe);
        }
        run(&mut gpu, &mut first);
        let &(voxel, face, _) = first.pick_tags.first().expect("a GPU-picked face");
        let probe = probes.iter().find(|p| p.voxel == voxel).unwrap();
        // Same classification quarter: the picked face is mirror-stale while
        // a CPU job traces it.
        let mut next = TraceWork::from_probes(std::slice::from_ref(probe));
        next.batch_id = 5;
        run(&mut gpu, &mut next);
        let (mirror, _) = next
            .picked
            .iter()
            .find(|(f, _)| f.voxel == voxel && f.face == face)
            .expect("the CPU-traced face is also in the mirror");
        let mut published = probe.clone();
        next.probes[0].publish(&next.faces, &mut published);
        assert!(mirror.sequence < published.stats().sequence[face]);
        assert!(
            mirror.predates(&published),
            "dispatch would publish the older record"
        );
        let mut slim = SurfaceProbe::new_slim(probe.voxel, probe.face_mask);
        next.probes[0].publish(&next.faces, &mut slim);
        assert!(slim.stats.is_none());
        assert!(
            mirror.predates(&slim),
            "slim publication must also reject the older mirror"
        );
    }

    #[test]
    #[ignore = "native: slim HUD progress follows GPU counts and scheduling without full statistics"]
    fn slim_progress_receives_gpu_stability_without_changing_sample_counts() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, _, probes) = super::super::tests::open_floor_fixture();
        let sources = TransportSources::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut probe = SurfaceProbe::new_slim(probes[0].voxel, probes[0].face_mask);
        let mut settled = false;
        for batch in 1..=16 {
            let mut work = TraceWork::from_probes(std::slice::from_ref(&probe));
            work.stable_error = 0.12;
            work.stable_samples = 32;
            work.batch_id = batch;
            gpu.trace_compact(
                scene.clone(),
                &sources,
                &default(),
                &mut work,
                64,
                &emitters,
            )
            .unwrap();
            assert_eq!(
                work.faces[0].samples,
                batch * 64,
                "readback flag must not alter counts"
            );
            settled |= work.faces[0].gpu_settled == Some(true);
            work.probes[0].publish(&work.faces, &mut probe);
            assert!(probe.stats.is_none());
            let (_, bootstrap, covered, stable) = crate::probe_metrics(&probe, 0);
            assert_eq!((bootstrap, covered), (8, 1));
            assert_eq!(stable > 0, work.faces[0].gpu_settled.unwrap());
        }
        assert!(
            settled,
            "constant illumination must deliver a settled verdict"
        );
    }

    #[test]
    #[ignore = "native: a CPU job's face is never picked by the next submissions"]
    fn cpu_job_owns_its_face_for_the_in_flight_submissions() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut pending = TraceWork::from_probes(&probes[..1]);
        pending.batch_id = 1;
        pending.picks = 1;
        for probe in &probes {
            pending.ensure_slots(probe);
        }
        let owned: Vec<_> = pending.faces.iter().map(|f| (f.voxel, f.face)).collect();
        let _unpublished = gpu
            .submit_compact(
                scene.clone(),
                &sources,
                &materials,
                &mut pending,
                16,
                &emitters,
            )
            .unwrap()
            .unwrap();
        // At most three batches are in flight, so batch 4 is the last one
        // submitted before job 1 publishes. Batch IDs count submissions only
        // (`start_trace_task`); poll-only frames once expired this window.
        let mut next = TraceWork::default();
        next.batch_id = 4;
        next.picks = 1024;
        next.mirror_budget = 1024;
        gpu.trace_compact(scene, &sources, &materials, &mut next, 16, &emitters)
            .unwrap();
        assert!(!next.pick_tags.is_empty());
        assert!(
            next.pick_tags
                .iter()
                .all(|&(p, f, _)| !owned.contains(&(p, f)))
        );
    }

    #[test]
    #[ignore = "native: eagerly ensured faces get a slot and seeded GPU state without a trace"]
    fn ensured_faces_get_seeded_slots_before_their_first_trace() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, mut probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        for (i, probe) in probes.iter_mut().enumerate() {
            for face in 0..6 {
                probe.display[face].indirect = Vec3::splat(0.25 + i as f32 * 0.01);
                probe.display[face].confidence = 0.5;
            }
        }
        // Trace one probe; ensure eight others (and one face that does not exist).
        let mut work = TraceWork::from_probes(&probes[..1]);
        // A model reset has advanced the source revision past the untraced
        // probes' own (0): bindings carry the batch's, or publication would
        // reject them and the faces would never get an address.
        work.source_revision = 3;
        for probe in &probes[1..9] {
            work.ensure_slots(probe);
        }
        let ensured: Vec<_> = work
            .ensure
            .iter()
            .map(|e| (e.face.voxel, e.face.face))
            .collect();
        assert!(!ensured.is_empty());
        let mut ghost = SurfaceProbe::new(IVec3::new(5, 40, 5), 4);
        ghost.revision = 1;
        work.ensure_slots(&ghost);
        gpu.trace_compact(
            Arc::clone(&scene),
            &sources,
            &materials,
            &mut work,
            16,
            &emitters,
        )
        .unwrap();
        let resident = gpu.resident.as_ref().unwrap();
        assert!(
            !resident.slots.contains_key(&(ghost.voxel, 2)),
            "absent faces get no slot"
        );
        assert_slot_metadata(&gpu);
        let words = read_state(&gpu);
        let bound: HashMap<_, _> = resident
            .bindings
            .iter()
            .map(|b| ((b.voxel, b.face), b.address))
            .collect();
        assert!(
            resident
                .bindings
                .iter()
                .filter(|b| b.voxel != probes[0].voxel)
                .all(|b| b.source_revision == 3)
        );
        for key in ensured {
            let slot = resident.slots[&key];
            assert_eq!(resident.face_index.get(key.0, key.1), Some(slot.index));
            assert_eq!(bound[&key], UVec2::new(slot.index, slot.incarnation));
            let base = slot.index as usize * (RECORD_BYTES / 4);
            assert_eq!(words[base + 28], slot.incarnation, "seeded incarnation");
            assert_eq!(words[base + 5], 0, "no samples yet");
            let probe = probes.iter().find(|p| p.voxel == key.0).unwrap();
            assert_eq!(
                f32::from_bits(words[base + 24]),
                probe.display[key.1].indirect.x,
                "seeded presentation"
            );
        }
    }

    #[test]
    #[ignore = "native GPU retention events on every face against the CPU statement"]
    fn retention_events_match_the_cpu_rule_on_every_face() {
        use crate::geometry::{Bounds, ChangeKind, ChangeWeight, GeometryChange};
        use crate::retention::{RetentionEvent, SpatialHistory, resolve};
        let edit = RetentionEvent::single(
            Bounds::point(IVec3::new(8, 6, 8)),
            ChangeWeight {
                kind: ChangeKind::Edit {
                    voxels: 1,
                    measure: false,
                },
                flux: 3.0,
            },
        );
        // Many bounds; receivers lie outside their union.
        let mut loads = GeometryChange::default();
        let ring: Vec<_> = (0..8)
            .map(|x| Bounds::point(IVec3::new(x * 4, 10, 40)))
            .collect();
        for &b in &ring {
            loads.weights.insert(
                b,
                ChangeWeight {
                    kind: ChangeKind::Load { top: 12 },
                    flux: 0.5,
                },
            );
        }
        loads.nearby.insert(
            IVec3::ZERO,
            crate::geometry::PageChanges::new(IVec3::ZERO, ring),
        );
        let mut history = SpatialHistory::default();
        history.record(loads, &[]);
        let loads = history.take_events().pop().unwrap();
        let unknown =
            RetentionEvent::single(Bounds::point(IVec3::new(60, 4, 20)), ChangeWeight::UNKNOWN);
        for (measured, events) in [
            (0.0, vec![edit.clone(), loads.clone()]),
            (1.0, vec![edit.clone(), loads.clone()]),
            (1.0, vec![edit.clone(), unknown]),
        ] {
            let mut gpu = super::super::tests::native_tracer();
            gpu.enable_resident().unwrap();
            gpu.without_footprints();
            let (scene, sources, probes) = super::super::tests::open_floor_fixture();
            let materials = VoxelMaterialTransportTable::default();
            let emitters = Arc::new(EmissionDistribution::default());
            let trace =
                |gpu: &mut GpuTracer, probes: &[SurfaceProbe], events: Vec<RetentionEvent>| {
                    let mut work = TraceWork::from_probes(probes);
                    work.measured_retention = measured;
                    work.retention_events = events;
                    gpu.trace_compact(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut work,
                        16,
                        &emitters,
                    )
                    .unwrap();
                };
            // Uneven histories: probe i is traced (i % 4) + 1 times.
            for round in 0..4 {
                let subset: Vec<_> = probes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 4 >= round)
                    .map(|(_, p)| p.clone())
                    .collect();
                trace(&mut gpu, &subset, Vec::new());
            }
            let before = read_state(&gpu);
            // The traced probe's own statistics also accumulate: skip it.
            let traced = probes[1023].clone();
            trace(&mut gpu, std::slice::from_ref(&traced), events.clone());
            let after = read_state(&gpu);
            let resident = gpu.resident.as_ref().unwrap();
            let meta = read_words(
                &gpu,
                &resident.meta.buffer,
                resident.high_water as u64 * META_BYTES as u64,
            );
            let (mut capped, mut changed) = (0, 0);
            for (key, slot) in &resident.slots {
                if key.0 == traced.voxel {
                    continue;
                }
                let base = slot.index as usize * (RECORD_BYTES / 4);
                let old = before[base + 5] & 0x7fff_ffff;
                let new = after[base + 5];
                let pending = f32::from_bits(meta[(slot.index * META_WORDS + 4) as usize]);
                let (cap, change) = resolve(&events, key.0);
                match change {
                    Some(change) if measured > 0.0 => {
                        assert_eq!(new, old, "measured changes wait for a trace: {key:?}");
                        assert!(
                            (pending - change).abs() <= 1e-5 * change.max(1e-6),
                            "{key:?} pending {pending} vs {change}"
                        );
                        changed += usize::from(change > 0.0);
                    }
                    _ => {
                        assert_eq!(new, old.min(cap), "{key:?} measured={measured}");
                        assert_eq!(pending, 0.0);
                        capped += usize::from(new < old);
                    }
                }
            }
            assert!(
                capped + changed > 100,
                "exercise the rule: {capped} {changed}"
            );
        }
    }

    #[test]
    #[ignore = "native GPU selection: coverage order, eventual coverage, CPU-owned faces, mirror"]
    fn gpu_picks_cover_nearest_first_never_race_cpu_jobs_and_reach_the_mirror() {
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let (scene, sources, probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let focus = IVec3::new(3, 5, 3);
        let mut batch = 0u32;
        let mut run =
            |gpu: &mut GpuTracer, cpu: &[SurfaceProbe], ensure: &[SurfaceProbe], picks: u32| {
                batch += 1;
                let mut work = TraceWork::from_probes(cpu);
                for probe in ensure {
                    work.ensure_slots(probe);
                }
                work.picks = picks;
                work.focus = focus;
                work.batch_id = batch;
                work.stable_error = 0.12;
                work.stable_samples = 32;
                work.audit_ms = 1_000_000;
                work.mirror_budget = 1024;
                gpu.trace_compact(
                    Arc::clone(&scene),
                    &sources,
                    &materials,
                    &mut work,
                    16,
                    &emitters,
                )
                .unwrap();
                work.picked
                    .iter()
                    .map(|(face, _)| (face.voxel, face.face))
                    .collect::<Vec<_>>()
            };
        let counts = |gpu: &GpuTracer| -> HashMap<(IVec3, usize), u32> {
            if gpu.resident.as_ref().unwrap().high_water == 0 {
                return HashMap::new();
            }
            let words = read_state(gpu);
            gpu.resident
                .as_ref()
                .unwrap()
                .slots
                .iter()
                .map(|(key, slot)| {
                    (
                        *key,
                        words[slot.index as usize * (RECORD_BYTES / 4) + 5] & 0x3fff_ffff,
                    )
                })
                .collect()
        };
        // Every face resident and unmeasured (seeded before selection in the
        // first batch); the CPU traces one probe then. Classification
        // rotates over quarters of the slots, so the first four batches pick
        // little while every slot gets ranked.
        let owned = (probes[0].voxel, 2);
        let mut mirrored = HashSet::new();
        for round in 0..124 {
            let budget = if round < 4 { 1 } else { 64 };
            let before = counts(&gpu);
            mirrored.extend(if round == 0 {
                run(&mut gpu, std::slice::from_ref(&probes[0]), &probes, budget)
            } else {
                run(&mut gpu, &[], &[], budget)
            });
            let after = counts(&gpu);
            let picked: Vec<_> = after
                .iter()
                .filter(|(key, n)| {
                    before.get(*key).is_some_and(|b| *n > b) && (round > 0 || **key != owned)
                })
                .map(|(key, _)| *key)
                .collect();
            assert!(
                picked.len() <= budget as usize,
                "round {round}: {}",
                picked.len()
            );
            if (1..4).contains(&round) {
                assert!(
                    !picked.contains(&owned),
                    "a face with a CPU job in flight is never picked"
                );
            }
            if round == 4 {
                // Coverage goes nearest the focus first (log-distance buckets).
                let d2 = |v: IVec3| (v - focus).length_squared();
                let bucket = |d: i32| ((d as f32 + 1.0).log2() * 8.0) as u32;
                let farthest = picked.iter().map(|(v, _)| d2(*v)).max().unwrap();
                // Still candidates: unmeasured, with a live key (a pick whose
                // paths all left the loaded world stays unmeasured, keyless).
                let resident = gpu.resident.as_ref().unwrap();
                let keys = read_words(
                    &gpu,
                    &resident.slot_keys.buffer,
                    resident.high_water as u64 * 4,
                );
                let nearest_left = after
                    .iter()
                    .filter(|(key, n)| {
                        **n == 0
                            && !picked.contains(key)
                            && keys[resident.slots[*key].index as usize] != u32::MAX
                    })
                    .map(|((v, _), _)| d2(*v))
                    .min()
                    .unwrap();
                assert!(
                    bucket(farthest) <= bucket(nearest_left),
                    "{farthest} vs {nearest_left}"
                );
            }
        }
        let after = counts(&gpu);
        assert!(
            after.values().all(|n| *n >= 16),
            "every face covered by GPU picks"
        );
        // Every face the GPU picked reached the CPU mirror (the owned face's
        // results came back with its CPU job instead, unless also picked).
        let missing: Vec<_> = after
            .keys()
            .filter(|k| **k != owned && !mirrored.contains(*k))
            .collect();
        assert!(
            missing.is_empty(),
            "{} picked faces never mirrored",
            missing.len()
        );
    }

    #[test]
    #[ignore = "native GPU change footprints: a skylight's sunlight lands far below it"]
    fn skylight_footprint_lands_on_the_floor_below_and_a_covered_one_nowhere() {
        use crate::geometry::{Bounds, ChangeKind, ChangeWeight};
        use crate::naadf::{linear_index, volume_len};
        use crate::sources::DirectionalTransportLight;
        use crate::world::{AIR, PAGE_SIZE};
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::Y,
                radiance: Vec3::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let hole = Bounds { min: IVec3::new(8, 20, 8), max: IVec3::new(10, 20, 10) };
        // A floor 16 voxels under a ceiling whose hole is the edit, with or
        // without a roof that shadows the hole anyway.
        let footprint = |roof: bool| {
            let mut page = vec![AIR; volume_len(PAGE_SIZE)];
            for z in 0..32 {
                for x in 0..32 {
                    page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
                    let p = IVec3::new(x as i32, 20, z as i32);
                    if p.cmplt(hole.min).any() || p.cmpgt(hole.max).any() {
                        page[linear_index(UVec3::new(x, 20, z), PAGE_SIZE)] = 2;
                    }
                    if roof {
                        page[linear_index(UVec3::new(x, 28, z), PAGE_SIZE)] = 2;
                    }
                }
            }
            let scene = Arc::new(TransportScene::with_page_delta(
                &TransportScene::default(),
                vec![(IVec3::ZERO, page)],
                &[],
            ));
            let probes: Vec<_> = (0..32)
                .flat_map(|z| (0..32).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
                .collect();
            let mut gpu = super::super::tests::native_tracer();
            gpu.enable_resident().unwrap();
            let mut work = TraceWork::from_probes(&probes);
            work.retention_events = vec![crate::retention::RetentionEvent::single(hole, ChangeWeight {
                kind: ChangeKind::Edit { voxels: 9, measure: false },
                flux: 0.0,
            })];
            gpu.trace_compact(scene, &sources, &materials, &mut work, 4, &emitters)
                .unwrap();
            // Two events (the edit, its footprint): six header records, the
            // edit's bound, then the footprint's.
            let words = read_words(&gpu, &gpu.resident.as_ref().unwrap().events_buffer.buffer, 10 * 16);
            let ints = |r: usize| IVec3::new(words[r * 4] as i32, words[r * 4 + 1] as i32, words[r * 4 + 2] as i32);
            (ints(8), ints(9), f32::from_bits(words[8 * 4 + 3]), words[9 * 4 + 3] & 7)
        };
        let (min, max, flux, kind) = footprint(false);
        assert_eq!((min, max, kind), (IVec3::new(8, 4, 8), IVec3::new(10, 4, 10), 4));
        // Nine voxels of white sunlight reflected by the floor's albedo.
        let albedo = materials.get(2).diffuse.dot(Vec3::new(0.2126, 0.7152, 0.0722));
        assert!((flux - 9.0 * albedo).abs() <= 0.25 * 9.0 * albedo, "{flux} vs {}", 9.0 * albedo);
        let (min, _, _, _) = footprint(true);
        assert_eq!(min, IVec3::splat(1 << 29), "a hole under a roof changes no sunlight");
    }

    #[test]
    fn loads_and_light_changes_get_no_footprint() {
        use crate::geometry::{Bounds, ChangeKind, ChangeWeight};
        let bounds = Bounds::point(IVec3::ZERO);
        let event = |kind| crate::retention::RetentionEvent::single(bounds, ChangeWeight { kind, flux: 0.0 });
        let mut events = vec![
            event(ChangeKind::Load { top: 4 }),
            event(ChangeKind::Light),
            event(ChangeKind::Edit { voxels: 1, measure: true }),
        ];
        let planned = plan_footprints(&mut events);
        assert_eq!(planned, vec![(bounds, 3)], "only the edit, with its reserved fourth event");
        assert_eq!(events.len(), 4);
    }

    #[test]
    #[ignore = "native GPU change footprints: a block placed on the floor shades nothing below it"]
    fn placed_block_footprint_does_not_land_inside_its_support() {
        use crate::geometry::{Bounds, ChangeKind, ChangeWeight};
        use crate::naadf::{linear_index, volume_len};
        use crate::sources::DirectionalTransportLight;
        use crate::world::{AIR, PAGE_SIZE};
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
            }
        }
        for z in 8..=10 {
            for x in 8..=10 {
                page[linear_index(UVec3::new(x, 5, z), PAGE_SIZE)] = 2;
            }
        }
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::Y,
                radiance: Vec3::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let probes: Vec<_> = (0..32)
            .flat_map(|z| (0..32).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        let mut work = TraceWork::from_probes(&probes);
        let placed = Bounds { min: IVec3::new(8, 5, 8), max: IVec3::new(10, 5, 10) };
        work.retention_events = vec![crate::retention::RetentionEvent::single(placed, ChangeWeight {
            kind: ChangeKind::Edit { voxels: 9, measure: false },
            flux: 0.0,
        })];
        let emitters = Arc::new(EmissionDistribution::default());
        gpu.trace_compact(scene, &sources, &VoxelMaterialTransportTable::default(), &mut work, 4, &emitters)
            .unwrap();
        let words = read_words(&gpu, &gpu.resident.as_ref().unwrap().events_buffer.buffer, 10 * 16);
        assert_eq!(words[8 * 4] as i32, 1 << 29, "the edit's own faces are the edit event's");
    }

    #[test]
    #[ignore = "native GPU edit flux measurement and edit urgency against their CPU statements"]
    fn gpu_measures_edit_flux_and_marks_faces_whose_history_it_binds() {
        use crate::geometry::{
            Bounds, ChangeKind, ChangeWeight, GeometryChange, PageChanges,
        };
        use crate::retention::SpatialHistory;
        let mut gpu = super::super::tests::native_tracer();
        gpu.enable_resident().unwrap();
        gpu.without_footprints();
        let (scene, sources, probes) = super::super::tests::open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let trace = |gpu: &mut GpuTracer, probes: &[SurfaceProbe], events| {
            let mut work = TraceWork::from_probes(probes);
            work.measured_retention = 1.0;
            work.retention_events = events;
            gpu.trace_compact(
                Arc::clone(&scene),
                &sources,
                &materials,
                &mut work,
                16,
                &emitters,
            )
            .unwrap();
        };
        for _ in 0..4 {
            trace(&mut gpu, &probes, Vec::new());
        }
        let before = read_state(&gpu);
        let edit = IVec3::new(8, 4, 8);
        let mut change = GeometryChange::default();
        change.nearby.insert(
            IVec3::ZERO,
            PageChanges::new(IVec3::ZERO, vec![Bounds::point(edit)]),
        );
        change.weights.insert(
            Bounds::point(edit),
            ChangeWeight {
                kind: ChangeKind::Edit {
                    voxels: 1,
                    measure: true,
                },
                flux: 0.25,
            },
        );
        let mut history = SpatialHistory::default();
        history.record(change, &[edit]);
        let events = history.take_events();
        assert!(events[0].urgent[0]);
        // The CPU statement of the measurement: the brightest outgoing
        // luminance (albedo times incident, plus emission) within one voxel.
        let w = Vec3::new(0.2126, 0.7152, 0.0722);
        let light = |base: usize| {
            let v = |o: usize| {
                Vec3::new(
                    f32::from_bits(before[o]),
                    f32::from_bits(before[o + 1]),
                    f32::from_bits(before[o + 2]),
                )
            };
            if before[base + 5] & 0x7fff_ffff > 0 {
                w.dot(v(base)) + w.dot(v(base + 16))
            } else {
                0.0
            }
        };
        let material = materials.get(2);
        let resident = gpu.resident.as_ref().unwrap();
        let mut brightest = 0.0f32;
        for (key, slot) in &resident.slots {
            if (key.0 - edit).abs().max_element() <= 1 {
                let base = slot.index as usize * (RECORD_BYTES / 4);
                brightest = brightest.max(
                    w.dot(material.diffuse) * light(base)
                        + w.dot(material.emission),
                );
            }
        }
        assert!(brightest > 0.0);
        let flux = 0.25 + brightest * 2.0;
        let traced = probes[1023].clone();
        trace(&mut gpu, std::slice::from_ref(&traced), events);
        let resident = gpu.resident.as_ref().unwrap();
        let encoded = read_words(&gpu, &resident.events_buffer.buffer, 5 * 16);
        let measured = f32::from_bits(encoded[3 * 4 + 3]);
        assert!(
            (measured - flux).abs() <= 1e-4 * flux,
            "{measured} vs {flux}"
        );
        assert_eq!(
            encoded[4 * 4 + 3] & 7,
            3,
            "measured: an ordinary edit for retention"
        );
        // Urgency: within 64 voxels, faces whose history this change binds.
        let meta = read_words(
            &gpu,
            &resident.meta.buffer,
            resident.high_water as u64 * META_BYTES as u64,
        );
        let (mut urgent, mut checked) = (0, 0);
        for (key, slot) in &resident.slots {
            if key.0 == traced.voxel {
                continue;
            }
            let base = slot.index as usize * (RECORD_BYTES / 4);
            let d2 = (key.0 - edit).length_squared() as f32;
            let change = flux / (std::f32::consts::PI * d2.max(1.0));
            let v = |o: usize| {
                Vec3::new(
                    f32::from_bits(before[o]),
                    f32::from_bits(before[o + 1]),
                    f32::from_bits(before[o + 2]),
                )
            };
            let mean = v(base);
            let variance = w.dot((v(base + 8) - mean * mean).max(Vec3::ZERO));
            let count = (before[base + 5] & 0x7fff_ffff) as f32;
            let binds = w.dot(mean) < 0.05 * change || variance / (change * change) < count;
            let expected = d2 <= 64.0 * 64.0 && binds;
            let marked = meta[(slot.index * META_WORDS + 11) as usize] & 0x4000_0000 != 0;
            // Single-precision boundary cases may differ; none in this fixture.
            assert_eq!(marked, expected, "{key:?} d2 {d2}");
            urgent += usize::from(marked);
            checked += 1;
        }
        assert!(urgent > 0 && urgent < checked, "{urgent} of {checked}");
    }

    fn check_resident_lifecycle() {
        let mut control = super::super::tests::native_tracer();
        let mut candidate = GpuTracer::new(control.context.clone()).unwrap();
        candidate.enable_resident().unwrap();
        let (mut scene, mut sources, initial) = super::super::tests::open_floor_fixture();
        sources.environment = Vec3::new(0.3, 0.1, 0.8);
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
            );
            let mut b = TraceWork::from_probes(
                &indices
                    .iter()
                    .map(|&i| actual[i].clone())
                    .collect::<Vec<_>>(),
            );
            for work in [&mut a, &mut b] {
                work.response_seconds = 0.075;
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
                        face.radiance = Vec3::ZERO;
                        face.second_moment = Vec3::ZERO;
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
                    )
                    .unwrap();
                for face in &mut work.faces {
                    face.updated_at = Some(measured_at);
                }
            }
            // Incremental face-index changes reach the GPU table exactly
            // (scatter pass), through growth, retirement and recycling.
            let resident = candidate.resident.as_ref().unwrap();
            assert_slot_metadata(&candidate);
            let expected_index = resident.face_index.table_words();
            assert_eq!(
                read_words(
                    &candidate,
                    &resident.face_index_buffer.buffer,
                    expected_index.len() as u64 * 4
                ),
                expected_index,
                "step={step}"
            );
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
                        delta.abs().max_element() < 1e-5 * (1.0 + a.max_element()),
                        "step={step} {a:?} vs {b:?}"
                    );
                }
            }
            for (resident, work, probes) in [(false, &a, &mut expected), (true, &b, &mut actual)] {
                for (job, &i) in work.probes.iter().zip(&indices) {
                    for face in &work.faces[job.faces()] {
                        let previous_time = probes[i].stats().face_updated_at[face.face];
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
            let cap = match step {
                1 => 3,
                3 => 0,
                _ => u32::MAX,
            };
            for (tracer, probe) in [(&mut control, &mut a), (&mut resident, &mut b)] {
                let mut work = TraceWork::default();
                work.push(
                    probe,
                    mask,
                    0,
                    u64::from(step >= 3),
                    step == 3,
                    u32::MAX,
                    cap,
                );
                work.probes[0].retention = Some(step as u64);
                work.response_seconds = 0.075;
                tracer
                    .trace_compact(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut work,
                        16,
                        &emitters,
                    )
                    .unwrap();
                work.probes[0].publish(&work.faces, probe);
                // A source cap covers the probe's unselected faces too. The
                // GPU applied it (cap records); mirror that in both CPU
                // copies, so the next trace of that face checks the GPU.
                for face in 0..6 {
                    if mask & (1 << face) == 0 {
                        probe.stats_mut().samples[face] = probe.stats().samples[face].min(cap);
                    }
                }
            }
            assert_eq!(
                a.stats().samples,
                b.stats().samples,
                "partial cap step={step}"
            );
            assert_eq!(a.stats().sequence, b.stats().sequence);
            for face in [2, 3] {
                assert!(
                    (a.stats().radiance[face] - b.stats().radiance[face])
                        .abs()
                        .max_element()
                        < 1e-5
                );
            }
        }
        assert_eq!(b.stats().samples[3], 16);
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
        // Batches each probe has had in its current slot: GPU state carries
        // the sample sequence across a revision cap, as the runtime CPU
        // mirror does, so the control traces from the same sequence.
        let mut traced = vec![0u32; 1024];
        for (step, count) in [17, 1024, 128].into_iter().enumerate() {
            sources.environment = Vec3::splat(step as f32 + 0.25);
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
                traced.fill(0);
            }
            let mut probes = initial[..count].to_vec();
            for (probe, traced) in probes.iter_mut().zip(&mut traced) {
                probe.revision = step as u64;
                probe.stats_mut().sequence = [*traced * 16; 6];
                *traced += 1;
            }
            let mut expected = TraceWork::from_probes(&probes);
            let mut actual = TraceWork::from_probes(&probes);
            for work in [&mut expected, &mut actual] {
                work.response_seconds = 0.075;
                // Each step's revision change restarts the probes (the
                // control traces them fresh): an explicit zero cap.
                for face in &mut work.faces {
                    face.revision_cap = face.samples;
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
                        d.abs().max_element() < 1e-5
                    );
                }
            }
        }
        assert_eq!(candidate.readbacks.len(), 3);
    }
}
