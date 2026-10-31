//! Asynchronous, bounded compute transport on Bevy's device. GPU geometry and
//! scratch allocations persist across batches. Only requested face estimates
//! are read back; neither the render thread nor gameplay waits for the GPU.
use std::borrow::Cow;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroup, BindGroupEntry, BindGroupLayout, Buffer, BufferDescriptor, BufferId, BufferUsages,
    ComputePipeline, ShaderModuleDescriptor, ShaderSource, ShaderType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::ShaderBuffer;

use super::color::FiniteColor;
use super::emission::EmissionDistribution;
use super::gpu_scene::PackedScene;
use super::sources::TransportSources;
use super::tracer::HISTORY_LIMIT;
use super::work::TraceWork;
#[cfg(test)]
use super::world::SurfaceProbe;
use super::world::{TransportScene, VoxelMaterialTransportTable};

/// The world-space consumer with the shared transport and traversal included.
static SHADER: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| include_paths(include_str!("../shaders/voxel_trace.wgsl")));

/// Compose a transport consumer without duplicating the lighting estimator.
pub(crate) fn include_paths(source: &str) -> String {
    const MARKER: &str = "// #include voxel_path.wgsl\n";
    assert_eq!(
        source.matches(MARKER).count(),
        1,
        "shader needs one path include"
    );
    include_traversal(&source.replacen(MARKER, include_str!("../shaders/voxel_path.wgsl"), 1))
}

/// Insert the shared traversal (`voxel_traversal.wgsl`) at its include marker,
/// for a material shader that reads `VoxelGiMaterialData`'s geometry.
pub fn include_traversal(source: &str) -> String {
    const MARKER: &str = "// #include voxel_traversal.wgsl\n";
    assert!(
        source.contains(MARKER),
        "shader lacks the traversal include marker"
    );
    source.replacen(MARKER, include_str!("../shaders/voxel_traversal.wgsl"), 1)
}

fn include_retention(source: &str) -> String {
    source.replace(
        "// #include voxel_retention.wgsl\n",
        include_str!("../shaders/voxel_retention.wgsl"),
    )
}
const RESULT_BYTES: usize = 96;
const TASK_BYTES: usize = 32;

mod face_index;
mod geometry;
mod readback;
pub(crate) mod reflect;
mod resident;
pub(crate) mod view;
pub(crate) use readback::Readback;

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod geometry_tests;
#[cfg(test)]
mod maintenance_tests;
#[cfg(test)]
mod task_tests;
#[cfg(test)]
mod traversal_tests;
#[cfg(test)]
mod upload_tests;
#[cfg(test)]
mod view_tests;

#[derive(Resource, Clone)]
pub(crate) struct GpuTransportDevice {
    pub device: RenderDevice,
    pub queue: RenderQueue,
}

pub(crate) struct ResidentBinding {
    pub voxel: IVec3,
    pub face: usize,
    pub revision: u64,
    pub source_revision: u64,
    pub address: UVec2,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuParams {
    counts: UVec4,
    minimum: IVec4,
    maximum: IVec4,
    environment: Vec4,
    extra: Vec4,
    transport: Vec4,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuMaterial {
    diffuse: Vec4,
    diffuse_extra: Vec4,
    emission: Vec4,
    emission_extra: Vec4,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuLight {
    position_range: Vec4,
    energy: Vec4,
    extra: Vec4,
}

#[cfg(test)]
#[derive(Clone, Copy, Default, ShaderType)]
struct GpuFaceTask {
    voxel: IVec3,
    face: u32,
    sequence: UVec4,
}

/// Encode the fixed WGSL task layout directly into reusable upload storage.
/// Integer words are explicitly little-endian, with all three padding words
/// initialized. No Rust struct layout or unsafe cast.
fn encode_tasks(bytes: &mut Vec<u8>, work: &TraceWork) {
    bytes.clear();
    bytes.reserve((work.faces.len() + work.picks as usize) * TASK_BYTES);
    // GPU pick capacity follows the CPU's faces; `select_emit` fills it and
    // the tracer skips what stays face 0xffffffff.
    let tasks = work
        .faces
        .iter()
        .map(|face| (face.voxel, face.face as u32, face.sequence))
        .chain((0..work.picks).map(|_| (IVec3::ZERO, u32::MAX, 0)));
    for (voxel, face, sequence) in tasks {
        let record = [
            voxel.x as u32,
            voxel.y as u32,
            voxel.z as u32,
            face,
            sequence,
            0,
            0,
            0,
        ]
        .map(u32::to_le_bytes);
        bytes.extend_from_slice(record.as_flattened());
    }
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuEmitter {
    voxel: IVec3,
    face: u32,
    sampling: Vec4,
}

struct UploadBuffer {
    buffer: Buffer,
    capacity: u64,
    label: &'static str,
    usage: BufferUsages,
    #[cfg(test)]
    uploads: usize,
}

/// Bindings are valid until an allocation changes, not merely until its contents
/// change. Cache by buffer identity so growth can never retain a stale binding.
struct CachedBindings<const N: usize> {
    buffers: [BufferId; N],
    group: BindGroup,
}

fn cached_bindings<'a, const N: usize>(
    cache: &'a mut Option<CachedBindings<N>>,
    device: &RenderDevice,
    layout: impl FnOnce() -> BindGroupLayout,
    buffers: [(u32, &UploadBuffer); N],
) -> &'a BindGroup {
    let ids = buffers.map(|(_, buffer)| buffer.buffer.id());
    if cache.as_ref().is_none_or(|cached| cached.buffers != ids) {
        let entries = buffers.map(|(binding, buffer)| BindGroupEntry {
            binding,
            resource: buffer.buffer.as_entire_binding(),
        });
        *cache = Some(CachedBindings {
            buffers: ids,
            group: device.create_bind_group("GI persistent dispatch bindings", &layout(), &entries),
        });
    }
    &cache.as_ref().unwrap().group
}

impl UploadBuffer {
    /// Model buffers travel with their counts and sampling metadata. Publish a
    /// fresh allocation on a model change so an extracted view cannot observe
    /// a new table through an older descriptor. Ordinary frames share handles.
    fn upload_generation(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        bytes: &[u8],
    ) -> Result<(), String> {
        self.capacity = 0;
        self.upload(device, queue, bytes)
    }

    fn new(device: &RenderDevice, label: &'static str, usage: BufferUsages) -> Self {
        let capacity = 256;
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: capacity,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            capacity,
            label,
            usage,
            #[cfg(test)]
            uploads: 0,
        }
    }

    fn ensure(&mut self, device: &RenderDevice, bytes: usize) -> Result<bool, String> {
        let size = (bytes as u64).max(16).next_power_of_two();
        if size > device.limits().max_buffer_size
            || (self.usage.contains(BufferUsages::STORAGE)
                && size > device.limits().max_storage_buffer_binding_size)
        {
            return Err(format!(
                "{} requires {size} bytes, exceeding this device's buffer limit",
                self.label
            ));
        }
        if size <= self.capacity {
            return Ok(false);
        }
        self.capacity = size;
        self.buffer = device.create_buffer(&BufferDescriptor {
            label: Some(self.label),
            size,
            usage: self.usage,
            mapped_at_creation: false,
        });
        Ok(true)
    }

    /// Another handle to the same allocation, for read-only diagnostic binding.
    #[cfg(test)]
    fn shared(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            capacity: self.capacity,
            label: self.label,
            usage: self.usage,
            #[cfg(test)]
            uploads: 0,
        }
    }

    fn upload(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        bytes: &[u8],
    ) -> Result<(), String> {
        self.ensure(device, bytes.len())?;
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
            #[cfg(test)]
            {
                self.uploads += 1;
            }
        }
        Ok(())
    }
}

pub(crate) struct GpuTracer {
    resident: Option<resident::ResidentAccumulator>,
    context: GpuTransportDevice,
    pipeline: ComputePipeline,
    module: wgpu::ShaderModule,
    entry: &'static str,
    pipeline_key: (u32, bool, u32),
    /// Three depths × finite/default horizon × two lane layouts.
    /// Reach itself stays a uniform; changing 16 to 32 never compiles another
    /// variant. Pipelines and compilation stay on the transport worker.
    quality_pipelines: bevy::platform::collections::HashMap<(u32, bool, u32), ComputePipeline>,
    faces_per_workgroup: u32,
    adaptive_lanes: bool,
    #[cfg(test)]
    fixed_path_lanes: Option<u32>,
    footprint_pipeline: ComputePipeline,
    finish_footprint_pipeline: ComputePipeline,
    footprint_bindings: Option<CachedBindings<8>>,
    finish_footprint_bindings: Option<CachedBindings<3>>,
    #[cfg(test)]
    full_geometry_uploads_for_benchmark: bool,
    #[cfg(test)]
    legacy_task_encoding_for_benchmark: bool,
    packed: PackedScene,
    scene: Option<Arc<TransportScene>>,
    params: UploadBuffer,
    directory: UploadBuffer,
    regions: UploadBuffer,
    cells: UploadBuffer,
    geometry_staging: UploadBuffer,
    materials: UploadBuffer,
    material_table: Option<VoxelMaterialTransportTable>,
    view_transport: super::TransportSettings,
    lights: UploadBuffer,
    sources: Option<TransportSources>,
    emitters: UploadBuffer,
    emission: Option<Arc<EmissionDistribution>>,
    tasks: UploadBuffer,
    task_bytes: Vec<u8>,
    output: UploadBuffer,
    bindings: Option<CachedBindings<11>>,
    /// Bound when this tracer has no resident cache. By default every face
    /// lookup misses, so paths continue explicitly (synchronous comparison
    /// backend). Diagnostics may bind another tracer's frozen cache instead.
    external_cache: (UploadBuffer, UploadBuffer),
    readbacks: Vec<UploadBuffer>,
    timestamps: Option<wgpu::QuerySet>,
    timestamp_resolve: UploadBuffer,
    pub last_dispatch_ms: f32,
    /// GPU compute-pass time only. Unlike the scheduling estimate above, this
    /// never falls back to submission/readback wall time when timestamps fail.
    pub last_gpu_ms: Option<f32>,
    pub uploaded_geometry_bytes: u64,
    pub last_prepare_ms: f32,
    /// Submission-to-decode age, not time blocking the worker or render thread.
    pub last_snapshot_latency_ms: f32,
    pub last_decode_ms: f32,
}

impl GpuTracer {
    pub(super) fn backend_name(&self) -> &'static str {
        if self.resident.is_some() {
            "gpu-resident"
        } else {
            "gpu-sync"
        }
    }

    pub(super) fn take_resident_bindings(&mut self) -> Vec<ResidentBinding> {
        self.resident
            .as_mut()
            .map_or_else(Vec::new, |resident| std::mem::take(&mut resident.bindings))
    }
    pub(super) fn snapshot_ready(&self, snapshot: &mut Readback) -> Result<bool, String> {
        snapshot.poll(&self.context.device)
    }
    /// The geometry buffers and traversal bounds the terrain material shares
    /// (far sun); None before the first scene upload.
    pub(crate) fn geometry_display(
        &self,
    ) -> Option<([Buffer; 3], super::render::GpuTraversalParams)> {
        let scene = self.scene.as_ref()?;
        Some((
            [
                self.directory.buffer.clone(),
                self.regions.buffer.clone(),
                self.cells.buffer.clone(),
            ],
            super::render::GpuTraversalParams {
                counts: UVec4::new(
                    super::tracer::MAX_PATH_DISTANCE as u32,
                    0,
                    self.packed.directory.len() as u32 - 1,
                    0,
                ),
                minimum: scene.top_level.minimum.extend(0),
                maximum: scene.top_level.maximum.extend(0),
            },
        ))
    }
    /// Published retention changes, applied on the GPU before the next
    /// batch's rays (kept until a batch with resident inputs runs).
    pub(super) fn queue_retention(&mut self, events: &mut Vec<super::retention::RetentionEvent>) {
        if let Some(resident) = &mut self.resident {
            resident.events.append(events);
        } else {
            events.clear();
        }
    }

    /// Plan only the CPU's retention events (tests of their exact rule).
    #[cfg(test)]
    pub(crate) fn without_footprints(&mut self) {
        if let Some(resident) = &mut self.resident {
            resident.trace_footprints = false;
        }
    }

    pub(super) fn enable_resident(&mut self) -> Result<(), String> {
        if self.resident.is_none() {
            self.resident = Some(resident::ResidentAccumulator::new(&self.context)?);
        }
        Ok(())
    }

    /// Diagnostics: terminate this tracer's paths on another tracer's resident
    /// measurements, read-only. The source must not trace while this is bound
    /// if its cache is meant to stay frozen.
    #[cfg(test)]
    pub(crate) fn share_cache_from(&mut self, source: &GpuTracer) -> Result<(), String> {
        let resident = source
            .resident
            .as_ref()
            .ok_or("cache source has no resident measurements")?;
        let (state, index) = resident.cache_buffers();
        self.external_cache = (state.shared(), index.shared());
        Ok(())
    }

    pub fn new(context: GpuTransportDevice) -> Result<Self, String> {
        Self::with_kernel(context, true)
    }

    /// `cooperative` selects the production multi-lane kernel; the serial
    /// one-invocation-per-face kernel serves path-matched native tests.
    fn with_kernel(context: GpuTransportDevice, cooperative: bool) -> Result<Self, String> {
        let (entry, faces_per_workgroup) = if cooperative {
            ("trace_faces_parallel", 16)
        } else {
            ("trace_faces", 64)
        };
        let device = &context.device;
        if device.limits().max_storage_buffers_per_shader_stage < 10 {
            return Err("GPU transport needs ten storage-buffer bindings".into());
        }
        let error_scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("World-space voxel transport"),
            source: ShaderSource::Wgsl(Cow::Borrowed(SHADER.as_str())),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("World-space voxel face tracing"),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        });
        let entry_pipeline = |label, entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: default(),
                cache: None,
            })
        };
        let footprint_pipeline = entry_pipeline("GI change footprints", "trace_footprints");
        let finish_footprint_pipeline =
            entry_pipeline("GI change footprint events", "finish_footprints");
        if let Some(error) = bevy::tasks::block_on(error_scope.pop()) {
            return Err(error.to_string());
        }
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let timestamps = device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            .then(|| {
                device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("GI transport timing"),
                        ty: wgpu::QueryType::Timestamp,
                        count: 8,
                    })
            });
        Ok(Self {
            resident: None,
            quality_pipelines: [((3, false, 64 / faces_per_workgroup), pipeline.clone())]
                .into_iter()
                .collect(),
            pipeline_key: (3, false, 64 / faces_per_workgroup),
            module,
            entry,
            pipeline,
            faces_per_workgroup,
            // Legacy test/reference shaders keep their original fixed layout.
            adaptive_lanes: cooperative,
            #[cfg(test)]
            fixed_path_lanes: None,
            footprint_pipeline,
            finish_footprint_pipeline,
            footprint_bindings: None,
            finish_footprint_bindings: None,
            #[cfg(test)]
            #[cfg(test)]
            full_geometry_uploads_for_benchmark: false,
            #[cfg(test)]
            legacy_task_encoding_for_benchmark: false,
            packed: PackedScene::default(),
            scene: None,
            params: UploadBuffer::new(
                device,
                "GI uniforms",
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            directory: UploadBuffer::new(
                device,
                "GI sparse directory",
                storage | BufferUsages::COPY_SRC,
            ),
            regions: UploadBuffer::new(
                device,
                "GI region headers",
                storage | BufferUsages::COPY_SRC,
            ),
            cells: UploadBuffer::new(
                device,
                "GI packed NAADF cells",
                storage | BufferUsages::COPY_SRC,
            ),
            geometry_staging: UploadBuffer::new(
                device,
                "GI geometry transaction",
                BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            ),
            materials: UploadBuffer::new(
                device,
                "GI material transport",
                storage | BufferUsages::COPY_SRC,
            ),
            material_table: None,
            view_transport: default(),
            lights: UploadBuffer::new(device, "GI analytic sources", storage),
            sources: None,
            emitters: UploadBuffer::new(device, "GI emissive face distribution", storage),
            emission: None,
            tasks: UploadBuffer::new(device, "GI face work queue", storage),
            task_bytes: Vec::new(),
            output: UploadBuffer::new(
                device,
                "GI face estimates",
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            bindings: None,
            external_cache: {
                let mut state = UploadBuffer::new(device, "GI empty resident cache", storage);
                let mut index = UploadBuffer::new(device, "GI empty face index", storage);
                state.ensure(device, 16)?;
                // Mask zero, then one empty entry (face 7): every lookup misses.
                let words: [u32; 8] = [0, 1, 0, 0, 0, 0, 0, u32::MAX];
                index.ensure(device, 32)?;
                context
                    .queue
                    .write_buffer(&index.buffer, 0, &encode_words(&words));
                (state, index)
            },
            readbacks: Vec::new(),
            timestamps,
            timestamp_resolve: UploadBuffer::new(
                device,
                "GI timestamp resolve",
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            ),
            last_dispatch_ms: 0.0,
            last_gpu_ms: None,
            context,
            uploaded_geometry_bytes: 0,
            last_prepare_ms: 0.0,
            last_snapshot_latency_ms: 0.0,
            last_decode_ms: 0.0,
        })
    }

    #[cfg(test)]
    pub fn trace(
        &mut self,
        scene: Arc<TransportScene>,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        probes: &mut [SurfaceProbe],
        samples: u32,
        emitters: &Arc<EmissionDistribution>,
    ) -> Result<(), String> {
        self.trace_work(scene, sources, materials, probes, samples, emitters)
    }

    #[cfg(test)]
    pub fn trace_work(
        &mut self,
        scene: Arc<TransportScene>,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        probes: &mut [SurfaceProbe],
        samples: u32,
        emitters: &Arc<EmissionDistribution>,
    ) -> Result<(), String> {
        let mut work = TraceWork::from_probes(probes);
        self.trace_compact(scene, sources, materials, &mut work, samples, emitters)?;
        work.copy_measurements(probes);
        for probe in probes {
            probe.emission = materials.get(probe.material).emission;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn trace_compact(
        &mut self,
        scene: Arc<TransportScene>,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        work: &mut TraceWork,
        samples: u32,
        emitters: &Arc<EmissionDistribution>,
    ) -> Result<(), String> {
        if let Some(mut snapshot) =
            self.submit_compact(scene, sources, materials, work, samples, emitters)?
        {
            snapshot.wait(&self.context.device)?;
            self.complete_readback(snapshot, work)?;
        }
        Ok(())
    }

    pub(super) fn submit_compact(
        &mut self,
        scene: Arc<TransportScene>,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        work: &mut TraceWork,
        samples: u32,
        emitters: &Arc<EmissionDistribution>,
    ) -> Result<Option<readback::Readback>, String> {
        let work_started = Instant::now();
        let geometry_changed = self.needs_geometry(&scene);
        self.synchronize_geometry(&scene)?;
        self.select_quality_pipeline(work.transport.normalized(), samples)?;
        // Reconcile ownership even without rays. A retirement must remove the
        // GPU face index and live metadata, not merely the CPU slot map.
        let prepare_resident = self.resident.is_some()
            && (!work.faces.is_empty()
                || work.picks > 0
                || geometry_changed
                || !work.retire_pages.is_empty()
                || self
                    .resident
                    .as_ref()
                    .is_some_and(|resident| !resident.events.is_empty()));
        if prepare_resident && let Some(resident) = &mut self.resident {
            resident.set_materials(&self.context, materials)?;
            resident.prepare_inputs(&self.context, &scene, work, samples)?;
        }
        // Model publication is independent of receiver work. Far views may
        // have no near receivers, but still need the current optical scene.
        self.view_transport = work.transport.normalized();
        let context = self.context.clone();
        let device = &context.device;
        let queue = &context.queue;
        if self
            .emission
            .as_ref()
            .is_none_or(|previous| !Arc::ptr_eq(previous, emitters))
        {
            let mut data: Vec<_> = emitters
                .faces
                .iter()
                .map(|face| GpuEmitter {
                    voxel: face.voxel,
                    face: face.face,
                    sampling: Vec4::new(
                        face.cumulative,
                        face.probability,
                        face.material as f32,
                        emitters.tree.len() as f32,
                    ),
                })
                .collect();
            if data.is_empty() {
                data.push(GpuEmitter::default());
            }
            let mut bytes = ShaderBuffer::from(data).data.unwrap();
            bytes.extend(
                emitters
                    .tree
                    .iter()
                    .flat_map(|node| node.words().into_iter().flat_map(u32::to_le_bytes)),
            );
            self.emitters.upload_generation(device, queue, &bytes)?;
            self.emission = Some(Arc::clone(emitters));
        }
        if self.material_table.as_ref() != Some(materials) {
            let gpu_materials: Vec<_> = (0..=255)
                .map(|i| {
                    let material = materials.get(i);
                    GpuMaterial {
                        diffuse: spectrum_vec(material.diffuse),
                        diffuse_extra: Vec4::ZERO,
                        emission: spectrum_vec(material.emission),
                        emission_extra: Vec4::new(0.0, material.emission.max_element(), 0.0, 0.0),
                    }
                })
                .collect();
            self.materials.upload_generation(
                device,
                queue,
                &ShaderBuffer::from(gpu_materials).data.unwrap(),
            )?;
            self.material_table = Some(materials.clone());
        }
        let light_count = (sources.directional.len() + sources.points.len()) as u32;
        if self.sources.as_ref().is_none_or(|previous| {
            previous.directional != sources.directional || previous.points != sources.points
        }) {
            let mut gpu_lights: Vec<_> = sources
                .directional
                .iter()
                .map(|light| GpuLight {
                    position_range: light.direction.normalize_or_zero().extend(-1.0),
                    energy: spectrum_vec(light.radiance),
                    extra: Vec4::new(
                        0.0,
                        light.custom_visible.x,
                        light.custom_visible.y,
                        light.custom_visible.z,
                    ),
                })
                .chain(sources.points.iter().map(|light| GpuLight {
                    position_range: light.position.extend(light.range),
                    energy: spectrum_vec(light.intensity),
                    extra: Vec4::new(
                        0.0,
                        light.custom_visible.x,
                        light.custom_visible.y,
                        light.custom_visible.z,
                    ),
                }))
                .collect();
            if gpu_lights.is_empty() {
                gpu_lights.push(GpuLight::default());
            }
            self.lights.upload_generation(
                device,
                queue,
                &ShaderBuffer::from(gpu_lights).data.unwrap(),
            )?;
        }
        self.sources = Some(sources.clone());
        let picks = if self.resident.is_some() {
            work.picks
        } else {
            0
        };
        let face_count = work.faces.len() + picks as usize;
        let transport = work.transport.normalized();
        let params = GpuParams {
            counts: UVec4::new(
                face_count as u32,
                samples,
                self.packed.directory.len() as u32 - 1,
                light_count,
            ),
            minimum: scene.top_level.minimum.extend(0),
            maximum: scene
                .top_level
                .maximum
                .extend(sources.directional.len() as i32),
            environment: spectrum_vec(sources.environment),
            extra: Vec4::new(
                0.0,
                super::tracer::MAX_PATH_DISTANCE,
                emitters.total_weight,
                emitters.faces.len() as f32,
            ),
            transport: Vec4::new(transport.reach, 0.0, 0.0, 0.0),
        };
        self.params.upload(
            device,
            queue,
            &ShaderBuffer::from(vec![params]).data.unwrap(),
        )?;
        // Geometry, ownership and model are now current. Maintenance needs
        // neither a ray dispatch nor a readback allocation.
        if work.faces.is_empty() && work.picks == 0 {
            if prepare_resident {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GI receiver maintenance"),
                });
                dispatch_footprints(self, &mut encoder);
                self.resident.as_mut().unwrap().dispatch_prepare(
                    device,
                    &mut encoder,
                    &self.tasks,
                    0,
                    None,
                );
                queue.submit([encoder.finish()]);
            }
            self.last_dispatch_ms = 0.0;
            self.last_gpu_ms = Some(0.0);
            self.last_prepare_ms = work_started.elapsed().as_secs_f32() * 1000.0;
            self.last_snapshot_latency_ms = 0.0;
            self.last_decode_ms = 0.0;
            return Ok(None);
        }
        #[cfg(not(test))]
        encode_tasks(&mut self.task_bytes, work);
        #[cfg(test)]
        if self.legacy_task_encoding_for_benchmark {
            self.task_bytes = task_tests::legacy_task_bytes(work);
        } else {
            encode_tasks(&mut self.task_bytes, work);
        }
        self.tasks.upload(device, queue, &self.task_bytes)?;
        // Readback: the CPU's faces, then (with GPU picks) the
        // statistics-mirror count and records. Picks themselves are not read
        // back: the mirror catches up with them under a fixed budget.
        let face_bytes = face_count * RESULT_BYTES;
        let cpu_bytes = work.faces.len() * RESULT_BYTES;
        let mirror_offset = cpu_bytes;
        let mirror_records = if picks > 0 {
            work.mirror_budget.max(1)
        } else {
            0
        };
        let picks_offset =
            mirror_offset + 16 + mirror_records as usize * resident::MIRROR_RECORD_BYTES;
        let readback_bytes = if picks > 0 {
            picks_offset + picks as usize * 16
        } else {
            mirror_offset
        };
        // Resident batches: preparation, rays, accumulation, then (with GPU
        // picks) selection in its own pass.
        let timestamp_count = if self.resident.is_some() && face_count > 0 {
            if picks > 0 { 8 } else { 6 }
        } else {
            2
        };
        let timestamp_bytes = timestamp_count as u64 * 8;
        self.output.ensure(device, face_bytes)?;
        let mut readback = self.readbacks.pop().unwrap_or_else(|| {
            UploadBuffer::new(
                device,
                "GI asynchronous simulation snapshot",
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )
        });
        readback.ensure(device, readback_bytes + timestamp_bytes as usize)?;
        if prepare_resident
            && self
                .resident
                .as_ref()
                .is_some_and(|r| r.footprints().is_some())
        {
            // Its own submission, ahead of the batch that applies the events.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GI change footprints"),
            });
            dispatch_footprints(self, &mut encoder);
            queue.submit([encoder.finish()]);
        }
        let (cache_state, cache_index) = self.resident.as_ref().map_or(
            (&self.external_cache.0, &self.external_cache.1),
            |resident| resident.cache_buffers(),
        );
        let buffers = [
            &self.params,
            &self.directory,
            &self.regions,
            &self.cells,
            &self.materials,
            &self.lights,
            &self.tasks,
            &self.output,
            &self.emitters,
            cache_state,
            cache_index,
        ];
        // Retire stale groups: never retain an obsolete, large output buffer.
        let ids = buffers.map(|buffer| buffer.buffer.id());
        if self
            .bindings
            .as_ref()
            .is_some_and(|cached| cached.buffers != ids)
        {
            self.bindings = None;
        }
        let bindings = cached_bindings(
            &mut self.bindings,
            device,
            || self.pipeline.get_bind_group_layout(0).into(),
            std::array::from_fn(|index| (index as u32, buffers[index])),
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GI bounded transport batch"),
        });
        if prepare_resident && let Some(resident) = &mut self.resident {
            resident.dispatch_prepare(
                device,
                &mut encoder,
                &self.tasks,
                face_count,
                self.timestamps.as_ref().filter(|_| face_count > 0),
            );
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GI world-space rays"),
                timestamp_writes: self.timestamps.as_ref().map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(if timestamp_count >= 6 {
                            2
                        } else {
                            0
                        }),
                        end_of_pass_write_index: Some(if timestamp_count >= 6 { 3 } else { 1 }),
                    }
                }),
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &**bindings, &[]);
            pass.dispatch_workgroups((face_count as u32).div_ceil(self.faces_per_workgroup), 1, 1);
        }
        if face_count > 0
            && let Some(resident) = &mut self.resident
        {
            resident.dispatch_accumulate(
                device,
                &mut encoder,
                &self.output,
                face_count,
                self.timestamps.as_ref(),
            );
        }
        if cpu_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.output.buffer,
                0,
                &readback.buffer,
                0,
                cpu_bytes as u64,
            );
        }
        if picks > 0
            && let Some(resident) = &self.resident
        {
            let (count, count_offset, records, budget) = resident.mirror_buffers();
            debug_assert_eq!(budget, mirror_records);
            encoder.copy_buffer_to_buffer(
                count,
                count_offset,
                &readback.buffer,
                mirror_offset as u64,
                4,
            );
            encoder.copy_buffer_to_buffer(
                records,
                0,
                &readback.buffer,
                mirror_offset as u64 + 16,
                (mirror_records as usize * resident::MIRROR_RECORD_BYTES) as u64,
            );
            let (records, offset, count) = resident.pick_records();
            encoder.copy_buffer_to_buffer(
                records,
                offset,
                &readback.buffer,
                picks_offset as u64,
                u64::from(count) * 16,
            );
        }
        if let Some(query) = &self.timestamps {
            encoder.resolve_query_set(query, 0..timestamp_count, &self.timestamp_resolve.buffer, 0);
            encoder.copy_buffer_to_buffer(
                &self.timestamp_resolve.buffer,
                0,
                &readback.buffer,
                readback_bytes as u64,
                timestamp_bytes,
            );
        }
        let submitted = Instant::now();
        self.last_prepare_ms = work_started.elapsed().as_secs_f32() * 1000.0;
        #[cfg_attr(not(test), allow(unused_variables))]
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        readback
            .buffer
            .slice(0..readback_bytes as u64 + timestamp_bytes)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        Ok(Some(readback::Readback {
            buffer: readback,
            #[cfg(test)]
            submission,
            completion: std::sync::Mutex::new(receiver),
            submitted,
            face_bytes: cpu_bytes,
            picks,
            mirror_records,
            bytes: readback_bytes,
            timestamp_bytes: timestamp_bytes as usize,
            samples,
            prepare_ms: self.last_prepare_ms,
            ready: false,
        }))
    }

    pub(super) fn complete_readback(
        &mut self,
        snapshot: readback::Readback,
        work: &mut TraceWork,
    ) -> Result<(), String> {
        debug_assert!(snapshot.ready, "cannot decode a pending mapping");
        let readback::Readback {
            buffer: readback,
            submitted,
            face_bytes,
            picks,
            mirror_records,
            bytes: readback_bytes,
            timestamp_bytes,
            samples,
            prepare_ms,
            ..
        } = snapshot;
        let queue = &self.context.queue;
        self.last_prepare_ms = prepare_ms;
        let view = readback
            .buffer
            .slice(0..readback_bytes as u64 + timestamp_bytes as u64)
            .get_mapped_range();
        self.last_snapshot_latency_ms = submitted.elapsed().as_secs_f32() * 1000.0;
        let timestamp_ms = if self.timestamps.is_some() {
            // Sum actual compute passes, not the gaps between encoders. Some
            // Metal devices cannot pair timestamps across separate passes.
            view[readback_bytes..]
                .chunks_exact(16)
                .map(|pair| {
                    let begin = u64::from_le_bytes(pair[..8].try_into().unwrap());
                    let end = u64::from_le_bytes(pair[8..].try_into().unwrap());
                    end.saturating_sub(begin) as f32 * queue.get_timestamp_period() / 1_000_000.0
                })
                .sum::<f32>()
        } else {
            0.0
        };
        // Some drivers expose timestamp queries but return identical/zero
        // endpoints. Never interpret that as free work and ramp without bound.
        self.last_gpu_ms =
            (timestamp_ms > 0.001 && timestamp_ms.is_finite()).then_some(timestamp_ms);
        self.last_dispatch_ms = self
            .last_gpu_ms
            .unwrap_or_else(|| submitted.elapsed().as_secs_f32() * 1000.0);
        let now = Instant::now();
        for (face, bytes) in work
            .faces
            .iter_mut()
            .zip(view[..face_bytes].chunks_exact(RESULT_BYTES))
        {
            let values: [f32; 24] = std::array::from_fn(|i| {
                f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
            });
            let mean = decode_spectrum(&values[0..4], values[4]);
            let second = decode_spectrum(&values[8..12], values[12]);
            // Unloaded-space samples are excluded from both mean and count.
            let known = values[5] as u32;
            let count = face.samples.saturating_add(known).min(HISTORY_LIMIT);
            let alpha = if known == 0 {
                0.0
            } else {
                (known as f32 / count as f32).min(1.0)
            };
            if self.resident.is_some() {
                let count = values[5].to_bits() & 0x3fff_ffff;
                face.radiance = mean;
                face.second_moment = second;
                face.samples = count;
                face.gpu_settled = Some(values[5].to_bits() & 0x8000_0000 != 0);
                face.sequence = values[6].to_bits();
                face.presentation = Vec4::new(values[7], values[13], values[14], values[15]);
            } else {
                face.radiance += (mean - face.radiance) * alpha;
                face.second_moment += (second - face.second_moment) * alpha;
                face.samples = count;
                face.sequence = face.sequence.wrapping_add(samples);
            }
            face.direct = decode_spectrum(&values[16..20], values[20]);
            face.custom_direct = Vec3::new(values[21], values[22], values[23]);
            face.updated_at = Some(submitted);
        }
        let mirror_offset = face_bytes;
        if picks > 0
            && let Some(resident) = &self.resident
        {
            let word = |at: usize| u32::from_le_bytes(view[at..at + 4].try_into().unwrap());
            let count = word(mirror_offset).min(mirror_records) as usize;
            for record in view[mirror_offset + 16..]
                .chunks_exact(resident::MIRROR_RECORD_BYTES)
                .take(count)
            {
                let word =
                    |i: usize| u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap());
                let (slot, incarnation, tag) = (word(0), word(1), word(2));
                let Some((voxel, face)) = resident.picked_face(slot, incarnation) else {
                    continue;
                };
                let values: [f32; 24] = std::array::from_fn(|i| f32::from_bits(word(4 + i)));
                let mirrored = super::work::FaceWork {
                    gpu_settled: Some(values[5].to_bits() & 0x8000_0000 != 0),
                    voxel,
                    face,
                    radiance: decode_spectrum(&values[0..4], values[4]),
                    second_moment: decode_spectrum(&values[8..12], values[12]),
                    direct: decode_spectrum(&values[16..20], values[20]),
                    custom_direct: Vec3::new(values[21], values[22], values[23]),
                    samples: values[5].to_bits() & 0x3fff_ffff,
                    history_cap: u32::MAX,
                    rollback: false,
                    material: 0,
                    revision_cap: u32::MAX,
                    sequence: values[6].to_bits(),
                    updated_at: Some(submitted),
                    presentation: Vec4::new(values[7], values[13], values[14], values[15]),
                    resident: UVec2::new(slot, incarnation),
                };
                work.picked.push((mirrored, tag));
            }
        }
        // Picked faces' donor lists build in the next batch, from the pick
        // records (slot, merged count after any retention cap, incarnation).
        if picks > 0
            && let Some(resident) = &mut self.resident
        {
            let picks_offset =
                mirror_offset + 16 + mirror_records as usize * resident::MIRROR_RECORD_BYTES;
            for record in view[picks_offset..readback_bytes].chunks_exact(16) {
                let word =
                    |i: usize| u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap());
                let (slot, tag, incarnation) = (word(0), word(1), word(3));
                if slot == u32::MAX {
                    continue;
                }
                if let Some((voxel, face)) = resident.picked_face(slot, incarnation) {
                    work.pick_tags.push((voxel, face, tag));
                }
            }
        }
        drop(view);
        readback.buffer.unmap();
        self.readbacks.push(readback);
        self.last_decode_ms = now.elapsed().as_secs_f32() * 1000.0;
        Ok(())
    }

    fn select_quality_pipeline(
        &mut self,
        transport: super::TransportSettings,
        samples: u32,
    ) -> Result<(), String> {
        // Larger batches benefit from more independent path lanes. Keep small
        // batches compact instead of allocating lanes with no samples to trace.
        let lanes = if self.adaptive_lanes {
            if samples >= 16 { 8 } else { 4 }
        } else {
            self.pipeline_key.2
        };
        #[cfg(test)]
        let lanes = self.fixed_path_lanes.unwrap_or(lanes);
        let key = (
            transport.bounces,
            transport.reach < super::tracer::MAX_PATH_DISTANCE,
            lanes,
        );
        if key == self.pipeline_key {
            return Ok(());
        }
        let pipeline = if let Some(pipeline) = self.quality_pipelines.get(&key) {
            pipeline.clone()
        } else {
            let device = &self.context.device;
            let error_scope = device
                .wgpu_device()
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let mut constants = vec![
                ("DIFFUSE_HITS", key.0 as f64),
                ("LIMITED_REACH", f64::from(key.1)),
            ];
            if self.adaptive_lanes {
                constants.push(("PATH_LANES", f64::from(lanes)));
            }
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Voxel transport quality variant"),
                layout: None,
                module: &self.module,
                entry_point: Some(self.entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    ..default()
                },
                cache: None,
            });
            if let Some(error) = bevy::tasks::block_on(error_scope.pop()) {
                return Err(error.to_string());
            }
            self.quality_pipelines.insert(key, pipeline.clone());
            pipeline
        };
        self.pipeline = pipeline;
        self.pipeline_key = key;
        self.faces_per_workgroup = 64 / lanes;

        // Auto-layout bind groups belong to their pipeline, even when every
        // buffer ID and binding type is unchanged.
        self.bindings = None;
        Ok(())
    }

    pub fn geometry_storage_bytes(&self) -> usize {
        self.packed.cells.len() * 4
            + self.packed.directory.len() * 16
            + self.packed.headers.len() * 16
    }
}

/// RGB in a vec4 lane group; the fourth lane (formerly infrared) is zero.
/// Trace a batch's change footprints into their reserved Light events, before
/// preparation applies (and the view tier copies) the events.
fn dispatch_footprints(tracer: &mut GpuTracer, encoder: &mut wgpu::CommandEncoder) {
    let GpuTracer {
        resident: Some(resident),
        context,
        params,
        directory,
        regions,
        cells,
        materials,
        lights,
        footprint_pipeline,
        finish_footprint_pipeline,
        footprint_bindings,
        finish_footprint_bindings,
        ..
    } = tracer
    else {
        return;
    };
    let Some((threads, tasks)) = resident.footprints() else {
        return;
    };
    let bindings = cached_bindings(
        footprint_bindings,
        &context.device,
        || footprint_pipeline.get_bind_group_layout(0).into(),
        [
            (0, &*params),
            (1, &*directory),
            (2, &*regions),
            (3, &*cells),
            (4, &*materials),
            (5, &*lights),
            (20, &resident.footprint_tasks),
            (21, &resident.footprint_sums),
        ],
    );
    let finish = cached_bindings(
        finish_footprint_bindings,
        &context.device,
        || finish_footprint_pipeline.get_bind_group_layout(0).into(),
        [
            (20, &resident.footprint_tasks),
            (21, &resident.footprint_sums),
            (22, resident.events_buffer()),
        ],
    );
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("GI change footprints"),
        timestamp_writes: None,
    });
    pass.set_pipeline(footprint_pipeline);
    pass.set_bind_group(0, &**bindings, &[]);
    pass.dispatch_workgroups(threads.div_ceil(64), 1, 1);
    pass.set_pipeline(finish_footprint_pipeline);
    pass.set_bind_group(0, &**finish, &[]);
    pass.dispatch_workgroups(tasks.div_ceil(64), 1, 1);
}

fn spectrum_vec(energy: Vec3) -> Vec4 {
    energy.extend(0.0)
}
fn decode_spectrum(value: &[f32], _unused: f32) -> Vec3 {
    Vec3::new(value[0], value[1], value[2]).finite_or_zero()
}
fn encode_words(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    #[test]
    fn compute_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(&SHADER)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(&SHADER)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }

    #[test]
    fn transport_params_mirror_the_shader() {
        let f = |i: u32| Vec4::splat(i as f32);
        let params = GpuParams {
            counts: UVec4::splat(1),
            minimum: IVec4::splat(2),
            maximum: IVec4::splat(3),
            environment: f(4),
            extra: f(5),
            transport: f(6),
        };
        let reflection = super::reflect::Reflection::new(&SHADER);
        super::reflect::assert_mirrors(
            &ShaderBuffer::from(params).data.unwrap(),
            reflection.layout("Params"),
            &[
                "counts",
                "minimum",
                "maximum",
                "environment",
                "extra",
                "transport",
            ],
        );
    }

    #[test]
    fn gpu_task_and_uniform_layouts_are_stable() {
        assert_eq!(
            ShaderBuffer::from(vec![GpuFaceTask::default()])
                .data
                .unwrap()
                .len(),
            32
        );
        assert_eq!(
            ShaderBuffer::from(vec![GpuParams::default()])
                .data
                .unwrap()
                .len(),
            96
        );
        assert_eq!(
            ShaderBuffer::from(vec![GpuMaterial::default()])
                .data
                .unwrap()
                .len(),
            64
        );
    }

    /// The native test adapter. It honours wgpu's environment overrides
    /// (`WGPU_BACKEND`, `WGPU_DX12_COMPILER`, `WGPU_ADAPTER_NAME`,
    /// `WGPU_POWER_PREF`), so a run that names a backend tests that backend
    /// or fails, never silently another.
    pub(crate) fn native_adapter() -> wgpu::Adapter {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = bevy::tasks::block_on(wgpu::util::initialize_adapter_from_env_or_default(
            &instance, None,
        ))
        .expect("native compute adapter for the backends WGPU_BACKEND allows");
        eprintln!("GI benchmark adapter: {:?}", adapter.get_info());
        adapter
    }

    pub(crate) fn native_tracer() -> GpuTracer {
        let adapter = native_adapter();
        let (device, queue) =
            bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("GI shared corner regression tests"),
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: adapter.limits(),
                ..default()
            }))
            .unwrap();
        let context = GpuTransportDevice {
            device: device.into(),
            queue: RenderQueue(Arc::new(bevy::render::renderer::WgpuWrapper::new(queue))),
        };
        GpuTracer::new(context).unwrap()
    }

    #[test]
    #[ignore = "requires a native compute adapter"]
    fn empty_frontier_streaming_does_not_relight_native_receivers() {
        use super::super::{
            naadf::{linear_index, volume_len},
            world::{AIR, PAGE_SIZE},
        };
        let mut gpu = native_tracer();
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        let voxel = IVec3::new(31, 8, 8);
        page[linear_index(voxel.as_uvec3(), PAGE_SIZE)] = 1;
        let base = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let extended = Arc::new(TransportScene::with_page_delta(
            &base,
            vec![(IVec3::new(2, 0, 0), vec![AIR; volume_len(PAGE_SIZE)])],
            &[],
        ));
        let sources = TransportSources::default();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut before = vec![SurfaceProbe::new(voxel, 1)];
        let mut after = before.clone();
        gpu.trace(base, &sources, &materials, &mut before, 256, &emitters)
            .unwrap();
        gpu.trace(extended, &sources, &materials, &mut after, 256, &emitters)
            .unwrap();
        assert_eq!(before[0].stats().radiance[0], after[0].stats().radiance[0]);
        assert_eq!(
            before[0].stats().second_moment[0],
            after[0].stats().second_moment[0]
        );
        assert!(before[0].stats().radiance[0].max_element() > 0.1);
    }

    #[test]
    #[ignore = "requires a native compute adapter"]
    fn error_scopes_are_thread_local() {
        // GI validates its pipelines inside error scopes on worker threads
        // while Bevy's pipeline cache does the same on its own. Shared scope
        // stacks (wgpu without "std") interleave them: a pop out of order
        // panics ("Mismatched pop_error_scope call").
        let device = native_tracer().context.device;
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let outer = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let worker = {
            let (device, barrier) = (device.clone(), barrier.clone());
            std::thread::spawn(move || {
                let held = device
                    .wgpu_device()
                    .push_error_scope(wgpu::ErrorFilter::Validation);
                barrier.wait(); // held while the other thread pushes and pops
                barrier.wait();
                assert!(bevy::tasks::block_on(held.pop()).is_none());
            })
        };
        barrier.wait();
        let inner = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        assert!(bevy::tasks::block_on(inner.pop()).is_none());
        // With one shared stack the worker's scope is on top here: panic.
        assert!(bevy::tasks::block_on(outer.pop()).is_none());
        barrier.wait();
        worker.join().unwrap();
    }

    #[test]
    #[ignore = "requires a native compute adapter"]
    fn static_dispatch_resources_reuse_and_rebind_after_growth() {
        let mut gpu = native_tracer();
        let (scene, mut sources, mut probes) = super::super::sky::contact_shadow_fixture();
        let mut materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        gpu.trace_work(
            scene.clone(),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
        )
        .unwrap();
        assert_eq!(gpu.materials.uploads, 1);
        gpu.trace(
            scene.clone(),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
        )
        .unwrap();
        let initial_faces = gpu.bindings.as_ref().unwrap().group.id();
        gpu.trace(
            scene.clone(),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
        )
        .unwrap();
        assert_eq!(gpu.materials.uploads, 1);
        assert_eq!(gpu.lights.uploads, 1);
        assert_eq!(gpu.bindings.as_ref().unwrap().group.id(), initial_faces);

        let mut many = vec![probes[0].clone(); 1024];
        gpu.trace(scene.clone(), &sources, &materials, &mut many, 4, &emitters)
            .unwrap();
        let grown_faces = gpu.bindings.as_ref().unwrap().group.id();
        assert_ne!(grown_faces, initial_faces, "work/output growth must rebind");
        gpu.trace(
            scene.clone(),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
        )
        .unwrap();
        assert_eq!(gpu.bindings.as_ref().unwrap().group.id(), grown_faces);

        sources.directional[0].radiance *= 2.0;
        sources.directional[0].direction =
            Quat::from_rotation_z(2.0_f32.to_radians()) * sources.directional[0].direction;
        let mut optical = materials.get(2);
        optical.diffuse = Vec3::ONE * 0.2;
        materials.set(2, optical);
        let mut fresh = super::super::sky::contact_shadow_fixture().2;
        gpu.trace(scene, &sources, &materials, &mut fresh, 4, &emitters)
            .unwrap();
        assert_eq!(gpu.materials.uploads, 2);
        assert_eq!(gpu.lights.uploads, 2);
    }

    #[test]
    #[ignore = "requires a native compute adapter"]
    fn explicit_measurement_restart_clears_energy_and_fades_display() {
        use super::super::{
            naadf::{linear_index, volume_len},
            world::{AIR, MaterialTransport, PAGE_SIZE},
        };
        let mut gpu = native_tracer();
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 4..=12 {
            for y in 4..=12 {
                for x in 4..=12 {
                    if [x, y, z].into_iter().any(|v| v == 4 || v == 12) {
                        page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 2;
                    }
                }
            }
        }
        let source = linear_index(UVec3::new(8, 9, 8), PAGE_SIZE);
        page[source] = 4;
        let lit = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        ));
        page[source] = AIR;
        let dark = Arc::new(TransportScene::with_page_delta(
            &lit,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            4,
            MaterialTransport {
                emission: Vec3::new(12.0, 5.0, 1.0),
                ..default()
            },
        );
        let materials = Arc::new(materials);
        let sources = TransportSources::default();
        let emitters = Arc::new(EmissionDistribution::build(&lit, &materials));
        let mut probes = vec![SurfaceProbe::new(IVec3::new(8, 4, 8), 4)];
        gpu.trace(lit, &sources, &materials, &mut probes, 128, &emitters)
            .unwrap();
        let mut history = probes.remove(0);
        assert!(history.stats().radiance[2].x > 0.01);
        history.display[2].indirect = history.stats().radiance[2];
        let original = history.display[2].indirect;
        let now = Instant::now();
        history.stats_mut().face_updated_at[2] = Some(now);
        let mut next = history.clone();
        next.restart_measurements();
        let emitters = Arc::new(EmissionDistribution::build(&dark, &materials));
        for batch in 1..=12 {
            let mut probes = vec![next];
            gpu.trace(
                dark.clone(),
                &sources,
                &materials,
                &mut probes,
                4,
                &emitters,
            )
            .unwrap();
            next = probes.remove(0);
            assert_eq!(
                next.stats().radiance[2],
                Vec3::ZERO,
                "no retained RGB after removal"
            );
            let at = now + std::time::Duration::from_millis(batch * 50);
            next.resolve_display(&history, at, 0.075);
            assert!(
                next.display[2]
                    .indirect
                    .cmple(history.display[2].indirect)
                    .all()
            );
            if batch == 1 {
                assert!(
                    next.display[2].indirect.x > original.x * 0.75,
                    "no black snap"
                );
            }
            if batch == 8 {
                assert!(
                    next.display[2].indirect.x < original.x * 0.1,
                    "less than 10% ghost after 32 fresh paths"
                );
            }
            next.stats_mut().face_updated_at[2] = Some(at);
            history = next.clone();
        }
        assert!(next.display[2].indirect.x < original.x * 0.01);
        eprintln!(
            "Explicit measurement restart: exact zero transport in first 4-path batch; display <10% after 8 batches, <1% after 12 (virtual 50ms cadence, not gameplay latency)."
        );
    }

    pub(super) fn open_floor_fixture() -> (Arc<TransportScene>, TransportSources, Vec<SurfaceProbe>)
    {
        use super::super::{
            naadf::{linear_index, volume_len},
            sources::DirectionalTransportLight,
            world::{AIR, PAGE_SIZE},
        };
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
            }
        }
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::new(0.2, 1.0, 0.3).normalize(),
                radiance: Vec3::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let initial: Vec<_> = (0..32)
            .flat_map(|z| (0..32).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        (scene, sources, initial)
    }
}
