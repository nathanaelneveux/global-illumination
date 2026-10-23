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

use super::emission::EmissionDistribution;
use super::gpu_scene::PackedScene;
use super::sources::TransportSources;
use super::spectrum::EnergySpectrum;
use super::tracer::HISTORY_LIMIT;
use super::work::TraceWork;
#[cfg(test)]
use super::world::SurfaceProbe;
use super::world::{TransportScene, VoxelMaterialTransportTable};

/// The world-space consumer with the shared transport and traversal included.
static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_paths(include_str!("../../assets/shaders/voxel_trace.wgsl"))
});

/// Compose a transport consumer without duplicating the lighting estimator.
pub(crate) fn include_paths(source: &str) -> String {
    const MARKER: &str = "// #include voxel_path.wgsl\n";
    assert_eq!(
        source.matches(MARKER).count(),
        1,
        "shader needs one path include"
    );
    include_traversal(&source.replacen(
        MARKER,
        include_str!("../../assets/shaders/voxel_path.wgsl"),
        1,
    ))
}

/// Insert the shared traversal (`voxel_traversal.wgsl`) at its include marker.
pub(crate) fn include_traversal(source: &str) -> String {
    const MARKER: &str = "// #include voxel_traversal.wgsl\n";
    assert!(
        source.contains(MARKER),
        "shader lacks the traversal include marker"
    );
    source.replacen(
        MARKER,
        include_str!("../../assets/shaders/voxel_traversal.wgsl"),
        1,
    )
}

fn include_retention(source: &str) -> String {
    source.replace(
        "// #include voxel_retention.wgsl\n",
        include_str!("../../assets/shaders/voxel_retention.wgsl"),
    )
}
const RESULT_BYTES: usize = 96;
const SKY_RESULT_BYTES: usize = 32;
const TASK_BYTES: usize = 32;

mod face_index;
mod geometry;
mod readback;
mod resident;
pub(crate) mod view;
pub(crate) use readback::Readback;

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod correctness_audit;
#[cfg(test)]
mod geometry_tests;
#[cfg(test)]
mod maintenance_tests;
#[cfg(test)]
mod task_tests;
#[cfg(test)]
mod traversal_tests;
#[cfg(test)]
mod tuning_tests;
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
/// Faces precede sky corners; integer words are explicitly little-endian, with
/// all three padding words initialized. No Rust struct layout or unsafe cast.
fn encode_tasks(bytes: &mut Vec<u8>, work: &TraceWork, sky: &[IVec3]) {
    bytes.clear();
    bytes.reserve((work.faces.len() + work.picks as usize + sky.len()) * TASK_BYTES);
    // GPU pick capacity follows the CPU's faces; `select_emit` fills it and
    // the tracer skips what stays face 0xffffffff.
    let tasks = work
        .faces
        .iter()
        .map(|face| (face.voxel, face.face as u32, face.sequence))
        .chain((0..work.picks).map(|_| (IVec3::ZERO, u32::MAX, 0)))
        .chain(sky.iter().map(|&position| (position, 0, 0)));
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
    sky_pipeline: ComputePipeline,
    pub sky_cache: super::sky_cache::SkyCache,
    #[cfg(test)]
    duplicate_sky_for_benchmark: bool,
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
    material_importance: super::EnergySpectrum,
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
    sky_bindings: Option<CachedBindings<7>>,
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

    fn with_kernel(context: GpuTransportDevice, cooperative: bool) -> Result<Self, String> {
        Self::with_shader(context, cooperative, &SHADER)
    }

    fn with_shader(
        context: GpuTransportDevice,
        cooperative: bool,
        shader: &str,
    ) -> Result<Self, String> {
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
            source: ShaderSource::Wgsl(Cow::Borrowed(shader)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("World-space voxel face tracing"),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: default(),
            cache: None,
        });
        let sky_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Shared world-space sky corners"),
            layout: None,
            module: &module,
            entry_point: Some("trace_sky_corners"),
            compilation_options: default(),
            cache: None,
        });
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
            adaptive_lanes: cooperative && shader.contains("override PATH_LANES"),
            #[cfg(test)]
            fixed_path_lanes: None,
            sky_pipeline,
            sky_cache: default(),
            #[cfg(test)]
            duplicate_sky_for_benchmark: false,
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
            material_importance: super::EnergySpectrum::ONE,
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
            sky_bindings: None,
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
        self.trace_work(scene, sources, materials, probes, samples, emitters, false)
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
        sky_only: bool,
    ) -> Result<(), String> {
        let mut work = TraceWork::from_probes(probes, sky_only);
        self.trace_compact(
            scene, sources, materials, &mut work, samples, emitters, sky_only,
        )?;
        work.copy_measurements(probes);
        if !sky_only {
            for probe in probes {
                probe.emission = materials.get(probe.material).emission;
            }
        }
        Ok(())
    }

    pub fn trace_compact(
        &mut self,
        scene: Arc<TransportScene>,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        work: &mut TraceWork,
        samples: u32,
        emitters: &Arc<EmissionDistribution>,
        sky_only: bool,
    ) -> Result<(), String> {
        if let Some(mut snapshot) =
            self.submit_compact(scene, sources, materials, work, samples, emitters, sky_only)?
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
        sky_only: bool,
    ) -> Result<Option<readback::Readback>, String> {
        let work_started = Instant::now();
        let geometry_changed = self.needs_geometry(&scene);
        self.synchronize_geometry(&scene)?;
        if !sky_only {
            self.select_quality_pipeline(work.transport.normalized(), samples)?;
        }
        self.sky_cache.set_materials(materials);
        self.sky_cache.retire_pages(&work.retire_pages);
        let sky_pending = self.sky_cache.prepare(&scene, sources, &work.probes);
        #[cfg(test)]
        let sky_pending = if self.duplicate_sky_for_benchmark {
            let duplicate: Vec<_> = work
                .faces
                .iter()
                .filter(|face| {
                    self.sky_cache.lights.iter().any(|light| {
                        super::world::FACE_OFFSETS[face.face]
                            .as_vec3()
                            .dot(light.direction)
                            > 0.0
                    })
                })
                .flat_map(|face| {
                    (0..4).map(move |corner| {
                        super::sky::corner_position(face.voxel, face.face, corner)
                    })
                })
                .collect();
            self.sky_cache.stats.traced = duplicate.len();
            self.sky_cache.stats.rays = duplicate.len() * self.sky_cache.lights.len();
            duplicate
        } else {
            sky_pending
        };
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
        let device = &self.context.device;
        let queue = &self.context.queue;
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
        if self.material_table.as_ref() != Some(materials)
            || self.material_importance != emitters.importance
        {
            let gpu_materials: Vec<_> = (0..=255)
                .map(|i| {
                    let material = materials.get(i);
                    GpuMaterial {
                        diffuse: spectrum_vec(material.diffuse),
                        diffuse_extra: Vec4::new(material.diffuse.ionizing, 0.0, 0.0, 0.0),
                        emission: spectrum_vec(material.emission),
                        emission_extra: Vec4::new(
                            material.emission.ionizing,
                            (material.emission * emitters.importance).max_component(),
                            0.0,
                            0.0,
                        ),
                    }
                })
                .collect();
            self.materials.upload_generation(
                device,
                queue,
                &ShaderBuffer::from(gpu_materials).data.unwrap(),
            )?;
            self.material_table = Some(materials.clone());
            self.material_importance = emitters.importance;
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
                        light.radiance.ionizing,
                        light.custom_visible.x,
                        light.custom_visible.y,
                        light.custom_visible.z,
                    ),
                })
                .chain(sources.points.iter().map(|light| GpuLight {
                    position_range: light.position.extend(light.range),
                    energy: spectrum_vec(light.intensity),
                    extra: Vec4::new(
                        light.intensity.ionizing,
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
        // Geometry, ownership and model are now current. Maintenance needs
        // neither a ray dispatch nor a readback allocation.
        if work.faces.is_empty() && work.picks == 0 && sky_pending.is_empty() {
            if prepare_resident {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("GI receiver maintenance"),
                });
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
            self.sky_cache.apply(&mut work.probes);
            return Ok(None);
        }
        let picks = if self.resident.is_some() {
            work.picks
        } else {
            0
        };
        let face_count = work.faces.len() + picks as usize;
        #[cfg(not(test))]
        encode_tasks(&mut self.task_bytes, work, &sky_pending);
        #[cfg(test)]
        if self.legacy_task_encoding_for_benchmark {
            self.task_bytes = task_tests::legacy_task_bytes(work, &sky_pending);
        } else {
            encode_tasks(&mut self.task_bytes, work, &sky_pending);
        }
        let transport = work.transport.normalized();
        let params = GpuParams {
            counts: UVec4::new(
                face_count as u32,
                samples,
                self.packed.directory.len() as u32 - 1,
                light_count,
            ),
            minimum: scene.top_level.minimum.extend(sky_pending.len() as i32),
            maximum: scene
                .top_level
                .maximum
                .extend(sources.directional.len() as i32),
            environment: spectrum_vec(sources.environment),
            extra: Vec4::new(
                sources.environment.ionizing,
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
        self.tasks.upload(device, queue, &self.task_bytes)?;
        // Readback: the CPU's faces, sky corners, then (with GPU picks) the
        // statistics-mirror count and records. Picks themselves are not read
        // back: the mirror catches up with them under a fixed budget.
        let face_bytes = face_count * RESULT_BYTES;
        let cpu_bytes = work.faces.len() * RESULT_BYTES;
        let sky_bytes = sky_pending.len() * SKY_RESULT_BYTES;
        let mirror_offset = cpu_bytes + sky_bytes;
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
        self.output.ensure(device, face_bytes + sky_bytes)?;
        let mut readback = self.readbacks.pop().unwrap_or_else(|| {
            UploadBuffer::new(
                device,
                "GI asynchronous simulation snapshot",
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            )
        });
        readback.ensure(device, readback_bytes + timestamp_bytes as usize)?;
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
        // Retire stale groups even when their pass is idle. Otherwise a warm
        // sky cache could retain an obsolete, large output buffer indefinitely.
        let ids = buffers.map(|buffer| buffer.buffer.id());
        if self
            .bindings
            .as_ref()
            .is_some_and(|cached| cached.buffers != ids)
        {
            self.bindings = None;
        }
        let sky_ids = [0, 1, 2, 3, 5, 6, 7].map(|index| ids[index]);
        if self
            .sky_bindings
            .as_ref()
            .is_some_and(|cached| cached.buffers != sky_ids)
        {
            self.sky_bindings = None;
        }
        let bindings = (!sky_only).then(|| {
            cached_bindings(
                &mut self.bindings,
                device,
                || self.pipeline.get_bind_group_layout(0).into(),
                std::array::from_fn(|index| (index as u32, buffers[index])),
            )
        });
        let sky_bindings = if sky_pending.is_empty() {
            None
        } else {
            Some(cached_bindings(
                &mut self.sky_bindings,
                device,
                || self.sky_pipeline.get_bind_group_layout(0).into(),
                [0, 1, 2, 3, 5, 6, 7].map(|index| (index as u32, buffers[index])),
            ))
        };
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
            if let Some(bindings) = bindings {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &**bindings, &[]);
                pass.dispatch_workgroups(
                    (face_count as u32).div_ceil(self.faces_per_workgroup),
                    1,
                    1,
                );
            }
            if let Some(bindings) = sky_bindings {
                pass.set_pipeline(&self.sky_pipeline);
                pass.set_bind_group(0, &**bindings, &[]);
                pass.dispatch_workgroups((sky_pending.len() as u32).div_ceil(64), 1, 1);
            }
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
        if sky_bytes > 0 {
            encoder.copy_buffer_to_buffer(
                &self.output.buffer,
                face_bytes as u64,
                &readback.buffer,
                cpu_bytes as u64,
                sky_bytes as u64,
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
            submission,
            completion: std::sync::Mutex::new(receiver),
            submitted,
            face_bytes: cpu_bytes,
            picks,
            mirror_records,
            bytes: readback_bytes,
            timestamp_bytes: timestamp_bytes as usize,
            samples,
            sky_pending,
            sky_ticket: self.sky_cache.ticket(),
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
            sky_pending,
            sky_ticket,
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
            if let Some(resident) = &mut self.resident {
                // A retention cap can leave a CPU job's face short of the
                // evidence target its list was planned without: list it
                // for its next trace from the GPU's merged count.
                let count = values[5].to_bits() & 0x3fff_ffff;
                resident.request_capped_list(face.resident, count);
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
        let mirror_offset = face_bytes + sky_pending.len() * SKY_RESULT_BYTES;
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
                let (slot, tag, count, incarnation) = (word(0), word(1), word(2), word(3));
                if slot == u32::MAX {
                    continue;
                }
                if let Some((voxel, face)) = resident.picked_face(slot, incarnation) {
                    work.pick_tags.push((voxel, face, tag));
                }
                resident.request_donor_list(UVec2::new(slot, incarnation), count);
            }
        }
        for (position, bytes) in sky_pending
            .into_iter()
            .zip(view[face_bytes..mirror_offset].chunks_exact(SKY_RESULT_BYTES))
        {
            // The first word is the corner's visibility; the rest is padding.
            self.sky_cache.store_ticket(
                sky_ticket,
                position,
                u32::from_le_bytes(bytes[..4].try_into().unwrap()),
            );
        }
        drop(view);
        readback.buffer.unmap();
        self.readbacks.push(readback);
        self.sky_cache.apply(&mut work.probes);
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
        // buffer ID and binding type is unchanged. Sky has its own pipeline.
        self.bindings = None;
        Ok(())
    }

    pub fn geometry_storage_bytes(&self) -> usize {
        self.packed.cells.len() * 4
            + self.packed.directory.len() * 16
            + self.packed.headers.len() * 16
    }
}

fn spectrum_vec(energy: EnergySpectrum) -> Vec4 {
    energy.visible.extend(energy.infrared)
}
fn decode_spectrum(value: &[f32], radiation: f32) -> EnergySpectrum {
    EnergySpectrum::new(Vec3::new(value[0], value[1], value[2]), value[3], radiation)
        .finite_or_zero()
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

    pub(crate) fn native_tracer() -> GpuTracer {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = bevy::tasks::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("native compute adapter");
        eprintln!("GI benchmark adapter: {:?}", adapter.get_info());
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
        assert!(before[0].stats().radiance[0].visible.max_element() > 0.1);
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
            true,
        )
        .unwrap();
        assert_eq!(
            gpu.materials.uploads, 1,
            "sky-only work still publishes a complete optical scene for views"
        );
        let initial_sky = gpu.sky_bindings.as_ref().unwrap().group.id();
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
        assert_eq!(gpu.sky_bindings.as_ref().unwrap().group.id(), initial_sky);

        let mut many = vec![probes[0].clone(); 1024];
        gpu.trace(scene.clone(), &sources, &materials, &mut many, 4, &emitters)
            .unwrap();
        let grown_faces = gpu.bindings.as_ref().unwrap().group.id();
        assert!(
            gpu.sky_bindings.is_none(),
            "idle sky pass must release superseded buffers"
        );
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
        // Color alone keeps cached visibility; leaving the sun's quantization
        // cell makes the sky cold again.
        sources.directional[0].direction =
            Quat::from_rotation_z(2.0_f32.to_radians()) * sources.directional[0].direction;
        let mut optical = materials.get(2);
        optical.diffuse = EnergySpectrum::ONE * 0.2;
        materials.set(2, optical);
        let mut fresh = super::super::sky::contact_shadow_fixture().2;
        let expected = super::super::tracer::trace_probe_batch(
            scene.clone(),
            sources.clone(),
            Arc::new(materials.clone()),
            fresh.clone(),
            4,
        );
        gpu.trace(scene, &sources, &materials, &mut fresh, 4, &emitters)
            .unwrap();
        assert_eq!(gpu.materials.uploads, 2);
        assert_eq!(gpu.lights.uploads, 2);
        assert_ne!(
            gpu.sky_bindings.as_ref().unwrap().group.id(),
            initial_sky,
            "cold sky must bind the grown work/output buffers"
        );
        for (actual, expected) in fresh.iter().zip(expected) {
            assert_eq!(actual.display[2].sky, expected.display[2].sky);
            assert!(
                (actual.stats().direct[2].visible - expected.stats().direct[2].visible).length()
                    < 0.002
            );
            assert!(
                (actual.stats().radiance[2].visible - expected.stats().radiance[2].visible)
                    .length()
                    < 0.002
            );
        }
    }

    #[test]
    #[ignore = "requires a native compute adapter"]
    fn explicit_measurement_restart_clears_energy_and_fades_display() {
        use super::super::{
            naadf::{linear_index, volume_len},
            tracer::trace_probe_batch,
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
                emission: EnergySpectrum::new(Vec3::new(12.0, 5.0, 1.0), 4.0, 2.0),
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
        assert!(
            history.stats().radiance[2].visible.x > 0.01
                && history.stats().radiance[2].infrared > 0.0
                && history.stats().radiance[2].ionizing > 0.0
        );
        history.display[2].indirect = history.stats().radiance[2].visible;
        let original = history.display[2].indirect;
        let now = Instant::now();
        history.stats_mut().face_updated_at[2] = Some(now);
        let mut next = history.clone();
        next.restart_measurements();
        let emitters = Arc::new(EmissionDistribution::build(&dark, &materials));
        let expected = trace_probe_batch(
            dark.clone(),
            sources.clone(),
            materials.clone(),
            vec![next.clone()],
            4,
        )
        .remove(0);
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
                EnergySpectrum::ZERO,
                "no retained RGB/IR/radiation after removal"
            );
            if batch == 1 {
                assert_eq!(next.stats().radiance[2], expected.stats().radiance[2]);
            }
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

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn sky_only_and_partial_face_dispatch_preserve_simulation_and_match_cpu() {
        use super::super::{sky::contact_shadow_fixture, tracer::trace_probe_batch};
        let mut gpu = native_tracer();
        let (scene, sources, mut probes) = contact_shadow_fixture();
        let materials = Arc::new(VoxelMaterialTransportTable::default());
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let reference = trace_probe_batch(
            Arc::clone(&scene),
            sources.clone(),
            Arc::clone(&materials),
            probes.clone(),
            0,
        );
        gpu.trace_work(
            Arc::clone(&scene),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
            true,
        )
        .unwrap();
        for (actual, expected) in probes.iter().zip(&reference) {
            assert_eq!(actual.display[2].sky, expected.display[2].sky);
            assert_eq!(actual.stats().samples, [0; 6]);
            assert_eq!(actual.stats().sequence, [0; 6]);
            assert_eq!(actual.stats().direct, [EnergySpectrum::ZERO; 6]);
            assert!(actual.updated_at.is_none());
            assert!(actual.stats().face_updated_at.iter().all(Option::is_none));
        }
        gpu.trace_work(
            Arc::clone(&scene),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
            true,
        )
        .unwrap();
        assert_eq!(gpu.sky_cache.stats.traced, 0);
        assert_eq!(
            gpu.last_dispatch_ms, 0.0,
            "warm sky-only work needs no GPU dispatch"
        );
        probes[0].face_mask = 5;
        probes[0].trace_mask = 4;
        probes[0].stats_mut().samples[0] = 32;
        probes[0].stats_mut().radiance[0] = EnergySpectrum::ONE;
        let untouched = probes[0].clone();
        probes[1].trace_mask = 0;
        probes[1].stats_mut().samples[2] = 128;
        probes[1].stats_mut().sequence[2] = 1000;
        let sky_receiver = probes[1].clone();
        let reference = trace_probe_batch(
            Arc::clone(&scene),
            sources.clone(),
            Arc::clone(&materials),
            probes.clone(),
            4,
        );
        gpu.trace(
            Arc::clone(&scene),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
        )
        .unwrap();
        assert_eq!(probes[0].stats().samples[0], untouched.stats().samples[0]);
        assert_eq!(probes[0].stats().radiance[0], untouched.stats().radiance[0]);
        assert_eq!(probes[0].stats().sequence[0], untouched.stats().sequence[0]);
        assert_eq!(probes[0].face_mask, 5);
        assert_eq!(probes[0].stats().samples[2], 4);
        assert!(
            (reference[0].stats().radiance[2].visible - probes[0].stats().radiance[2].visible)
                .length()
                < 0.002
        );
        assert_eq!(reference[0].display[2].sky, probes[0].display[2].sky);
        for actual in [&probes[1], &reference[1]] {
            assert_eq!(actual.stats().samples, sky_receiver.stats().samples);
            assert_eq!(actual.stats().sequence, sky_receiver.stats().sequence);
            assert_eq!(actual.stats().radiance, sky_receiver.stats().radiance);
            assert_eq!(actual.stats().direct, sky_receiver.stats().direct);
            assert_eq!(actual.updated_at, sky_receiver.updated_at);
            assert_eq!(
                actual.stats().face_updated_at,
                sky_receiver.stats().face_updated_at
            );
        }
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn adaptive_idle_and_edit_workload() {
        use super::super::{
            naadf::{linear_index, volume_len},
            scheduling::{due_faces, record_batch},
            settings::VoxelGiSettings,
            sources::DirectionalTransportLight,
            world::{AIR, PAGE_SIZE},
        };
        let mut gpu = native_tracer();
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
            }
        }
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        ));
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::Y,
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let mut probes: Vec<_> = (8..24)
            .flat_map(|z| (8..24).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        let settings = VoxelGiSettings::default();
        let start = Instant::now();
        let mut paths = 0;
        let mut idle = 0;
        let mut workers = Vec::new();
        // Virtual 60 Hz scheduling clock, real Metal ray tracing. This measures
        // work reduction, NOT whole-game FPS or real-time edit latency.
        for frame in 0..600 {
            let now = start + Duration::from_secs_f64(frame as f64 / 60.0);
            let mut selected = Vec::new();
            let mut indices = Vec::new();
            for (index, probe) in probes.iter().enumerate() {
                let mask = due_faces(probe, 0, now, &settings, false);
                if mask != 0 {
                    let mut probe = probe.clone();
                    probe.trace_mask = mask;
                    selected.push(probe);
                    indices.push(index);
                    paths += mask.count_ones() * 4;
                }
            }
            if selected.is_empty() {
                idle += 1;
                continue;
            }
            let started = Instant::now();
            gpu.trace(
                Arc::clone(&scene),
                &sources,
                &materials,
                &mut selected,
                4,
                &emitters,
            )
            .unwrap();
            workers.push(started.elapsed().as_secs_f32() * 1000.0);
            for (index, mut probe) in indices.into_iter().zip(selected) {
                probe.resolve_display(&probes[index], now, settings.display_response_seconds());
                record_batch(&mut probe, &probes[index], now, &settings);
                probes[index] = probe;
            }
        }
        assert!(
            idle > 500,
            "settled terrain should stop dispatching continuously: {idle}"
        );
        assert!(paths < 60 * 256 * 4);
        let expected = sources.environment.visible * (0.65 + 0.35 * 2.0 / 3.0);
        let rms = (probes
            .iter()
            .map(|probe| (probe.stats().radiance[2].visible - expected).length_squared())
            .sum::<f32>()
            / (probes.len() * 3) as f32)
            .sqrt();
        let max_error = probes
            .iter()
            .map(|probe| {
                (probe.stats().radiance[2].visible - expected)
                    .abs()
                    .max_element()
            })
            .fold(0.0, f32::max);
        eprintln!("Analytic open-sky fixture error: RMS={rms:.6}, max channel={max_error:.6}");
        // This policy trades sampling for bounded observed fixture error, not
        // exact integration. CPU/GPU sample parity is tested separately.
        assert!(rms < 0.003 && max_error < 0.015);
        workers.sort_by(f32::total_cmp);
        eprintln!(
            "Adaptive Metal workload: {paths} paths versus 614400 continuous paths, {idle}/600 idle scheduling ticks, worker median {:.3} ms; virtual 10-second static 256-face fixture",
            workers[workers.len() / 2]
        );

        // An inserted roof must update sky with no new simulation samples, then
        // wake settled indirect faces. Visibility crosses previously empty air.
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 8, z), PAGE_SIZE)] = 2;
            }
        }
        let edited = Arc::new(TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        let before = probes.clone();
        let started = Instant::now();
        gpu.trace_work(
            Arc::clone(&edited),
            &sources,
            &materials,
            &mut probes,
            4,
            &emitters,
            true,
        )
        .unwrap();
        eprintln!(
            "Roof sky-only worker {:.3} ms; GPU {:.3} ms; prepare {:.3}, snapshot age {:.3}, decode {:.3} ms",
            started.elapsed().as_secs_f32() * 1000.0,
            gpu.last_dispatch_ms,
            gpu.last_prepare_ms,
            gpu.last_snapshot_latency_ms,
            gpu.last_decode_ms
        );
        for (probe, old) in probes.iter().zip(&before) {
            assert_eq!(probe.display[2].sky, [super::super::sky::KNOWN; 4]);
            assert_eq!(probe.stats().samples, old.stats().samples);
            assert_eq!(probe.stats().radiance, old.stats().radiance);
            assert_eq!(probe.stats().face_updated_at, old.stats().face_updated_at);
            assert_eq!(
                due_faces(probe, 1, start + Duration::from_secs(10), &settings, false),
                4
            );
        }
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
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let initial: Vec<_> = (0..32)
            .flat_map(|z| (0..32).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        (scene, sources, initial)
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn cooperative_path_layout_benchmark() {
        let parallel = native_tracer();
        let serial = GpuTracer::with_kernel(parallel.context.clone(), false).unwrap();
        compare_kernels(serial, parallel, ["serial", "cooperative"]);
    }

    /// Compare an archived shader to the current kernel with identical work,
    /// alternating order and full five-band moment parity. The baseline path is
    /// explicit so experiment shaders never become a production runtime mode.
    #[test]
    #[ignore = "requires a native adapter and VOXEL_GI_BASELINE_SHADER"]
    fn shader_candidate_benchmark() {
        let path = std::env::var("VOXEL_GI_BASELINE_SHADER").expect("baseline WGSL path");
        let source = std::fs::read_to_string(path).expect("read baseline shader");
        let candidate = native_tracer();
        let baseline = GpuTracer::with_shader(candidate.context.clone(), true, &source).unwrap();
        compare_kernels(baseline, candidate, ["baseline", "candidate"]);
    }

    fn compare_kernels(mut serial: GpuTracer, mut parallel: GpuTracer, labels: [&str; 2]) {
        use super::super::{
            naadf::{linear_index, volume_len},
            world::{AIR, MaterialTransport, PAGE_SIZE},
        };
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            4,
            MaterialTransport {
                emission: EnergySpectrum::new(Vec3::new(12.0, 3.0, 1.0), 4.0, 2.0),
                ..default()
            },
        );
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for y in 4..=20 {
                for x in 0..32 {
                    if y == 4 || y == 20 || x == 0 || x == 31 || z == 0 || z == 31 {
                        page[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 2;
                    }
                }
            }
        }
        page[linear_index(UVec3::new(16, 19, 16), PAGE_SIZE)] = 4;
        let pages: Vec<_> = (0..3)
            .flat_map(|z| (0..3).map(move |x| IVec3::new(x, 0, z)))
            .collect();
        let closed = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            pages.iter().map(|p| (*p, page.clone())).collect(),
            &[],
        ));
        let closed_probes: Vec<_> = pages
            .iter()
            .flat_map(|page| {
                (1..31).flat_map(move |z| {
                    (1..31).map(move |x| SurfaceProbe::new(*page * 32 + IVec3::new(x, 4, z), 4))
                })
            })
            .collect();
        let (open, sources, open_probes) = open_floor_fixture();
        for (name, scene, initial) in [
            ("open floor", open, open_probes),
            ("nine emissive rooms", closed, closed_probes),
        ] {
            let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
            let mut worker_ms: [Vec<f32>; 2] = default();
            let mut gpu_ms: [Vec<f32>; 2] = default();
            for round in 0..36 {
                // Untimed tail/count cases also exercise inactive barrier lanes
                // and the generalized subsequence loop used by reference tests.
                let samples = match round {
                    33 => 1,
                    34 => 7,
                    35 => 64,
                    _ => 4,
                };
                let receivers = if round >= 33 {
                    &initial[..17]
                } else {
                    &initial[..]
                };
                let mut outputs: [Vec<SurfaceProbe>; 2] = default();
                // Alternate order, warm both caches, and use matching advancing
                // sequences. Keep ray count and all transport bands fixed.
                for mode in (0..2).map(|offset| (offset + round) % 2) {
                    let tracer = if mode == 0 {
                        &mut serial
                    } else {
                        &mut parallel
                    };
                    let mut actual = receivers.to_vec();
                    for probe in &mut actual {
                        probe.stats_mut().sequence = [(round * 4) as u32; 6];
                    }
                    let started = Instant::now();
                    tracer
                        .trace(
                            scene.clone(),
                            &sources,
                            &materials,
                            &mut actual,
                            samples,
                            &emitters,
                        )
                        .unwrap();
                    if (3..33).contains(&round) {
                        worker_ms[mode].push(started.elapsed().as_secs_f32() * 1000.0);
                        gpu_ms[mode].push(tracer.last_dispatch_ms);
                    }
                    outputs[mode] = actual;
                }
                for (a, b) in outputs[0].iter().zip(&outputs[1]) {
                    assert_eq!(a.stats().samples, b.stats().samples);
                    assert_eq!(a.stats().sequence, b.stats().sequence);
                    assert_eq!(a.stats().direct, b.stats().direct);
                    assert_eq!(a.stats().custom_direct, b.stats().custom_direct);
                    assert_eq!(a.display[2].sky, b.display[2].sky);
                    for (a, b) in a
                        .stats()
                        .radiance
                        .iter()
                        .chain(&a.stats().second_moment)
                        .zip(b.stats().radiance.iter().chain(&b.stats().second_moment))
                    {
                        let delta = *a - *b;
                        let error = delta
                            .visible
                            .abs()
                            .max_element()
                            .max(delta.infrared.abs())
                            .max(delta.ionizing.abs());
                        assert!(
                            error <= 0.00001 * (1.0 + a.max_component()),
                            "parallel path moments differ: {a:?} vs {b:?}"
                        );
                    }
                }
            }
            for (mode, label) in labels.into_iter().enumerate() {
                worker_ms[mode].sort_by(f32::total_cmp);
                gpu_ms[mode].sort_by(f32::total_cmp);
                eprintln!(
                    "{name}, {label}, {} faces x 4 paths: GPU p50/p95 {:.3}/{:.3} ms; worker p50/p95 {:.3}/{:.3} ms (30 alternating pairs)",
                    initial.len(),
                    gpu_ms[mode][15],
                    gpu_ms[mode][28],
                    worker_ms[mode][15],
                    worker_ms[mode][28]
                );
            }
        }
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn fused_sky_transport_benchmark() {
        let mut gpu = native_tracer();
        let (scene, sources, initial) = open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let mut worker_ms: [Vec<f32>; 2] = default();
        let mut gpu_ms: [Vec<f32>; 2] = default();
        let mut expected: Option<Vec<SurfaceProbe>> = None;
        for round in 0..43 {
            for mode in (0..2).map(|offset| (offset + round) % 2) {
                gpu.sky_cache = default();
                let mut actual = initial.clone();
                for probe in &mut actual[..512] {
                    probe.trace_mask = 0;
                }
                let started = Instant::now();
                let (rays, execution) = if mode == 0 {
                    gpu.trace_work(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut actual[..512],
                        4,
                        &emitters,
                        true,
                    )
                    .unwrap();
                    let rays = gpu.sky_cache.stats.rays;
                    let execution = gpu.last_dispatch_ms;
                    gpu.trace(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut actual[512..],
                        4,
                        &emitters,
                    )
                    .unwrap();
                    (
                        rays + gpu.sky_cache.stats.rays,
                        execution + gpu.last_dispatch_ms,
                    )
                } else {
                    gpu.trace(
                        Arc::clone(&scene),
                        &sources,
                        &materials,
                        &mut actual,
                        4,
                        &emitters,
                    )
                    .unwrap();
                    (gpu.sky_cache.stats.rays, gpu.last_dispatch_ms)
                };
                let elapsed = started.elapsed().as_secs_f32() * 1000.0;
                assert_eq!(rays, 1089);
                if let Some(expected) = &expected {
                    for (a, b) in actual.iter().zip(expected) {
                        assert_eq!(a.stats().samples, b.stats().samples);
                        assert_eq!(a.stats().sequence, b.stats().sequence);
                        assert_eq!(a.stats().radiance, b.stats().radiance);
                        assert_eq!(a.stats().second_moment, b.stats().second_moment);
                        assert_eq!(a.stats().direct, b.stats().direct);
                        assert_eq!(a.display[2].sky, b.display[2].sky);
                    }
                } else {
                    expected = Some(actual);
                }
                if round >= 3 {
                    worker_ms[mode].push(elapsed);
                    gpu_ms[mode].push(execution);
                }
            }
        }
        for (mode, name) in ["separate", "fused"].iter().enumerate() {
            worker_ms[mode].sort_by(f32::total_cmp);
            gpu_ms[mode].sort_by(f32::total_cmp);
            eprintln!(
                "Mixed-queue {name}: 512 GI faces x 4 paths + 512 sky-only receivers, 1089 sky rays; worker median {:.3} ms, GPU median {:.3} ms (40 rounds)",
                worker_ms[mode][20], gpu_ms[mode][20]
            );
        }
    }

    #[test]
    #[ignore = "native equal-work benchmark for explicit sky demand"]
    fn explicit_sky_demand_benchmark() {
        use super::super::{
            naadf::{linear_index, volume_len},
            world::{AIR, PAGE_SIZE},
        };
        let mut gpu = native_tracer();
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
            }
        }
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            (0..8)
                .map(|x| (IVec3::new(x, 0, 0), page.clone()))
                .collect(),
            &[],
        ));
        let sources = open_floor_fixture().1;
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::default());
        let mut warm: Vec<_> = (0..32)
            .flat_map(|z| (0..256).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        gpu.trace(scene.clone(), &sources, &materials, &mut warm, 4, &emitters)
            .unwrap();
        assert!(warm.iter().all(|p| !p.display[2].sky.contains(&0)));
        let mut cpu_ms: [Vec<f32>; 2] = default();
        let mut worker_ms: [Vec<f32>; 2] = default();
        let mut gpu_ms: [Vec<f32>; 2] = default();
        for round in 0..43 {
            let mut results: [Vec<SurfaceProbe>; 2] = default();
            for mode in (0..2).map(|offset| (offset + round) % 2) {
                let mut probes = warm.clone();
                for p in &mut probes {
                    p.stats_mut().sequence[2] = (round * 4) as u32;
                }
                let mut work = TraceWork::from_probes(&probes, false);
                for job in &mut work.probes {
                    job.sky_requested = mode == 0;
                }
                let start = Instant::now();
                gpu.trace_compact(
                    scene.clone(),
                    &sources,
                    &materials,
                    &mut work,
                    4,
                    &emitters,
                    false,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f32() * 1000.0;
                assert_eq!(
                    gpu.sky_cache.stats.requested,
                    if mode == 0 { 32768 } else { 0 }
                );
                assert_eq!(
                    gpu.sky_cache.stats.rays, 0,
                    "both modes reuse measured sunlight"
                );
                if round >= 3 {
                    cpu_ms[mode].push(gpu.last_prepare_ms + gpu.last_decode_ms);
                    worker_ms[mode].push(elapsed);
                    gpu_ms[mode].push(gpu.last_gpu_ms.unwrap());
                }
                work.copy_measurements(&mut probes);
                results[mode] = probes;
            }
            for (a, b) in results[0].iter().zip(&results[1]) {
                assert_eq!(a.stats().radiance, b.stats().radiance);
                assert_eq!(a.stats().second_moment, b.stats().second_moment);
                assert_eq!(a.stats().direct, b.stats().direct);
                assert_eq!(a.display[2].sky, b.display[2].sky);
                assert_eq!(a.stats().sequence, b.stats().sequence);
            }
        }
        for mode in 0..2 {
            cpu_ms[mode].sort_by(f32::total_cmp);
            worker_ms[mode].sort_by(f32::total_cmp);
            gpu_ms[mode].sort_by(f32::total_cmp);
            eprintln!(
                "Explicit sky demand {}: 8192faces x4paths, CPU prepare+decode p50/p95 {:.3}/{:.3}ms; worker {:.3}/{:.3}ms; GPU {:.3}/{:.3}ms (40 alternating pairs)",
                ["legacy", "explicit"][mode],
                cpu_ms[mode][20],
                cpu_ms[mode][38],
                worker_ms[mode][20],
                worker_ms[mode][38],
                gpu_ms[mode][20],
                gpu_ms[mode][38]
            );
        }
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn shared_sky_corner_benchmark() {
        let mut gpu = native_tracer();
        let (scene, sources, initial) = open_floor_fixture();
        let materials = VoxelMaterialTransportTable::default();
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let mut gpu_ms: [Vec<f32>; 3] = default();
        let mut worker_ms: [Vec<f32>; 3] = default();
        let mut expected = None;
        // Same transport paths and GPU kernel in all modes; only sky task
        // multiplicity/cache reuse differs. This is not the previous shader.
        // Rotate order and discard initial rounds to avoid startup bias.
        for round in 0..33 {
            for mode in (0..3).map(|offset| (offset + round) % 3) {
                gpu.duplicate_sky_for_benchmark = mode == 0;
                if mode < 2 {
                    gpu.sky_cache = default();
                }
                let mut actual = initial.clone();
                let started = Instant::now();
                gpu.trace(
                    Arc::clone(&scene),
                    &sources,
                    &materials,
                    &mut actual,
                    4,
                    &emitters,
                )
                .unwrap();
                let wall_ms = started.elapsed().as_secs_f32() * 1000.0;
                let expected_rays = [4096, 1089, 0][mode];
                assert_eq!(gpu.sky_cache.stats.rays, expected_rays);
                let colors: Vec<_> = actual.iter().map(|probe| probe.display[2].sky).collect();
                if let Some(expected) = &expected {
                    assert_eq!(
                        &colors, expected,
                        "sharing/reuse must not change corner colors"
                    );
                } else {
                    expected = Some(colors);
                }
                if round >= 3 {
                    gpu_ms[mode].push(gpu.last_dispatch_ms);
                    worker_ms[mode].push(wall_ms);
                }
            }
        }
        for (mode, name) in ["duplicated", "shared-cold", "shared-warm"]
            .iter()
            .enumerate()
        {
            gpu_ms[mode].sort_by(f32::total_cmp);
            worker_ms[mode].sort_by(f32::total_cmp);
            let mid = gpu_ms[mode].len() / 2;
            eprintln!(
                "Sky benchmark {name}: 1024 faces × 4 GI paths, {} scheduled sky rays, GPU median {:.3} ms, worker median {:.3} ms (30 samples)",
                [4096, 1089, 0][mode],
                gpu_ms[mode][mid],
                worker_ms[mode][mid]
            );
        }
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn gpu_matches_cpu_reference_and_reuses_geometry_after_edit() {
        use super::super::{
            naadf::{linear_index, volume_len},
            sources::{DirectionalTransportLight, PointTransportLight},
            tracer::trace_probe_batch,
            world::{AIR, PAGE_SIZE},
        };
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = bevy::tasks::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .expect("native compute adapter");
        eprintln!("GI reference adapter: {:?}", adapter.get_info());
        let (device, queue) =
            bevy::tasks::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("GI regression tests"),
                required_features: adapter.features() & wgpu::Features::TIMESTAMP_QUERY,
                required_limits: adapter.limits(),
                ..default()
            }))
            .unwrap();
        let mut gpu = GpuTracer::new(GpuTransportDevice {
            device: device.into(),
            queue: RenderQueue(Arc::new(bevy::render::renderer::WgpuWrapper::new(queue))),
        })
        .unwrap();
        let mut page = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 0..32 {
            for x in 0..32 {
                page[linear_index(UVec3::new(x, 4, z), PAGE_SIZE)] = 2;
            }
        }
        for z in 4..28 {
            for y in 5..18 {
                page[linear_index(UVec3::new(8, y, z), PAGE_SIZE)] = 1;
            }
        }
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, page.clone())],
            &[],
        ));
        let sources = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::new(0.2, 1.0, 0.3).normalize(),
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::new(0.125, 0.2, 0.3),
            }],
            points: vec![PointTransportLight {
                position: Vec3::new(16.5, 12.0, 15.5),
                intensity: EnergySpectrum::new(Vec3::splat(8.0), 4.0, 2.0),
                range: 40.0,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let materials = Arc::new(VoxelMaterialTransportTable::default());
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let probes: Vec<_> = (0..28)
            .flat_map(|z| (10..28).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        let cpu = trace_probe_batch(
            Arc::clone(&scene),
            sources.clone(),
            Arc::clone(&materials),
            probes.clone(),
            64,
        );
        gpu.sky_cache.track_changes = true;
        let mut actual = probes;
        gpu.trace(
            Arc::clone(&scene),
            &sources,
            &materials,
            &mut actual,
            64,
            &emitters,
        )
        .unwrap();
        let mut squared = 0.0;
        for (expected, actual) in cpu.iter().zip(&actual) {
            squared += (expected.stats().radiance[2].visible - actual.stats().radiance[2].visible)
                .length_squared();
            assert!(
                (expected.stats().direct[2].infrared - actual.stats().direct[2].infrared).abs()
                    < 0.002
            );
            assert!(
                (expected.stats().direct[2].ionizing - actual.stats().direct[2].ionizing).abs()
                    < 0.002
            );
            assert!(
                (expected.stats().direct[2].visible - actual.stats().direct[2].visible).length()
                    < 0.002
            );
            assert!(
                (expected.stats().custom_direct[2] - actual.stats().custom_direct[2]).length()
                    < 0.002
            );
            assert_eq!(
                expected.display[2].sky, actual.display[2].sky,
                "packed CPU/GPU sky corners at {:?}",
                actual.voxel
            );
        }
        let rms = (squared / (actual.len() * 3) as f32).sqrt();
        assert!(
            actual
                .iter()
                .any(|probe| probe.stats().radiance[2].visible.max_element() > 0.05)
        );
        eprintln!(
            "GPU reference RMS={rms:.6}, {} faces × 64 paths in {:.3} ms",
            actual.len(),
            gpu.last_dispatch_ms
        );
        assert!(rms < 0.005, "GPU and CPU transport diverged: {rms}");
        let uploaded = gpu.uploaded_geometry_bytes;
        assert!(gpu.sky_cache.take_changes(&scene).is_some());
        gpu.trace(
            Arc::clone(&scene),
            &sources,
            &materials,
            &mut actual,
            4,
            &emitters,
        )
        .unwrap();
        assert_eq!(
            gpu.uploaded_geometry_bytes, uploaded,
            "unchanged geometry must not be uploaded"
        );
        assert_eq!(gpu.sky_cache.stats.traced, 0, "warm sky must not retrace");
        assert!(gpu.sky_cache.take_changes(&scene).is_none());
        page[linear_index(UVec3::new(8, 8, 15), PAGE_SIZE)] = AIR;
        let edited = Arc::new(TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        gpu.trace(
            Arc::clone(&edited),
            &sources,
            &materials,
            &mut actual,
            4,
            &emitters,
        )
        .unwrap();
        let cold_edited = trace_probe_batch(
            edited,
            sources.clone(),
            Arc::clone(&materials),
            actual.clone(),
            1,
        );
        assert!(gpu.sky_cache.stats.traced > 0);
        for (expected, actual) in cold_edited.iter().zip(&actual) {
            assert_eq!(
                expected.display[2].sky, actual.display[2].sky,
                "incremental GPU sky must agree with a cold CPU solve after editing"
            );
        }
        assert!(
            gpu.uploaded_geometry_bytes - uploaded < 150_000,
            "single-page edit upload should be bounded"
        );

        let (contact_scene, contact_sources, mut contact) =
            super::super::sky::contact_shadow_fixture();
        let expected = trace_probe_batch(
            Arc::clone(&contact_scene),
            contact_sources.clone(),
            Arc::clone(&materials),
            contact.clone(),
            1,
        );
        let emitters = Arc::new(EmissionDistribution::build(&contact_scene, &materials));
        gpu.trace(
            contact_scene,
            &contact_sources,
            &materials,
            &mut contact,
            1,
            &emitters,
        )
        .unwrap();
        for (expected, actual) in expected.iter().zip(&contact) {
            assert_eq!(expected.display[2].sky, actual.display[2].sky);
            assert_eq!(actual.display[2].sky[0], super::super::sky::KNOWN);
            assert!(super::super::sky::lit(actual.display[2].sky[1], 0));
        }

        // Shared crease corners, all six normals, negative coordinates, and
        // two differently colored suns must match the reference projection.
        let mut cube = vec![AIR; volume_len(PAGE_SIZE)];
        cube[linear_index(UVec3::splat(23), PAGE_SIZE)] = 2;
        let cube_scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::NEG_ONE, cube)],
            &[],
        ));
        let mut two_suns = TransportSources {
            directional: vec![
                DirectionalTransportLight {
                    direction: Vec3::new(1.0, 2.0, 3.0).normalize(),
                    radiance: EnergySpectrum::new(Vec3::new(1.0, 0.5, 0.3), 2.0, 0.0),
                    custom_visible: Vec3::ZERO,
                },
                DirectionalTransportLight {
                    direction: Vec3::new(-3.0, -2.0, -1.0).normalize(),
                    radiance: EnergySpectrum::new(Vec3::new(0.3, 0.5, 1.0), 0.0, 1.0),
                    custom_visible: Vec3::ZERO,
                },
            ],
            ..default()
        };
        let emitters = Arc::new(EmissionDistribution::build(&cube_scene, &materials));
        let mut cube_probes = vec![SurfaceProbe::new(IVec3::splat(-9), 63)];
        let expected = trace_probe_batch(
            Arc::clone(&cube_scene),
            two_suns.clone(),
            Arc::clone(&materials),
            cube_probes.clone(),
            1,
        );
        gpu.trace(
            Arc::clone(&cube_scene),
            &two_suns,
            &materials,
            &mut cube_probes,
            1,
            &emitters,
        )
        .unwrap();
        assert_eq!(gpu.sky_cache.stats.requested, 24);
        assert_eq!(gpu.sky_cache.stats.traced, 8);
        assert_eq!(gpu.sky_cache.stats.rays, 16);
        for face in 0..6 {
            assert_eq!(
                expected[0].display[face].sky,
                cube_probes[0].display[face].sky
            );
        }
        two_suns.points = sources.points.clone();
        gpu.trace(
            Arc::clone(&cube_scene),
            &two_suns,
            &materials,
            &mut cube_probes,
            1,
            &emitters,
        )
        .unwrap();
        assert_eq!(
            gpu.sky_cache.stats.rays, 0,
            "changing point lights must reuse sky"
        );

        // Exercise the tagged UNIFORM BRICK path within a mixed region, not
        // just uniform whole regions and mixed voxel payloads.
        let mut uniform_bricks = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 4..12 {
            for y in 4..12 {
                for x in 4..12 {
                    uniform_bricks[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 255;
                }
            }
        }
        uniform_bricks[linear_index(UVec3::new(14, 4, 8), PAGE_SIZE)] = 2;
        let uniform_scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, uniform_bricks)],
            &[],
        ));
        let mut grazing_sun = sources.clone();
        grazing_sun.directional[0].direction = Vec3::new(-1.0, 0.1, 0.0).normalize();
        let emitters = Arc::new(EmissionDistribution::build(&uniform_scene, &materials));
        let mut receiver = vec![SurfaceProbe::new(IVec3::new(14, 4, 8), 4)];
        let reference = trace_probe_batch(
            Arc::clone(&uniform_scene),
            grazing_sun.clone(),
            Arc::clone(&materials),
            receiver.clone(),
            64,
        );
        gpu.trace(
            uniform_scene,
            &grazing_sun,
            &materials,
            &mut receiver,
            64,
            &emitters,
        )
        .unwrap();
        assert_eq!(receiver[0].display[2].sky, [super::super::sky::KNOWN; 4]);
        assert_eq!(receiver[0].display[2].sky, reference[0].display[2].sky);
        assert!(
            (receiver[0].stats().radiance[2].visible - reference[0].stats().radiance[2].visible)
                .length()
                < 0.002
        );

        // Display shadows cross streamed gaps; simulation energy stays strict.
        // Also exercise sparse-directory reuse after the preceding edit.
        let empty = vec![AIR; volume_len(PAGE_SIZE)];
        let mut distant_geometry = empty.clone();
        // An occupied region across the gap, but not on this sun ray. Empty
        // page bookkeeping alone must never create an unknown occluder.
        distant_geometry[linear_index(UVec3::new(8, 12, 8), PAGE_SIZE)] = 1;
        let sparse = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![
                (IVec3::ZERO, empty.clone()),
                (IVec3::new(2, 0, 0), distant_geometry),
            ],
            &[],
        ));
        let horizontal = TransportSources {
            directional: vec![DirectionalTransportLight {
                direction: Vec3::X,
                radiance: EnergySpectrum::ONE,
                custom_visible: Vec3::ZERO,
            }],
            ..default()
        };
        let emitters = Arc::new(EmissionDistribution::build(&sparse, &materials));
        let mut boundary = vec![SurfaceProbe::new(IVec3::new(31, 8, 8), 1)];
        gpu.trace(
            Arc::clone(&sparse),
            &horizontal,
            &materials,
            &mut boundary,
            1,
            &emitters,
        )
        .unwrap();
        assert_eq!(
            boundary[0].display[0].sky,
            [super::super::sky::KNOWN | 1; 4]
        );
        assert_eq!(boundary[0].stats().direct[0], EnergySpectrum::ZERO);
        let filled = Arc::new(TransportScene::with_page_delta(
            &sparse,
            vec![(IVec3::X, empty)],
            &[],
        ));
        gpu.trace(
            Arc::clone(&filled),
            &horizontal,
            &materials,
            &mut boundary,
            1,
            &emitters,
        )
        .unwrap();
        assert_eq!(
            boundary[0].display[0].sky,
            [super::super::sky::KNOWN | 1; 4]
        );
        let blocked = Arc::new(TransportScene::with_page_delta(
            &filled,
            vec![(IVec3::new(2, 0, 0), vec![2; volume_len(PAGE_SIZE)])],
            &[IVec3::X],
        ));
        gpu.trace(
            blocked,
            &horizontal,
            &materials,
            &mut boundary,
            1,
            &emitters,
        )
        .unwrap();
        assert_eq!(
            boundary[0].display[0].sky,
            [super::super::sky::KNOWN; 4],
            "do not skip loaded occluders beyond streaming gaps"
        );

        // Exercise emissive MIS (not just sky and analytic sources) inside an
        // opaque enclosure, where any environment contribution is a leak.
        let mut room = vec![AIR; volume_len(PAGE_SIZE)];
        for z in 4..21 {
            for y in 4..17 {
                for x in 4..21 {
                    if x == 4 || x == 20 || y == 4 || y == 16 || z == 4 || z == 20 {
                        room[linear_index(UVec3::new(x, y, z), PAGE_SIZE)] = 3;
                    }
                }
            }
        }
        room[linear_index(UVec3::new(12, 12, 12), PAGE_SIZE)] = 9;
        let scene = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![(IVec3::ZERO, room)],
            &[],
        ));
        let mut materials = VoxelMaterialTransportTable::default();
        materials.set(
            9,
            super::super::world::MaterialTransport {
                emission: EnergySpectrum::new(Vec3::splat(8.0), 4.0, 2.0),
                ..default()
            },
        );
        let materials = Arc::new(materials);
        let emitters = Arc::new(EmissionDistribution::build(&scene, &materials));
        let sources = TransportSources::default();
        let probes: Vec<_> = (8..16)
            .flat_map(|z| (8..16).map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4)))
            .collect();
        let expected = trace_probe_batch(
            Arc::clone(&scene),
            sources.clone(),
            Arc::clone(&materials),
            probes.clone(),
            128,
        );
        let mut actual = probes;
        gpu.trace(scene, &sources, &materials, &mut actual, 128, &emitters)
            .unwrap();
        let squared = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| {
                (a.stats().radiance[2].visible - b.stats().radiance[2].visible).length_squared()
            })
            .sum::<f32>();
        let rms = (squared / (actual.len() * 3) as f32).sqrt();
        eprintln!("Emissive enclosed-room GPU/CPU RMS={rms:.6}");
        assert!(rms < 0.005);
        assert!(
            actual
                .iter()
                .all(|probe| probe.stats().radiance[2].visible.x > 0.01)
        );
    }
}
