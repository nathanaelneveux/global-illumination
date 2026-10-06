//! Asynchronous, bounded compute transport on Bevy's device. GPU geometry and
//! scratch allocations persist across batches. Only requested face estimates
//! are read back; neither the render thread nor gameplay waits for the GPU.
use std::borrow::Cow;
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::render_resource::{
    BindGroupEntry, Buffer, BufferDescriptor, BufferUsages, ComputePipeline,
    ShaderModuleDescriptor, ShaderSource, ShaderType,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::ShaderBuffer;

use super::emission::EmissionDistribution;
use super::gpu_scene::PackedScene;
use super::spectrum::EnergySpectrum;
use super::tracer::{HISTORY_LIMIT, TransportSources};
use super::world::{SurfaceProbe, TransportScene, VoxelMaterialTransportTable};

const SHADER: &str = include_str!("../../assets/shaders/voxel_trace.wgsl");
const RESULT_BYTES: usize = 96;
const SKY_RESULT_BYTES: usize = 32;

#[derive(Resource, Clone)]
pub(crate) struct GpuTransportDevice {
    pub device: RenderDevice,
    pub queue: RenderQueue,
}

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuParams {
    counts: UVec4,
    minimum: IVec4,
    maximum: IVec4,
    environment: Vec4,
    extra: Vec4,
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

#[derive(Clone, Copy, Default, ShaderType)]
struct GpuFaceTask {
    voxel: IVec3,
    face: u32,
    sequence: UVec4,
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
}

impl UploadBuffer {
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

    fn upload(
        &mut self,
        device: &RenderDevice,
        queue: &RenderQueue,
        bytes: &[u8],
    ) -> Result<(), String> {
        self.ensure(device, bytes.len())?;
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytes);
        }
        Ok(())
    }
}

pub(crate) struct GpuTracer {
    context: GpuTransportDevice,
    pipeline: ComputePipeline,
    sky_pipeline: ComputePipeline,
    pub sky_cache: super::sky_cache::SkyCache,
    #[cfg(test)]
    duplicate_sky_for_benchmark: bool,
    packed: PackedScene,
    scene: Option<Arc<TransportScene>>,
    params: UploadBuffer,
    directory: UploadBuffer,
    regions: UploadBuffer,
    cells: UploadBuffer,
    materials: UploadBuffer,
    lights: UploadBuffer,
    emitters: UploadBuffer,
    emission: Option<Arc<EmissionDistribution>>,
    tasks: UploadBuffer,
    output: UploadBuffer,
    readback: UploadBuffer,
    timestamps: Option<wgpu::QuerySet>,
    timestamp_resolve: UploadBuffer,
    pub last_dispatch_ms: f32,
    pub uploaded_geometry_bytes: u64,
    pub last_prepare_ms: f32,
    pub last_wait_ms: f32,
    pub last_decode_ms: f32,
}

impl GpuTracer {
    pub fn new(context: GpuTransportDevice) -> Result<Self, String> {
        let device = &context.device;
        if device.limits().max_storage_buffers_per_shader_stage < 8 {
            return Err("GPU transport needs eight storage-buffer bindings".into());
        }
        let error_scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_and_validate_shader_module(ShaderModuleDescriptor {
            label: Some("World-space voxel transport"),
            source: ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("World-space voxel face tracing"),
            layout: None,
            module: &module,
            entry_point: Some("trace_faces"),
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
                        count: 2,
                    })
            });
        Ok(Self {
            pipeline,
            sky_pipeline,
            sky_cache: default(),
            #[cfg(test)]
            duplicate_sky_for_benchmark: false,
            packed: PackedScene::default(),
            scene: None,
            params: UploadBuffer::new(
                device,
                "GI uniforms",
                BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            ),
            directory: UploadBuffer::new(device, "GI sparse directory", storage),
            regions: UploadBuffer::new(device, "GI region headers", storage),
            cells: UploadBuffer::new(device, "GI packed NAADF cells", storage),
            materials: UploadBuffer::new(device, "GI material transport", storage),
            lights: UploadBuffer::new(device, "GI analytic sources", storage),
            emitters: UploadBuffer::new(device, "GI emissive face distribution", storage),
            emission: None,
            tasks: UploadBuffer::new(device, "GI face work queue", storage),
            output: UploadBuffer::new(
                device,
                "GI face estimates",
                BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            ),
            readback: UploadBuffer::new(
                device,
                "GI asynchronous simulation snapshot",
                BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            ),
            timestamps,
            timestamp_resolve: UploadBuffer::new(
                device,
                "GI timestamp resolve",
                BufferUsages::QUERY_RESOLVE | BufferUsages::COPY_SRC,
            ),
            last_dispatch_ms: 0.0,
            context,
            uploaded_geometry_bytes: 0,
            last_prepare_ms: 0.0,
            last_wait_ms: 0.0,
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

    #[allow(clippy::too_many_arguments)]
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
        let work_started = Instant::now();
        let mut mapping = Vec::new();
        let mut tasks = Vec::new();
        for (index, probe) in probes.iter_mut().enumerate() {
            if !sky_only {
                probe.emission = materials.get(probe.material).emission;
            }
            for face in 0..6 {
                if sky_only || probe.face_mask & probe.trace_mask & (1 << face) == 0 {
                    continue;
                }
                mapping.push((index, face));
                tasks.push(GpuFaceTask {
                    voxel: probe.voxel,
                    face: face as u32,
                    sequence: UVec4::new(probe.sequence[face], 0, 0, 0),
                });
            }
        }
        let sky_pending = self.sky_cache.prepare(&scene, sources, probes);
        #[cfg(test)]
        let sky_pending = if self.duplicate_sky_for_benchmark {
            let duplicate: Vec<_> = mapping
                .iter()
                .filter(|(_, face)| {
                    self.sky_cache.lights.iter().any(|light| {
                        super::world::FACE_OFFSETS[*face]
                            .as_vec3()
                            .dot(light.direction)
                            > 0.0
                    })
                })
                .flat_map(|(probe, face)| {
                    let voxel = probes[*probe].voxel;
                    (0..4).map(move |corner| super::sky::corner_position(voxel, *face, corner))
                })
                .collect();
            self.sky_cache.stats.traced = duplicate.len();
            self.sky_cache.stats.rays = duplicate.len() * self.sky_cache.lights.len();
            duplicate
        } else {
            sky_pending
        };
        // Reusing measured corners needs no queue writes, geometry upload, or
        // GPU submission. Leave geometry synchronization until rays need it.
        if tasks.is_empty() && sky_pending.is_empty() {
            self.last_dispatch_ms = 0.0;
            self.last_prepare_ms = work_started.elapsed().as_secs_f32() * 1000.0;
            self.last_wait_ms = 0.0;
            self.last_decode_ms = 0.0;
            self.sky_cache.apply(probes);
            return Ok(());
        }
        let device = &self.context.device;
        let queue = &self.context.queue;
        if !sky_only
            && self
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
                        0.0,
                    ),
                })
                .collect();
            if data.is_empty() {
                data.push(GpuEmitter::default());
            }
            self.emitters
                .upload(device, queue, &ShaderBuffer::from(data).data.unwrap())?;
            self.emission = Some(Arc::clone(emitters));
        }
        if self
            .scene
            .as_ref()
            .is_none_or(|previous| !Arc::ptr_eq(previous, &scene))
        {
            let dirty = self.packed.update(&scene);
            let directory = ShaderBuffer::from(self.packed.directory.clone())
                .data
                .unwrap();
            let headers = ShaderBuffer::from(self.packed.headers.clone())
                .data
                .unwrap();
            self.directory.upload(device, queue, &directory)?;
            self.regions.upload(device, queue, &headers)?;
            self.uploaded_geometry_bytes += (directory.len() + headers.len()) as u64;
            if self.cells.ensure(device, self.packed.cells.len() * 4)? {
                let bytes = encode_words(&self.packed.cells);
                queue.write_buffer(&self.cells.buffer, 0, &bytes);
                self.uploaded_geometry_bytes += bytes.len() as u64;
            } else {
                for range in dirty {
                    let bytes = encode_words(&self.packed.cells[range.clone()]);
                    queue.write_buffer(&self.cells.buffer, (range.start * 4) as u64, &bytes);
                    self.uploaded_geometry_bytes += bytes.len() as u64;
                }
            }
            self.scene = Some(Arc::clone(&scene));
        }
        if !sky_only {
            let gpu_materials: Vec<_> = (0..=255)
                .map(|i| {
                    let material = materials.get(i);
                    GpuMaterial {
                        diffuse: spectrum_vec(material.diffuse),
                        diffuse_extra: Vec4::new(material.diffuse.ionizing, 0.0, 0.0, 0.0),
                        emission: spectrum_vec(material.emission),
                        emission_extra: Vec4::new(material.emission.ionizing, 0.0, 0.0, 0.0),
                    }
                })
                .collect();
            self.materials.upload(
                device,
                queue,
                &ShaderBuffer::from(gpu_materials).data.unwrap(),
            )?;
        }
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
        let light_count = gpu_lights.len() as u32;
        if gpu_lights.is_empty() {
            gpu_lights.push(GpuLight::default());
        }
        self.lights
            .upload(device, queue, &ShaderBuffer::from(gpu_lights).data.unwrap())?;
        let face_count = tasks.len();
        tasks.extend(sky_pending.iter().map(|&position| GpuFaceTask {
            voxel: position,
            ..default()
        }));
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
        };
        self.params.upload(
            device,
            queue,
            &ShaderBuffer::from(vec![params]).data.unwrap(),
        )?;
        self.tasks
            .upload(device, queue, &ShaderBuffer::from(tasks).data.unwrap())?;
        let face_bytes = face_count * RESULT_BYTES;
        let readback_bytes = face_bytes + sky_pending.len() * SKY_RESULT_BYTES;
        self.output.ensure(device, readback_bytes)?;
        self.readback.ensure(device, readback_bytes + 16)?;
        let layout = self.pipeline.get_bind_group_layout(0).into();
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
        ];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| BindGroupEntry {
                binding: binding as u32,
                resource: buffer.buffer.as_entire_binding(),
            })
            .collect();
        let bindings =
            (!sky_only).then(|| device.create_bind_group("GI face dispatch", &layout, &entries));
        let sky_bindings = if sky_pending.is_empty() {
            None
        } else {
            let layout = self.sky_pipeline.get_bind_group_layout(0).into();
            let entries: Vec<_> = [0, 1, 2, 3, 5, 6, 7]
                .into_iter()
                .map(|binding| BindGroupEntry {
                    binding: binding as u32,
                    resource: buffers[binding].buffer.as_entire_binding(),
                })
                .collect();
            Some(device.create_bind_group("GI shared corner dispatch", &layout, &entries))
        };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GI bounded transport batch"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GI world-space rays"),
                timestamp_writes: self.timestamps.as_ref().map(|query_set| {
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }
                }),
            });
            if let Some(bindings) = &bindings {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &**bindings, &[]);
                pass.dispatch_workgroups((face_count as u32).div_ceil(64), 1, 1);
            }
            if let Some(bindings) = &sky_bindings {
                pass.set_pipeline(&self.sky_pipeline);
                pass.set_bind_group(0, &**bindings, &[]);
                pass.dispatch_workgroups((sky_pending.len() as u32).div_ceil(64), 1, 1);
            }
        }
        encoder.copy_buffer_to_buffer(
            &self.output.buffer,
            0,
            &self.readback.buffer,
            0,
            readback_bytes as u64,
        );
        if let Some(query) = &self.timestamps {
            encoder.resolve_query_set(query, 0..2, &self.timestamp_resolve.buffer, 0);
            encoder.copy_buffer_to_buffer(
                &self.timestamp_resolve.buffer,
                0,
                &self.readback.buffer,
                readback_bytes as u64,
                16,
            );
        }
        let submitted = Instant::now();
        self.last_prepare_ms = work_started.elapsed().as_secs_f32() * 1000.0;
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::sync_channel(1);
        self.readback
            .buffer
            .slice(0..readback_bytes as u64 + 16)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        // This method is only executed by an async worker, never an ECS/render
        // system. The timeout also provides a bounded device-loss failure path.
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(5)),
            })
            .map_err(|error| error.to_string())?;
        receiver
            .recv_timeout(Duration::from_secs(1))
            .map_err(|error| error.to_string())?
            .map_err(|error| error.to_string())?;
        let view = self
            .readback
            .buffer
            .slice(0..readback_bytes as u64 + 16)
            .get_mapped_range();
        self.last_wait_ms = submitted.elapsed().as_secs_f32() * 1000.0;
        let timestamp_ms = if self.timestamps.is_some() {
            let begin =
                u64::from_le_bytes(view[readback_bytes..readback_bytes + 8].try_into().unwrap());
            let end = u64::from_le_bytes(
                view[readback_bytes + 8..readback_bytes + 16]
                    .try_into()
                    .unwrap(),
            );
            end.saturating_sub(begin) as f32 * queue.get_timestamp_period() / 1_000_000.0
        } else {
            0.0
        };
        // Some drivers expose timestamp queries but return identical/zero
        // endpoints. Never interpret that as free work and ramp without bound.
        self.last_dispatch_ms = if timestamp_ms > 0.001 {
            timestamp_ms
        } else {
            submitted.elapsed().as_secs_f32() * 1000.0
        };
        let now = Instant::now();
        for ((index, face), bytes) in mapping
            .into_iter()
            .zip(view[..face_bytes].chunks_exact(RESULT_BYTES))
        {
            let values: [f32; 24] = std::array::from_fn(|i| {
                f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
            });
            let probe = &mut probes[index];
            let mean = decode_spectrum(&values[0..4], values[4]);
            let second = decode_spectrum(&values[8..12], values[12]);
            let count = probe.samples[face]
                .saturating_add(samples)
                .min(HISTORY_LIMIT);
            let alpha = (samples as f32 / count as f32).min(1.0);
            probe.radiance[face] += (mean - probe.radiance[face]) * alpha;
            probe.second_moment[face] += (second - probe.second_moment[face]) * alpha;
            probe.direct[face] = decode_spectrum(&values[16..20], values[20]);
            probe.custom_direct[face] = Vec3::new(values[21], values[22], values[23]);
            probe.samples[face] = count;
            probe.sequence[face] = probe.sequence[face].wrapping_add(samples);
            probe.updated_at = Some(now);
            probe.face_updated_at[face] = Some(now);
        }
        for (position, bytes) in sky_pending
            .into_iter()
            .zip(view[face_bytes..readback_bytes].chunks_exact(SKY_RESULT_BYTES))
        {
            self.sky_cache.store(
                position,
                std::array::from_fn(|face| {
                    u32::from_le_bytes(bytes[face * 4..face * 4 + 4].try_into().unwrap())
                }),
            );
        }
        drop(view);
        self.readback.buffer.unmap();
        self.sky_cache.apply(probes);
        self.last_decode_ms = now.elapsed().as_secs_f32() * 1000.0;
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
mod tests {
    use super::*;

    #[test]
    fn compute_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(SHADER)
            .unwrap_or_else(|error| panic!("{}", error.emit_to_string(SHADER)));
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
            80
        );
        assert_eq!(
            ShaderBuffer::from(vec![GpuMaterial::default()])
                .data
                .unwrap()
                .len(),
            64
        );
    }

    fn native_tracer() -> GpuTracer {
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
        GpuTracer::new(GpuTransportDevice {
            device: device.into(),
            queue: RenderQueue(Arc::new(bevy::render::renderer::WgpuWrapper::new(queue))),
        })
        .unwrap()
    }

    #[test]
    #[ignore = "requires a native compute adapter"]
    fn emitter_removal_replaces_energy_in_one_batch_and_fades_display() {
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
            history.radiance[2].visible.x > 0.01
                && history.radiance[2].infrared > 0.0
                && history.radiance[2].ionizing > 0.0
        );
        history.display[2].indirect = history.radiance[2].visible;
        let original = history.display[2].indirect;
        let now = Instant::now();
        history.face_updated_at[2] = Some(now);
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
                next.radiance[2],
                EnergySpectrum::ZERO,
                "no retained RGB/IR/radiation after removal"
            );
            if batch == 1 {
                assert_eq!(next.radiance[2], expected.radiance[2]);
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
            next.face_updated_at[2] = Some(at);
            history = next.clone();
        }
        assert!(next.display[2].indirect.x < original.x * 0.01);
        eprintln!(
            "Emitter removal: exact zero transport in first 4-path batch; display <10% after 8 batches, <1% after 12 (virtual 50ms cadence, not gameplay latency)."
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
            assert_eq!(actual.samples, [0; 6]);
            assert_eq!(actual.sequence, [0; 6]);
            assert_eq!(actual.direct, [EnergySpectrum::ZERO; 6]);
            assert!(actual.updated_at.is_none());
            assert!(actual.face_updated_at.iter().all(Option::is_none));
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
        probes[0].samples[0] = 32;
        probes[0].radiance[0] = EnergySpectrum::ONE;
        let untouched = probes[0].clone();
        probes[1].trace_mask = 0;
        probes[1].samples[2] = 128;
        probes[1].sequence[2] = 1000;
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
        assert_eq!(probes[0].samples[0], untouched.samples[0]);
        assert_eq!(probes[0].radiance[0], untouched.radiance[0]);
        assert_eq!(probes[0].sequence[0], untouched.sequence[0]);
        assert_eq!(probes[0].face_mask, 5);
        assert_eq!(probes[0].samples[2], 4);
        assert!(
            (reference[0].radiance[2].visible - probes[0].radiance[2].visible).length() < 0.002
        );
        assert_eq!(reference[0].display[2].sky, probes[0].display[2].sky);
        for actual in [&probes[1], &reference[1]] {
            assert_eq!(actual.samples, sky_receiver.samples);
            assert_eq!(actual.sequence, sky_receiver.sequence);
            assert_eq!(actual.radiance, sky_receiver.radiance);
            assert_eq!(actual.direct, sky_receiver.direct);
            assert_eq!(actual.updated_at, sky_receiver.updated_at);
            assert_eq!(actual.face_updated_at, sky_receiver.face_updated_at);
        }
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn adaptive_idle_and_edit_workload() {
        use super::super::{
            naadf::{linear_index, volume_len},
            scheduling::{due_faces, record_batch},
            settings::VoxelGiSettings,
            tracer::DirectionalTransportLight,
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
            .map(|probe| (probe.radiance[2].visible - expected).length_squared())
            .sum::<f32>()
            / (probes.len() * 3) as f32)
            .sqrt();
        let max_error = probes
            .iter()
            .map(|probe| (probe.radiance[2].visible - expected).abs().max_element())
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
            "Roof sky-only worker {:.3} ms; GPU {:.3} ms; prepare {:.3}, wait {:.3}, decode {:.3} ms",
            started.elapsed().as_secs_f32() * 1000.0,
            gpu.last_dispatch_ms,
            gpu.last_prepare_ms,
            gpu.last_wait_ms,
            gpu.last_decode_ms
        );
        for (probe, old) in probes.iter().zip(&before) {
            assert_eq!(probe.display[2].sky, [1; 4]);
            assert_eq!(probe.samples, old.samples);
            assert_eq!(probe.radiance, old.radiance);
            assert_eq!(probe.face_updated_at, old.face_updated_at);
            assert_eq!(
                due_faces(probe, 1, start + Duration::from_secs(10), &settings, false),
                4
            );
        }
    }

    fn open_floor_fixture() -> (Arc<TransportScene>, TransportSources, Vec<SurfaceProbe>) {
        use super::super::{
            naadf::{linear_index, volume_len},
            tracer::DirectionalTransportLight,
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
                        assert_eq!(a.samples, b.samples);
                        assert_eq!(a.sequence, b.sequence);
                        assert_eq!(a.radiance, b.radiance);
                        assert_eq!(a.second_moment, b.second_moment);
                        assert_eq!(a.direct, b.direct);
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
            tracer::{DirectionalTransportLight, PointTransportLight, trace_probe_batch},
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
            squared += (expected.radiance[2].visible - actual.radiance[2].visible).length_squared();
            assert!((expected.direct[2].infrared - actual.direct[2].infrared).abs() < 0.002);
            assert!((expected.direct[2].ionizing - actual.direct[2].ionizing).abs() < 0.002);
            assert!((expected.direct[2].visible - actual.direct[2].visible).length() < 0.002);
            assert!((expected.custom_direct[2] - actual.custom_direct[2]).length() < 0.002);
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
                .any(|probe| probe.radiance[2].visible.max_element() > 0.05)
        );
        eprintln!(
            "GPU reference RMS={rms:.6}, {} faces × 64 paths in {:.3} ms",
            actual.len(),
            gpu.last_dispatch_ms
        );
        assert!(rms < 0.005, "GPU and CPU transport diverged: {rms}");
        let uploaded = gpu.uploaded_geometry_bytes;
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
            assert_eq!(actual.display[2].sky[0], 1);
            assert!(super::super::sky::unpack(actual.display[2].sky[1]).x > 0.9);
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
        assert_eq!(receiver[0].display[2].sky, [1; 4]);
        assert_eq!(receiver[0].display[2].sky, reference[0].display[2].sky);
        assert!(
            (receiver[0].radiance[2].visible - reference[0].radiance[2].visible).length() < 0.002
        );

        // Display shadows cross streamed gaps; simulation energy stays strict.
        // Also exercise sparse-directory reuse after the preceding edit.
        let empty = vec![AIR; volume_len(PAGE_SIZE)];
        let sparse = Arc::new(TransportScene::with_page_delta(
            &TransportScene::default(),
            vec![
                (IVec3::ZERO, empty.clone()),
                (IVec3::new(2, 0, 0), empty.clone()),
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
            [super::super::sky::pack(Vec3::ONE); 4]
        );
        assert_eq!(boundary[0].direct[0], EnergySpectrum::ZERO);
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
            [super::super::sky::pack(Vec3::ONE); 4]
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
            boundary[0].display[0].sky, [1; 4],
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
            .map(|(a, b)| (a.radiance[2].visible - b.radiance[2].visible).length_squared())
            .sum::<f32>();
        let rms = (squared / (actual.len() * 3) as f32).sqrt();
        eprintln!("Emissive enclosed-room GPU/CPU RMS={rms:.6}");
        assert!(rms < 0.005);
        assert!(
            actual
                .iter()
                .all(|probe| probe.radiance[2].visible.x > 0.01)
        );
    }
}
