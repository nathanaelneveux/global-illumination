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
        })
    }

    pub fn trace(
        &mut self,
        scene: Arc<TransportScene>,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        probes: &mut [SurfaceProbe],
        samples: u32,
        emitters: &Arc<EmissionDistribution>,
    ) -> Result<(), String> {
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
        let mut mapping = Vec::new();
        let mut tasks = Vec::new();
        for (index, probe) in probes.iter_mut().enumerate() {
            probe.emission = materials.get(probe.material).emission;
            for face in 0..6 {
                if probe.face_mask & (1 << face) == 0 {
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
        if tasks.is_empty() {
            return Ok(());
        }
        let params = GpuParams {
            counts: UVec4::new(
                tasks.len() as u32,
                samples,
                self.packed.directory.len() as u32 - 1,
                light_count,
            ),
            minimum: scene.top_level.minimum.extend(0),
            maximum: scene.top_level.maximum.extend(0),
            environment: spectrum_vec(sources.environment),
            extra: Vec4::new(
                sources.environment.ionizing,
                768.0,
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
        let readback_bytes = mapping.len() * RESULT_BYTES;
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
        let bindings = device.create_bind_group("GI face dispatch", &layout, &entries);
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
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &*bindings, &[]);
            pass.dispatch_workgroups((mapping.len() as u32).div_ceil(64), 1, 1);
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
        for ((index, face), bytes) in mapping.into_iter().zip(view.chunks_exact(RESULT_BYTES)) {
            let values: [f32; 24] = std::array::from_fn(|i| {
                f32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
            });
            let mean = decode_spectrum(&values[0..4], values[4]);
            let second = decode_spectrum(&values[8..12], values[12]);
            let direct = decode_spectrum(&values[16..20], values[20]);
            let probe = &mut probes[index];
            let count = probe.samples[face]
                .saturating_add(samples)
                .min(HISTORY_LIMIT);
            let alpha = (samples as f32 / count as f32).min(1.0);
            probe.radiance[face] += (mean - probe.radiance[face]) * alpha;
            probe.second_moment[face] += (second - probe.second_moment[face]) * alpha;
            probe.direct[face] = direct;
            probe.custom_direct[face] = Vec3::new(values[21], values[22], values[23]);
            probe.direct_sky[face] = Vec3::new(values[13], values[14], values[15]);
            probe.samples[face] = count;
            probe.sequence[face] = probe.sequence[face].wrapping_add(samples);
            probe.updated_at = Some(now);
        }
        drop(view);
        self.readback.buffer.unmap();
        Ok(())
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
            .flat_map(|z| {
                (10..28)
                    .map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4, EnergySpectrum::ZERO))
            })
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
            assert!((expected.direct_sky[2] - actual.direct_sky[2]).length() < 0.002);
            assert!((expected.custom_direct[2] - actual.custom_direct[2]).length() < 0.002);
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
        page[linear_index(UVec3::new(8, 8, 15), PAGE_SIZE)] = AIR;
        let edited = Arc::new(TransportScene::with_page_delta(
            &scene,
            vec![(IVec3::ZERO, page)],
            &[],
        ));
        gpu.trace(edited, &sources, &materials, &mut actual, 4, &emitters)
            .unwrap();
        assert!(
            gpu.uploaded_geometry_bytes - uploaded < 150_000,
            "single-page edit upload should be bounded"
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
            .flat_map(|z| {
                (8..16)
                    .map(move |x| SurfaceProbe::new(IVec3::new(x, 4, z), 4, EnergySpectrum::ZERO))
            })
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
