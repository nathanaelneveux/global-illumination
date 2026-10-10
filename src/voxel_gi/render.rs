#[cfg(test)]
use bevy::platform::collections::HashMap;
use std::sync::{Arc, Mutex};

use bevy::image::{ImageAddressMode, ImageSampler, TextureReinterpretationError};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::platform::collections::{HashMap as VoxelMap, HashSet};
use bevy::prelude::*;
use bevy::render::erased_render_asset::prepare_erased_assets;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{RenderAssets, prepare_assets};
use bevy::render::render_resource::{
    AsBindGroup, Buffer, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{GpuResourceAppExt, Render, RenderApp, RenderSystems};
use bevy::shader::ShaderRef;
use bevy_voxel_world::rendering::vertex_layout;

use super::world::SurfaceProbe;

pub const VOXEL_TRANSPORT_SHADER: &str = "shaders/voxel_transport.wgsl";
const MAX_SHADER_HASH_PROBES: usize = 32;
const GPU_PROBE_BYTES: usize = 208;

pub type VoxelTransportMaterial = ExtendedMaterial<StandardMaterial, VoxelTransportExtension>;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct VoxelTransportExtension {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    pub voxel_texture: Handle<Image>,
    #[storage(102, read_only)]
    pub hash_entries: Handle<ShaderBuffer>,
    #[storage(103, read_only)]
    pub probes: Handle<ShaderBuffer>,
    #[uniform(104)]
    pub settings: GpuTransportSettings,
}

impl MaterialExtension for VoxelTransportExtension {
    fn fragment_shader() -> ShaderRef {
        VOXEL_TRANSPORT_SHADER.into()
    }

    fn vertex_shader() -> ShaderRef {
        VOXEL_TRANSPORT_SHADER.into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct GpuTransportSettings {
    pub hash_mask: u32,
    /// Allocated bound, fixed for this buffer generation (not live occupancy).
    pub probe_capacity: u32,
    pub enabled: u32,
    pub reconstruction: u32,
    pub intensity: f32,
    pub debug_irradiance: u32,
    /// 0: Bevy per-pixel direct sun; 1: cached diffuse direct sun.
    pub voxel_sky: u32,
}

impl Default for GpuTransportSettings {
    fn default() -> Self {
        Self {
            hash_mask: 0,
            probe_capacity: 0,
            enabled: 1,
            reconstruction: 1,
            intensity: 1.0,
            debug_irradiance: 0,
            voxel_sky: 1,
        }
    }
}

impl GpuTransportSettings {
    /// Reset presentation without replacing the live cache's addressing data.
    #[cfg(any(debug_assertions, test))]
    pub(crate) fn reset_display(&mut self) {
        *self = Self {
            hash_mask: self.hash_mask,
            probe_capacity: self.probe_capacity,
            ..default()
        };
    }
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct GpuHashEntry {
    pub voxel: IVec3,
    pub probe_index: u32,
}

impl GpuHashEntry {
    const EMPTY: Self = Self {
        voxel: IVec3::ZERO,
        probe_index: u32::MAX,
    };
    const TOMBSTONE: Self = Self {
        voxel: IVec3::ZERO,
        probe_index: u32::MAX - 1,
    };
}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct GpuProbe {
    pub positive_x: Vec4,
    pub negative_x: Vec4,
    pub positive_y: Vec4,
    pub negative_y: Vec4,
    pub positive_z: Vec4,
    pub negative_z: Vec4,
    pub emission: Vec4,
    pub direct_sky: [UVec4; 6],
}

#[derive(Resource, Clone)]
pub struct VoxelTransportAssets {
    pub texture: Handle<Image>,
    pub hash_entries: Handle<ShaderBuffer>,
    pub probes: Handle<ShaderBuffer>,
}

/// Stable GPU indices are rebuilt only when probe topology changes. Radiance
/// updates use a render-thread buffer subwrite, so an ordinary trace batch
/// never rehashes or reuploads the entire loaded world.
#[derive(Resource, Default)]
pub struct GpuTransportCache {
    slots: VoxelMap<IVec3, usize>,
    hash_slots: VoxelMap<IVec3, usize>,
    hash_entries: Vec<GpuHashEntry>,
    free_probe_slots: Vec<usize>,
    probe_capacity: usize,
    high_water: usize,
    pub ready: bool,
}

impl GpuTransportCache {
    /// Bounded diagnostics read of the publication mirror. This does not wait
    /// for GPU consumption; screenshots remain the end-to-end evidence.
    pub(super) fn validation_mirror(
        &self,
        queue: &GpuProbePatchQueue,
        sites: &[(IVec3, usize)],
    ) -> Vec<[f32; 4]> {
        let state = queue.state.lock().unwrap();
        sites
            .iter()
            .map(|(voxel, face)| {
                let Some(slot) = self.slots.get(voxel) else {
                    return [0.0; 4];
                };
                let offset = slot * GPU_PROBE_BYTES + face * 16;
                std::array::from_fn(|i| {
                    f32::from_le_bytes(
                        state.probe_mirror[offset + i * 4..offset + i * 4 + 4]
                            .try_into()
                            .unwrap(),
                    )
                })
            })
            .collect()
    }
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct GpuProbePatchQueue {
    probe_buffer: Handle<ShaderBuffer>,
    hash_buffer: Handle<ShaderBuffer>,
    /// Fully initialized and unmapped before extraction. Bevy registers these
    /// exact buffers instead of allocating another world-sized generation.
    prepared: Option<Arc<[Buffer; 2]>>,
    // A generation owns its buffers AND its pending writes. Extraction clones
    // this Arc; replacing a generation must never mutate an older frame's Arc.
    state: Arc<Mutex<ProbePatchState>>,
}

#[derive(Default)]
struct ProbePatchState {
    // Mirrors own the authoritative bytes. Dirty ledgers need only slot IDs;
    // retaining another 208-byte GpuProbe per slot duplicates a large payload
    // and forces expensive hash-table moves during bursty publication.
    pending_probes: HashSet<usize>,
    pending_hash: HashSet<usize>,
    probe_mirror: Vec<u8>,
    hash_mirror: Vec<u8>,
}

/// Render-local, sparse transaction snapshot. Copy both buffers under the
/// mirror lock, then release it before calling the graphics driver. Retaining
/// the scratch allocation avoids allocator churn; capacity follows dirty
/// ranges, not the size of the world-sized mirrors.
#[derive(Default)]
struct ProbeUpload {
    bytes: Vec<u8>,
    writes: Vec<(bool, usize, std::ops::Range<usize>)>,
}

impl ProbeUpload {
    fn stage(
        &mut self,
        state: &Mutex<ProbePatchState>,
        probe_bytes: u64,
        hash_bytes: u64,
    ) -> Result<(), &'static str> {
        self.bytes.clear();
        self.writes.clear();
        state
            .lock()
            .unwrap()
            .upload(probe_bytes, hash_bytes, |probe, offset, bytes| {
                let start = self.bytes.len();
                self.bytes.extend_from_slice(bytes);
                self.writes.push((probe, offset, start..self.bytes.len()));
            })
    }

    fn submit(&self, mut write: impl FnMut(bool, usize, &[u8])) {
        for (probe, offset, range) in &self.writes {
            write(*probe, *offset, &self.bytes[range.clone()]);
        }
    }

    /// Suballocate sparse writes from reusable mapped staging chunks. The
    /// complete hash/probe transaction is submitted before terrain rendering;
    /// recall is asynchronous and never waits for the device on the game thread.
    fn submit_staged(
        &self,
        belt: &mut wgpu::util::StagingBelt,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        probes: &wgpu::Buffer,
        hash: &wgpu::Buffer,
    ) {
        if self.writes.is_empty() {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Voxel GI sparse display upload"),
        });
        self.submit(|probe, offset, bytes| {
            belt.write_buffer(
                &mut encoder,
                if probe { probes } else { hash },
                offset as u64,
                wgpu::BufferSize::new(bytes.len() as u64).unwrap(),
            )
            .copy_from_slice(bytes);
        });
        belt.finish();
        queue.submit([encoder.finish()]);
        belt.recall();
    }
}

#[derive(Resource)]
struct ProbeStaging(wgpu::util::StagingBelt);

impl FromWorld for ProbeStaging {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>().wgpu_device().clone();
        Self(wgpu::util::StagingBelt::new(device, 2 * 1024 * 1024))
    }
}

impl ProbePatchState {
    /// Validate the entire transaction before submitting either buffer. On a
    /// missing/undersized destination the caller retries without losing writes.
    fn upload(
        &mut self,
        probe_bytes: u64,
        hash_bytes: u64,
        mut write: impl FnMut(bool, usize, &[u8]),
    ) -> Result<(), &'static str> {
        if probe_bytes < self.probe_mirror.len() as u64
            || hash_bytes < self.hash_mirror.len() as u64
        {
            return Err("GPU destination is smaller than its cache generation");
        }
        if self
            .pending_probes
            .iter()
            .any(|slot| *slot >= self.probe_mirror.len() / GPU_PROBE_BYTES)
            || self
                .pending_hash
                .iter()
                .any(|slot| *slot >= self.hash_mirror.len() / 16)
        {
            return Err("dirty slot is outside its cache generation");
        }
        for range in patch_ranges(
            self.pending_probes.iter().copied().collect(),
            GPU_PROBE_BYTES,
        ) {
            write(true, range.start, &self.probe_mirror[range]);
        }
        for range in patch_ranges(self.pending_hash.iter().copied().collect(), 16) {
            write(false, range.start, &self.hash_mirror[range]);
        }
        self.pending_probes.clear();
        self.pending_hash.clear();
        Ok(())
    }
}

#[derive(SystemSet, Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct DisplayBuffersPrepared;

pub fn install_render_assets(app: &mut App, texture: Handle<Image>) {
    app.add_plugins((
        MaterialPlugin::<VoxelTransportMaterial>::default(),
        ExtractResourcePlugin::<GpuProbePatchQueue>::default(),
    ));
    let hash_entries = app
        .world_mut()
        .resource_mut::<Assets<ShaderBuffer>>()
        .add(ShaderBuffer::from(vec![GpuHashEntry::EMPTY]));
    let probes = app
        .world_mut()
        .resource_mut::<Assets<ShaderBuffer>>()
        .add(ShaderBuffer::from(vec![GpuProbe::default()]));

    app.insert_resource(VoxelTransportAssets {
        texture,
        hash_entries: hash_entries.clone(),
        probes: probes.clone(),
    });
    app.insert_resource(GpuProbePatchQueue {
        probe_buffer: probes,
        hash_buffer: hash_entries,
        prepared: None,
        state: default(),
    });
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        // Recreate device-owned staging on renderer recovery as well as startup.
        render_app.init_gpu_resource::<ProbeStaging>();
        // Both new storage assets must exist before the terrain material binds
        // its generation. Otherwise preparation order can hide it for a frame.
        render_app.configure_sets(
            Render,
            DisplayBuffersPrepared
                .after(prepare_assets::<GpuShaderBuffer>)
                .before(prepare_erased_assets::<MeshMaterial3d<VoxelTransportMaterial>>),
        );
        render_app.add_systems(
            Render,
            (
                install_prepared_buffers
                    .in_set(RenderSystems::PrepareAssets)
                    .before(prepare_assets::<GpuShaderBuffer>),
                apply_probe_patches.in_set(RenderSystems::PrepareBindGroups),
            ),
        );
    }
}

pub fn material_template(assets: &VoxelTransportAssets) -> VoxelTransportMaterial {
    ExtendedMaterial {
        base: StandardMaterial {
            reflectance: 0.05,
            metallic: 0.05,
            perceptual_roughness: 0.95,
            ..default()
        },
        extension: VoxelTransportExtension {
            voxel_texture: assets.texture.clone(),
            hash_entries: assets.hash_entries.clone(),
            probes: assets.probes.clone(),
            settings: GpuTransportSettings::default(),
        },
    }
}

pub(crate) struct GpuCacheBuild {
    mapped: Option<[Buffer; 2]>,
    patch_state: ProbePatchState,
    hash_data: Vec<u8>,
    probe_data: Vec<u8>,
    slots: VoxelMap<IVec3, usize>,
    hash_slots: VoxelMap<IVec3, usize>,
    hash_entries: Vec<GpuHashEntry>,
    probe_capacity: usize,
    high_water: usize,
}

/// A replacement generation stays private until its topology and values have
/// caught up. Its byte arrays can then be published without a stale first frame.
pub(crate) struct PreparedGpuCache {
    cache: GpuTransportCache,
    patches: GpuProbePatchQueue,
    hash_data: Vec<u8>,
    probe_data: Vec<u8>,
    // Still mapped at creation: private CPU catch-up writes directly into the
    // initial GPU allocation. No GPU submission may reference these yet.
    mapped: Option<[Buffer; 2]>,
}

impl PreparedGpuCache {
    pub fn new(build: GpuCacheBuild) -> Self {
        Self {
            cache: GpuTransportCache {
                slots: build.slots,
                hash_slots: build.hash_slots,
                hash_entries: build.hash_entries,
                probe_capacity: build.probe_capacity,
                high_water: build.high_water,
                ready: true,
                ..default()
            },
            patches: GpuProbePatchQueue {
                probe_buffer: default(),
                hash_buffer: default(),
                prepared: None,
                state: Arc::new(Mutex::new(build.patch_state)),
            },
            hash_data: build.hash_data,
            probe_data: build.probe_data,
            mapped: build.mapped,
        }
    }

    /// Reconcile a bounded set of keys against current authoritative state.
    /// Event order does not matter: remove/re-add, mask and material changes
    /// all converge to the same latest probe. No renderer can see this cache.
    pub fn synchronize(&mut self, keys: &[IVec3], probes: &impl super::field::ProbeLookup) -> bool {
        let mut removed = Vec::new();
        let mut added = Vec::new();
        let mut changed = HashSet::new();
        for &voxel in keys {
            if probes.lookup(&voxel).is_none() {
                removed.push(voxel);
            } else if self.cache.slots.contains_key(&voxel) {
                changed.insert(voxel);
            } else {
                added.push(voxel);
            }
        }
        if !patch_gpu_topology(
            &mut self.cache,
            &self.patches,
            probes,
            &removed,
            &added,
            &changed,
        ) {
            return false;
        }
        // Fold private patches into the initial asset bytes in the same bounded
        // batch. Installing must not stage a world-sized catch-up upload later.
        let probe_bytes = (self.cache.probe_capacity * GPU_PROBE_BYTES) as u64;
        let hash_bytes = (self.cache.hash_entries.len() * 16) as u64;
        self.patches
            .state
            .lock()
            .unwrap()
            .upload(probe_bytes, hash_bytes, |probe, offset, bytes| {
                if let Some(buffers) = &self.mapped {
                    buffers[usize::from(!probe)]
                        .slice(offset as u64..(offset + bytes.len()) as u64)
                        .get_mapped_range_mut()
                        .copy_from_slice(bytes);
                    return;
                }
                let data = if probe {
                    &mut self.probe_data
                } else {
                    &mut self.hash_data
                };
                data[offset..offset + bytes.len()].copy_from_slice(bytes);
            })
            .expect("private cache bytes and mirrors have the same capacity");
        true
    }
}

/// Snapshot packing has a bounded game-thread cost. The key list is stable even
/// if live order indices are compacted by removals; the caller records deltas
/// from construction until installation to catch up both topology and values.
pub(crate) struct GpuSnapshot {
    order: Vec<IVec3>,
    cursor: usize,
    probes: Vec<(IVec3, GpuProbe)>,
}

impl GpuSnapshot {
    pub fn new(order: &[IVec3]) -> Self {
        Self {
            order: order.to_vec(),
            cursor: 0,
            probes: Vec::with_capacity(order.len()),
        }
    }
    pub fn advance(
        &mut self,
        probes: &impl super::field::ProbeLookup,
        mut captured: impl FnMut(IVec3),
    ) -> bool {
        let started = std::time::Instant::now();
        let end = (self.cursor + 8192).min(self.order.len());
        while self.cursor < end {
            let voxel = self.order[self.cursor];
            if let Some(probe) = probes.lookup(&voxel) {
                self.probes.push((voxel, gpu_probe(probe)));
            }
            // The snapshot now includes this key's latest membership/value.
            // Only changes after this point require catch-up.
            captured(voxel);
            self.cursor += 1;
            if self.cursor.is_multiple_of(128) && started.elapsed().as_micros() >= 1500 {
                break;
            }
        }
        self.cursor == self.order.len()
    }
    pub fn finish(self) -> Vec<(IVec3, GpuProbe)> {
        self.probes
    }
}

#[cfg(test)]
pub(crate) fn snapshot_gpu_probes(
    probes: &impl super::field::ProbeLookup,
    order: &[IVec3],
) -> Vec<(IVec3, GpuProbe)> {
    order
        .iter()
        .filter_map(|voxel| probes.lookup(voxel).map(|probe| (*voxel, gpu_probe(probe))))
        .collect()
}

/// CPU encoding is deliberately run in an async task. `ShaderBuffer::set_data`
/// on 250k probes was responsible for multi-frame stalls in ordinary builds.
#[cfg(test)]
pub(crate) fn build_gpu_cache(snapshot: Vec<(IVec3, GpuProbe)>) -> GpuCacheBuild {
    let resident_count = snapshot.len();
    build_gpu_cache_for_residency(snapshot, resident_count)
}

pub(crate) fn build_gpu_cache_for_residency(
    snapshot: Vec<(IVec3, GpuProbe)>,
    resident_count: usize,
) -> GpuCacheBuild {
    // Geometric headroom avoids rebuilding the entire cache every 16k streamed
    // receivers. Keep hash occupancy <= 50% even after using that headroom.
    let resident_count = resident_count.max(snapshot.len());
    let probe_capacity = resident_count
        .saturating_add((resident_count / 2).max(16_384))
        .max(1);
    let mut hash_capacity = probe_capacity.saturating_mul(2).next_power_of_two().max(1);
    loop {
        let mut entries = vec![GpuHashEntry::EMPTY; hash_capacity];
        // Reserve the same growth headroom as the GPU slots, on this worker.
        // Catch-up must not rehash world-sized CPU indices on the game thread.
        let mut slots = VoxelMap::with_capacity(probe_capacity);
        let mut hash_slots = VoxelMap::with_capacity(probe_capacity);
        let mut longest_probe = 0;
        for (probe_index, (voxel, _)) in snapshot.iter().enumerate() {
            let mut slot = hash_voxel(*voxel) as usize & (hash_capacity - 1);
            let mut distance = 0;
            while entries[slot].probe_index != u32::MAX {
                slot = (slot + 1) & (hash_capacity - 1);
                distance += 1;
            }
            entries[slot] = GpuHashEntry {
                voxel: *voxel,
                probe_index: probe_index as u32,
            };
            slots.insert(*voxel, probe_index);
            hash_slots.insert(*voxel, slot);
            longest_probe = longest_probe.max(distance);
        }
        if longest_probe >= MAX_SHADER_HASH_PROBES {
            hash_capacity *= 2;
            continue;
        }
        let mut gpu_probes = vec![GpuProbe::default(); probe_capacity];
        for (index, (_, probe)) in snapshot.iter().enumerate() {
            gpu_probes[index] = *probe;
        }
        let hash_data = ShaderBuffer::from(entries.clone()).data.unwrap();
        let probe_data = ShaderBuffer::from(gpu_probes).data.unwrap();
        return GpuCacheBuild {
            mapped: None,
            patch_state: ProbePatchState {
                probe_mirror: probe_data.clone(),
                hash_mirror: hash_data.clone(),
                ..default()
            },
            hash_data,
            probe_data,
            slots,
            hash_slots,
            hash_entries: entries,
            probe_capacity,
            high_water: snapshot.len(),
        };
    }
}

impl GpuCacheBuild {
    /// Allocate and initialize on the worker, but leave buffers mapped until
    /// bounded catch-up completes. The owning private generation is the only
    /// writer; all mapped views are dropped before transfer to the renderer.
    pub fn prepare_buffers(mut self, device: &RenderDevice) -> Self {
        let make = |label, data: &[u8]| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: data.len() as u64,
                mapped_at_creation: true,
                ..ShaderBuffer::default().buffer_description
            });
            buffer
                .slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(data);
            buffer
        };
        self.mapped = Some([
            make("Voxel GI display probes", &self.probe_data),
            make("Voxel GI display hash", &self.hash_data),
        ]);
        // Mirrors retain authoritative bytes; no second initial asset payload.
        self.probe_data = Vec::new();
        self.hash_data = Vec::new();
        self
    }
}

fn install_prepared_buffers(
    patches: Res<GpuProbePatchQueue>,
    mut assets: ResMut<RenderAssets<GpuShaderBuffer>>,
) {
    let Some(buffers) = &patches.prepared else {
        return;
    };
    for (handle, buffer, label) in [
        (
            &patches.probe_buffer,
            &buffers[0],
            "Voxel GI display probes",
        ),
        (&patches.hash_buffer, &buffers[1], "Voxel GI display hash"),
    ] {
        if assets.get(handle).is_some() {
            continue;
        }
        assets.insert(
            handle.id(),
            GpuShaderBuffer {
                buffer: buffer.clone(),
                buffer_descriptor: wgpu::BufferDescriptor {
                    label: Some(label),
                    size: buffer.size(),
                    ..ShaderBuffer::default().buffer_description
                },
                had_data: false,
            },
        );
    }
    // Descriptor-only preparation must reuse the exact existing RenderAsset.
    // The native generation test guards this Bevy integration contract.
}

#[cfg(test)]
pub(crate) fn install_gpu_cache(
    build: GpuCacheBuild,
    cache: &mut GpuTransportCache,
    patches: &mut GpuProbePatchQueue,
    assets: &mut VoxelTransportAssets,
    buffers: &mut Assets<ShaderBuffer>,
    materials: &mut Assets<VoxelTransportMaterial>,
    material_handle: &Handle<VoxelTransportMaterial>,
) {
    install_prepared_gpu_cache(
        PreparedGpuCache::new(build),
        cache,
        patches,
        assets,
        buffers,
        materials,
        material_handle,
    );
}

pub(crate) fn install_prepared_gpu_cache(
    mut build: PreparedGpuCache,
    cache: &mut GpuTransportCache,
    patches: &mut GpuProbePatchQueue,
    assets: &mut VoxelTransportAssets,
    buffers: &mut Assets<ShaderBuffer>,
    materials: &mut Assets<VoxelTransportMaterial>,
    material_handle: &Handle<VoxelTransportMaterial>,
) {
    if let Some(mapped) = build.mapped.take() {
        for buffer in &mapped {
            buffer.unmap();
        }
        build.patches.prepared = Some(Arc::new(mapped));
    }
    // Fresh asset identities also handle equal-size rebuilds: a size check alone
    // cannot tell that hash indices have been reassigned. Old handles stay alive
    // for as long as the extracted frame/material needs them.
    assets.hash_entries = buffers.add(ShaderBuffer {
        data: build.patches.prepared.is_none().then_some(build.hash_data),
        buffer_description: bevy::render::render_resource::BufferDescriptor {
            label: Some("Voxel GI display hash"),
            size: (build.cache.hash_entries.len() * 16) as u64,
            ..ShaderBuffer::default().buffer_description
        },
        ..default()
    });
    assets.probes = buffers.add(ShaderBuffer {
        data: build.patches.prepared.is_none().then_some(build.probe_data),
        buffer_description: bevy::render::render_resource::BufferDescriptor {
            label: Some("Voxel GI display probes"),
            size: (build.cache.probe_capacity * GPU_PROBE_BYTES) as u64,
            ..ShaderBuffer::default().buffer_description
        },
        ..default()
    });
    build.patches.probe_buffer = assets.probes.clone();
    build.patches.hash_buffer = assets.hash_entries.clone();
    *patches = build.patches;
    *cache = build.cache;
    if let Some(mut material) = materials.get_mut(material_handle) {
        material.extension.hash_entries = assets.hash_entries.clone();
        material.extension.probes = assets.probes.clone();
        material.extension.settings.hash_mask = cache.hash_entries.len() as u32 - 1;
        material.extension.settings.probe_capacity = cache.probe_capacity as u32;
    }
}

/// Apply sparse probe additions/removals without replacing either GPU buffer.
/// Tombstones preserve collision chains for the shader's linear probing.
pub(crate) fn patch_gpu_topology(
    cache: &mut GpuTransportCache,
    patches: &GpuProbePatchQueue,
    probes: &impl super::field::ProbeLookup,
    removed: &[IVec3],
    added: &[IVec3],
    changed: &HashSet<IVec3>,
) -> bool {
    // Preflight only the delta. A failed allocation/hash insertion must leave
    // the published cache completely intact until its replacement is ready.
    // Cloning the full world-sized mirrors would defeat sparse edit uploads.
    let mut removed_once = HashSet::new();
    let removed: Vec<_> = removed
        .iter()
        .copied()
        .filter(|voxel| removed_once.insert(*voxel))
        .collect();
    let released: Vec<_> = removed
        .iter()
        .filter_map(|voxel| cache.slots.get(voxel).copied())
        .collect();
    let mut free = released
        .iter()
        .rev()
        .chain(cache.free_probe_slots.iter().rev());
    let mut high_water = cache.high_water;
    let mut hash_delta: VoxelMap<_, _> = removed
        .iter()
        .filter_map(|voxel| cache.hash_slots.get(voxel).copied())
        .map(|slot| (slot, GpuHashEntry::TOMBSTONE))
        .collect();
    let mut planned = Vec::new();
    let mut added_once = HashSet::new();
    for &voxel in added {
        if !added_once.insert(voxel)
            || (cache.slots.contains_key(&voxel) && !removed_once.contains(&voxel))
        {
            continue;
        }
        let Some(probe) = probes.lookup(&voxel) else {
            continue;
        };
        let probe_slot = if let Some(&slot) = free.next() {
            slot
        } else {
            if high_water >= cache.probe_capacity {
                return false;
            }
            let slot = high_water;
            high_water += 1;
            slot
        };
        let mut hash_slot = hash_voxel(voxel) as usize & (cache.hash_entries.len() - 1);
        let mut found = false;
        for _ in 0..MAX_SHADER_HASH_PROBES {
            let entry = hash_delta
                .get(&hash_slot)
                .unwrap_or(&cache.hash_entries[hash_slot]);
            if entry.probe_index >= u32::MAX - 1 {
                found = true;
                break;
            }
            hash_slot = (hash_slot + 1) & (cache.hash_entries.len() - 1);
        }
        if !found {
            return false;
        }
        let entry = GpuHashEntry {
            voxel,
            probe_index: probe_slot as u32,
        };
        hash_delta.insert(hash_slot, entry);
        planned.push((entry, hash_slot, gpu_probe(probe)));
    }

    let mut state = patches.state.lock().unwrap();
    for voxel in &removed {
        if let Some(probe_slot) = cache.slots.remove(voxel) {
            cache.free_probe_slots.push(probe_slot);
        }
        if let Some(hash_slot) = cache.hash_slots.remove(voxel) {
            cache.hash_entries[hash_slot] = GpuHashEntry::TOMBSTONE;
            state.pending_hash.insert(hash_slot);
            write_hash_entry(&mut state.hash_mirror, hash_slot, GpuHashEntry::TOMBSTONE);
        }
    }
    for (entry, hash_slot, gpu_probe) in planned {
        let probe_slot = entry.probe_index as usize;
        if let Some(slot) = cache.free_probe_slots.pop() {
            debug_assert_eq!(slot, probe_slot);
        } else {
            debug_assert_eq!(cache.high_water, probe_slot);
            cache.high_water += 1;
        }
        cache.hash_entries[hash_slot] = entry;
        cache.slots.insert(entry.voxel, probe_slot);
        cache.hash_slots.insert(entry.voxel, hash_slot);
        state.pending_hash.insert(hash_slot);
        write_hash_entry(&mut state.hash_mirror, hash_slot, entry);
        state.pending_probes.insert(probe_slot);
        write_probe(&mut state.probe_mirror, probe_slot, gpu_probe);
    }
    // Retire old donors and publish newly exposed faces on EXISTING probes in
    // one transaction. The ordinary value-upload timer must not split these.
    patch_values_locked(&mut state, cache, probes, changed);
    // Addressing is generation-stable: an extracted material must already
    // accept a freshly allocated slot before its new hash entry becomes live.
    true
}

pub(crate) fn patch_probe_values(
    cache: &GpuTransportCache,
    patches: &GpuProbePatchQueue,
    probes: &impl super::field::ProbeLookup,
    dirty_voxels: &HashSet<IVec3>,
) {
    let mut state = patches.state.lock().unwrap();
    patch_values_locked(&mut state, cache, probes, dirty_voxels);
}

fn patch_values_locked(
    state: &mut ProbePatchState,
    cache: &GpuTransportCache,
    probes: &impl super::field::ProbeLookup,
    dirty_voxels: &HashSet<IVec3>,
) {
    for voxel in dirty_voxels {
        let (Some(slot), Some(probe)) = (cache.slots.get(voxel), probes.lookup(voxel)) else {
            continue;
        };
        let gpu_probe = gpu_probe(probe);
        state.pending_probes.insert(*slot);
        write_probe(&mut state.probe_mirror, *slot, gpu_probe);
    }
}

fn apply_probe_patches(
    patches: Res<GpuProbePatchQueue>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    queue: Res<RenderQueue>,
    device: Res<RenderDevice>,
    mut staging: ResMut<ProbeStaging>,
    mut failed: Local<bool>,
    mut upload: Local<ProbeUpload>,
) {
    let (Some(probe_buffer), Some(hash_buffer)) = (
        buffers.get(&patches.probe_buffer),
        buffers.get(&patches.hash_buffer),
    ) else {
        return;
    };
    // Capture hash and probe bytes atomically, including coalesced gaps. Later
    // topology changes cannot alter this transaction, and remain queued for
    // the next frame. Driver staging/submission never holds the shared lock.
    let result = upload.stage(
        &patches.state,
        probe_buffer.buffer.size(),
        hash_buffer.buffer.size(),
    );
    if let Err(reason) = result {
        if !*failed {
            error!(
                reason,
                "voxel GI display upload deferred; pending writes retained"
            );
        }
        *failed = true;
    } else {
        upload.submit_staged(
            &mut staging.0,
            device.wgpu_device(),
            &queue,
            &probe_buffer.buffer,
            &hash_buffer.buffer,
        );
        *failed = false;
    }
}

fn patch_ranges(mut slots: Vec<usize>, stride: usize) -> Vec<std::ops::Range<usize>> {
    slots.sort_unstable();
    let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
    for slot in slots {
        let start = slot * stride;
        if let Some(last) = ranges.last_mut()
            && start <= last.end + 1024
        {
            last.end = start + stride;
        } else {
            ranges.push(start..start + stride);
        }
    }
    ranges
}

fn encode_gpu_probe(probe: GpuProbe, bytes: &mut [u8]) {
    assert_eq!(bytes.len(), GPU_PROBE_BYTES);
    // Pack on the stack and copy once. Fifty-two tiny slice copies per dirty
    // receiver are costly in ordinary development builds. Keep explicit little
    // endian encoding, without assuming Rust/glam layout matches WGSL.
    let words = [
        probe.positive_x.to_array().map(f32::to_bits),
        probe.negative_x.to_array().map(f32::to_bits),
        probe.positive_y.to_array().map(f32::to_bits),
        probe.negative_y.to_array().map(f32::to_bits),
        probe.positive_z.to_array().map(f32::to_bits),
        probe.negative_z.to_array().map(f32::to_bits),
        probe.emission.to_array().map(f32::to_bits),
        probe.direct_sky[0].to_array(),
        probe.direct_sky[1].to_array(),
        probe.direct_sky[2].to_array(),
        probe.direct_sky[3].to_array(),
        probe.direct_sky[4].to_array(),
        probe.direct_sky[5].to_array(),
    ];
    let encoded = words.map(|group| group.map(u32::to_le_bytes));
    bytes.copy_from_slice(encoded.as_flattened().as_flattened());
}

fn write_probe(mirror: &mut [u8], slot: usize, probe: GpuProbe) {
    encode_gpu_probe(
        probe,
        &mut mirror[slot * GPU_PROBE_BYTES..(slot + 1) * GPU_PROBE_BYTES],
    );
}

fn write_hash_entry(mirror: &mut [u8], slot: usize, entry: GpuHashEntry) {
    let offset = slot * 16;
    for (axis, coordinate) in entry.voxel.to_array().into_iter().enumerate() {
        mirror[offset + axis * 4..offset + axis * 4 + 4].copy_from_slice(&coordinate.to_le_bytes());
    }
    mirror[offset + 12..offset + 16].copy_from_slice(&entry.probe_index.to_le_bytes());
}

pub fn set_enabled(
    enabled: bool,
    materials: &mut Assets<VoxelTransportMaterial>,
    material_handle: &Handle<VoxelTransportMaterial>,
) {
    if let Some(mut material) = materials.get_mut(material_handle) {
        material.extension.settings.enabled = enabled as u32;
    }
}

fn gpu_probe(probe: &SurfaceProbe) -> GpuProbe {
    let face = |i: usize| {
        if probe.face_mask & (1 << i) != 0 {
            (probe.display[i].indirect + probe.custom_direct[i])
                .extend(probe.display[i].confidence.max(0.03125))
        } else {
            Vec4::ZERO
        }
    };
    GpuProbe {
        positive_x: face(0),
        negative_x: face(1),
        positive_y: face(2),
        negative_y: face(3),
        positive_z: face(4),
        negative_z: face(5),
        emission: probe.emission.visible.extend(0.0),
        direct_sky: std::array::from_fn(|face| {
            if probe.face_mask & (1 << face) != 0 {
                UVec4::from_array(probe.display[face].sky)
            } else {
                UVec4::ZERO
            }
        }),
    }
}

fn hash_voxel(voxel: IVec3) -> u32 {
    let mut hash = pcg(voxel.x as u32);
    hash = pcg(hash ^ voxel.y as u32);
    pcg(hash ^ voxel.z as u32)
}

fn pcg(value: u32) -> u32 {
    let state = value.wrapping_mul(747_796_405).wrapping_add(2_891_336_453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277_803_737);
    (word >> 22) ^ word
}

/// The host can check this without marking an asset modified. Asset loading,
/// layer-count policy and hot reload belong to the host, not GI cache lifetime.
pub(crate) fn texture_array_ready(image: &Image, layers: u32) -> bool {
    image.texture_descriptor.size.depth_or_array_layers == layers
        && image
            .texture_view_descriptor
            .as_ref()
            .is_some_and(|view| view.dimension == Some(TextureViewDimension::D2Array))
        && matches!(&image.sampler, ImageSampler::Descriptor(sampler)
            if sampler.address_mode_u == ImageAddressMode::Repeat
                && sampler.address_mode_v == ImageAddressMode::Repeat
                && sampler.address_mode_w == ImageAddressMode::Repeat)
}

pub(crate) fn prepare_texture_array(
    image: &mut Image,
    layers: u32,
) -> Result<(), TextureReinterpretationError> {
    if image.texture_descriptor.dimension != wgpu::TextureDimension::D2 {
        return Err(TextureReinterpretationError::WrongDimension);
    }
    if layers == 0 {
        return Err(TextureReinterpretationError::NotEnoughLayers);
    }
    if image.texture_descriptor.size.depth_or_array_layers != layers {
        let old = image.texture_descriptor.size;
        let height = old.height * old.depth_or_array_layers;
        if !height.is_multiple_of(layers) {
            return Err(TextureReinterpretationError::HeightNotDivisibleByLayers {
                height,
                layers,
            });
        }
        image.reinterpret_size(wgpu::Extent3d {
            width: old.width,
            height: height / layers,
            depth_or_array_layers: layers,
        })?;
    }
    let descriptor = image.sampler.get_or_init_descriptor();
    descriptor.address_mode_u = ImageAddressMode::Repeat;
    descriptor.address_mode_v = ImageAddressMode::Repeat;
    descriptor.address_mode_w = ImageAddressMode::Repeat;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel_gi::spectrum::EnergySpectrum;

    #[test]
    fn terrain_shader_composes_with_bevys_real_pbr_modules_without_a_window() {
        use bevy::render::{RenderPlugin, settings::WgpuSettings};
        use bevy::shader::{Shader, ShaderCache, ShaderCacheSource, ShaderDefVal};
        // Load the exact libraries registered by this Bevy build. No copied
        // upstream shader fixtures, window, GPU, or unlocked desktop required.
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<bevy::log::LogPlugin>()
                .disable::<bevy::audio::AudioPlugin>()
                .disable::<bevy::gilrs::GilrsPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: WgpuSettings {
                        backends: None,
                        ..default()
                    }
                    .into(),
                    ..default()
                }),
        );
        app.finish();
        app.cleanup();
        let mut cache = ShaderCache::new(
            (),
            wgpu::Features::empty(),
            wgpu::DownlevelFlags::all(),
            |_, source, _| {
                let ShaderCacheSource::Naga(module) = source else {
                    panic!("expected Naga module")
                };
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap();
                Ok(module)
            },
        );
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wgsl(
                include_str!("../../assets/shaders/voxel_transport.wgsl"),
                VOXEL_TRANSPORT_SHADER,
            ));
        let mut defs: Vec<ShaderDefVal> = [
            "VERTEX_POSITIONS",
            "VERTEX_NORMALS",
            "VERTEX_UVS",
            "VERTEX_UVS_A",
            "VERTEX_COLORS",
            "VERTEX_OUTPUT_INSTANCE_INDEX",
        ]
        .into_iter()
        .map(Into::into)
        .collect();
        defs.extend([
            ShaderDefVal::UInt("MATERIAL_BIND_GROUP".into(), 3),
            ShaderDefVal::UInt("MAX_DIRECTIONAL_LIGHTS".into(), 10),
            ShaderDefVal::UInt("MAX_CASCADES_PER_LIGHT".into(), 4),
            ShaderDefVal::UInt("AVAILABLE_STORAGE_BUFFER_BINDINGS".into(), 8),
        ]);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            app.update();
            for (id, shader) in app.world().resource::<Assets<Shader>>().iter() {
                cache.set_shader(id, shader.clone());
            }
            match cache.get(0, handle.id(), &defs) {
                Ok(_) => break,
                Err(bevy::shader::ShaderCacheError::ShaderImportNotYetAvailable)
                    if std::time::Instant::now() < deadline =>
                {
                    std::thread::yield_now()
                }
                Err(bevy::shader::ShaderCacheError::ProcessShaderError(error)) => {
                    panic!("{}", error.emit_to_string(&cache.composer))
                }
                Err(error) => panic!("terrain shader failed to compose: {error:?}"),
            }
        }
        // Compiling the ordinary PBR material afterwards must not retain our
        // override or world-space bindings in the shared shader composer.
        let standard_id = app
            .world()
            .resource::<Assets<Shader>>()
            .iter()
            .find(|(_, shader)| {
                shader.path.ends_with("bevy_pbr/src/render/pbr.wgsl")
                    || shader.path.ends_with("bevy_pbr/render/pbr.wgsl")
            })
            .map(|(id, _)| id)
            .expect("Bevy's PBR shader was loaded");
        let standard = cache.get(1, standard_id, &defs).unwrap();
        assert!(!standard.global_variables.iter().any(|(_, global)| {
            global
                .name
                .as_deref()
                .is_some_and(|name| name.starts_with("transport_"))
        }));
    }

    #[test]
    fn sparse_patch_bytes_match_bevy_shader_buffer_layout() {
        let probe = GpuProbe {
            positive_x: Vec4::new(1.0, 2.0, 3.0, 4.0),
            negative_x: Vec4::new(5.0, 6.0, 7.0, 8.0),
            positive_y: Vec4::new(9.0, 10.0, 11.0, 12.0),
            negative_y: Vec4::new(13.0, 14.0, 15.0, 16.0),
            positive_z: Vec4::new(17.0, 18.0, 19.0, 20.0),
            negative_z: Vec4::new(21.0, 22.0, 23.0, 24.0),
            emission: Vec4::new(25.0, 26.0, 27.0, 28.0),
            direct_sky: std::array::from_fn(|face| UVec4::splat(29 + face as u32)),
        };
        let mut encoded = vec![0; GPU_PROBE_BYTES];
        encode_gpu_probe(probe, &mut encoded);
        assert_eq!(encoded.len(), GPU_PROBE_BYTES);
        assert_eq!(encoded, ShaderBuffer::from(vec![probe]).data.unwrap());

        let entry = GpuHashEntry {
            voxel: IVec3::new(-7, 12, 35),
            probe_index: 42,
        };
        let mut encoded = vec![0; 16];
        write_hash_entry(&mut encoded, 0, entry);
        assert_eq!(encoded, ShaderBuffer::from(vec![entry]).data.unwrap());
    }

    #[test]
    fn direct_sky_is_separate_from_gi_and_preserves_exposed_face_mask() {
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4);
        probe.radiance[2] = EnergySpectrum::new(Vec3::X, 3.0, 4.0);
        probe.custom_direct[2] = Vec3::Y;
        probe.display[2] = super::super::history::FaceDisplay {
            indirect: Vec3::X,
            sky: [super::super::sky::pack(Vec3::splat(10.0)); 4],
            confidence: 0.5,
        };
        let encoded = gpu_probe(&probe);
        assert_eq!(encoded.positive_y, Vec4::new(1.0, 1.0, 0.0, 0.5));
        assert_eq!(
            encoded.direct_sky[2],
            UVec4::from_array(probe.display[2].sky)
        );
        assert_eq!(encoded.direct_sky[0], UVec4::ZERO);
        let material = material_template(&VoxelTransportAssets {
            texture: default(),
            hash_entries: default(),
            probes: default(),
        });
        assert_eq!(material.extension.settings.voxel_sky, 1);
        assert_eq!(material.extension.settings.reconstruction, 1);
        assert_eq!(material.extension.settings.enabled, 1);
        assert_eq!(material.extension.settings.debug_irradiance, 0);
    }

    #[test]
    fn resetting_presentation_preserves_live_cache_addresses() {
        let mut settings = GpuTransportSettings {
            hash_mask: 1023,
            probe_capacity: 700,
            enabled: 0,
            reconstruction: 0,
            debug_irradiance: 1,
            voxel_sky: 0,
            intensity: 0.5,
        };
        settings.reset_display();
        assert_eq!((settings.hash_mask, settings.probe_capacity), (1023, 700));
        assert_eq!(
            (
                settings.enabled,
                settings.reconstruction,
                settings.voxel_sky
            ),
            (1, 1, 1)
        );
        assert_eq!(settings.debug_irradiance, 0);
        assert_eq!(settings.intensity, 1.0);
    }

    #[test]
    fn incremental_snapshot_uses_stable_keys_across_live_order_compaction() {
        let mut probes: HashMap<_, _> = (0..9000)
            .map(|x| {
                let voxel = IVec3::new(x, 0, 0);
                (voxel, SurfaceProbe::new(voxel, 4))
            })
            .collect();
        let mut order: Vec<_> = (0..9000).map(|x| IVec3::new(x, 0, 0)).collect();
        let mut snapshot = GpuSnapshot::new(&order);
        assert!(
            !snapshot.advance(&probes, |_| {}),
            "snapshot work is bounded"
        );
        let removed = order.swap_remove(8999);
        probes.remove(&removed);
        order.swap(0, 8000);
        while !snapshot.advance(&probes, |_| {}) {}
        let result = snapshot.finish();
        assert_eq!(result.len(), 8999);
        assert!(!result.iter().any(|(voxel, _)| *voxel == removed));
        assert_eq!(
            result.iter().map(|(v, _)| *v).collect::<HashSet<_>>().len(),
            8999
        );
    }

    #[test]
    fn topology_patch_reuses_probe_slot_and_keeps_hash_chain_searchable() {
        let voxel_a = IVec3::new(0, 0, 0);
        let bucket = hash_voxel(voxel_a) & 3;
        let colliding: Vec<_> = (1..100)
            .map(|x| IVec3::new(x, 0, 0))
            .filter(|voxel| hash_voxel(*voxel) & 3 == bucket)
            .take(2)
            .collect();
        assert_eq!(colliding.len(), 2);
        let voxel_b = colliding[0];
        let voxel_c = colliding[1];
        let mut probes = HashMap::new();
        for voxel in [voxel_a, voxel_b, voxel_c] {
            probes.insert(voxel, SurfaceProbe::new(voxel, 1));
        }
        let build = build_gpu_cache(snapshot_gpu_probes(&probes, &[voxel_a, voxel_b]));
        let patches = GpuProbePatchQueue {
            probe_buffer: Handle::default(),
            hash_buffer: Handle::default(),
            prepared: None,
            state: Arc::new(Mutex::new(ProbePatchState {
                probe_mirror: build.probe_data.clone(),
                hash_mirror: build.hash_data.clone(),
                ..default()
            })),
        };
        let old_slot = build.slots[&voxel_a];
        let mut cache = GpuTransportCache {
            slots: build.slots,
            hash_slots: build.hash_slots,
            hash_entries: build.hash_entries,
            free_probe_slots: Vec::new(),
            probe_capacity: build.probe_capacity,
            high_water: build.high_water,
            ready: true,
        };
        assert!(patch_gpu_topology(
            &mut cache,
            &patches,
            &probes,
            &[voxel_a],
            &[voxel_c],
            &HashSet::new(),
        ));
        assert_eq!(cache.slots[&voxel_c], old_slot);
        assert!(!cache.slots.contains_key(&voxel_a));
        for voxel in [voxel_b, voxel_c] {
            let mut slot = hash_voxel(voxel) as usize & (cache.hash_entries.len() - 1);
            let mut found = false;
            for _ in 0..MAX_SHADER_HASH_PROBES {
                let entry = cache.hash_entries[slot];
                if entry.probe_index == u32::MAX {
                    break;
                }
                if entry.probe_index != u32::MAX - 1 && entry.voxel == voxel {
                    found = true;
                    break;
                }
                slot = (slot + 1) & (cache.hash_entries.len() - 1);
            }
            assert!(found, "hash chain lost {voxel:?}");
        }
    }

    #[test]
    fn failed_topology_patch_leaves_the_published_generation_untouched() {
        let a = IVec3::ZERO;
        let b = IVec3::X;
        let c = IVec3::Y;
        let probes: HashMap<_, _> = [a, b, c]
            .into_iter()
            .map(|voxel| (voxel, SurfaceProbe::new(voxel, 4)))
            .collect();
        for hash_failure in [false, true] {
            let build = build_gpu_cache(snapshot_gpu_probes(&probes, &[a]));
            let patches = GpuProbePatchQueue {
                probe_buffer: default(),
                hash_buffer: default(),
                prepared: None,
                state: Arc::new(Mutex::new(ProbePatchState {
                    probe_mirror: build.probe_data.clone(),
                    hash_mirror: build.hash_data.clone(),
                    ..default()
                })),
            };
            let mut cache = GpuTransportCache {
                slots: build.slots,
                hash_slots: build.hash_slots,
                hash_entries: build.hash_entries,
                high_water: build.high_water,
                probe_capacity: if hash_failure { 3 } else { 1 },
                ready: true,
                ..default()
            };
            // Simulate a directory with no insertion slot except the removal.
            if hash_failure {
                for entry in &mut cache.hash_entries {
                    entry.probe_index = 0;
                }
            }
            let before_slots = cache.slots.clone();
            let before_hash = ShaderBuffer::from(cache.hash_entries.clone()).data.unwrap();
            assert!(!patch_gpu_topology(
                &mut cache,
                &patches,
                &probes,
                &[a],
                &[b, c],
                &HashSet::new()
            ));
            assert_eq!(cache.slots, before_slots);
            assert_eq!(
                ShaderBuffer::from(cache.hash_entries.clone()).data.unwrap(),
                before_hash
            );
            assert_eq!(cache.high_water, 1);
            assert!(cache.free_probe_slots.is_empty());
            let state = patches.state.lock().unwrap();
            assert!(state.pending_hash.is_empty());
            assert!(state.pending_probes.is_empty());
            assert_eq!(state.probe_mirror, build.probe_data);
            assert_eq!(state.hash_mirror, build.hash_data);
        }
    }

    #[test]
    fn gpu_hash_contains_every_probe_without_exceeding_shader_search_limit() {
        let mut probes = HashMap::new();
        for z in -8..8 {
            for x in -32..32 {
                let voxel = IVec3::new(x, 7, z);
                probes.insert(
                    voxel,
                    SurfaceProbe {
                        voxel,
                        face_mask: 4,
                        radiance: [EnergySpectrum::ZERO; 6],
                        direct: [EnergySpectrum::ZERO; 6],
                        samples: [0; 6],
                        ..SurfaceProbe::new(voxel, 4)
                    },
                );
            }
        }

        let mut order: Vec<_> = probes.keys().copied().collect();
        order.sort_unstable_by_key(|voxel| (voxel.z, voxel.y, voxel.x));
        let build = build_gpu_cache(snapshot_gpu_probes(&probes, &order));
        let entries = build.hash_entries;
        assert_eq!(build.high_water, probes.len());
        for voxel in probes.keys() {
            let mut slot = hash_voxel(*voxel) as usize & (entries.len() - 1);
            let mut found = false;
            for _ in 0..MAX_SHADER_HASH_PROBES {
                let entry = entries[slot];
                if entry.probe_index == u32::MAX {
                    break;
                }
                if entry.voxel == *voxel {
                    found = true;
                    break;
                }
                slot = (slot + 1) & (entries.len() - 1);
            }
            assert!(found, "missing GPU hash entry for {voxel:?}");
        }
    }

    #[test]
    fn stale_cache_snapshot_accepts_streaming_deltas_and_newer_lighting() {
        let a = IVec3::ZERO;
        let b = IVec3::X;
        let c = IVec3::Y;
        let transient = IVec3::Z;
        let mut probes: HashMap<_, _> = [a, b]
            .into_iter()
            .map(|voxel| (voxel, SurfaceProbe::new(voxel, 4)))
            .collect();
        // Simulate the immutable snapshot spending time on a worker while
        // main-world topology and radiance continue changing.
        let snapshot = snapshot_gpu_probes(&probes, &[a, b]);
        let old_display = ShaderBuffer::from(vec![gpu_probe(&probes[&b])])
            .data
            .unwrap();
        probes.remove(&a);
        probes.insert(c, SurfaceProbe::new(c, 4));
        let updated = probes.get_mut(&b).unwrap();
        updated.radiance[2] = EnergySpectrum::ONE * 3.0;
        updated.display[2].indirect = Vec3::splat(2.0);
        updated.display[2].confidence = 0.75;
        assert_ne!(
            old_display,
            ShaderBuffer::from(vec![gpu_probe(updated)]).data.unwrap(),
            "the stale snapshot must differ from the latest presentation history"
        );
        let build = build_gpu_cache(snapshot);
        let mut buffers = Assets::<ShaderBuffer>::default();
        let mut assets = VoxelTransportAssets {
            texture: Handle::default(),
            hash_entries: buffers.add(ShaderBuffer::from(vec![GpuHashEntry::EMPTY])),
            probes: buffers.add(ShaderBuffer::from(vec![GpuProbe::default()])),
        };
        let mut patches = GpuProbePatchQueue {
            probe_buffer: assets.probes.clone(),
            hash_buffer: assets.hash_entries.clone(),
            prepared: None,
            state: default(),
        };
        let mut cache = GpuTransportCache::default();
        let mut materials = Assets::default();
        install_gpu_cache(
            build,
            &mut cache,
            &mut patches,
            &mut assets,
            &mut buffers,
            &mut materials,
            &Handle::default(),
        );
        assert!(patch_gpu_topology(
            &mut cache,
            &patches,
            &probes,
            &[a, transient],
            &[c, transient],
            &HashSet::from([b, c]),
        ));
        assert_eq!(cache.slots.len(), probes.len());
        assert!(!cache.slots.contains_key(&a));
        assert!(!cache.slots.contains_key(&transient));
        let state = patches.state.lock().unwrap();
        for (voxel, probe) in &probes {
            let offset = cache.slots[voxel] * GPU_PROBE_BYTES;
            let expected = ShaderBuffer::from(vec![gpu_probe(probe)]).data.unwrap();
            assert_eq!(
                &state.probe_mirror[offset..offset + GPU_PROBE_BYTES],
                &expected
            );
        }
    }

    #[test]
    fn replacement_catchup_is_bounded_and_publishes_latest_initial_bytes() {
        use super::super::{
            VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, upload_transport_cache,
        };
        use bevy::ecs::system::RunSystemOnce;
        use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
        use std::time::{Duration, Instant};

        let mut field = VoxelEnergyField::default();
        for x in 0..12_000 {
            let voxel = IVec3::new(x, 0, 0);
            let mut probe = SurfaceProbe::new(voxel, 4);
            probe.display[2].indirect = Vec3::ONE;
            field.order.push(voxel);
            field.probes.insert(voxel, probe);
        }
        let replacement = PreparedGpuCache::new(build_gpu_cache(snapshot_gpu_probes(
            &field.probes,
            &field.order,
        )));
        let mut buffers = Assets::<ShaderBuffer>::default();
        let mut assets = VoxelTransportAssets {
            texture: default(),
            probes: default(),
            hash_entries: default(),
        };
        let mut patches = GpuProbePatchQueue {
            probe_buffer: default(),
            hash_buffer: default(),
            prepared: None,
            state: default(),
        };
        let mut cache = GpuTransportCache::default();
        let mut materials = Assets::default();
        let handle = materials.add(material_template(&assets));
        install_gpu_cache(
            build_gpu_cache(snapshot_gpu_probes(&field.probes, &[IVec3::ZERO])),
            &mut cache,
            &mut patches,
            &mut assets,
            &mut buffers,
            &mut materials,
            &handle,
        );
        let old_id = assets.probes.id();
        for &voxel in &field.order {
            field.probes.get_mut(&voxel).unwrap().display[2].indirect = Vec3::splat(2.0);
        }
        let mut world = World::new();
        world.insert_resource(VoxelGiRuntime {
            gpu_prepared: Some(replacement),
            gpu_catchup_dirty: field.order.iter().copied().collect(),
            gpu_value_dirty: field.order.iter().copied().collect(),
            last_gpu_upload: Instant::now() - Duration::from_secs(1),
            ..default()
        });
        world.insert_resource(field);
        world.insert_resource(assets);
        world.insert_resource(patches);
        world.insert_resource(cache);
        world.insert_resource(buffers);
        world.insert_resource(materials);
        world.insert_resource(VoxelWorldMaterialHandle { handle });
        world.insert_resource(VoxelGiSettings::default());
        world.init_resource::<super::super::telemetry::GiWorkTimings>();

        world.run_system_once(upload_transport_cache).unwrap();
        assert_eq!(
            world.resource::<VoxelTransportAssets>().probes.id(),
            old_id,
            "more than a frame's catch-up budget must not publish a partial generation"
        );
        assert!(world.resource::<VoxelGiRuntime>().gpu_prepared.is_some());
        // Editing during preparation must requeue already-processed keys and
        // reconcile final membership, including remove/re-add of the same key.
        let added = IVec3::new(12_000, 0, 0);
        {
            let mut field = world.resource_mut::<VoxelEnergyField>();
            field.probes.get_mut(&IVec3::ZERO).unwrap().display[2].indirect = Vec3::splat(3.0);
            field.probes.remove(&IVec3::X);
            field.probes.insert(added, SurfaceProbe::new(added, 4));
        }
        {
            let mut runtime = world.resource_mut::<VoxelGiRuntime>();
            runtime.mark_display_dirty(IVec3::ZERO);
            runtime.gpu_removed.extend([IVec3::X, added]);
            runtime.gpu_added.push(added);
            runtime.gpu_topology_dirty = true;
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while world.resource::<VoxelGiRuntime>().gpu_prepared.is_some() {
            world.run_system_once(upload_transport_cache).unwrap();
            assert!(
                Instant::now() < deadline,
                "private catch-up made no progress"
            );
        }
        let assets = world.resource::<VoxelTransportAssets>();
        assert_ne!(assets.probes.id(), old_id);
        let cache = world.resource::<GpuTransportCache>();
        assert!(!cache.slots.contains_key(&IVec3::X));
        assert!(cache.slots.contains_key(&added));
        let state = world.resource::<GpuProbePatchQueue>().state.lock().unwrap();
        let buffers = world.resource::<Assets<ShaderBuffer>>();
        assert_eq!(
            buffers.get(&assets.probes).unwrap().data.as_ref().unwrap(),
            &state.probe_mirror
        );
        assert_eq!(
            buffers
                .get(&assets.hash_entries)
                .unwrap()
                .data
                .as_ref()
                .unwrap(),
            &state.hash_mirror
        );
        assert!(
            state.pending_probes.is_empty() && state.pending_hash.is_empty(),
            "initial upload must already contain catch-up, not defer it to a massive render patch"
        );
        for probe in world.resource::<VoxelEnergyField>().probes.values() {
            let offset = cache.slots[&probe.voxel] * GPU_PROBE_BYTES;
            let expected = ShaderBuffer::from(vec![gpu_probe(probe)]).data.unwrap();
            assert_eq!(
                &state.probe_mirror[offset..offset + GPU_PROBE_BYTES],
                &expected
            );
        }
    }

    #[test]
    fn rebuild_installation_merges_latest_values_without_upload_interval_flash() {
        use super::super::{
            VoxelEnergyField, VoxelGiRuntime, VoxelGiSettings, upload_transport_cache,
        };
        use bevy::ecs::system::RunSystemOnce;
        use bevy::tasks::{AsyncComputeTaskPool, TaskPool};
        use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
        use std::time::{Duration, Instant};
        let pool = AsyncComputeTaskPool::get_or_init(TaskPool::new);
        let voxel = IVec3::ZERO;
        let mut field = VoxelEnergyField::default();
        field.order.push(voxel);
        field.probes.insert(voxel, SurfaceProbe::new(voxel, 4));
        let snapshot = snapshot_gpu_probes(&field.probes, &field.order);
        let mut buffers = Assets::<ShaderBuffer>::default();
        let mut assets = VoxelTransportAssets {
            texture: default(),
            probes: default(),
            hash_entries: default(),
        };
        let mut patches = GpuProbePatchQueue {
            probe_buffer: default(),
            hash_buffer: default(),
            prepared: None,
            state: default(),
        };
        let mut cache = GpuTransportCache::default();
        let mut materials = Assets::default();
        let handle = materials.add(material_template(&assets));
        install_gpu_cache(
            build_gpu_cache(snapshot.clone()),
            &mut cache,
            &mut patches,
            &mut assets,
            &mut buffers,
            &mut materials,
            &handle,
        );
        let old_id = assets.probes.id();
        field.probes.get_mut(&voxel).unwrap().display[2].indirect = Vec3::splat(7.0);
        let expected = ShaderBuffer::from(vec![gpu_probe(&field.probes[&voxel])])
            .data
            .unwrap();
        let mut world = World::new();
        world.insert_resource(VoxelGiRuntime {
            gpu_task: Some(pool.spawn(std::future::pending())),
            gpu_catchup_dirty: [voxel].into_iter().collect(),
            gpu_value_dirty: HashSet::from([voxel]),
            last_gpu_upload: Instant::now() - Duration::from_secs(1),
            ..default()
        });
        world.insert_resource(field);
        world.insert_resource(assets);
        world.insert_resource(patches);
        world.insert_resource(cache);
        world.insert_resource(buffers);
        world.insert_resource(materials);
        world.insert_resource(VoxelWorldMaterialHandle { handle });
        world.insert_resource(VoxelGiSettings::default());
        world.init_resource::<super::super::telemetry::GiWorkTimings>();
        world.run_system_once(upload_transport_cache).unwrap();
        assert!(
            world
                .resource::<VoxelGiRuntime>()
                .gpu_catchup_dirty
                .contains(&voxel),
            "updating the old generation must retain the replacement's catch-up ledger"
        );
        let assert_current = |world: &World| {
            let queue = world.resource::<GpuProbePatchQueue>();
            let state = queue.state.lock().unwrap();
            assert_eq!(&state.probe_mirror[..GPU_PROBE_BYTES], &expected);
        };
        assert_current(&world);
        // A deliberately huge interval proves this is a transaction guarantee,
        // not lucky timing or an extra debounce.
        world
            .resource_mut::<VoxelGiSettings>()
            .display_upload_interval = Duration::from_secs(3600);
        world.resource_mut::<VoxelGiRuntime>().gpu_task =
            Some(pool.spawn(async move { build_gpu_cache(snapshot) }));
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            world.run_system_once(upload_transport_cache).unwrap();
            if world.resource::<VoxelGiRuntime>().gpu_task.is_none() {
                break;
            }
            assert!(Instant::now() < deadline, "cache build did not finish");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert_ne!(world.resource::<VoxelTransportAssets>().probes.id(), old_id);
        assert_current(&world);
        assert!(
            world
                .resource::<VoxelGiRuntime>()
                .gpu_value_dirty
                .is_empty()
        );
        // A measured sunlight repair must bypass the ordinary upload cadence
        // without changing it or requiring an indirect sample batch.
        world
            .resource_mut::<VoxelEnergyField>()
            .probes
            .get_mut(&voxel)
            .unwrap()
            .display[2]
            .sky = [super::super::sky::pack(Vec3::ONE); 4];
        let expected_sky = ShaderBuffer::from(vec![gpu_probe(
            &world.resource::<VoxelEnergyField>().probes[&voxel],
        )])
        .data
        .unwrap();
        {
            let mut runtime = world.resource_mut::<VoxelGiRuntime>();
            runtime.sky_upload_pending = true;
            runtime.mark_display_dirty(voxel);
            runtime.last_gpu_upload = Instant::now();
        }
        world.run_system_once(upload_transport_cache).unwrap();
        let state = world.resource::<GpuProbePatchQueue>().state.lock().unwrap();
        assert_eq!(&state.probe_mirror[..GPU_PROBE_BYTES], &expected_sky);
        assert!(!world.resource::<VoxelGiRuntime>().sky_upload_pending);
        drop(state);

        // Face-mask-only changes need the same publication guarantee, even
        // without a hash insertion and with the value timer still held closed.
        {
            let mut field = world.resource_mut::<VoxelEnergyField>();
            let probe = field.probes.get_mut(&voxel).unwrap();
            probe.update_face_mask(5);
            probe.display[0].indirect = Vec3::splat(3.0);
            probe.display[0].confidence = 0.125;
        }
        {
            let mut runtime = world.resource_mut::<VoxelGiRuntime>();
            runtime.gpu_topology_dirty = true;
            runtime.mark_display_dirty(voxel);
        }
        world.run_system_once(upload_transport_cache).unwrap();
        let expected_faces = ShaderBuffer::from(vec![gpu_probe(
            &world.resource::<VoxelEnergyField>().probes[&voxel],
        )])
        .data
        .unwrap();
        assert_eq!(
            &world
                .resource::<GpuProbePatchQueue>()
                .state
                .lock()
                .unwrap()
                .probe_mirror[..GPU_PROBE_BYTES],
            &expected_faces,
            "newly exposed face cannot wait for the indirect upload timer"
        );
        assert!(
            world
                .resource::<VoxelGiRuntime>()
                .gpu_value_dirty
                .is_empty(),
            "atomic publication must not enqueue a redundant value upload"
        );

        // A previously extracted material must accept sparse growth. Requiring
        // a new high-water uniform creates a black frame even with atomic data.
        let handle = world
            .resource::<VoxelWorldMaterialHandle<VoxelTransportMaterial>>()
            .handle
            .clone();
        let extracted_settings = world
            .resource::<Assets<VoxelTransportMaterial>>()
            .get(&handle)
            .unwrap()
            .extension
            .settings;
        let added = IVec3::X;
        world
            .resource_mut::<VoxelEnergyField>()
            .probes
            .insert(added, SurfaceProbe::new(added, 4));
        {
            let mut runtime = world.resource_mut::<VoxelGiRuntime>();
            runtime.gpu_topology_dirty = true;
            runtime.gpu_added.push(added);
        }
        world.run_system_once(upload_transport_cache).unwrap();
        let slot = world.resource::<GpuTransportCache>().slots[&added];
        assert!(slot < extracted_settings.probe_capacity as usize);
        assert_eq!(
            world
                .resource::<Assets<VoxelTransportMaterial>>()
                .get(&handle)
                .unwrap()
                .extension
                .settings,
            extracted_settings
        );
    }

    #[test]
    fn extracted_generations_survive_growth_shrink_and_equal_size_rebuilds() {
        let mut buffers = Assets::<ShaderBuffer>::default();
        let mut assets = VoxelTransportAssets {
            texture: default(),
            hash_entries: buffers.add(ShaderBuffer::from(vec![GpuHashEntry::EMPTY])),
            probes: buffers.add(ShaderBuffer::from(vec![GpuProbe::default()])),
        };
        let mut materials = Assets::default();
        let material = materials.add(material_template(&assets));
        let mut patches = GpuProbePatchQueue {
            probe_buffer: assets.probes.clone(),
            hash_buffer: assets.hash_entries.clone(),
            prepared: None,
            state: default(),
        };
        let mut cache = GpuTransportCache::default();
        for (generation, count) in [1, 7, 3, 3, 0, 9].into_iter().enumerate() {
            // This is the previous frame, still rendering while main installs
            // a new cache. It must not observe any new generation's mutation.
            let extracted = patches.clone();
            let old_probe = extracted.state.lock().unwrap().probe_mirror.clone();
            let old_hash = extracted.state.lock().unwrap().hash_mirror.clone();
            let probes: HashMap<_, _> = (0..count)
                .map(|x| {
                    let voxel = IVec3::new(x, generation as i32, 0);
                    (voxel, SurfaceProbe::new(voxel, 4))
                })
                .collect();
            let order: Vec<_> = probes.keys().copied().collect();
            install_gpu_cache(
                build_gpu_cache(snapshot_gpu_probes(&probes, &order)),
                &mut cache,
                &mut patches,
                &mut assets,
                &mut buffers,
                &mut materials,
                &material,
            );
            patch_probe_values(&cache, &patches, &probes, &order.iter().copied().collect());
            assert_ne!(extracted.probe_buffer.id(), patches.probe_buffer.id());
            assert_ne!(extracted.hash_buffer.id(), patches.hash_buffer.id());
            assert!(!Arc::ptr_eq(&extracted.state, &patches.state));
            assert_eq!(extracted.state.lock().unwrap().probe_mirror, old_probe);
            assert_eq!(extracted.state.lock().unwrap().hash_mirror, old_hash);
            let settings = &materials.get(&material).unwrap().extension;
            assert_eq!(settings.probes.id(), patches.probe_buffer.id());
            assert_eq!(settings.hash_entries.id(), patches.hash_buffer.id());
            assert_eq!(
                settings.settings.probe_capacity,
                cache.probe_capacity as u32
            );
            // A delayed old-frame drain cannot consume new-frame updates.
            extracted
                .state
                .lock()
                .unwrap()
                .upload(old_probe.len() as u64, old_hash.len() as u64, |_, _, _| {})
                .unwrap();
            assert_eq!(
                patches.state.lock().unwrap().pending_probes.len(),
                count as usize
            );
        }
    }

    #[test]
    fn dirty_slots_publish_the_latest_mirror_once_after_retry() {
        let mut state = ProbePatchState {
            probe_mirror: vec![0; GPU_PROBE_BYTES * 2],
            hash_mirror: vec![0; 16],
            ..default()
        };
        let mut latest = GpuProbe::default();
        for energy in [1.0, 2.0, 3.0] {
            latest.positive_y = Vec4::splat(energy);
            state.pending_probes.insert(1);
            write_probe(&mut state.probe_mirror, 1, latest);
        }
        assert_eq!(state.pending_probes.len(), 1);
        assert!(
            state
                .upload(GPU_PROBE_BYTES as u64, 16, |_, _, _| panic!("early write"))
                .is_err()
        );
        let expected = ShaderBuffer::from(vec![latest]).data.unwrap();
        let mut writes = 0;
        state
            .upload((GPU_PROBE_BYTES * 2) as u64, 16, |probe, offset, bytes| {
                assert!(probe);
                assert_eq!(offset, GPU_PROBE_BYTES);
                assert_eq!(bytes, expected);
                writes += 1;
            })
            .unwrap();
        assert_eq!(writes, 1);
        assert!(state.pending_probes.is_empty());
    }

    #[test]
    fn upload_validates_both_buffers_before_writing_and_preserves_retry() {
        let mut state = ProbePatchState {
            probe_mirror: vec![7; GPU_PROBE_BYTES * 2],
            hash_mirror: vec![3; 64],
            pending_probes: HashSet::from([1]),
            pending_hash: HashSet::from([3]),
        };
        // Smaller old hash allocation, even though the probe buffer fits.
        assert!(
            state
                .upload((GPU_PROBE_BYTES * 2) as u64, 32, |_, _, _| panic!(
                    "partial upload"
                ))
                .is_err()
        );
        assert_eq!(state.pending_probes.len(), 1);
        assert_eq!(state.pending_hash.len(), 1);
        let mut writes = Vec::new();
        state
            .upload((GPU_PROBE_BYTES * 2) as u64, 64, |probe, offset, bytes| {
                writes.push((probe, offset, bytes.len()))
            })
            .unwrap();
        assert_eq!(
            writes,
            [(true, GPU_PROBE_BYTES, GPU_PROBE_BYTES), (false, 48, 16)]
        );
        assert!(state.pending_probes.is_empty() && state.pending_hash.is_empty());
        state.pending_hash.insert(usize::MAX);
        assert!(
            state
                .upload((GPU_PROBE_BYTES * 2) as u64, 64, |_, _, _| panic!(
                    "invalid slot uploaded"
                ))
                .is_err()
        );
        assert_eq!(state.pending_hash.len(), 1);
    }

    #[test]
    fn sparse_uploads_are_bounded_by_changed_ranges_not_world_size() {
        let ranges = patch_ranges(vec![0, 1, 10, 100_000, 100_001], 96);
        assert_eq!(ranges, vec![0..1056, 9_600_000..9_600_192]);
        assert_eq!(ranges.iter().map(|r| r.len()).sum::<usize>(), 1248);
    }

    #[test]
    fn staged_upload_releases_mirror_lock_and_preserves_next_transaction() {
        let state = Mutex::new(ProbePatchState {
            probe_mirror: vec![7; GPU_PROBE_BYTES * 2],
            hash_mirror: vec![3; 64],
            pending_probes: HashSet::from([1]),
            pending_hash: HashSet::from([3]),
        });
        let mut upload = ProbeUpload::default();
        upload
            .stage(&state, (GPU_PROBE_BYTES * 2) as u64, 64)
            .unwrap();
        assert_eq!(upload.bytes.len(), GPU_PROBE_BYTES + 16);
        // Simulate the main world publishing a new transaction while the
        // graphics driver consumes the immutable previous one.
        upload.submit(|probe, _, bytes| {
            let mut next = state
                .try_lock()
                .expect("driver must not own the mirror lock");
            if probe {
                assert_eq!(bytes, vec![7; GPU_PROBE_BYTES]);
                next.probe_mirror.fill(9);
                next.pending_probes.insert(1);
            } else {
                assert_eq!(bytes, [3; 16]);
                next.hash_mirror.fill(5);
                next.pending_hash.insert(3);
            }
        });
        assert!(
            upload
                .stage(&state, (GPU_PROBE_BYTES * 2) as u64, 32)
                .is_err()
        );
        assert!(upload.bytes.is_empty() && upload.writes.is_empty());
        upload
            .stage(&state, (GPU_PROBE_BYTES * 2) as u64, 64)
            .unwrap();
        upload.submit(|probe, _, bytes| {
            assert!(bytes.iter().all(|&b| b == if probe { 9 } else { 5 }));
        });
        let next = state.lock().unwrap();
        assert!(next.pending_hash.is_empty() && next.pending_probes.is_empty());
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn gpu_headless_renderer_prepares_cache_and_material_generations_together() {
        use bevy::pbr::PreparedMaterial;
        use bevy::render::erased_render_asset::ErasedRenderAssets;
        use bevy::render::pipelined_rendering::PipelinedRenderingPlugin;
        use bevy::render::renderer::RenderDevice;
        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .build()
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<bevy::log::LogPlugin>()
                .disable::<bevy::audio::AudioPlugin>()
                .disable::<bevy::gilrs::GilrsPlugin>()
                .disable::<PipelinedRenderingPlugin>()
                .set(WindowPlugin {
                    primary_window: None,
                    ..default()
                }),
        );
        install_render_assets(&mut app, Handle::default());
        let mut assets = app.world().resource::<VoxelTransportAssets>().clone();
        let mut texture = Image::new_fill(
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 4,
            },
            wgpu::TextureDimension::D2,
            &[255, 255, 255, 255],
            wgpu::TextureFormat::Rgba8UnormSrgb,
            default(),
        );
        prepare_texture_array(&mut texture, 4).unwrap();
        assets.texture = app.world_mut().resource_mut::<Assets<Image>>().add(texture);
        let material = app
            .world_mut()
            .resource_mut::<Assets<VoxelTransportMaterial>>()
            .add(material_template(&assets));
        let mut patches = app.world().resource::<GpuProbePatchQueue>().clone();
        let mut cache = GpuTransportCache::default();
        app.finish();
        app.cleanup();
        let device = app
            .sub_app(RenderApp)
            .world()
            .resource::<RenderDevice>()
            .clone();
        let scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        // Prime the standalone image/material before testing buffer generations.
        for _ in 0..4 {
            app.update();
        }
        for (generation, count) in [1, 64, 3, 3, 128, 8].into_iter().enumerate() {
            let mut probes: HashMap<_, _> = (0..count)
                .map(|x| {
                    let voxel = IVec3::new(x, generation as i32, 0);
                    (voxel, SurfaceProbe::new(voxel, 4))
                })
                .collect();
            let order: Vec<_> = probes.keys().copied().collect();
            let build = build_gpu_cache(snapshot_gpu_probes(&probes, &order));
            // Growth, shrink and equal-size replacement all use the worker's
            // private mapped buffers; the last generation covers CPU fallback.
            let build = if generation < 5 {
                build.prepare_buffers(&device)
            } else {
                build
            };
            let mut build = PreparedGpuCache::new(build);
            for probe in probes.values_mut() {
                probe.display[2].indirect = Vec3::splat(3.0);
                probe.display[2].confidence = 0.75;
            }
            let added = IVec3::new(count, generation as i32, 0);
            probes.insert(added, SurfaceProbe::new(added, 4));
            let mut changed = order.clone();
            changed.push(added);
            assert!(build.synchronize(&changed, &probes));
            let prepared = build.mapped.clone();
            if let Some(mapped) = &prepared {
                let state = build.patches.state.lock().unwrap();
                assert_eq!(
                    &*mapped[0].slice(..).get_mapped_range(),
                    &state.probe_mirror
                );
                assert_eq!(&*mapped[1].slice(..).get_mapped_range(), &state.hash_mirror);
                assert!(state.pending_probes.is_empty() && state.pending_hash.is_empty());
            }
            app.world_mut()
                .resource_scope(|world, mut buffers: Mut<Assets<ShaderBuffer>>| {
                    install_prepared_gpu_cache(
                        build,
                        &mut cache,
                        &mut patches,
                        &mut assets,
                        &mut buffers,
                        &mut world.resource_mut::<Assets<VoxelTransportMaterial>>(),
                        &material,
                    );
                });
            patch_probe_values(&cache, &patches, &probes, &order.iter().copied().collect());
            app.insert_resource(patches.clone());
            app.insert_resource(assets.clone());
            {
                let mut materials = app
                    .world_mut()
                    .resource_mut::<Assets<VoxelTransportMaterial>>();
                let mut material = materials.get_mut(&material).unwrap();
                material.extension.settings.enabled = (generation & 1) as u32;
                material.extension.settings.reconstruction = ((generation >> 1) & 1) as u32;
                material.extension.settings.debug_irradiance = ((generation >> 2) & 1) as u32;
                material.extension.settings.voxel_sky = (generation & 1) as u32;
            }
            app.update();
            let render_world = app.sub_app(RenderApp).world();
            let buffers = render_world.resource::<RenderAssets<GpuShaderBuffer>>();
            if let Some(prepared) = prepared {
                assert_eq!(
                    buffers.get(&assets.probes).unwrap().buffer.id(),
                    prepared[0].id()
                );
                assert_eq!(
                    buffers.get(&assets.hash_entries).unwrap().buffer.id(),
                    prepared[1].id()
                );
                // Actual submission validates that no mapped buffer escaped
                // private preparation (bind-group creation alone does not).
                let destination = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 32,
                    usage: wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&default());
                encoder.copy_buffer_to_buffer(&prepared[0], 0, &destination, 0, 16);
                encoder.copy_buffer_to_buffer(&prepared[1], 0, &destination, 16, 16);
                render_world
                    .resource::<RenderQueue>()
                    .submit([encoder.finish()]);
            }
            assert_eq!(
                buffers.get(&assets.probes).unwrap().buffer.size(),
                (cache.probe_capacity * GPU_PROBE_BYTES) as u64
            );
            assert_eq!(
                buffers.get(&assets.hash_entries).unwrap().buffer.size(),
                (cache.hash_entries.len() * 16) as u64
            );
            assert!(
                render_world
                    .resource::<ErasedRenderAssets<PreparedMaterial>>()
                    .get(material.id())
                    .is_some(),
                "material disappeared for generation {generation}"
            );
            assert!(patches.state.lock().unwrap().pending_probes.is_empty());
        }
        assert!(bevy::tasks::block_on(scope.pop()).is_none());
    }

    #[test]
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn gpu_display_upload_retries_the_reported_out_of_bounds_offset() {
        use bevy::tasks::block_on;
        const OFFSET: usize = 63_420_448;
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).unwrap();
        let (device, queue) =
            block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let create = |size, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("display generation regression"),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        let usage = wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let probe_buffer = create(GPU_PROBE_BYTES as u64, usage);
        let old_hash = create(33_554_432, usage);
        let new_hash = create(67_108_864, usage);
        let expected = GpuHashEntry {
            voxel: IVec3::new(12, -5, 8),
            probe_index: 0,
        };
        let mut state = ProbePatchState {
            probe_mirror: vec![0; GPU_PROBE_BYTES],
            hash_mirror: vec![0; new_hash.size() as usize],
            pending_hash: HashSet::from([OFFSET / 16]),
            ..default()
        };
        write_hash_entry(&mut state.hash_mirror, OFFSET / 16, expected);
        let state = Mutex::new(state);
        let mut upload = ProbeUpload::default();
        assert!(
            upload
                .stage(&state, probe_buffer.size(), old_hash.size())
                .is_err()
        );
        assert!(upload.writes.is_empty());
        assert_eq!(state.lock().unwrap().pending_hash.len(), 1);
        upload
            .stage(&state, probe_buffer.size(), new_hash.size())
            .unwrap();
        let mut belt = wgpu::util::StagingBelt::new(device.clone(), 2 * 1024 * 1024);
        upload.submit_staged(&mut belt, &device, &queue, &probe_buffer, &new_hash);
        let readback = create(
            16,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let mut encoder = device.create_command_encoder(&default());
        encoder.copy_buffer_to_buffer(&new_hash, OFFSET as u64, &readback, 0, 16);
        let submission = queue.submit([encoder.finish()]);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(std::time::Duration::from_secs(5)),
            })
            .unwrap();
        receiver
            .recv_timeout(std::time::Duration::from_secs(1))
            .unwrap()
            .unwrap();
        assert_eq!(
            &*readback.slice(..).get_mapped_range(),
            &state.lock().unwrap().hash_mirror[OFFSET..OFFSET + 16]
        );
        readback.unmap();
        assert!(block_on(scope.pop()).is_none());
    }
}
