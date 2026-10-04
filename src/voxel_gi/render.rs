use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use bevy::image::{ImageAddressMode, TextureReinterpretationError};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp, RenderSystems};
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

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct GpuTransportSettings {
    pub hash_mask: u32,
    pub probe_count: u32,
    pub enabled: u32,
    pub reconstruction: u32,
    pub intensity: f32,
    pub debug_irradiance: u32,
    /// 0: Bevy per-pixel direct sun; 1: cached diffuse direct sun.
    pub voxel_sky: u32,
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
    pub direct_sky: [Vec4; 6],
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
    slots: HashMap<IVec3, usize>,
    hash_slots: HashMap<IVec3, usize>,
    hash_entries: Vec<GpuHashEntry>,
    free_probe_slots: Vec<usize>,
    probe_capacity: usize,
    high_water: usize,
    pub ready: bool,
}

#[derive(Resource, Clone, ExtractResource)]
pub(crate) struct GpuProbePatchQueue {
    probe_buffer: Handle<ShaderBuffer>,
    hash_buffer: Handle<ShaderBuffer>,
    pending_probes: Arc<Mutex<HashMap<usize, GpuProbe>>>,
    pending_hash: Arc<Mutex<HashMap<usize, GpuHashEntry>>>,
    probe_mirror: Arc<Mutex<Vec<u8>>>,
    hash_mirror: Arc<Mutex<Vec<u8>>>,
}

#[derive(Resource, Default)]
struct TexturePrepared(bool);

pub fn install_render_assets(app: &mut App) {
    app.add_plugins((
        MaterialPlugin::<VoxelTransportMaterial>::default(),
        ExtractResourcePlugin::<GpuProbePatchQueue>::default(),
    ))
    .init_resource::<TexturePrepared>()
    .add_systems(Update, prepare_voxel_texture);

    let texture = app
        .world()
        .resource::<AssetServer>()
        .load("example_voxel_texture.png");
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
        pending_probes: Arc::new(Mutex::new(HashMap::new())),
        pending_hash: Arc::new(Mutex::new(HashMap::new())),
        probe_mirror: Arc::new(Mutex::new(Vec::new())),
        hash_mirror: Arc::new(Mutex::new(Vec::new())),
    });
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            apply_probe_patches.in_set(RenderSystems::PrepareBindGroups),
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
            settings: GpuTransportSettings {
                enabled: 1,
                reconstruction: 1,
                intensity: 1.0,
                ..default()
            },
        },
    }
}

pub(crate) struct GpuCacheBuild {
    hash_data: Vec<u8>,
    probe_data: Vec<u8>,
    slots: HashMap<IVec3, usize>,
    hash_slots: HashMap<IVec3, usize>,
    hash_entries: Vec<GpuHashEntry>,
    probe_capacity: usize,
    high_water: usize,
}

pub(crate) fn snapshot_gpu_probes(
    probes: &HashMap<IVec3, SurfaceProbe>,
    order: &[IVec3],
) -> Vec<(IVec3, GpuProbe)> {
    order
        .iter()
        .map(|voxel| (*voxel, gpu_probe(&probes[voxel])))
        .collect()
}

/// CPU encoding is deliberately run in an async task. `ShaderBuffer::set_data`
/// on 250k probes was responsible for multi-frame stalls in ordinary builds.
pub(crate) fn build_gpu_cache(snapshot: Vec<(IVec3, GpuProbe)>) -> GpuCacheBuild {
    let probe_capacity = snapshot.len().saturating_add(16_384).max(1);
    let mut hash_capacity = snapshot.len().saturating_mul(2).next_power_of_two().max(1);
    loop {
        let mut entries = vec![GpuHashEntry::EMPTY; hash_capacity];
        let mut slots = HashMap::with_capacity(snapshot.len());
        let mut hash_slots = HashMap::with_capacity(snapshot.len());
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

pub(crate) fn install_gpu_cache(
    build: GpuCacheBuild,
    cache: &mut GpuTransportCache,
    patches: &GpuProbePatchQueue,
    assets: &VoxelTransportAssets,
    buffers: &mut Assets<ShaderBuffer>,
    materials: &mut Assets<VoxelTransportMaterial>,
    material_handle: &Handle<VoxelTransportMaterial>,
) {
    patches.pending_probes.lock().unwrap().clear();
    patches.pending_hash.lock().unwrap().clear();
    *patches.probe_mirror.lock().unwrap() = build.probe_data.clone();
    *patches.hash_mirror.lock().unwrap() = build.hash_data.clone();
    cache.slots = build.slots;
    cache.hash_slots = build.hash_slots;
    cache.hash_entries = build.hash_entries;
    cache.free_probe_slots.clear();
    cache.probe_capacity = build.probe_capacity;
    cache.high_water = build.high_water;
    cache.ready = true;
    if let Some(mut buffer) = buffers.get_mut(&assets.hash_entries) {
        buffer.data = Some(build.hash_data);
    }
    if let Some(mut buffer) = buffers.get_mut(&assets.probes) {
        buffer.data = Some(build.probe_data);
    }
    if let Some(mut material) = materials.get_mut(material_handle) {
        material.extension.settings.hash_mask = cache.hash_entries.len() as u32 - 1;
        material.extension.settings.probe_count = cache.high_water as u32;
    }
}

/// Apply sparse probe additions/removals without replacing either GPU buffer.
/// Tombstones preserve collision chains for the shader's linear probing.
pub(crate) fn patch_gpu_topology(
    cache: &mut GpuTransportCache,
    patches: &GpuProbePatchQueue,
    probes: &HashMap<IVec3, SurfaceProbe>,
    removed: &[IVec3],
    added: &[IVec3],
    materials: &mut Assets<VoxelTransportMaterial>,
    material_handle: &Handle<VoxelTransportMaterial>,
) -> bool {
    let mut hash_pending = patches.pending_hash.lock().unwrap();
    let mut probe_pending = patches.pending_probes.lock().unwrap();
    let mut hash_mirror = patches.hash_mirror.lock().unwrap();
    let mut probe_mirror = patches.probe_mirror.lock().unwrap();
    for voxel in removed {
        if let Some(probe_slot) = cache.slots.remove(voxel) {
            cache.free_probe_slots.push(probe_slot);
        }
        if let Some(hash_slot) = cache.hash_slots.remove(voxel) {
            cache.hash_entries[hash_slot] = GpuHashEntry::TOMBSTONE;
            hash_pending.insert(hash_slot, GpuHashEntry::TOMBSTONE);
            write_hash_entry(&mut hash_mirror, hash_slot, GpuHashEntry::TOMBSTONE);
        }
    }
    for voxel in added {
        if cache.slots.contains_key(voxel) {
            continue;
        }
        let Some(probe) = probes.get(voxel) else {
            continue;
        };
        let probe_slot = if let Some(slot) = cache.free_probe_slots.pop() {
            slot
        } else {
            if cache.high_water >= cache.probe_capacity {
                return false;
            }
            let slot = cache.high_water;
            cache.high_water += 1;
            slot
        };
        let mut hash_slot = hash_voxel(*voxel) as usize & (cache.hash_entries.len() - 1);
        let mut found = false;
        for _ in 0..MAX_SHADER_HASH_PROBES {
            if cache.hash_entries[hash_slot].probe_index >= u32::MAX - 1 {
                found = true;
                break;
            }
            hash_slot = (hash_slot + 1) & (cache.hash_entries.len() - 1);
        }
        if !found {
            return false;
        }
        let entry = GpuHashEntry {
            voxel: *voxel,
            probe_index: probe_slot as u32,
        };
        cache.hash_entries[hash_slot] = entry;
        cache.slots.insert(*voxel, probe_slot);
        cache.hash_slots.insert(*voxel, hash_slot);
        hash_pending.insert(hash_slot, entry);
        write_hash_entry(&mut hash_mirror, hash_slot, entry);
        let gpu_probe = gpu_probe(probe);
        probe_pending.insert(probe_slot, gpu_probe);
        write_probe(&mut probe_mirror, probe_slot, gpu_probe);
    }
    if let Some(mut material) = materials.get_mut(material_handle) {
        material.extension.settings.probe_count = cache.high_water as u32;
    }
    true
}

pub(crate) fn patch_probe_values(
    cache: &GpuTransportCache,
    patches: &GpuProbePatchQueue,
    probes: &HashMap<IVec3, SurfaceProbe>,
    dirty_voxels: &HashSet<IVec3>,
) {
    let mut pending = patches.pending_probes.lock().unwrap();
    let mut mirror = patches.probe_mirror.lock().unwrap();
    for voxel in dirty_voxels {
        let (Some(slot), Some(probe)) = (cache.slots.get(voxel), probes.get(voxel)) else {
            continue;
        };
        let gpu_probe = gpu_probe(probe);
        pending.insert(*slot, gpu_probe);
        write_probe(&mut mirror, *slot, gpu_probe);
    }
}

fn apply_probe_patches(
    patches: Res<GpuProbePatchQueue>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    queue: Res<RenderQueue>,
) {
    let (Some(probe_buffer), Some(hash_buffer)) = (
        buffers.get(&patches.probe_buffer),
        buffers.get(&patches.hash_buffer),
    ) else {
        return;
    };
    // Lock in the same order as producers so a mirror and its dirty indices
    // form one coherent snapshot. Copying/uploading a whole 30MB buffer merely
    // because 2,049 probes changed is deliberately avoided.
    let mut pending = patches.pending_probes.lock().unwrap();
    let mirror = patches.probe_mirror.lock().unwrap();
    for range in patch_ranges(pending.keys().copied().collect(), GPU_PROBE_BYTES) {
        queue.write_buffer(&probe_buffer.buffer, range.start as u64, &mirror[range]);
    }
    pending.clear();
    drop(mirror);
    drop(pending);
    let mut pending = patches.pending_hash.lock().unwrap();
    let mirror = patches.hash_mirror.lock().unwrap();
    for range in patch_ranges(pending.keys().copied().collect(), 16) {
        queue.write_buffer(&hash_buffer.buffer, range.start as u64, &mirror[range]);
    }
    pending.clear();
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

fn encode_gpu_probe(probe: GpuProbe, bytes: &mut Vec<u8>) {
    for value in [
        probe.positive_x,
        probe.negative_x,
        probe.positive_y,
        probe.negative_y,
        probe.positive_z,
        probe.negative_z,
        probe.emission,
    ]
    .into_iter()
    .chain(probe.direct_sky)
    {
        for component in value.to_array() {
            bytes.extend_from_slice(&component.to_le_bytes());
        }
    }
}

fn write_probe(mirror: &mut [u8], slot: usize, probe: GpuProbe) {
    let mut bytes = Vec::with_capacity(GPU_PROBE_BYTES);
    encode_gpu_probe(probe, &mut bytes);
    mirror[slot * GPU_PROBE_BYTES..(slot + 1) * GPU_PROBE_BYTES].copy_from_slice(&bytes);
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
            (probe.radiance[i].visible + probe.custom_direct[i]).extend(1.0)
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
                probe.direct_sky[face].extend(1.0)
            } else {
                Vec4::ZERO
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

fn prepare_voxel_texture(
    assets: Res<VoxelTransportAssets>,
    asset_server: Res<AssetServer>,
    mut prepared: ResMut<TexturePrepared>,
    mut images: ResMut<Assets<Image>>,
) {
    if prepared.0
        || !matches!(
            asset_server.get_load_state(assets.texture.id()),
            Some(bevy::asset::LoadState::Loaded)
        )
    {
        return;
    }

    let Some(mut image) = images.get_mut(&assets.texture) else {
        return;
    };
    if let Err(error) = prepare_texture_array(&mut image, 4) {
        error!(%error, "failed to prepare voxel texture array");
        return;
    }
    prepared.0 = true;
}

fn prepare_texture_array(
    image: &mut Image,
    layers: u32,
) -> Result<(), TextureReinterpretationError> {
    let descriptor = image.sampler.get_or_init_descriptor();
    descriptor.address_mode_u = ImageAddressMode::Repeat;
    descriptor.address_mode_v = ImageAddressMode::Repeat;
    descriptor.address_mode_w = ImageAddressMode::Repeat;
    if image.texture_descriptor.size.depth_or_array_layers == 1 {
        image.reinterpret_stacked_2d_as_array(layers)?;
    }
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
            direct_sky: std::array::from_fn(|face| Vec4::splat(29.0 + face as f32)),
        };
        let mut encoded = Vec::new();
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
        let mut probe = SurfaceProbe::new(IVec3::ZERO, 4, EnergySpectrum::ZERO);
        probe.radiance[2] = EnergySpectrum::new(Vec3::X, 3.0, 4.0);
        probe.custom_direct[2] = Vec3::Y;
        probe.direct_sky = [Vec3::splat(10.0); 6];
        let encoded = gpu_probe(&probe);
        assert_eq!(encoded.positive_y, Vec4::new(1.0, 1.0, 0.0, 1.0));
        assert_eq!(encoded.direct_sky[2], Vec4::new(10.0, 10.0, 10.0, 1.0));
        assert_eq!(encoded.direct_sky[0], Vec4::ZERO);
        let material = material_template(&VoxelTransportAssets {
            texture: default(),
            hash_entries: default(),
            probes: default(),
        });
        assert_eq!(material.extension.settings.voxel_sky, 0);
        assert_eq!(material.extension.settings.reconstruction, 1);
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
            probes.insert(voxel, SurfaceProbe::new(voxel, 1, EnergySpectrum::ZERO));
        }
        let build = build_gpu_cache(snapshot_gpu_probes(&probes, &[voxel_a, voxel_b]));
        let patches = GpuProbePatchQueue {
            probe_buffer: Handle::default(),
            hash_buffer: Handle::default(),
            pending_probes: Arc::new(Mutex::new(HashMap::new())),
            pending_hash: Arc::new(Mutex::new(HashMap::new())),
            probe_mirror: Arc::new(Mutex::new(build.probe_data.clone())),
            hash_mirror: Arc::new(Mutex::new(build.hash_data.clone())),
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
            &mut Assets::default(),
            &Handle::default(),
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
                        ..SurfaceProbe::new(voxel, 4, EnergySpectrum::ZERO)
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
            .map(|voxel| (voxel, SurfaceProbe::new(voxel, 4, EnergySpectrum::ZERO)))
            .collect();
        // Simulate the immutable snapshot spending time on a worker while
        // main-world topology and radiance continue changing.
        let snapshot = snapshot_gpu_probes(&probes, &[a, b]);
        probes.remove(&a);
        probes.insert(c, SurfaceProbe::new(c, 4, EnergySpectrum::ONE));
        probes.get_mut(&b).unwrap().radiance[2] = EnergySpectrum::ONE * 3.0;
        let build = build_gpu_cache(snapshot);
        let mut buffers = Assets::<ShaderBuffer>::default();
        let assets = VoxelTransportAssets {
            texture: Handle::default(),
            hash_entries: buffers.add(ShaderBuffer::from(vec![GpuHashEntry::EMPTY])),
            probes: buffers.add(ShaderBuffer::from(vec![GpuProbe::default()])),
        };
        let patches = GpuProbePatchQueue {
            probe_buffer: assets.probes.clone(),
            hash_buffer: assets.hash_entries.clone(),
            pending_probes: default(),
            pending_hash: default(),
            probe_mirror: default(),
            hash_mirror: default(),
        };
        let mut cache = GpuTransportCache::default();
        let mut materials = Assets::default();
        install_gpu_cache(
            build,
            &mut cache,
            &patches,
            &assets,
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
            &mut materials,
            &Handle::default(),
        ));
        patch_probe_values(&cache, &patches, &probes, &HashSet::from([b, c]));
        assert_eq!(cache.slots.len(), probes.len());
        assert!(!cache.slots.contains_key(&a));
        assert!(!cache.slots.contains_key(&transient));
        let mirror = patches.probe_mirror.lock().unwrap();
        for (voxel, probe) in &probes {
            let offset = cache.slots[voxel] * GPU_PROBE_BYTES;
            let expected = ShaderBuffer::from(vec![gpu_probe(probe)]).data.unwrap();
            assert_eq!(&mirror[offset..offset + GPU_PROBE_BYTES], &expected);
        }
    }

    #[test]
    fn sparse_uploads_are_bounded_by_changed_ranges_not_world_size() {
        let ranges = patch_ranges(vec![0, 1, 10, 100_000, 100_001], 96);
        assert_eq!(ranges, vec![0..1056, 9_600_000..9_600_192]);
        assert_eq!(ranges.iter().map(|r| r.len()).sum::<usize>(), 1248);
    }
}
