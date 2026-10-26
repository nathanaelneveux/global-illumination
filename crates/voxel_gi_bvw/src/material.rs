//! The `bevy_voxel_world` mesh material lit by voxel GI: BVW's texture array
//! and vertex layout over `StandardMaterial`, with GI's bindings copied from
//! `VoxelGiMaterialData` whenever it changes.
use voxel_gi::{GpuTransportSettings, GpuTraversalParams, VoxelGiMaterialData};
use bevy::image::{ImageAddressMode, ImageSampler, TextureReinterpretationError};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError, TextureViewDescriptor,
    TextureViewDimension,
};
use bevy::render::storage::ShaderBuffer;
use bevy::shader::ShaderRef;
use bevy_voxel_world::rendering::vertex_layout;

pub const VOXEL_TRANSPORT_SHADER: &str = "shaders/voxel_transport.wgsl";
/// The terrain material shader with the shared voxel traversal included
/// (inserted by `install_render_assets`).
pub const VOXEL_TRANSPORT_SHADER_HANDLE: Handle<Shader> =
    bevy::asset::uuid_handle!("2ae5baf3-fcaf-4036-8021-b3296b833947");

pub type VoxelTransportMaterial = ExtendedMaterial<StandardMaterial, VoxelTransportExtension>;

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct VoxelTransportExtension {
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    pub voxel_texture: Handle<Image>,
    #[uniform(104)]
    pub settings: GpuTransportSettings,
    /// Transport geometry (region directory, headers, cells) and its bounds:
    /// faces missing from the view tier trace sun visibility through it.
    #[storage(106, read_only, visibility(fragment))]
    pub directory: Handle<ShaderBuffer>,
    #[storage(107, read_only, visibility(fragment))]
    pub regions: Handle<ShaderBuffer>,
    #[storage(108, read_only, visibility(fragment))]
    pub cells: Handle<ShaderBuffer>,
    #[uniform(109)]
    pub traversal: GpuTraversalParams,
    /// Current view's GPU-resident indirect reconstruction (zero header when
    /// disabled). Camera passes write it in encoder order before opaque draws.
    #[storage(110, read_only, visibility(fragment))]
    pub far: Handle<ShaderBuffer>,
    /// The far tier's cells: per-pixel lighting for faces small on screen.
    #[storage(111, read_only, visibility(fragment))]
    pub far_pixels: Handle<ShaderBuffer>,
}

/// `voxel_traversal.wgsl`'s `params`: the sun visibility distance
/// (`counts.x`, 0 before the first scene), directory mask (`counts.z`) and
/// region bounds. The default has empty bounds, so nothing is traversed.
impl MaterialExtension for VoxelTransportExtension {
    fn fragment_shader() -> ShaderRef {
        VOXEL_TRANSPORT_SHADER_HANDLE.into()
    }

    fn vertex_shader() -> ShaderRef {
        VOXEL_TRANSPORT_SHADER_HANDLE.into()
    }

    fn specialize(
        _pipeline: &MaterialExtensionPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &bevy::mesh::MeshVertexBufferLayoutRef,
        _key: MaterialExtensionKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        // Prepasses use Bevy's own vertex shader/layout (UV at 1, normal at
        // 3), not the custom forward terrain IO. Replacing that layout would
        // corrupt the normal prepass used for view receiver discovery.
        if descriptor
            .vertex
            .shader_defs
            .contains(&"PREPASS_PIPELINE".into())
        {
            return Ok(());
        }
        let vertex_layout = layout.0.get_layout(&vertex_layout())?;
        descriptor.vertex.buffers = vec![vertex_layout];
        Ok(())
    }
}

/// Register the material and keep every instance's GI bindings current. A
/// host may add (and configure) `MaterialPlugin<VoxelTransportMaterial>`
/// itself; the shader and binding synchronization are installed either way.
pub(crate) fn install(app: &mut App) {
    if !app.is_plugin_added::<MaterialPlugin<VoxelTransportMaterial>>() {
        app.add_plugins(MaterialPlugin::<VoxelTransportMaterial>::default());
    }
    app.add_systems(PostUpdate, sync_material_data);
    let _ = app.world_mut().resource_mut::<Assets<Shader>>().insert(
        VOXEL_TRANSPORT_SHADER_HANDLE.id(),
        Shader::from_wgsl(
            voxel_gi::include_traversal(include_str!("../shaders/voxel_transport.wgsl")),
            VOXEL_TRANSPORT_SHADER,
        ),
    );
}

fn sync_material_data(data: Res<VoxelGiMaterialData>, mut materials: ResMut<Assets<VoxelTransportMaterial>>) {
    if !data.is_changed() {
        return;
    }
    for (_, material) in materials.iter_mut() {
        let extension = &mut material.extension;
        extension.directory = data.directory.clone();
        extension.regions = data.regions.clone();
        extension.cells = data.cells.clone();
        extension.traversal = data.traversal;
        extension.far = data.far.clone();
        extension.far_pixels = data.far_pixels.clone();
        extension.settings = data.settings;
    }
}

/// A BVW voxel material over `texture` (a 2D array, see `prepare_texture_array`).
pub fn material_template(texture: Handle<Image>, data: &VoxelGiMaterialData) -> VoxelTransportMaterial {
    ExtendedMaterial {
        base: StandardMaterial {
            reflectance: 0.05,
            metallic: 0.05,
            perceptual_roughness: 0.95,
            ..default()
        },
        extension: VoxelTransportExtension {
            voxel_texture: texture,
            directory: data.directory.clone(),
            regions: data.regions.clone(),
            cells: data.cells.clone(),
            settings: data.settings,
            traversal: data.traversal,
            far: data.far.clone(),
            far_pixels: data.far_pixels.clone(),
        },
    }
}

pub fn texture_array_ready(image: &Image, layers: u32) -> bool {
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

pub fn prepare_texture_array(
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
                voxel_gi::include_traversal(include_str!(
                    "../shaders/voxel_transport.wgsl"
                )),
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
                Ok(module) => {
                    let fragment = module.entry_points.iter()
                        .find(|entry| entry.name == "fragment").unwrap();
                    let naga::TypeInner::Struct { members, .. } =
                        &module.types[fragment.function.arguments[0].ty].inner else {
                            panic!("terrain fragment must receive its vertex output struct");
                        };
                    assert!(members.iter().any(|member| matches!(member.binding,
                        Some(naga::Binding::Location {
                            location: 0,
                            interpolation: Some(naga::Interpolation::Perspective),
                            sampling: Some(naga::Sampling::Centroid), ..
                        })
                    )), "world position must not extrapolate from an uncovered MSAA pixel centre");
                    break;
                }
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
    #[ignore = "requires a native compute adapter; run explicitly on supported hardware"]
    fn gpu_headless_renderer_binds_geometry_generations_with_the_material() {
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
        voxel_gi::testing::install_render(&mut app);
        // A host that registers (and may configure) the material plugin
        // itself still gets the connector's shader and binding sync.
        app.add_plugins(MaterialPlugin::<VoxelTransportMaterial>::default());
        install(&mut app);
        let data = app.world().resource::<VoxelGiMaterialData>().clone();
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
        let texture = app.world_mut().resource_mut::<Assets<Image>>().add(texture);
        let material = app
            .world_mut()
            .resource_mut::<Assets<VoxelTransportMaterial>>()
            .add(material_template(texture, &data));
        app.finish();
        app.cleanup();
        let device = app
            .sub_app(bevy::render::RenderApp)
            .world()
            .resource::<RenderDevice>()
            .clone();
        let scope = device
            .wgpu_device()
            .push_error_scope(wgpu::ErrorFilter::Validation);
        for _ in 0..4 {
            app.update();
        }
        // Growth, shrink and equal-size replacement each get a new identity;
        // descriptor-only preparation must bind the shared allocations.
        for (generation, size) in [64u64, 4096, 256, 256].into_iter().enumerate() {
            let buffers = [0, 1, 2].map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("geometry generation regression"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            });
            voxel_gi::testing::publish_geometry(
                app.world_mut(),
                buffers.clone().map(|buffer| buffer.into()),
                768,
            );
            {
                let mut data = app.world_mut().resource_mut::<VoxelGiMaterialData>();
                data.settings.enabled = (generation & 1) as u32;
                data.settings.debug_irradiance = ((generation >> 1) & 1) as u32;
            }
            app.update();
            app.update();
            let render_world = app.sub_app(bevy::render::RenderApp).world();
            let prepared = render_world
                .resource::<bevy::render::render_asset::RenderAssets<bevy::render::storage::GpuShaderBuffer>>();
            let material_assets = app.world().resource::<Assets<VoxelTransportMaterial>>();
            let extension = &material_assets.get(&material).unwrap().extension;
            assert_eq!(extension.traversal.counts.x, 768);
            for (handle, buffer) in [&extension.directory, &extension.regions, &extension.cells]
                .into_iter()
                .zip(&buffers)
            {
                assert_eq!(prepared.get(handle).unwrap().buffer.id(), buffer.id());
            }
            assert!(
                render_world
                    .resource::<ErasedRenderAssets<PreparedMaterial>>()
                    .get(material.id())
                    .is_some(),
                "material disappeared for generation {generation}"
            );
        }
        assert!(bevy::tasks::block_on(scope.pop()).is_none());
    }
}
