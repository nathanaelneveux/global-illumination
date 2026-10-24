use bevy::image::{ImageAddressMode, ImageSampler, TextureReinterpretationError};
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{RenderAssets, prepare_assets};
use bevy::render::render_resource::{
    AsBindGroup, Buffer, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
    TextureViewDescriptor, TextureViewDimension,
};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp, RenderSystems};
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
}

/// `voxel_traversal.wgsl`'s `params`: the sun visibility distance
/// (`counts.x`, 0 before the first scene), directory mask (`counts.z`) and
/// region bounds. The default has empty bounds, so nothing is traversed.
#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct GpuTraversalParams {
    pub counts: UVec4,
    pub minimum: IVec4,
    pub maximum: IVec4,
}

impl Default for GpuTraversalParams {
    fn default() -> Self {
        Self {
            counts: UVec4::ZERO,
            minimum: IVec4::ONE,
            maximum: IVec4::ZERO,
        }
    }
}

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

/// How GI-lit materials present lighting. Presentation only: never changes
/// transport data or history.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct VoxelGiDisplay {
    /// Apply GI lighting at all.
    pub enabled: bool,
    /// Interpolate lighting across voxel corners (false: flat per face).
    pub corner_reconstruction: bool,
    /// GI's voxel-corner sun visibility replaces Bevy's shadowed sun.
    pub voxel_sun: bool,
    /// Indirect light multiplier.
    pub intensity: f32,
    pub debug_view: GiDebugView,
}

impl Default for VoxelGiDisplay {
    fn default() -> Self {
        Self {
            enabled: true,
            corner_reconstruction: true,
            voxel_sun: true,
            intensity: 1.0,
            debug_view: GiDebugView::None,
        }
    }
}

/// Diagnostic shading, replacing the lit color (the material's mode index).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GiDebugView {
    #[default]
    None,
    /// Combined irradiance (sun and indirect) without albedo.
    Irradiance,
    /// Which lookup supplied each pixel's indirect light.
    Coverage,
    Indirect,
    Sun,
    /// Sun traced from the face's own corners (reference for `Sun`).
    ExactCornerSun,
    /// Missing indirect support and invalid exposed-face lookups.
    FaceValidity,
}

impl GiDebugView {
    pub fn index(self) -> u32 {
        self as u32
    }

    pub fn from_index(index: u32) -> Self {
        [Self::None, Self::Irradiance, Self::Coverage, Self::Indirect, Self::Sun, Self::ExactCornerSun, Self::FaceValidity]
            .get(index as usize)
            .copied()
            .unwrap_or_default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, ShaderType)]
pub struct GpuTransportSettings {
    pub enabled: u32,
    pub reconstruction: u32,
    pub intensity: f32,
    pub debug_irradiance: u32,
    /// 0: Bevy per-pixel direct sun; 1: cached diffuse direct sun.
    pub voxel_sky: u32,
    /// Directional lights the cached corner visibility refers to (source
    /// order): current direction and transport color, applied per pixel.
    pub sun_count: u32,
    pub sun_direction: [Vec4; super::sky::MAX_SKY_LIGHTS],
    pub sun_color: [Vec4; super::sky::MAX_SKY_LIGHTS],
}

impl Default for GpuTransportSettings {
    fn default() -> Self {
        Self {
            enabled: 1,
            reconstruction: 1,
            intensity: 1.0,
            debug_irradiance: 0,
            voxel_sky: 1,
            sun_count: 0,
            sun_direction: [Vec4::ZERO; super::sky::MAX_SKY_LIGHTS],
            sun_color: [Vec4::ZERO; super::sky::MAX_SKY_LIGHTS],
        }
    }
}


#[derive(Resource, Clone)]
pub struct VoxelTransportAssets {
    pub texture: Handle<Image>,
    /// Placeholder geometry until the GI worker publishes its own.
    pub geometry: Handle<ShaderBuffer>,
    pub far: Handle<ShaderBuffer>,
}

/// The GI worker's geometry buffers, shared with the terrain material like
/// the resident buffer (each new allocation gets a new asset identity).
#[derive(Resource, Default, Clone, ExtractResource)]
pub(crate) struct GeometryDisplay {
    pub buffers: Option<[Buffer; 3]>,
    pub params: GpuTraversalParams,
    installed: Option<[(Buffer, Handle<ShaderBuffer>); 3]>,
    installed_params: GpuTraversalParams,
}

fn publish_geometry_buffers(
    mut geometry: ResMut<GeometryDisplay>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
) {
    let unchanged_buffers = match (&geometry.buffers, &geometry.installed) {
        (Some(current), Some(installed)) => current
            .iter()
            .zip(installed)
            .all(|(a, (b, _))| a.id() == b.id()),
        (None, None) => true,
        _ => false,
    };
    if unchanged_buffers && geometry.params == geometry.installed_params {
        return;
    }
    if !unchanged_buffers {
        geometry.installed = geometry.buffers.as_ref().map(|current| {
            current.clone().map(|buffer| {
                let mut asset = ShaderBuffer::default();
                asset.buffer_description.label = Some("Voxel GI shared geometry");
                asset.buffer_description.size = buffer.size();
                asset.buffer_description.usage = buffer.usage();
                (buffer.clone(), buffers.add(asset))
            })
        });
    }
    geometry.installed_params = geometry.params;
    let Some(installed) = geometry.installed.clone() else {
        return;
    };
    for (_, material) in materials.iter_mut() {
        let extension = &mut material.extension;
        extension.directory = installed[0].1.clone();
        extension.regions = installed[1].1.clone();
        extension.cells = installed[2].1.clone();
        extension.traversal = geometry.params;
    }
}

fn install_geometry_buffers(
    geometry: Res<GeometryDisplay>,
    mut assets: ResMut<RenderAssets<GpuShaderBuffer>>,
) {
    for (buffer, handle) in geometry.installed.iter().flatten() {
        if assets.get(handle).is_some() {
            continue;
        }
        assets.insert(
            handle.id(),
            GpuShaderBuffer {
                buffer: buffer.clone(),
                buffer_descriptor: wgpu::BufferDescriptor {
                    label: Some("Voxel GI shared geometry"),
                    size: buffer.size(),
                    usage: buffer.usage(),
                    ..ShaderBuffer::default().buffer_description
                },
                had_data: false,
            },
        );
    }
}

/// Compose the material's settings from `VoxelGiDisplay` and the current suns.
/// The display applies each cached sun's current direction and transport
/// color per pixel; corner visibility comes from its quantized direction.
fn synchronize_material_settings(
    display: Res<VoxelGiDisplay>,
    runtime: Option<Res<super::VoxelGiRuntime>>,
    handle: Option<
        Res<bevy_voxel_world::rendering::VoxelWorldMaterialHandle<VoxelTransportMaterial>>,
    >,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
) {
    let (Some(runtime), Some(handle)) = (runtime, handle) else {
        return;
    };
    let lights = super::sky_cache::sky_lights(&runtime.sources);
    let mut next = GpuTransportSettings {
        enabled: u32::from(display.enabled),
        reconstruction: u32::from(display.corner_reconstruction),
        intensity: display.intensity,
        debug_irradiance: display.debug_view.index(),
        voxel_sky: u32::from(display.voxel_sun),
        sun_count: lights.len() as u32,
        ..default()
    };
    for (index, light) in lights.iter().enumerate() {
        next.sun_direction[index] = light.direction.extend(0.0);
        next.sun_color[index] = light.color.extend(0.0);
    }
    // Idle frames must not mark the material changed.
    if materials.get(&handle.handle).is_some_and(|material| material.extension.settings != next)
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings = next;
    }
}

pub fn install_render_assets(app: &mut App, texture: Handle<Image>) {
    app.add_plugins((
        MaterialPlugin::<VoxelTransportMaterial>::default(),
        ExtractResourcePlugin::<GeometryDisplay>::default(),
        ExtractResourcePlugin::<super::gpu::view::ViewTransport>::default(),
    ));
    app.init_resource::<GeometryDisplay>();
    app.init_resource::<VoxelGiDisplay>();
    app.init_resource::<super::gpu::view::ViewTransport>();
    app.add_systems(
        Update,
        (
            publish_geometry_buffers.after(super::dispatch::apply_trace_task),
            synchronize_material_settings.after(super::sources::synchronize_transport_sources),
        ),
    );
    let geometry = app
        .world_mut()
        .resource_mut::<Assets<ShaderBuffer>>()
        .add(ShaderBuffer::from(vec![UVec4::ZERO; 4]));
    let far = app
        .world_mut()
        .resource_mut::<Assets<ShaderBuffer>>()
        .add(ShaderBuffer::from(vec![UVec4::ZERO; super::gpu::view::EMPTY_OUTPUT_BYTES as usize / 16]));
    let _ = app.world_mut().resource_mut::<Assets<Shader>>().insert(
        VOXEL_TRANSPORT_SHADER_HANDLE.id(),
        Shader::from_wgsl(
            super::gpu::include_traversal(include_str!(
                "../../assets/shaders/voxel_transport.wgsl"
            )),
            VOXEL_TRANSPORT_SHADER,
        ),
    );
    app.insert_resource(VoxelTransportAssets {
        texture,
        geometry,
        far: far.clone(),
    });
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            install_geometry_buffers
                .in_set(RenderSystems::PrepareAssets)
                .before(prepare_assets::<GpuShaderBuffer>),
        );
    }
    super::gpu::view::install(app, far);
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
            directory: assets.geometry.clone(),
            regions: assets.geometry.clone(),
            cells: assets.geometry.clone(),
            settings: GpuTransportSettings::default(),
            traversal: GpuTraversalParams::default(),
            far: assets.far.clone(),
        },
    }
}

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
#[path = "render/coverage_tests.rs"]
mod coverage_tests;

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
                crate::voxel_gi::gpu::include_traversal(include_str!(
                    "../../assets/shaders/voxel_transport.wgsl"
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
            {
                let mut geometry = app.world_mut().resource_mut::<GeometryDisplay>();
                geometry.buffers = Some(buffers.clone().map(|buffer| buffer.into()));
                geometry.params.counts.x = 768;
            }
            {
                let mut materials = app
                    .world_mut()
                    .resource_mut::<Assets<VoxelTransportMaterial>>();
                let mut material = materials.get_mut(&material).unwrap();
                material.extension.settings.enabled = (generation & 1) as u32;
                material.extension.settings.debug_irradiance = ((generation >> 1) & 1) as u32;
            }
            app.update();
            app.update();
            let render_world = app.sub_app(RenderApp).world();
            let prepared = render_world.resource::<RenderAssets<GpuShaderBuffer>>();
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
