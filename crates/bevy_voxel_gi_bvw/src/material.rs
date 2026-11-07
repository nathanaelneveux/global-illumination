//! The `bevy_voxel_world` mesh material lit by voxel GI: BVW's texture array
//! and vertex layout over `StandardMaterial`, with GI's bindings
//! (`VoxelGiBindings`) and its presentation module (`bevy_voxel_gi::presentation`).
use bevy::image::{ImageAddressMode, ImageSampler, TextureReinterpretationError};
use bevy::material::OpaqueRendererMethod;
use bevy::pbr::{
    ExtendedMaterial, MaterialExtension, MaterialExtensionKey, MaterialExtensionPipeline,
};
use bevy::prelude::*;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, SpecializedMeshPipelineError, TextureDataOrder,
    TextureFormat, TextureUsages, TextureViewDescriptor, TextureViewDimension,
};
use bevy::shader::ShaderRef;
use bevy_voxel_gi::{VoxelGiBindings, VoxelGiMaterialData, VoxelGiMaterialPlugin};
use bevy_voxel_world::rendering::vertex_layout;

/// The terrain material shader's path, which is also its WESL import path
/// (`bevy_voxel_gi_bvw::voxel_transport`).
pub const VOXEL_TRANSPORT_SHADER: &str = "embedded://bevy_voxel_gi_bvw/voxel_transport.wesl";
/// The terrain material shader (inserted by `install`).
pub const VOXEL_TRANSPORT_SHADER_HANDLE: Handle<Shader> =
    bevy::asset::uuid_handle!("2ae5baf3-fcaf-4036-8021-b3296b833947");

/// `StandardMaterial` with GI's bindings, then BVW's texture array and shader.
pub type VoxelTransportMaterial =
    ExtendedMaterial<ExtendedMaterial<StandardMaterial, VoxelGiBindings>, VoxelTransportExtension>;

/// BVW's half of `VoxelTransportMaterial`: its block texture array and
/// shader, at bindings 100 and 101.
#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct VoxelTransportExtension {
    /// The block texture array (`prepare_texture_array`).
    #[texture(100, dimension = "2d_array")]
    #[sampler(101)]
    pub voxel_texture: Handle<Image>,
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

/// Register the material and keep every instance's GI bindings current. A
/// host may add (and configure) `MaterialPlugin<VoxelTransportMaterial>`
/// itself; the shader and binding synchronization are installed either way.
pub(crate) fn install(app: &mut App) {
    if !app.is_plugin_added::<MaterialPlugin<VoxelTransportMaterial>>() {
        app.add_plugins(MaterialPlugin::<VoxelTransportMaterial>::default());
    }
    app.add_plugins(VoxelGiMaterialPlugin::<VoxelTransportMaterial>::new(
        |material| &mut material.base.extension,
    ));
    let _ = app.world_mut().resource_mut::<Assets<Shader>>().insert(
        VOXEL_TRANSPORT_SHADER_HANDLE.id(),
        Shader::from_wesl(
            include_str!("../shaders/voxel_transport.wesl"),
            VOXEL_TRANSPORT_SHADER,
        ),
    );
}

/// A BVW voxel material over `texture` (a 2D array, see `prepare_texture_array`),
/// rendered forward: GI's presentation is in the forward fragment shader.
pub fn material_template(
    texture: Handle<Image>,
    data: &VoxelGiMaterialData,
) -> VoxelTransportMaterial {
    ExtendedMaterial {
        base: ExtendedMaterial {
            base: StandardMaterial {
                reflectance: 0.05,
                metallic: 0.05,
                perceptual_roughness: 0.95,
                opaque_render_method: OpaqueRendererMethod::Forward,
                ..default()
            },
            extension: VoxelGiBindings::from(data),
        },
        extension: VoxelTransportExtension {
            voxel_texture: texture,
        },
    }
}

/// Whether `prepare_texture_array` has already prepared `image` for `layers`
/// (check before mutably borrowing the asset, which would reupload it).
pub fn texture_array_ready(image: &Image, layers: u32) -> bool {
    image.texture_descriptor.size.depth_or_array_layers == layers
        && (image.texture_descriptor.mip_level_count > 1
            || !mipmapped(image.texture_descriptor.format))
        && image
            .texture_view_descriptor
            .as_ref()
            .is_some_and(|view| view.dimension == Some(TextureViewDimension::D2Array))
        && matches!(&image.sampler, ImageSampler::Descriptor(sampler)
            if sampler.address_mode_u == ImageAddressMode::Repeat
                && sampler.address_mode_v == ImageAddressMode::Repeat
                && sampler.address_mode_w == ImageAddressMode::Repeat)
}

/// Reinterprets a vertically stacked image as a `layers`-layer 2D array with
/// repeat addressing, a full mip chain (8-bit RGBA formats; sRGB averaged in
/// linear light) and 8x anisotropic filtering, so distant and grazing faces
/// do not alias.
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
        // A new layer count re-lays out level 0; the old mip chain is stale.
        strip_mips(image);
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
    descriptor.set_anisotropic_filter(8);
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::D2Array),
        ..default()
    });
    let format = image.texture_descriptor.format;
    if image.texture_descriptor.mip_level_count == 1 && mipmapped(format) {
        let size = image.texture_descriptor.size;
        if let Some((data, levels)) = image.data.as_deref().and_then(|base| {
            mip_chain(
                base,
                size.width,
                size.height,
                layers,
                format == TextureFormat::Rgba8UnormSrgb,
            )
        }) {
            image.data = Some(data);
            image.data_order = TextureDataOrder::LayerMajor;
            image.texture_descriptor.mip_level_count = levels;
            // Bevy 0.19 updates a same-descriptor COPY_DST texture by writing
            // the whole buffer into mip 0. Without COPY_DST a reload recreates
            // the texture with every level (wgpu adds COPY_DST for creation).
            image
                .texture_descriptor
                .usage
                .remove(TextureUsages::COPY_DST);
        }
    }
    Ok(())
}

/// Keeps each layer's level 0 of a mip chain this module built.
fn strip_mips(image: &mut Image) {
    let levels = image.texture_descriptor.mip_level_count;
    if levels <= 1 || !mipmapped(image.texture_descriptor.format) {
        return;
    }
    let size = image.texture_descriptor.size;
    let level_bytes = |level: u32| {
        (size.width >> level).max(1) as usize * (size.height >> level).max(1) as usize * 4
    };
    let layer_bytes: usize = (0..levels).map(level_bytes).sum();
    let Some(data) = image.data.as_deref() else {
        return;
    };
    if data.len() != layer_bytes * size.depth_or_array_layers as usize {
        return;
    }
    let base = data
        .chunks_exact(layer_bytes)
        .flat_map(|layer| &layer[..level_bytes(0)])
        .copied()
        .collect();
    image.data = Some(base);
    image.texture_descriptor.mip_level_count = 1;
}

fn mipmapped(format: TextureFormat) -> bool {
    matches!(
        format,
        TextureFormat::Rgba8Unorm | TextureFormat::Rgba8UnormSrgb
    )
}

/// Every layer's full mip chain, layer-major (layer 0's levels, then layer
/// 1's), box-filtered from level 0; colour in linear light when `srgb`.
fn mip_chain(
    base: &[u8],
    width: u32,
    height: u32,
    layers: u32,
    srgb: bool,
) -> Option<(Vec<u8>, u32)> {
    let layer_bytes = width as usize * height as usize * 4;
    if width == 0 || height == 0 || base.len() != layer_bytes * layers as usize {
        return None;
    }
    let decode: [f32; 256] = std::array::from_fn(|v| {
        let c = v as f32 / 255.0;
        if !srgb {
            c
        } else if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    });
    let encode = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        let c = if !srgb {
            c
        } else if c <= 0.003_130_8 {
            c * 12.92
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (c * 255.0 + 0.5) as u8
    };
    let levels = width.max(height).ilog2() + 1;
    let mut out = Vec::with_capacity(base.len() * 4 / 3 + 4 * layers as usize);
    for layer in base.chunks_exact(layer_bytes) {
        out.extend_from_slice(layer);
        let (mut w, mut h) = (width, height);
        let mut previous = layer.to_vec();
        while w > 1 || h > 1 {
            let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
            let mut next = vec![0u8; nw as usize * nh as usize * 4];
            for y in 0..nh {
                for x in 0..nw {
                    let texels = [
                        (2 * x, 2 * y),
                        (2 * x + 1, 2 * y),
                        (2 * x, 2 * y + 1),
                        (2 * x + 1, 2 * y + 1),
                    ]
                    .map(|(sx, sy)| ((sx.min(w - 1) + sy.min(h - 1) * w) * 4) as usize);
                    let at = ((x + y * nw) * 4) as usize;
                    for channel in 0..4 {
                        // Alpha is linear coverage in either format.
                        if channel == 3 || !srgb {
                            let sum: u32 =
                                texels.iter().map(|t| previous[t + channel] as u32).sum();
                            next[at + channel] = ((sum + 2) / 4) as u8;
                        } else {
                            let sum: f32 = texels
                                .iter()
                                .map(|t| decode[previous[t + channel] as usize])
                                .sum();
                            next[at + channel] = encode(sum / 4.0);
                        }
                    }
                }
            }
            out.extend_from_slice(&next);
            previous = next;
            (w, h) = (nw, nh);
        }
    }
    Some((out, levels))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn texture_array_gets_a_linear_light_mip_chain_and_stays_prepared() {
        use bevy::asset::RenderAssetUsages;
        // Four 4x4 layers stacked vertically; layer 2 is a black/white checker.
        let mut image = Image::new_fill(
            wgpu::Extent3d {
                width: 4,
                height: 16,
                depth_or_array_layers: 1,
            },
            wgpu::TextureDimension::D2,
            &[0, 0, 0, 255],
            TextureFormat::Rgba8UnormSrgb,
            RenderAssetUsages::default(),
        );
        let source: Vec<u8> = (0..64u32)
            .flat_map(|texel| {
                let (layer, x, y) = (texel / 16, texel % 4, texel / 4 % 4);
                let v = if layer == 2 {
                    255 * ((x + y) % 2) as u8
                } else {
                    layer as u8 * 60
                };
                [v, v, v, 255]
            })
            .collect();
        image.data = Some(source.clone());
        image
            .texture_descriptor
            .usage
            .insert(TextureUsages::COPY_DST);
        assert!(!texture_array_ready(&image, 4));
        prepare_texture_array(&mut image, 4).unwrap();
        assert!(texture_array_ready(&image, 4));
        assert_eq!(image.texture_descriptor.mip_level_count, 3);
        assert!(
            !image
                .texture_descriptor
                .usage
                .contains(TextureUsages::COPY_DST)
        );
        let data = image.data.as_ref().unwrap();
        let stride = (16 + 4 + 1) * 4;
        assert_eq!(data.len(), 4 * stride);
        for layer in 0..4 {
            // Level 0 of every layer is the original tile.
            assert_eq!(
                &data[layer * stride..layer * stride + 64],
                &source[layer * 64..(layer + 1) * 64]
            );
        }
        // Half black, half white is sRGB 188 in linear light (not 128).
        let checker_1x1 = &data[2 * stride + 80..2 * stride + 84];
        assert_eq!(checker_1x1, &[188, 188, 188, 255]);
        let prepared = image.clone();
        prepare_texture_array(&mut image, 4).unwrap();
        assert_eq!(image, prepared, "preparing twice changes nothing");
        // A new layer count starts again from level 0: two 4x8 layers.
        prepare_texture_array(&mut image, 2).unwrap();
        assert!(texture_array_ready(&image, 2));
        assert_eq!(image.texture_descriptor.mip_level_count, 4);
        let data = image.data.as_ref().unwrap();
        let stride = (32 + 8 + 2 + 1) * 4;
        assert_eq!(data.len(), 2 * stride);
        assert_eq!(&data[..128], &source[..128]);
        assert_eq!(&data[stride..stride + 128], &source[128..]);
    }

    #[test]
    fn terrain_shader_composes_with_bevys_real_pbr_modules_without_a_window() {
        use bevy::render::{RenderPlugin, settings::WgpuSettings};
        use bevy::shader::{
            Shader, ShaderCache, ShaderCacheError, ShaderCacheSource, ShaderDefVal,
        };
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
        let mut cache = ShaderCache::new((), |_, source, _| {
            let ShaderCacheSource::Wgsl(source) = source else {
                panic!("expected composed WGSL")
            };
            let module = naga::front::wgsl::parse_str(&source).map_err(|error| {
                ShaderCacheError::CreateShaderModule(error.emit_to_string(&source))
            })?;
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .map_err(|error| ShaderCacheError::CreateShaderModule(error.emit_to_string(&source)))?;
            Ok(module)
        });
        // The material imports GI's modules (VoxelGiPlugin registers them in
        // an app; here, by hand).
        bevy_voxel_gi::insert_voxel_gi_shaders(
            &mut app.world_mut().resource_mut::<Assets<Shader>>(),
        );
        let handle = app
            .world_mut()
            .resource_mut::<Assets<Shader>>()
            .add(Shader::from_wesl(
                include_str!("../shaders/voxel_transport.wesl"),
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
                    let fragment = module
                        .entry_points
                        .iter()
                        .find(|entry| entry.name == "fragment")
                        .unwrap();
                    let naga::TypeInner::Struct { members, .. } =
                        &module.types[fragment.function.arguments[0].ty].inner
                    else {
                        panic!("terrain fragment must receive its vertex output struct");
                    };
                    assert!(
                        members.iter().any(|member| matches!(
                            member.binding,
                            Some(naga::Binding::Location {
                                location: 0,
                                interpolation: Some(naga::Interpolation::Perspective),
                                sampling: Some(naga::Sampling::Centroid),
                                ..
                            })
                        )),
                        "world position must not extrapolate from an uncovered MSAA pixel centre"
                    );
                    break;
                }
                Err(ShaderCacheError::ShaderImportNotYetAvailable)
                    if std::time::Instant::now() < deadline =>
                {
                    std::thread::yield_now()
                }
                Err(error) => panic!("terrain shader failed to compose: {error}"),
            }
        }
        // Compiling the ordinary PBR material afterwards must not pick up GI's
        // bindings from the shared shader cache.
        let standard_id = app
            .world()
            .resource::<Assets<Shader>>()
            .iter()
            .find(|(_, shader)| {
                shader.path.ends_with("bevy_pbr/src/render/pbr.wesl")
                    || shader.path.ends_with("bevy_pbr/render/pbr.wesl")
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
        bevy_voxel_gi::testing::install_render(&mut app);
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
        let template = material_template(default(), &data);
        assert_eq!(
            Material::opaque_render_method(&template),
            OpaqueRendererMethod::Forward,
            "a deferred default must not take GI's presentation away"
        );
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
            bevy_voxel_gi::testing::publish_geometry(app.world_mut(), buffers.clone(), 768);
            bevy_voxel_gi::testing::set_display(
                app.world_mut(),
                generation & 1 == 1,
                ((generation >> 1) & 1) as u32,
            );
            app.update();
            app.update();
            let render_world = app.sub_app(bevy::render::RenderApp).world();
            let prepared =
                render_world.resource::<bevy::render::render_asset::RenderAssets<
                    bevy::render::storage::GpuShaderBuffer,
                >>();
            let material_assets = app.world().resource::<Assets<VoxelTransportMaterial>>();
            let extension = &material_assets.get(&material).unwrap().base.extension;
            let (handles, nodes) = bevy_voxel_gi::testing::geometry_bindings(extension);
            assert_eq!(nodes, 768);
            for (handle, buffer) in handles.iter().zip(&buffers) {
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
