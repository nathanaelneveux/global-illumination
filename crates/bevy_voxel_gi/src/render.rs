//! What GI-lit materials bind: shared geometry, the visible-face output and
//! the presentation uniform, published as `VoxelGiMaterialData`.
use bevy::prelude::*;
use bevy::render::RenderSystems;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{RenderAssets, prepare_assets};
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp};

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

/// How GI-lit materials present lighting. Presentation only: never changes
/// transport data or history.
#[derive(Resource, Clone, Copy, Debug, PartialEq)]
pub struct VoxelGiDisplay {
    /// Apply the indirect-light term in GI's presentation shader.
    ///
    /// [`Self::voxel_sun`] controls the directional term separately.
    pub enabled: bool,
    /// Interpolate lighting across voxel corners (false: flat per face).
    pub corner_reconstruction: bool,
    /// GI's voxel-corner sun visibility replaces Bevy's shadowed sun.
    pub voxel_sun: bool,
    /// Indirect light multiplier.
    pub intensity: f32,
    /// A diagnostic view in place of the lit colour, or `None`.
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
    /// Normal lit shading.
    #[default]
    None,
    /// Combined irradiance (sun and indirect) without albedo.
    Irradiance,
    /// Which lookup supplied each pixel's indirect light.
    Coverage,
    /// Indirect light alone.
    Indirect,
    /// GI's sunlight alone.
    Sun,
    /// Sun traced from the face's own corners (reference for `Sun`).
    ExactCornerSun,
    /// Missing indirect support and invalid exposed-face lookups.
    FaceValidity,
}

impl GiDebugView {
    /// The view's index (the material's mode number).
    pub fn index(self) -> u32 {
        self as u32
    }

    /// The view at `index`; `None` past the last.
    pub fn from_index(index: u32) -> Self {
        [
            Self::None,
            Self::Irradiance,
            Self::Coverage,
            Self::Indirect,
            Self::Sun,
            Self::ExactCornerSun,
            Self::FaceValidity,
        ]
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

/// Everything a GI-lit material binds, kept current by voxel GI: the shared
/// transport geometry (directory, regions, cells), its traversal parameters,
/// the visible-face output (`far`) and the presentation uniform composed from
/// `VoxelGiDisplay` and the current suns. Materials copy these when this
/// resource changes; each new GPU allocation gets a new handle.
#[derive(Resource, Clone)]
pub struct VoxelGiMaterialData {
    pub(crate) directory: Handle<ShaderBuffer>,
    pub(crate) regions: Handle<ShaderBuffer>,
    pub(crate) cells: Handle<ShaderBuffer>,
    pub(crate) traversal: GpuTraversalParams,
    pub(crate) far: Handle<ShaderBuffer>,
    /// The far tier's cells (`FarTierSettings`): a 64-byte header (zero grid
    /// when off), then one packed cell per `cell`x`cell` pixels.
    pub(crate) far_pixels: Handle<ShaderBuffer>,
    pub(crate) settings: GpuTransportSettings,
}

/// The GI worker's geometry buffers, shared with the terrain material like
/// the resident buffer (each new allocation gets a new asset identity).
#[derive(Resource, Default, Clone, ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub(crate) struct GeometryDisplay {
    pub buffers: Option<[Buffer; 3]>,
    pub params: GpuTraversalParams,
    installed: Option<[(Buffer, Handle<ShaderBuffer>); 3]>,
    installed_params: GpuTraversalParams,
}

fn publish_geometry_buffers(
    mut geometry: ResMut<GeometryDisplay>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut data: ResMut<VoxelGiMaterialData>,
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
                let mut asset = ShaderBuffer::with_size(buffer.size(), default());
                asset.label = "Voxel GI shared geometry".into();
                asset.buffer_usage = buffer.usage();
                (buffer.clone(), buffers.add(asset))
            })
        });
    }
    geometry.installed_params = geometry.params;
    let Some(installed) = geometry.installed.clone() else {
        return;
    };
    data.directory = installed[0].1.clone();
    data.regions = installed[1].1.clone();
    data.cells = installed[2].1.clone();
    data.traversal = geometry.params;
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
                label: "Voxel GI shared geometry".into(),
                buffer_usage: buffer.usage(),
                had_data: false,
            },
        );
    }
}

/// Compose the materials' presentation uniform from `VoxelGiDisplay` and the
/// current suns. The display applies each cached sun's current direction and
/// transport color per pixel; corner visibility comes from its quantized
/// direction. Idle frames leave `VoxelGiMaterialData` unchanged.
fn synchronize_material_settings(
    display: Res<VoxelGiDisplay>,
    runtime: Option<Res<super::VoxelGiRuntime>>,
    mut data: ResMut<VoxelGiMaterialData>,
) {
    let Some(runtime) = runtime else {
        return;
    };
    let lights = super::sky::sky_lights(&runtime.sources);
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
    if data.settings != next {
        data.settings = next;
    }
}

pub fn install_render_assets(app: &mut App) {
    app.add_plugins((
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
        .add(ShaderBuffer::from(vec![
            UVec4::ZERO;
            super::gpu::view::EMPTY_OUTPUT_BYTES
                as usize
                / 16
        ]));
    let far_pixels =
        app.world_mut()
            .resource_mut::<Assets<ShaderBuffer>>()
            .add(ShaderBuffer::from(vec![
                UVec4::ZERO;
                super::gpu::view::PIXEL_HEADER_BYTES
                    as usize
                    / 16
                    + 1
            ]));
    app.insert_resource(VoxelGiMaterialData {
        directory: geometry.clone(),
        regions: geometry.clone(),
        cells: geometry,
        traversal: GpuTraversalParams::default(),
        far: far.clone(),
        far_pixels: far_pixels.clone(),
        settings: GpuTransportSettings::default(),
    });
    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app.add_systems(
            Render,
            install_geometry_buffers
                .in_set(RenderSystems::PrepareAssets)
                .before(prepare_assets::<GpuShaderBuffer>),
        );
    }
    super::gpu::view::install(app, far, far_pixels);
    if let Some(mut shaders) = app.world_mut().get_resource_mut::<Assets<Shader>>() {
        insert_voxel_gi_shaders(&mut shaders);
    }
}

/// Registers GI's WESL modules in `shaders`, as `VoxelGiPlugin` does: for
/// composing a material shader without the plugin (a composition test). Their
/// `embedded://` paths are their import paths: `bevy_voxel_gi::presentation`,
/// `bevy_voxel_gi::pbr` and the stock material's `bevy_voxel_gi::standard`.
pub fn insert_voxel_gi_shaders(shaders: &mut Assets<Shader>) {
    let _ = shaders.insert(
        PRESENTATION_SHADER.id(),
        Shader::from_wesl(
            presentation_shader_source(),
            "embedded://bevy_voxel_gi/presentation.wesl",
        ),
    );
    let _ = shaders.insert(
        PBR_SHADER.id(),
        Shader::from_wesl(PBR_SHADER_SOURCE, "embedded://bevy_voxel_gi/pbr.wesl"),
    );
    let _ = shaders.insert(
        STANDARD_SHADER.id(),
        Shader::from_wesl(
            STANDARD_SHADER_SOURCE,
            "embedded://bevy_voxel_gi/standard.wesl",
        ),
    );
}

/// The `bevy_voxel_gi::pbr` WESL module (registered by `VoxelGiPlugin`):
/// Bevy's `apply_pbr_lighting` with GI's cached voxel sun in place of Bevy's
/// first directional light while GI's sky is on. Host materials call it
/// instead of Bevy's.
pub const PBR_SHADER: Handle<Shader> =
    bevy::asset::uuid_handle!("4d2b8f61-93c7-4e0a-b1f5-7a3c6e9d2b48");
/// The `bevy_voxel_gi::pbr` module's source.
pub const PBR_SHADER_SOURCE: &str = include_str!("../shaders/voxel_gi_pbr.wesl");

/// `VoxelGiStandardMaterial`'s fragment shader.
pub const STANDARD_SHADER: Handle<Shader> =
    bevy::asset::uuid_handle!("0c5f3d2a-8e61-4b7a-b3c9-6d1e4f2a7b85");
pub const STANDARD_SHADER_SOURCE: &str = include_str!("../shaders/voxel_gi_standard.wesl");

/// Bevy's `StandardMaterial` lit by voxel GI: indirect light and emission
/// added, GI's cached voxel sun in place of Bevy's shadowed sun. Build one
/// with `standard_material`; `VoxelGiStandardMaterialPlugin` registers it.
/// It renders forward only: GI's presentation is in the forward fragment
/// shader, and a deferred `StandardMaterial` would run Bevy's own deferred
/// shader without it.
pub type VoxelGiStandardMaterial = bevy::pbr::ExtendedMaterial<
    bevy::pbr::ExtendedMaterial<StandardMaterial, VoxelGiBindings>,
    VoxelGiStandardShader,
>;

/// The shader half of `VoxelGiStandardMaterial` (no bindings of its own).
#[derive(Asset, TypePath, bevy::render::render_resource::AsBindGroup, Debug, Clone, Default)]
pub struct VoxelGiStandardShader {}

impl bevy::pbr::MaterialExtension for VoxelGiStandardShader {
    fn fragment_shader() -> bevy::shader::ShaderRef {
        STANDARD_SHADER.into()
    }
}

/// A `VoxelGiStandardMaterial` over `base`, forced to forward rendering
/// (an `ExtendedMaterial` takes its render method from `base`, and `Auto`
/// follows a deferred `DefaultOpaqueRendererMethod`).
pub fn standard_material(
    base: StandardMaterial,
    data: &VoxelGiMaterialData,
) -> VoxelGiStandardMaterial {
    bevy::pbr::ExtendedMaterial {
        base: bevy::pbr::ExtendedMaterial {
            base: StandardMaterial {
                opaque_render_method: bevy::material::OpaqueRendererMethod::Forward,
                ..base
            },
            extension: VoxelGiBindings::from(data),
        },
        extension: VoxelGiStandardShader {},
    }
}

/// Registers `VoxelGiStandardMaterial` and keeps its GI bindings current.
pub struct VoxelGiStandardMaterialPlugin;

impl Plugin for VoxelGiStandardMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins((
            MaterialPlugin::<VoxelGiStandardMaterial>::default(),
            VoxelGiMaterialPlugin::<VoxelGiStandardMaterial>::new(|material| {
                &mut material.base.extension
            }),
        ));
    }
}

/// The `bevy_voxel_gi::presentation` WGSL import module (registered by
/// `VoxelGiPlugin`): indirect light, sunlight and emission at a fragment for
/// host materials, over the bindings `VoxelGiBindings` supplies.
pub const PRESENTATION_SHADER: Handle<Shader> =
    bevy::asset::uuid_handle!("6b7a1c2e-7d43-4f0e-9a51-2c8e5b9d0f14");

/// The `bevy_voxel_gi::presentation` module's source, for hosts composing
/// shaders outside Bevy's asset pipeline (e.g. composition tests).
pub fn presentation_shader_source() -> String {
    super::gpu::include_traversal(include_str!("../shaders/voxel_gi_presentation.wesl"))
}

/// GI's bindings for a host material (104 and 106-111 of the material bind
/// group, where `bevy_voxel_gi::presentation` reads them). Wrap the host material
/// as `ExtendedMaterial<HostMaterial, VoxelGiBindings>`: the host keeps its
/// own shaders and bindings, which must leave those numbers free. Build it
/// with `VoxelGiBindings::from(&VoxelGiMaterialData)` and add
/// `VoxelGiMaterialPlugin` to keep it current.
#[derive(Asset, TypePath, bevy::render::render_resource::AsBindGroup, Debug, Clone, PartialEq)]
pub struct VoxelGiBindings {
    #[uniform(104)]
    pub(crate) settings: GpuTransportSettings,
    /// Transport geometry (region directory, headers, cells) and its bounds,
    /// for sun visibility of faces the view tier does not cover.
    #[storage(106, read_only, visibility(fragment))]
    pub(crate) directory: Handle<ShaderBuffer>,
    #[storage(107, read_only, visibility(fragment))]
    pub(crate) regions: Handle<ShaderBuffer>,
    #[storage(108, read_only, visibility(fragment))]
    pub(crate) cells: Handle<ShaderBuffer>,
    #[uniform(109)]
    pub(crate) traversal: GpuTraversalParams,
    /// The current view's receiver output (zero header when disabled).
    #[storage(110, read_only, visibility(fragment))]
    pub(crate) far: Handle<ShaderBuffer>,
    /// The far tier's cells.
    #[storage(111, read_only, visibility(fragment))]
    pub(crate) far_pixels: Handle<ShaderBuffer>,
}

impl bevy::pbr::MaterialExtension for VoxelGiBindings {}

impl From<&VoxelGiMaterialData> for VoxelGiBindings {
    fn from(data: &VoxelGiMaterialData) -> Self {
        Self {
            settings: data.settings,
            directory: data.directory.clone(),
            regions: data.regions.clone(),
            cells: data.cells.clone(),
            traversal: data.traversal,
            far: data.far.clone(),
            far_pixels: data.far_pixels.clone(),
        }
    }
}

/// Keeps every `M`'s `VoxelGiBindings` current with `VoxelGiMaterialData`
/// (new GPU allocations get new handles). `access` finds the bindings in
/// an `M`; `extended()` covers `ExtendedMaterial<Base, VoxelGiBindings>`.
pub struct VoxelGiMaterialPlugin<M: Material> {
    access: fn(&mut M) -> &mut VoxelGiBindings,
}

impl<M: Material> VoxelGiMaterialPlugin<M> {
    /// Keeps `M`'s bindings current; `access` finds them in an `M`.
    pub fn new(access: fn(&mut M) -> &mut VoxelGiBindings) -> Self {
        Self { access }
    }
}

impl<B: Material> VoxelGiMaterialPlugin<bevy::pbr::ExtendedMaterial<B, VoxelGiBindings>> {
    /// For `ExtendedMaterial<Base, VoxelGiBindings>`, the usual wrapping.
    pub fn extended() -> Self {
        Self::new(|material| &mut material.extension)
    }
}

#[derive(Resource)]
struct BindingsAccess<M: Material>(fn(&mut M) -> &mut VoxelGiBindings);

impl<M: Material> Plugin for VoxelGiMaterialPlugin<M> {
    fn build(&self, app: &mut App) {
        app.insert_resource(BindingsAccess::<M>(self.access))
            .add_systems(PostUpdate, sync_bindings::<M>);
    }
}

fn sync_bindings<M: Material>(
    data: Res<VoxelGiMaterialData>,
    access: Res<BindingsAccess<M>>,
    mut materials: ResMut<Assets<M>>,
) {
    if !data.is_changed() {
        return;
    }
    let current = VoxelGiBindings::from(&*data);
    for (_, material) in materials.iter_mut() {
        let bindings = (access.0)(material);
        if *bindings != current {
            *bindings = current.clone();
        }
    }
}

#[cfg(test)]
#[path = "render/coverage_tests.rs"]
mod coverage_tests;

#[cfg(test)]
mod standard_tests {
    use super::*;

    #[test]
    fn standard_material_renders_forward_under_a_deferred_default() {
        // Explicit Forward wins over `DefaultOpaqueRendererMethod`; Auto or
        // Deferred would select Bevy's deferred shader, which has no GI.
        let data = VoxelGiMaterialData {
            directory: default(),
            regions: default(),
            cells: default(),
            traversal: default(),
            far: default(),
            far_pixels: default(),
            settings: default(),
        };
        let deferred = StandardMaterial {
            opaque_render_method: bevy::material::OpaqueRendererMethod::Deferred,
            ..default()
        };
        for base in [StandardMaterial::default(), deferred] {
            let material = standard_material(base, &data);
            assert_eq!(
                Material::opaque_render_method(&material),
                bevy::material::OpaqueRendererMethod::Forward
            );
        }
    }

    #[test]
    fn standard_material_shader_composes_with_bevys_real_pbr_modules() {
        use bevy::render::{RenderPlugin, settings::WgpuSettings};
        use bevy::shader::{ShaderCache, ShaderCacheError, ShaderCacheSource, ShaderDefVal};
        // Bevy's own registered PBR libraries plus bevy_voxel_gi::presentation; no
        // window or GPU.
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
        insert_voxel_gi_shaders(&mut app.world_mut().resource_mut::<Assets<Shader>>());
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
        let mut defs: Vec<ShaderDefVal> = [
            "VERTEX_POSITIONS",
            "VERTEX_NORMALS",
            "VERTEX_UVS",
            "VERTEX_UVS_A",
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
        // Every `@if` path of the copied `apply_pbr_lighting` that GI's sun
        // touches: the plain material, then the optional lobes.
        let mut featured = defs.clone();
        featured.extend(
            [
                "STANDARD_MATERIAL_DIFFUSE_TRANSMISSION",
                "STANDARD_MATERIAL_CLEARCOAT",
                "STANDARD_MATERIAL_ANISOTROPY",
                "VERTEX_TANGENTS",
            ]
            .map(Into::into),
        );
        for (pipeline, defs) in [defs, featured].iter().enumerate() {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                app.update();
                for (id, shader) in app.world().resource::<Assets<Shader>>().iter() {
                    cache.set_shader(id, shader.clone());
                }
                match cache.get(pipeline, STANDARD_SHADER.id(), defs) {
                    Ok(module) => {
                        assert!(
                            module
                                .entry_points
                                .iter()
                                .any(|entry| entry.name == "fragment")
                        );
                        assert!(
                            module.functions.iter().any(|(_, f)| f
                                .name
                                .as_deref()
                                .is_some_and(|n| n.contains("voxel_sun"))),
                            "GI's sun is part of the composed lighting"
                        );
                        break;
                    }
                    Err(ShaderCacheError::ShaderImportNotYetAvailable)
                        if std::time::Instant::now() < deadline =>
                    {
                        std::thread::yield_now()
                    }
                    Err(error) => panic!("standard material shader failed to compose: {error}"),
                }
            }
        }
    }
}
