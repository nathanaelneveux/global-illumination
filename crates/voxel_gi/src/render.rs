//! What GI-lit materials bind: shared geometry, the visible-face output and
//! the presentation uniform, published as `VoxelGiMaterialData`.
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::{RenderAssets, prepare_assets};
use bevy::render::render_resource::{Buffer, ShaderType};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp};
use bevy::render::RenderSystems;

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


/// Everything a GI-lit material binds, kept current by voxel GI: the shared
/// transport geometry (directory, regions, cells), its traversal parameters,
/// the visible-face output (`far`) and the presentation uniform composed from
/// `VoxelGiDisplay` and the current suns. Materials copy these when this
/// resource changes; each new GPU allocation gets a new handle.
#[derive(Resource, Clone)]
pub struct VoxelGiMaterialData {
    pub directory: Handle<ShaderBuffer>,
    pub regions: Handle<ShaderBuffer>,
    pub cells: Handle<ShaderBuffer>,
    pub traversal: GpuTraversalParams,
    pub far: Handle<ShaderBuffer>,
    pub settings: GpuTransportSettings,
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
        .add(ShaderBuffer::from(vec![UVec4::ZERO; super::gpu::view::EMPTY_OUTPUT_BYTES as usize / 16]));
    app.insert_resource(VoxelGiMaterialData {
        directory: geometry.clone(),
        regions: geometry.clone(),
        cells: geometry,
        traversal: GpuTraversalParams::default(),
        far: far.clone(),
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
    super::gpu::view::install(app, far);
}

#[cfg(test)]
#[path = "render/coverage_tests.rs"]
mod coverage_tests;

