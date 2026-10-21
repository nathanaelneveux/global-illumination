//! Coherent, zero-readback publication of the transport scene to view passes.
//! Geometry and raw-cache pairs preserve their allocation generations; optical
//! tables are immutable per generation. A view owns its uniforms, never the
//! worker's mutable dispatch parameters.
use super::*;
use bevy::render::extract_resource::ExtractResource;
use std::sync::Mutex;
mod runtime;
mod timing;
pub(in crate::voxel_gi) use runtime::FAR_SHARE;
pub(crate) use runtime::install;

#[derive(Resource, Clone, Default, ExtractResource)]
pub(crate) struct ViewTransport(pub Option<SceneBindings>, pub Arc<Mutex<Vec<ViewChange>>>);

#[derive(Clone)]
pub(crate) struct ViewChange {
    pub(super) events: Buffer,
    pub(super) control: Buffer,
}

#[derive(Clone)]
pub(crate) struct SceneBindings {
    pub(super) params: GpuParams,
    /// Directory, regions, cells, materials, lights, emitters, cache, index.
    pub(super) buffers: [Buffer; 8],
}

impl GpuTracer {
    pub(crate) fn take_view_changes(&mut self) -> Vec<ViewChange> {
        self.resident.as_mut().map_or_else(Vec::new, |resident| {
            std::mem::take(&mut resident.view_changes)
        })
    }
    pub(crate) fn needs_model(
        &self,
        sources: &TransportSources,
        materials: &VoxelMaterialTransportTable,
        importance: EnergySpectrum,
        transport: super::super::TransportSettings,
    ) -> bool {
        self.sources.as_ref() != Some(sources)
            || self.material_table.as_ref() != Some(materials)
            || self.material_importance != importance
            || self.view_transport != transport.normalized()
    }

    pub(crate) fn view_scene(&self) -> Option<SceneBindings> {
        let scene = self.scene.as_ref()?;
        let sources = self.sources.as_ref()?;
        let emitters = self.emission.as_ref()?;
        self.material_table.as_ref()?;
        let (cache, index) = self.resident.as_ref().map_or(
            (&self.external_cache.0, &self.external_cache.1),
            |resident| resident.cache_buffers(),
        );
        Some(SceneBindings {
            params: GpuParams {
                counts: UVec4::new(
                    0,
                    1,
                    self.packed.directory.len() as u32 - 1,
                    (sources.directional.len() + sources.points.len()) as u32,
                ),
                minimum: scene.top_level.minimum.extend(0),
                maximum: scene
                    .top_level
                    .maximum
                    .extend(sources.directional.len() as i32),
                environment: spectrum_vec(sources.environment),
                extra: Vec4::new(
                    sources.environment.ionizing,
                    super::super::tracer::MAX_PATH_DISTANCE,
                    emitters.total_weight,
                    emitters.faces.len() as f32,
                ),
                transport: Vec4::new(self.view_transport.reach, 0.0, 0.0, 0.0),
            },
            buffers: [
                &self.directory,
                &self.regions,
                &self.cells,
                &self.materials,
                &self.lights,
                &self.emitters,
                cache,
                index,
            ]
            .map(|buffer| buffer.buffer.clone()),
        })
    }
}
