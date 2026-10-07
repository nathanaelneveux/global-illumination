//! Camera-turn regression workload in the actual prototype house. Keep terrain,
//! sources, streaming, and scheduler unchanged; inspect fixed world-space faces
//! even when the camera looks away from them.
use super::*;

#[derive(Default)]
pub(super) struct HouseRun {
    initialized: bool,
    last_sample: f32,
    pub samples: Vec<serde_json::Value>,
}

impl HouseRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        field: &VoxelEnergyField,
        runtime: &super::super::VoxelGiRuntime,
        commands: &mut Commands,
        cameras: &Query<Entity, With<Camera3d>>,
        transforms: &mut Query<&mut Transform, Without<Player>>,
    ) {
        if !self.initialized {
            for camera in cameras {
                commands
                    .entity(camera)
                    .remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>();
            }
            self.initialized = true;
        }
        let opposite = (30.0..50.0).contains(&elapsed) || elapsed >= 70.0;
        let target = if opposite {
            Vec3::new(6.0, 17.0, -12.0)
        } else {
            Vec3::new(-5.0, 17.0, -23.0)
        };
        for camera in cameras {
            if let Ok(mut transform) = transforms.get_mut(camera) {
                *transform = Transform::from_xyz(0.5, 14.0, -17.5).looking_at(target, Vec3::Y);
            }
        }
        if elapsed - self.last_sample < 0.1 {
            return;
        }
        self.last_sample = elapsed;
        let mut faces = Vec::new();
        for z in [-23, -22, -18, -14, -13] {
            for x in [-5, -4, 0, 4, 5] {
                let voxel = IVec3::new(x, 17, z);
                let face = 3; // Downward-facing roof underside.
                let probe = field.probes.get(&voxel);
                faces.push(json!({"voxel": voxel.to_array(),
                    "samples": probe.map(|p| p.samples[face]),
                    "sequence": probe.map(|p| p.sequence[face]),
                    "measured": probe.map(|p| p.radiance[face].visible.to_array()),
                    "display": probe.map(|p| p.display[face].indirect.to_array()),
                    "stable_batches": probe.map(|p| p.stable_batches[face]),
                    "fresh_mask": probe.map(|p| p.fresh_mask),
                    "transport_revision": probe.map(|p| p.transport_revision),
                    "source_revision": probe.map(|p| p.source_revision),
                    "geometry_revision": probe.map(|p| p.revision)}));
            }
        }
        self.samples
            .push(json!({"at_s": elapsed, "opposite": opposite,
            "transport_revision": field.transport_revision,
            "source_revision": field.source_revision,
            "generation": runtime.generation, "pages": runtime.loaded_pages.len(),
            "publishing_pages": field.publishing_pages.len(), "faces": faces}));
    }
}
