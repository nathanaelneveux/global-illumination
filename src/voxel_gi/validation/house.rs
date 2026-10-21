//! Camera-turn and opening-edit workloads in the actual prototype house.
//! Inspect fixed world-space receivers even while offscreen. Sources, normal
//! terrain streaming and adaptive scheduling remain enabled and unchanged.
use super::*;

const OPENINGS: &[(f32, &str, &[IVec3])] = &[
    (90.0, "skylight", &[IVec3::new(0, 17, -18)]),
    (
        120.0,
        "west-door",
        &[
            IVec3::new(-6, 11, -18),
            IVec3::new(-6, 12, -18),
            IVec3::new(-6, 13, -18),
        ],
    ),
    (
        150.0,
        "east-door-turn",
        &[
            IVec3::new(6, 11, -18),
            IVec3::new(6, 12, -18),
            IVec3::new(6, 13, -18),
        ],
    ),
];

#[derive(Default)]
pub(super) struct HouseRun {
    initialized: bool,
    last_sample: f32,
    stage: usize,
    pub samples: Vec<serde_json::Value>,
    pub events: Vec<serde_json::Value>,
    pub captures: Vec<serde_json::Value>,
}

impl HouseRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        edits: bool,
        world: &mut VoxelWorld<PrototypeWorld>,
        dirty: &mut VoxelGiDirty,
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
        let target = if edits && elapsed >= 150.0 {
            // Previously unseen corner: allow real streaming to overlap the
            // last doorway edit, without manufacturing geometry revisions.
            Vec3::new(-5.0, 17.0, -12.0)
        } else if opposite {
            Vec3::new(6.0, 17.0, -12.0)
        } else {
            Vec3::new(-5.0, 17.0, -23.0)
        };
        for camera in cameras {
            if let Ok(mut transform) = transforms.get_mut(camera) {
                *transform = Transform::from_xyz(0.5, 14.0, -17.5).looking_at(target, Vec3::Y);
            }
        }
        if edits {
            // Fixed warm-up, not a claim of convergence. The recorded stable
            // batches/measurements show the actual pre-edit receiver state.
            while let Some((at, name, voxels)) = OPENINGS.get(self.stage) {
                if elapsed < *at {
                    break;
                }
                let mut removed = 0;
                for &voxel in *voxels {
                    if matches!(world.get_voxel(voxel), WorldVoxel::Solid(_)) {
                        removed += 1;
                    }
                    dirty.set_voxel(world, voxel, WorldVoxel::Air);
                }
                self.events.push(json!({"name": name, "scheduled_s": at,
                    "at_s": elapsed, "requested_voxels": voxels.iter().map(|v| v.to_array()).collect::<Vec<_>>(),
                    "removed_solid_voxels": removed, "transport_revision": field.transport_revision,
                    "pages": runtime.loaded_pages.len()}));
                self.stage += 1;
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
                let probe = field
                    .probes
                    .get(&voxel)
                    .filter(|p| p.face_mask & (1 << face) != 0);
                faces.push(json!({"voxel": voxel.to_array(),
                    "samples": probe.map(|p| p.stats().samples[face]),
                    "sequence": probe.map(|p| p.stats().sequence[face]),
                    "measured": probe.map(|p| p.stats().radiance[face].visible.to_array()),
                    "second_moment": probe.map(|p| p.stats().second_moment[face].visible.to_array()),
                    "display": probe.map(|p| p.display[face].indirect.to_array()),
                    "sky": probe.map(|p| p.display[face].sky),
                    "stable_batches": probe.map(|p| p.stats().stable_batches[face]),
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
            "publishing_pages": field.publishing_pages.len(),
            "scene_build_pending": runtime.scene_task.is_some(), "faces": faces}));
    }
}
