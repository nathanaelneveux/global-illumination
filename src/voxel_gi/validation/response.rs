//! Focused visual fixture for cavity initialization and voxel-emitter removal.
//! Opt-in only; never changes ordinary terrain, settings, or materials on disk.
use super::*;

const HOLE: IVec3 = IVec3::new(8, 38, -9);
const EMITTER: IVec3 = IVec3::new(10, 41, -9);

#[derive(Default)]
pub(super) struct ResponseRun {
    initialized: bool,
    stage: u8,
    last_sample: f32,
    pub samples: Vec<serde_json::Value>,
    pub captures: Vec<serde_json::Value>,
}

impl ResponseRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        edits: bool,
        world: &mut VoxelWorld<PrototypeWorld>,
        dirty: &mut VoxelGiDirty,
        field: &VoxelEnergyField,
        materials: &mut VoxelMaterialTransportTable,
        commands: &mut Commands,
        cameras: &Query<Entity, With<Camera3d>>,
        transforms: &mut Query<&mut Transform, Without<Player>>,
    ) {
        if !self.initialized {
            // Two-block-thick floor so the edit opens a five-face cavity, not
            // a skylight. A weak permanent emitter keeps dark-room captures
            // inspectable after the brighter warm test emitter is removed.
            for z in -12..=-4 {
                for y in 37..=45 {
                    for x in 4..=12 {
                        let position = IVec3::new(x, y, z);
                        let solid = x == 4 || x == 12 || y <= 38 || y == 45 || z == -12 || z == -4;
                        world.set_voxel(
                            position,
                            if solid {
                                WorldVoxel::Solid(2)
                            } else {
                                WorldVoxel::Air
                            },
                        );
                        dirty.mark_voxel(position);
                    }
                }
            }
            materials.set(
                251,
                MaterialTransport {
                    emission: EnergySpectrum::new(Vec3::splat(2.0), 0.5, 0.0),
                    ..default()
                },
            );
            let lamp = IVec3::new(6, 42, -8);
            world.set_voxel(lamp, WorldVoxel::Solid(251));
            dirty.mark_voxel(lamp);
            for camera in cameras {
                commands
                    .entity(camera)
                    .remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>();
            }
            self.initialized = true;
        }
        for camera in cameras {
            if let Ok(mut transform) = transforms.get_mut(camera) {
                *transform = if edits && elapsed >= 23.0 {
                    Transform::from_xyz(8.5, 48.5, -5.5)
                        .looking_at(Vec3::new(8.5, 45.0, -9.5), Vec3::Y)
                } else {
                    Transform::from_xyz(8.5, 41.5, -5.5)
                        .looking_at(Vec3::new(8.5, 38.8, -9.5), Vec3::Y)
                };
            }
        }
        let stage = if edits {
            [15.0, 17.0, 19.0, 21.0, 25.0]
                .into_iter()
                .filter(|&at| elapsed >= at)
                .count() as u8
        } else if elapsed >= 25.0 {
            3
        } else if elapsed >= 20.0 {
            2
        } else if elapsed >= 15.0 {
            1
        } else {
            0
        };
        if stage > self.stage {
            let (position, material) = if edits {
                match stage {
                    1 => (HOLE, WorldVoxel::Air),
                    2 => (HOLE, WorldVoxel::Solid(2)),
                    3 => (HOLE + IVec3::Y, WorldVoxel::Solid(2)),
                    4 => (HOLE + IVec3::Y, WorldVoxel::Air),
                    _ => (IVec3::new(8, 45, -9), WorldVoxel::Air),
                }
            } else {
                match stage {
                    1 => (HOLE, WorldVoxel::Air),
                    2 => (EMITTER, WorldVoxel::Solid(4)),
                    _ => (EMITTER, WorldVoxel::Air),
                }
            };
            world.set_voxel(position, material);
            dirty.mark_voxel(position);
            self.stage = stage;
            self.samples.push(json!({"event": stage, "at_s": elapsed}));
        }
        if elapsed - self.last_sample >= 0.05 {
            self.last_sample = elapsed;
            let faces: Vec<_> = FACE_OFFSETS
                .into_iter()
                .enumerate()
                .filter(|(_, normal)| *normal != IVec3::NEG_Y)
                .map(|(face, normal)| (HOLE - normal, face))
                .chain([(IVec3::new(9, 38, -9), 2), (IVec3::new(10, 38, -9), 2)])
                .chain((0..6).map(|face| (HOLE + IVec3::Y, face)))
                .map(|(voxel, face)| {
                    let probe = field
                        .probes
                        .get(&voxel)
                        .filter(|p| p.face_mask & (1 << face) != 0);
                    json!({"voxel": voxel.to_array(), "face": face,
                        "display": probe.map(|p| p.display[face].indirect.to_array()),
                        "measured": probe.map(|p| p.radiance[face].visible.to_array()),
                        "samples": probe.map(|p| p.samples[face]),
                        "sky": probe.map(|p| p.display[face].sky),
                        "source_revision": probe.map(|p| p.source_revision),
                        "confidence": probe.map(|p| p.display[face].confidence)})
                })
                .collect();
            self.samples.push(
                json!({"at_s": elapsed, "source_revision": field.source_revision, "faces": faces}),
            );
        }
    }
}
