//! Orthogonal terrain-streaming workload. An inactive, non-rendering BVW
//! camera loads real chunks without moving the measured view or GI focus.
//! Generation, meshing, residency, scene publication and tracing stay enabled.
use super::*;

#[derive(Default)]
pub(super) struct BackgroundRun {
    camera: Option<Entity>,
    pub moves: u32,
}

impl BackgroundRun {
    pub fn tick(
        &mut self,
        elapsed: f32,
        evidence: &CaptureEvidence,
        commands: &mut Commands,
        transforms: &mut Query<&mut Transform, Without<Player>>,
    ) {
        let transform = path(elapsed);
        if let Some(camera) = self.camera {
            if let Ok(mut current) = transforms.get_mut(camera) {
                *current = transform;
                self.moves += 1;
            }
        } else {
            self.camera = Some(spawn_loader(
                commands,
                evidence,
                "Validation background terrain loader (not rendered)",
                transform,
            ));
        }
    }
}

pub(super) fn spawn_loader(
    commands: &mut Commands,
    evidence: &CaptureEvidence,
    name: &'static str,
    transform: Transform,
) -> Entity {
    commands
        .spawn((
            Name::new(name),
            Camera {
                is_active: false,
                ..default()
            },
            bevy::render::camera::CameraRenderGraph::new(bevy::core_pipeline::schedule::Core3d),
            evidence
                .render_target
                .as_ref()
                .map_or_else(RenderTarget::default, |target| {
                    RenderTarget::Image(target.clone().into())
                }),
            Projection::Perspective(PerspectiveProjection::default()),
            bevy::camera::primitives::Frustum::default(),
            VoxelWorldCamera::<PrototypeWorld>::default(),
            transform,
        ))
        .id()
}

fn path(elapsed: f32) -> Transform {
    let eye = Vec3::new(-96.0, 40.0, -4.32 * elapsed);
    Transform::from_translation(eye).looking_at(eye + Vec3::new(0.0, -0.3, -1.0), Vec3::Y)
}

#[test]
fn background_path_keeps_walking_beyond_initial_residency() {
    let start = path(0.0).translation;
    assert!((path(100.0).translation.distance(start) - 432.0).abs() < 0.001);
    assert_eq!(path(100.0).translation.y, start.y);
}
