//! Which cameras GI lights.
use bevy::core_pipeline::prepass::{DepthPrepass, NormalPrepass};
use bevy::prelude::*;
use bevy::render::extract_component::ExtractComponent;

/// A camera voxel GI lights. Add it to the cameras that show your world:
///
/// ```no_run
/// # use bevy::prelude::*;
/// # use bevy_voxel_gi::VoxelGiCamera;
/// # fn setup(mut commands: Commands) {
/// commands.spawn((
///     Camera3d::default(),
///     VoxelGiCamera,
///     Transform::from_xyz(20.0, 12.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
/// ));
/// # }
/// ```
///
/// Each active `VoxelGiCamera` gets the view tier (lighting for the faces it
/// sees, and the far tier's per-pixel light), keeps the world cache around
/// it, and steers which pages GI samples first and ranks in
/// [`VoxelGiDemand`](crate::VoxelGiDemand). It brings the depth and normal
/// prepasses the view tier reads. Other cameras (a minimap, a portal view)
/// cost GI nothing; GI-lit materials they draw show GI's sun and emission
/// without indirect light.
#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Eq, ExtractComponent)]
#[extract_app(bevy::render::RenderApp)]
#[require(Camera3d, DepthPrepass, NormalPrepass)]
pub struct VoxelGiCamera;
