mod ambient_occlusion;
mod app_controls;
mod assets;
mod chunk_colliders;
mod player;
mod terrain;
mod terrain_meshing;
mod terrain_noise;
pub mod voxel_gi;
mod world_environment;

use app_controls::AppControlsPlugin;
use assets::AssetSupportPlugin;
use avian3d::prelude::*;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::render::{RenderPlugin, render_resource::WgpuFeatures, settings::WgpuSettings};
use bevy::window::WindowPlugin;
use bevy_ahoy::prelude::AhoyPlugins;
use bevy_enhanced_input::prelude::EnhancedInputPlugin;

use chunk_colliders::ChunkColliderPlugin;
use player::PlayerPlugin;
use terrain::TerrainPlugin;
use voxel_gi::VoxelGiPlugin;
use world_environment::WorldEnvironmentPlugin;

fn main() -> AppExit {
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "Voxel GI Lab | Sun: voxel corners (F7)".into(),
                        ..default()
                    }),
                    ..default()
                })
                .set(AssetPlugin {
                    meta_check: AssetMetaCheck::Never,
                    watch_for_changes_override: Some(true),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: WgpuSettings {
                        features: WgpuFeatures::POLYGON_MODE_LINE,
                        ..default()
                    }
                    .into(),
                    ..default()
                }),
        )
        .add_plugins((
            PhysicsPlugins::default(),
            // Uncomment the next line for physics debugging
            //PhysicsDebugPlugin,
            FrameTimeDiagnosticsPlugin::default(),
            EnhancedInputPlugin,
            AhoyPlugins::default(),
            AssetSupportPlugin,
            WorldEnvironmentPlugin,
            AppControlsPlugin,
            VoxelGiPlugin,
            TerrainPlugin,
            ChunkColliderPlugin,
            PlayerPlugin,
        ))
        .run()
}
