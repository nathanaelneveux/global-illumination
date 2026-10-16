mod ambient_occlusion;
mod app_controls;
mod assets;
mod cave_scene;
mod canyon_scene;
mod scene_presets;
mod chunk_colliders;
mod player;
mod structures;
mod terrain;
mod terrain_meshing;
mod terrain_noise;
pub mod voxel_gi;
mod world_environment;

use app_controls::AppControlsPlugin;
use assets::AssetSupportPlugin;
use avian3d::prelude::*;
use bevy::app::ScheduleRunnerPlugin;
use bevy::asset::AssetMetaCheck;
use bevy::diagnostic::FrameTimeDiagnosticsPlugin;
use bevy::prelude::*;
use bevy::render::{RenderPlugin, render_resource::WgpuFeatures, settings::WgpuSettings};
use bevy::window::{ExitCondition, WindowPlugin};
use bevy::winit::WinitPlugin;
use bevy_ahoy::prelude::AhoyPlugins;
use bevy_enhanced_input::prelude::EnhancedInputPlugin;

use chunk_colliders::ChunkColliderPlugin;
use player::PlayerPlugin;
use terrain::TerrainPlugin;
use voxel_gi::{VoxelGiDiagnosticsPlugin, VoxelGiPlugin};
use world_environment::WorldEnvironmentPlugin;

// Keep launch provenance available in release builds too; validation-only
// metadata cannot diagnose a stale executable loading another checkout's assets.
const BUILD_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

fn main() -> AppExit {
    let headless = headless_validation();
    if headless && (!cfg!(debug_assertions) || std::env::var_os("VOXEL_GI_VALIDATE").is_none()) {
        eprintln!(
            "VOXEL_GI_VALIDATE_HEADLESS=1 requires a development build and VOXEL_GI_VALIDATE=1."
        );
        return AppExit::error();
    }
    let mut plugins = DefaultPlugins
        .set(WindowPlugin {
            primary_window: (!headless).then(|| Window {
                title: "Voxel GI Lab | Sun: voxel corners (F7)".into(),
                ..default()
            }),
            exit_condition: if headless {
                ExitCondition::DontExit
            } else {
                ExitCondition::OnAllClosed
            },
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
        });
    if headless {
        plugins = plugins.disable::<WinitPlugin>();
    }
    let mut app = App::new();
    app.add_plugins(plugins);
    info!(
        compiled_manifest_dir = BUILD_MANIFEST_DIR,
        debug_assertions = cfg!(debug_assertions),
        "GI executable identity"
    );
    if headless {
        // No OS window, swapchain, focus events, cursor capture or display-server
        // runner. Keep the regular real-time workload cadence, not a fast-forward
        // offline simulation; report these timings separately from desktop FPS.
        app.add_plugins(ScheduleRunnerPlugin::run_loop(
            std::time::Duration::from_secs_f64(1.0 / 60.0),
        ));
    }
    app.add_plugins((
        PhysicsPlugins::default(),
        FrameTimeDiagnosticsPlugin::default(),
        EnhancedInputPlugin,
        AhoyPlugins::default(),
        AssetSupportPlugin,
        WorldEnvironmentPlugin,
    ));
    if !headless {
        app.add_plugins(AppControlsPlugin);
    }
    let voxel_texture = app
        .world()
        .resource::<assets::TemplateAssets>()
        .voxel_texture
        .clone();
    // All prototype writes use VoxelGiDirty::set_voxel; coarse external changes
    // explicitly request resampling. BVW remains the terrain/mesh host.
    app.insert_resource(voxel_gi::VoxelGiDirty::tracked());
    if let Err(error) = voxel_gi::configure_demo_launch(&mut app) {
        eprintln!("Invalid GI launch configuration: {error}");
        return AppExit::error();
    }
    app.add_plugins((
        VoxelGiPlugin { voxel_texture },
        TerrainPlugin,
        ChunkColliderPlugin,
        PlayerPlugin,
        VoxelGiDiagnosticsPlugin,
    ));
    app.run()
}

pub(crate) fn headless_validation() -> bool {
    std::env::var("VOXEL_GI_VALIDATE_HEADLESS").is_ok_and(|value| value == "1")
}
