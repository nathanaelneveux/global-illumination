//! The lab's voxel GI wiring: launch settings, the debug overlay's
//! configuration, scripted validation and the gameplay action buttons.
use crate::player::{BombAtAim, PlaceEmissiveBlock};
use crate::voxel_gi::{VoxelGiDebugPlugin, VoxelGiSettings};
#[cfg(debug_assertions)]
use crate::voxel_gi::VoxelGiSystems;
use bevy::feathers::controls::FeathersButton;
use bevy::prelude::*;
use bevy::ui_widgets::Activate;

/// Add before `VoxelGiPlugin`: launch settings must be in place first.
pub struct GiLabPlugin;

fn scripted() -> bool {
    cfg!(debug_assertions) && std::env::var_os("VOXEL_GI_VALIDATE").is_some()
}

impl Plugin for GiLabPlugin {
    fn build(&self, app: &mut App) {
        load_launch_settings(app);
        // Scripted validation owns presentation and keeps captures free of
        // overlay text; headless runs have no window for the UI.
        // VOXEL_GI_VALIDATE_DEBUG_UI=1 shows the overlay, panel open, in
        // validation captures (checks the UI itself, not lighting).
        let debug_ui_capture = std::env::var("VOXEL_GI_VALIDATE_DEBUG_UI").is_ok_and(|value| value == "1");
        let ui = (!scripted() && !crate::headless_validation()) || debug_ui_capture;
        app.add_plugins(VoxelGiDebugPlugin {
            ui,
            keyboard_toggles: !scripted(),
            terminal_stats: std::env::var("VOXEL_GI_LOG_STATS").is_ok_and(|value| value == "1"),
            panel_open: debug_ui_capture,
        });
        if ui {
            app.add_systems(Startup, spawn_actions);
        }
        #[cfg(debug_assertions)]
        if scripted() {
            crate::validation::install_frame_timing(app);
            app.init_resource::<crate::validation::CaptureEvidence>()
                .add_systems(PostStartup, crate::validation::setup_capture_target)
                .add_systems(Update, crate::validation::run_validation.after(VoxelGiSystems));
        }
    }
}

/// `VOXEL_GI_SETTINGS=path.ron` replaces the default settings at launch.
fn load_launch_settings(app: &mut App) {
    let Some(path) = std::env::var_os("VOXEL_GI_SETTINGS") else {
        app.init_resource::<VoxelGiSettings>();
        return;
    };
    let settings = std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|text| crate::voxel_gi::parse_settings(&text))
        .unwrap_or_else(|error| panic!("Cannot load VOXEL_GI_SETTINGS={path:?}: {error}"));
    // Fail visibly on invalid files rather than benchmarking a silent fallback.
    app.insert_resource(settings);
}

#[derive(Component, Clone, Copy, Default)]
enum LabAction {
    #[default]
    PlaceEmitter,
    Bomb,
}

fn action_button(action: LabAction, caption: &'static str) -> impl Scene {
    bsn! {
        @FeathersButton
        template_value(action)
        on(|activate: On<Activate>,
            actions: Query<&LabAction>,
            mut place: MessageWriter<PlaceEmissiveBlock>,
            mut bomb: MessageWriter<BombAtAim>| {
            match actions.get(activate.entity) {
                Ok(LabAction::PlaceEmitter) => { place.write(PlaceEmissiveBlock); }
                Ok(LabAction::Bomb) => { bomb.write(BombAtAim); }
                Err(_) => {}
            }
        })
        Children [ (Text(caption) TextFont { font_size: FontSize::Px(13.0) }) ]
    }
}

/// Gameplay shortcuts (E and B while playing; Esc frees the cursor).
fn spawn_actions(mut commands: Commands) {
    commands.spawn_scene(bsn! {
        Node {
            position_type: PositionType::Absolute,
            right: px(12),
            bottom: px(12),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: UiRect::all(px(10)),
            border_radius: BorderRadius::all(px(5)),
        }
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.72))
        Children [
            action_button(LabAction::PlaceEmitter, "Add emitting block (E)"),
            action_button(LabAction::Bomb, "Bomb: radius 5 (B)"),
        ]
    });
}
