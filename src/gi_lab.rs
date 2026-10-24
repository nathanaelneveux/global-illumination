//! The lab's voxel GI wiring: launch settings, the debug overlay's
//! configuration, scripted validation and the gameplay action buttons.
use crate::player::{BombAtAim, PlaceEmissiveBlock};
use crate::voxel_gi::{VoxelGiDebugPlugin, VoxelGiSettings};
#[cfg(debug_assertions)]
use crate::voxel_gi::VoxelGiSystems;
use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_inspector_egui::bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};

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
        app.add_plugins(VoxelGiDebugPlugin {
            ui: !scripted() && !crate::headless_validation(),
            keyboard_toggles: !scripted(),
            terminal_stats: std::env::var("VOXEL_GI_LOG_STATS").is_ok_and(|value| value == "1"),
        });
        if !crate::headless_validation() && !scripted() {
            app.add_systems(EguiPrimaryContextPass, draw_actions);
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

fn draw_actions(
    mut contexts: EguiContexts,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut place_emissive: MessageWriter<PlaceEmissiveBlock>,
    mut bomb: MessageWriter<BombAtAim>,
) {
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    egui::Area::new("lab_actions".into())
        .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
        .interactable(cursor.visible)
        .show(context, |ui| {
            egui::Frame::NONE
                .fill(egui::Color32::from_black_alpha(185))
                .inner_margin(10)
                .corner_radius(5)
                .show(ui, |ui| {
                    if ui.button("Add emitting block (E)")
                        .on_hover_text("Place a warm emitting voxel against the aimed-at face. E works while playing; Esc releases the cursor for this button.")
                        .clicked() {
                        place_emissive.write(PlaceEmissiveBlock);
                    }
                    if ui.button("Bomb: radius 5 (B)")
                        .on_hover_text("Remove a radius-5 sphere centered on the aimed-at voxel. B detonates immediately while playing; left-click always digs a single block.")
                        .clicked() {
                        bomb.write(BombAtAim);
                    }
                });
        });
}
