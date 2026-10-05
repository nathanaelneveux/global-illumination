//! HUD counters reuse existing stats; editing is queued through the player path.
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_inspector_egui::bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};

use super::VoxelGiStats;
use crate::player::PlaceEmissiveBlock;

pub(super) fn install(app: &mut App) {
    if terminal_stats_enabled() {
        app.add_plugins(LogDiagnosticsPlugin::default());
    }
    if !crate::headless_validation() {
        app.add_systems(EguiPrimaryContextPass, draw_hud);
    }
}

pub(super) fn terminal_stats_enabled() -> bool {
    std::env::var("VOXEL_GI_LOG_STATS").is_ok_and(|value| value == "1")
        || std::env::var_os("VOXEL_GI_VALIDATE").is_some()
}

struct HudState {
    visible: bool,
    next_update: f64,
    text: String,
}

impl Default for HudState {
    fn default() -> Self {
        Self {
            // Keep reference captures unobstructed, and don't let overlay text
            // make an otherwise blank validation screenshot appear valid.
            visible: std::env::var_os("VOXEL_GI_VALIDATE").is_none(),
            next_update: 0.0,
            text: String::new(),
        }
    }
}

fn draw_hud(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    stats: Res<VoxelGiStats>,
    time: Res<Time<Real>>,
    mut state: Local<HudState>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut place_emissive: MessageWriter<PlaceEmissiveBlock>,
) {
    if !state.visible {
        return;
    }
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    if time.elapsed_secs_f64() >= state.next_update {
        let fps = diagnostics
            .get(&FrameTimeDiagnosticsPlugin::FPS)
            .and_then(|fps| fps.smoothed());
        state.text = hud_text(&stats, fps);
        state.next_update = time.elapsed_secs_f64() + 0.25;
    }
    egui::Area::new("voxel_gi_hud".into())
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .interactable(cursor.visible)
        .show(context, |ui| {
            egui::Frame::NONE
                .fill(egui::Color32::from_black_alpha(185))
                .inner_margin(10)
                .corner_radius(5)
                .show(ui, |ui| {
                    ui.label(
                        egui::RichText::new(&state.text)
                            .monospace()
                            .size(16.0)
                            .color(egui::Color32::WHITE),
                    );
                    if ui.button("Add emitting block (E)")
                        .on_hover_text("Place a warm emitting voxel against the aimed-at face. E works while playing; Esc releases the cursor for this button.")
                        .clicked() {
                        place_emissive.write(PlaceEmissiveBlock);
                    }
                });
        });
}

fn hud_text(stats: &VoxelGiStats, fps: Option<f64>) -> String {
    let fps = fps
        .filter(|fps| fps.is_finite() && *fps >= 0.0)
        .map_or_else(|| "—".into(), |fps| format!("{fps:.0}"));
    if stats.surface_probes == 0 {
        return format!("FPS:       {fps}\nCoverage:  —\nBootstrap: —");
    }
    format!(
        "FPS:       {fps}\nCoverage:  {:.1}%\nBootstrap: {:.1}%",
        stats.interactive_coverage.clamp(0.0, 1.0) * 100.0,
        stats.bootstrap_coverage.clamp(0.0, 1.0) * 100.0
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_uses_coverage_and_bootstrap_not_history_or_convergence() {
        let stats = VoxelGiStats {
            surface_probes: 100,
            interactive_coverage: 0.75,
            bootstrap_coverage: 0.325,
            mean_history_samples: 200.0,
            ..default()
        };
        assert_eq!(
            hud_text(&stats, Some(59.7)),
            "FPS:       60\nCoverage:  75.0%\nBootstrap: 32.5%"
        );
    }

    #[test]
    fn empty_world_and_missing_fps_do_not_look_complete() {
        let stats = VoxelGiStats {
            interactive_coverage: 1.0,
            bootstrap_coverage: 1.0,
            ..default()
        };
        assert_eq!(
            hud_text(&stats, None),
            "FPS:       —\nCoverage:  —\nBootstrap: —"
        );
        assert_eq!(hud_text(&stats, Some(f64::NAN)), hud_text(&stats, None));
    }
}
