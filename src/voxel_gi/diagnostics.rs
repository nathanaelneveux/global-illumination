//! Opt-in debug tooling on GI's public API: a HUD overlay, the lighting
//! controls panel, keyboard presentation toggles and periodic terminal stats.
//! Nothing here runs unless `VoxelGiDebugPlugin` is added.
mod tuning;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_inspector_egui::bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use std::time::{Duration, Instant};

use super::{VoxelGiStats, VoxelGiSystems};

/// Debug overlay and controls for voxel GI.
pub struct VoxelGiDebugPlugin {
    /// The HUD overlay and the "Lighting controls" panel (egui).
    pub ui: bool,
    /// F4 GI on/off, F5 corner reconstruction, F6 irradiance view, F7 sun mode.
    pub keyboard_toggles: bool,
    /// An `info!` status line every five seconds.
    pub terminal_stats: bool,
}

impl Default for VoxelGiDebugPlugin {
    fn default() -> Self {
        Self {
            ui: true,
            keyboard_toggles: true,
            terminal_stats: false,
        }
    }
}

impl Plugin for VoxelGiDebugPlugin {
    fn build(&self, app: &mut App) {
        if self.terminal_stats {
            app.add_plugins(LogDiagnosticsPlugin::default())
                .add_systems(Update, log_voxel_gi_stats.after(VoxelGiSystems));
        }
        if self.keyboard_toggles {
            app.add_systems(Update, toggle_voxel_gi.after(VoxelGiSystems));
        }
        if self.ui {
            app.init_resource::<tuning::TuningPanel>().add_systems(
                EguiPrimaryContextPass,
                (draw_hud, tuning::draw_panel).chain(),
            );
        }
    }
}

/// Parse a RON `VoxelGiSettings` (missing fields default; values are clamped).
pub fn parse_settings(text: &str) -> Result<super::VoxelGiSettings, String> {
    tuning::parse_settings(text)
}

#[derive(Default)]
struct HudState {
    next_update: f64,
    text: String,
}

fn draw_hud(
    mut contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    stats: Res<VoxelGiStats>,
    time: Res<Time<Real>>,
    mut state: Local<HudState>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
    mut tuning: ResMut<tuning::TuningPanel>,
) {
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
                    ).on_hover_text(
                        "Coverage = faces with samples; bootstrap = progress toward 8 samples.\n\
                         Settled = faces passing the current scene's adaptive stability checks, not convergence or idle. Edits can reset this without clearing history; settled faces still receive audits.\n\
                         GI main/GPU = average milliseconds per app frame over 250 ms; they overlap, so do not add them. Main covers GI main-world systems, not background workers or render-world uploads/drawing. GPU covers completed transport/sky compute passes, not transfers or queue waits. — means no measurement or unavailable timestamps."
                    );
                    if ui.button("Lighting controls…")
                        .on_hover_text("Esc releases the cursor. Compare lighting reach, quality, work budget and presentation.")
                        .clicked() {
                        tuning.open = !tuning.open;
                    }
                });
        });
}

fn hud_text(stats: &VoxelGiStats, fps: Option<f64>) -> String {
    let fps = fps
        .filter(|fps| fps.is_finite() && *fps >= 0.0)
        .map_or_else(|| "—".into(), |fps| format!("{fps:.0}"));
    if stats.surface_probes == 0 {
        return format!(
            "FPS:       {fps}\nCoverage:  —\nBootstrap: —\nSettled:   —{}",
            hud_cost_text(stats)
        );
    }
    format!(
        "FPS:       {fps}\nCoverage:  {:.1}%\nBootstrap: {:.1}%\nSettled:   {:.1}%{}",
        stats.interactive_coverage.clamp(0.0, 1.0) * 100.0,
        stats.bootstrap_coverage.clamp(0.0, 1.0) * 100.0,
        stats.settled_coverage.clamp(0.0, 1.0) * 100.0,
        hud_cost_text(stats)
    )
}

fn hud_cost_text(stats: &VoxelGiStats) -> String {
    let cost = |value: Option<f32>| {
        value
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map_or_else(|| "—".into(), |value| format!("{value:.2} ms/frame"))
    };
    format!(
        "\nGI main:   {}\nGI GPU:    {}",
        cost(stats.main_thread_ms_per_frame),
        cost(stats.gpu_ms_per_frame)
    )
}

pub(super) fn toggle_voxel_gi(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut gi: ResMut<super::VoxelGiDisplay>,
    mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
) {
    if keyboard.just_pressed(KeyCode::F7) {
        gi.voxel_sun = !gi.voxel_sun;
        info!(voxel_sky = gi.voxel_sun, "direct sky lighting changed (F5 controls cached face/corner blending)");
    }
    if keyboard.just_pressed(KeyCode::F6) {
        gi.debug_view = if gi.debug_view == super::GiDebugView::None {
            super::GiDebugView::Irradiance
        } else {
            super::GiDebugView::None
        };
        info!(irradiance_only = gi.debug_view != super::GiDebugView::None, "voxel GI diagnostic view changed");
    }
    if keyboard.just_pressed(KeyCode::F5) {
        gi.corner_reconstruction = !gi.corner_reconstruction;
        info!(smooth_corners = gi.corner_reconstruction, "voxel GI reconstruction changed");
    }
    if keyboard.just_pressed(KeyCode::F5) || keyboard.just_pressed(KeyCode::F7) {
        let sky_mode = if !gi.voxel_sun {
            "per-pixel"
        } else if gi.corner_reconstruction {
            "voxel corners"
        } else {
            "voxel faces"
        };
        for mut window in &mut windows {
            window.title = format!("Voxel GI Lab | Sun: {sky_mode} (F7)");
        }
    }
    if keyboard.just_pressed(KeyCode::F4) {
        gi.enabled = !gi.enabled;
        info!(enabled = gi.enabled, "world-space voxel GI toggled");
    }
}

pub(super) fn log_voxel_gi_stats(stats: Res<VoxelGiStats>, mut last_log: Local<Option<Instant>>) {
    let now = Instant::now();
    if last_log.is_some_and(|last| now.duration_since(last) < Duration::from_secs(5)) {
        return;
    }
    *last_log = Some(now);
    info!(
        loaded_pages = stats.loaded_pages,
        surface_probes = stats.surface_probes,
        coverage_percent = format_args!("{:.1}", stats.interactive_coverage * 100.0),
        bootstrap_percent = format_args!("{:.1}", stats.bootstrap_coverage * 100.0),
        settled_percent = format_args!("{:.1}", stats.settled_coverage * 100.0),
        main_ms_per_frame = stats.main_thread_ms_per_frame,
        gpu_ms_per_frame = stats.gpu_ms_per_frame,
        pending_probes = stats.pending_probes,
        traced_paths = stats.traced_paths,
        refresh_percent = format_args!("{:.1}", stats.global_refresh * 100.0),
        last_trace_ms = format_args!("{:.1}", stats.last_trace_batch_ms),
        sky_requests = stats.sky_corner_requests,
        sky_traced = stats.sky_corners_traced,
        sky_rays_scheduled = stats.sky_rays_scheduled,
        sky_cached = stats.sky_cached_corners,
        sky_batches = stats.sky_only_batches,
        transport_batches = stats.transport_batches,
        idle_scheduler_frames = stats.idle_scheduler_frames,
        dispatch_budget = stats.dispatch_budget,
        sky_budget = stats.sky_dispatch_budget,
        worker_ms = stats.last_worker_ms,
        selection_ms = stats.last_selection_ms,
        merge_ms = stats.last_merge_ms,
        prepare_ms = stats.last_prepare_ms,
        snapshot_latency_ms = stats.last_snapshot_latency_ms,
        decode_ms = stats.last_decode_ms,
        packed_geometry_bytes = stats.packed_geometry_bytes,
        fully_bootstrapped = stats.fully_bootstrapped,
        backend = stats.backend,
        "voxel GI status"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_toggle_flips_the_current_display_state() {
        use bevy::ecs::system::RunSystemOnce;
        let mut world = World::new();
        // An external caller, not the keyboard or panel, changed presentation.
        world.insert_resource(super::super::VoxelGiDisplay {
            enabled: false,
            ..default()
        });
        let mut keyboard = ButtonInput::default();
        keyboard.press(KeyCode::F4);
        world.insert_resource(keyboard);
        for expected in [true, false] {
            world.run_system_once(toggle_voxel_gi).unwrap();
            assert_eq!(world.resource::<super::super::VoxelGiDisplay>().enabled, expected);
        }
    }

    #[test]
    fn hud_uses_coverage_and_bootstrap_not_history_or_convergence() {
        let stats = VoxelGiStats {
            surface_probes: 100,
            interactive_coverage: 0.75,
            bootstrap_coverage: 0.325,
            settled_coverage: 0.125,
            main_thread_ms_per_frame: Some(1.25),
            gpu_ms_per_frame: Some(0.375),
            ..default()
        };
        assert_eq!(
            hud_text(&stats, Some(59.7)),
            "FPS:       60\nCoverage:  75.0%\nBootstrap: 32.5%\nSettled:   12.5%\nGI main:   1.25 ms/frame\nGI GPU:    0.38 ms/frame"
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
            "FPS:       —\nCoverage:  —\nBootstrap: —\nSettled:   —\nGI main:   —\nGI GPU:    —"
        );
        assert_eq!(hud_text(&stats, Some(f64::NAN)), hud_text(&stats, None));
    }

    #[test]
    fn unavailable_and_zero_work_have_distinct_labels() {
        let stats = VoxelGiStats {
            main_thread_ms_per_frame: Some(0.0),
            gpu_ms_per_frame: Some(f32::NAN),
            ..default()
        };
        assert_eq!(
            hud_cost_text(&stats),
            "\nGI main:   0.00 ms/frame\nGI GPU:    —"
        );
    }
}
