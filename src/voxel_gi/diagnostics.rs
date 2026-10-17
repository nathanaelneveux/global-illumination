//! Optional demo instrumentation, separate from transport scheduling. The HUD
//! reuses O(1) counters; expensive field summaries and scripted edits are opt-in.
mod tuning;
#[cfg(debug_assertions)]
mod face_watch;
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::window::{CursorOptions, PrimaryWindow};
use bevy_inspector_egui::bevy_egui::{EguiContexts, EguiPrimaryContextPass, egui};
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::render::VoxelTransportMaterial;
use super::{VoxelEnergyField, VoxelGiRuntime, VoxelGiStats, VoxelGiSystems};
use crate::player::PlaceEmissiveBlock;

/// Demo-only UI, keyboard comparisons and opt-in validation. Install after GI
/// and the player plugins; the reusable transport does not own player input.
pub(crate) struct VoxelGiDiagnosticsPlugin;

/// Demo boundary only. Invalid input is reported by main rather than panicking
/// inside construction of the reusable GI plugin. No process-global mutation.
pub(crate) fn configure_demo_launch(app: &mut App) -> Result<(), String> {
    let startup = parse_startup(|name| std::env::var(name).ok())?;
    app.insert_resource(startup)
        .init_resource::<super::VoxelGiSettings>();
    if std::env::var("VOXEL_GI_CONTINUOUS").is_ok_and(|v| v == "1") {
        app.world_mut()
            .resource_mut::<super::VoxelGiSettings>()
            .adaptive_updates = false;
    }
    Ok(())
}

fn parse_startup(
    mut value: impl FnMut(&str) -> Option<String>,
) -> Result<super::VoxelGiStartup, String> {
    use super::{RetentionPolicy, RetentionSummary, TransportBackend, VoxelGiStartup};
    let backend = match value("VOXEL_GI_BACKEND").as_deref() {
        None | Some("auto" | "gpu") => TransportBackend::Auto,
        Some("cpu") => TransportBackend::Cpu,
        Some("resident") => TransportBackend::Resident,
        Some("sync") => TransportBackend::Synchronous,
        Some(other) => {
            return Err(format!(
                "unknown VOXEL_GI_BACKEND={other:?}; choose auto, gpu, resident, sync or cpu"
            ));
        }
    };
    let summary = RetentionSummary::launch(value("VOXEL_GI_RETENTION_SUMMARY").as_deref())?;
    let retention = if super::retention::launch_enabled(value("VOXEL_GI_RETENTION").as_deref())? {
        RetentionPolicy::WholeRange(summary)
    } else {
        RetentionPolicy::Legacy
    };
    Ok(VoxelGiStartup { backend, retention })
}

#[cfg(test)]
mod launch_tests {
    use super::*;
    #[test]
    fn typed_launch_preserves_defaults_aliases_and_rejects_typos() {
        assert_eq!(
            parse_startup(|_| None).unwrap(),
            super::super::VoxelGiStartup::default()
        );
        assert_eq!(
            parse_startup(|key| (key == "VOXEL_GI_BACKEND").then(|| "cpu".into()))
                .unwrap()
                .backend,
            super::super::TransportBackend::Cpu
        );
        for alias in ["auto", "gpu"] {
            assert_eq!(
                parse_startup(|key| (key == "VOXEL_GI_BACKEND").then(|| alias.into()))
                    .unwrap()
                    .backend,
                super::super::TransportBackend::Auto
            );
        }
        assert_eq!(
            parse_startup(|key| (key == "VOXEL_GI_BACKEND").then(|| "sync".into()))
                .unwrap()
                .backend,
            super::super::TransportBackend::Synchronous
        );
        for key in [
            "VOXEL_GI_BACKEND",
            "VOXEL_GI_RETENTION",
            "VOXEL_GI_RETENTION_SUMMARY",
        ] {
            assert!(parse_startup(|k| (k == key).then(|| "typo".into())).is_err());
        }
    }
}

impl Plugin for VoxelGiDiagnosticsPlugin {
    fn build(&self, app: &mut App) {
        tuning::load_launch_settings(app);
        #[cfg(debug_assertions)]
        face_watch::install(app);
        app.init_resource::<super::VoxelGiVisualWeights>()
            .add_systems(Update, tuning::update_visual_weights.before(VoxelGiSystems));
        if terminal_stats_enabled() {
            app.add_plugins(LogDiagnosticsPlugin::default())
                .add_systems(Update, log_voxel_gi_stats.after(VoxelGiSystems));
        }
        let scripted = cfg!(debug_assertions) && std::env::var_os("VOXEL_GI_VALIDATE").is_some();
        if !scripted {
            app.add_systems(Update, toggle_voxel_gi.after(VoxelGiSystems));
        }
        if !crate::headless_validation() {
            app.init_resource::<tuning::TuningPanel>().add_systems(
                EguiPrimaryContextPass,
                (draw_hud, tuning::draw_panel).chain(),
            );
        }
        #[cfg(debug_assertions)]
        if scripted {
            super::validation::install_frame_timing(app);
            super::validation::install_reference_image(app);
            app.init_resource::<super::validation::CaptureEvidence>()
                .add_systems(PostStartup, super::validation::setup_capture_target)
                .add_systems(
                    Update,
                    super::validation::run_validation.after(VoxelGiSystems),
                );
        }
    }
}

pub(super) fn terminal_stats_enabled() -> bool {
    // Launch configuration, not a hot-path environment lookup on every page.
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("VOXEL_GI_LOG_STATS").is_ok_and(|value| value == "1"))
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
    mut tuning: ResMut<tuning::TuningPanel>,
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
                    ).on_hover_text(
                        "Coverage = faces with samples; bootstrap = progress toward 8 samples.\n\
                         Settled = faces passing the current scene's adaptive stability checks (all energy bands), not convergence or idle. Edits can reset this without clearing history; settled faces still receive audits.\n\
                         GI main/GPU = average milliseconds per app frame over 250 ms; they overlap, so do not add them. Main covers GI main-world systems, not background workers or render-world uploads/drawing. GPU covers completed transport/sky compute passes, not transfers or queue waits. — means no measurement or unavailable timestamps."
                    );
                    if ui.button("Add emitting block (E)")
                        .on_hover_text("Place a warm emitting voxel against the aimed-at face. E works while playing; Esc releases the cursor for this button.")
                        .clicked() {
                        place_emissive.write(PlaceEmissiveBlock);
                    }
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
    material_handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    mut windows: Query<&mut Window, With<bevy::window::PrimaryWindow>>,
) {
    // Scripted validation owns presentation as well as the player path.
    #[cfg(debug_assertions)]
    if std::env::var_os("VOXEL_GI_VALIDATE").is_some() {
        return;
    }
    if keyboard.just_pressed(KeyCode::F7)
        && let Some(handle) = material_handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.voxel_sky ^= 1;
        info!(
            voxel_sky = material.extension.settings.voxel_sky != 0,
            "direct sky lighting changed (F5 controls cached face/corner blending)"
        );
    }
    if keyboard.just_pressed(KeyCode::F6)
        && let Some(handle) = material_handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.debug_irradiance ^= 1;
        info!(
            irradiance_only = material.extension.settings.debug_irradiance != 0,
            "voxel GI diagnostic view changed"
        );
    }
    if keyboard.just_pressed(KeyCode::F5)
        && let Some(handle) = material_handle.as_ref()
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        material.extension.settings.reconstruction ^= 1;
        info!(
            smooth_corners = material.extension.settings.reconstruction != 0,
            "voxel GI reconstruction changed"
        );
    }
    if (keyboard.just_pressed(KeyCode::F5) || keyboard.just_pressed(KeyCode::F7))
        && let Some(handle) = material_handle.as_ref()
        && let Some(material) = materials.get(&handle.handle)
    {
        let settings = material.extension.settings;
        let sky_mode = if settings.voxel_sky == 0 {
            "per-pixel"
        } else if settings.reconstruction != 0 {
            "voxel corners"
        } else {
            "voxel faces"
        };
        for mut window in &mut windows {
            window.title = format!("Voxel GI Lab | Sun: {sky_mode} (F7)");
        }
    }
    if !keyboard.just_pressed(KeyCode::F4) {
        return;
    }
    if let Some(handle) = material_handle
        && let Some(mut material) = materials.get_mut(&handle.handle)
    {
        let settings = &mut material.extension.settings;
        settings.enabled = u32::from(settings.enabled == 0);
        info!(
            enabled = settings.enabled != 0,
            "world-space voxel GI toggled"
        );
    }
}

pub(super) fn log_voxel_gi_stats(
    mut stats: ResMut<VoxelGiStats>,
    field: Res<VoxelEnergyField>,
    runtime: Res<VoxelGiRuntime>,
    mut last_log: Local<Option<Instant>>,
) {
    let now = Instant::now();
    if last_log.is_some_and(|last| now.duration_since(last) < Duration::from_secs(5)) {
        return;
    }
    *last_log = Some(now);
    // Diagnostic summaries must not stall the workload they measure. Sample a
    // bounded, evenly spaced subset of the live key array; transport/freshness
    // counters above remain exact and independent of this optional estimate.
    let (mean, oldest) = sampled_field_history(&field, now);
    stats.mean_history_samples = mean;
    stats.oldest_sample_age_ms = oldest.as_secs_f32() * 1000.0;
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
        dispatch_budget = runtime.batch_size,
        sky_budget = runtime.sky_batch_size,
        worker_ms = stats.last_worker_ms,
        selection_ms = stats.last_selection_ms,
        merge_ms = stats.last_merge_ms,
        prepare_ms = stats.last_prepare_ms,
        snapshot_latency_ms = stats.last_snapshot_latency_ms,
        decode_ms = stats.last_decode_ms,
        packed_geometry_bytes = stats.packed_geometry_bytes,
        fully_bootstrapped = stats.fully_bootstrapped,
        sampled_mean_history = stats.mean_history_samples,
        sampled_oldest_ms = stats.oldest_sample_age_ms,
        backend = stats.backend,
        "voxel GI status"
    );
}

fn sampled_field_history(field: &VoxelEnergyField, now: Instant) -> (f32, Duration) {
    let mut samples = 0_u64;
    let mut faces = 0_u64;
    let mut oldest = Duration::ZERO;
    for voxel in field
        .order
        .iter()
        .step_by(field.order.len().div_ceil(1024).max(1))
    {
        let Some(probe) = field.probes.get(voxel) else {
            continue;
        };
        for face in 0..6 {
            if probe.face_mask & (1 << face) != 0 {
                samples += u64::from(probe.samples[face]);
                faces += 1;
                if let Some(updated) = probe.face_updated_at[face].or(probe.updated_at) {
                    oldest = oldest.max(now.saturating_duration_since(updated));
                }
            }
        }
    }
    (samples as f32 / faces.max(1) as f32, oldest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyboard_toggle_uses_material_state_without_a_runtime_mirror() {
        use super::super::render::{VoxelTransportAssets, material_template};
        use bevy::ecs::system::RunSystemOnce;
        let mut world = World::new();
        let mut materials = Assets::<VoxelTransportMaterial>::default();
        let handle = materials.add(material_template(&VoxelTransportAssets {
            resident: default(),
            texture: default(),
            probes: default(),
            hash_entries: default(),
        }));
        // An external caller, not the keyboard or panel, changed presentation.
        materials
            .get_mut(&handle)
            .unwrap()
            .extension
            .settings
            .enabled = 0;
        world.insert_resource(materials);
        world.insert_resource(VoxelWorldMaterialHandle {
            handle: handle.clone(),
        });
        let mut keyboard = ButtonInput::default();
        keyboard.press(KeyCode::F4);
        world.insert_resource(keyboard);
        for expected in [1, 0] {
            world.run_system_once(toggle_voxel_gi).unwrap();
            assert_eq!(
                world
                    .resource::<Assets<VoxelTransportMaterial>>()
                    .get(&handle)
                    .unwrap()
                    .extension
                    .settings
                    .enabled,
                expected
            );
        }
    }

    #[test]
    fn optional_history_summary_is_bounded_and_explicitly_sampled() {
        let now = Instant::now();
        let mut field = VoxelEnergyField::default();
        assert_eq!(sampled_field_history(&field, now), (0.0, Duration::ZERO));
        for x in 0..2048 {
            let voxel = IVec3::new(x, 0, 0);
            let mut probe = super::super::world::SurfaceProbe::new(voxel, 4);
            probe.samples[2] = if x % 2 == 0 { 32 } else { 500 };
            probe.face_updated_at[2] =
                Some(now - Duration::from_secs(if x % 2 == 0 { 1 } else { 10 }));
            field.probes.insert(voxel, probe);
            field.order.push(voxel);
        }
        assert_eq!(
            sampled_field_history(&field, now),
            (32.0, Duration::from_secs(1))
        );
    }

    #[test]
    fn hud_uses_coverage_and_bootstrap_not_history_or_convergence() {
        let stats = VoxelGiStats {
            surface_probes: 100,
            interactive_coverage: 0.75,
            bootstrap_coverage: 0.325,
            mean_history_samples: 200.0,
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
