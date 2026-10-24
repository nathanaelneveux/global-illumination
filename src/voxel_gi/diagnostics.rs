//! Opt-in debug tooling on GI's public API: a HUD, the lighting controls
//! panel (Bevy UI with feathers widgets), keyboard presentation toggles and
//! periodic terminal stats. Nothing here runs unless `VoxelGiDebugPlugin` is
//! added.
use bevy::clipboard::{Clipboard, ClipboardRead};
use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::feathers::{
    FeathersPlugins,
    controls::{FeathersButton, FeathersCheckbox, FeathersSlider},
    dark_theme::create_dark_theme,
    theme::UiTheme,
};
use bevy::prelude::*;
use bevy::ui::Checked;
use bevy::ui_widgets::{Activate, SliderPrecision, SliderStep, SliderValue, ValueChange};
use std::time::{Duration, Instant};

use super::{GiDebugView, VoxelGiDisplay, VoxelGiQuality, VoxelGiSettings, VoxelGiStats, VoxelGiSystems};

/// Debug overlay and controls for voxel GI.
pub struct VoxelGiDebugPlugin {
    /// The HUD and the "Lighting controls" panel.
    pub ui: bool,
    /// F4 GI on/off, F5 corner reconstruction, F6 irradiance view, F7 sun mode.
    pub keyboard_toggles: bool,
    /// An `info!` status line every five seconds.
    pub terminal_stats: bool,
    /// Start with the lighting controls panel open.
    pub panel_open: bool,
}

impl Default for VoxelGiDebugPlugin {
    fn default() -> Self {
        Self {
            ui: true,
            keyboard_toggles: true,
            terminal_stats: false,
            panel_open: false,
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
            if !app.is_plugin_added::<bevy::feathers::FeathersCorePlugin>() {
                app.add_plugins(FeathersPlugins);
            }
            app.insert_resource(UiTheme(create_dark_theme()))
                .insert_resource(PanelOpen(self.panel_open))
                .init_resource::<PanelState>()
                .add_systems(Startup, spawn_ui)
                .add_systems(Update, (update_hud, sync_controls, finish_paste))
                .add_observer(on_slider)
                .add_observer(on_checkbox)
                .add_observer(on_button);
        }
    }
}

/// A RON `VoxelGiSettings` (missing fields default; values are clamped).
pub fn parse_settings(text: &str) -> Result<VoxelGiSettings, String> {
    VoxelGiSettings::from_ron(text)
}

/// A slider bound to one setting.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq)]
enum Knob {
    #[default]
    GpuShare,
    CacheRadius,
    ViewReceivers,
    CacheShare,
    Spatial,
    Evidence,
    Response,
    Gain,
    Reach,
    Bounces,
}

impl Knob {
    fn label(self) -> &'static str {
        match self {
            Self::GpuShare => "GI GPU share",
            Self::CacheRadius => "World cache radius",
            Self::ViewReceivers => "Visible receivers (k)",
            Self::CacheShare => "Cache share of budget",
            Self::Spatial => "Spatial smoothing",
            Self::Evidence => "Smoothing target",
            Self::Response => "Temporal response (ms)",
            Self::Gain => "Indirect gain",
            Self::Reach => "Light reach (voxels)",
            Self::Bounces => "Bounces",
        }
    }

    /// Range, step and displayed decimals.
    fn range(self) -> (f32, f32, f32, i32) {
        match self {
            Self::GpuShare => (0.05, 0.5, 0.01, 2),
            Self::CacheRadius => (0.0, 128.0, 8.0, 0),
            Self::ViewReceivers => (8.0, 128.0, 8.0, 0),
            Self::CacheShare => (0.1, 0.5, 0.05, 2),
            Self::Spatial => (0.0, 1.0, 0.05, 2),
            Self::Evidence => (0.0, 4096.0, 128.0, 0),
            Self::Response => (10.0, 500.0, 5.0, 0),
            Self::Gain => (0.0, 3.0, 0.05, 2),
            Self::Reach => (16.0, 768.0, 16.0, 0),
            Self::Bounces => (1.0, 3.0, 1.0, 0),
        }
    }

    fn get(self, settings: &VoxelGiSettings, display: &VoxelGiDisplay, panel: &PanelState) -> f32 {
        let model = panel.draft.unwrap_or(settings.transport);
        match self {
            Self::GpuShare => settings.gpu_time_share,
            Self::CacheRadius => settings.cache_radius,
            Self::ViewReceivers => settings.view_receivers as f32 / 1024.0,
            Self::CacheShare => settings.cache_share,
            Self::Spatial => settings.spatial_reconstruction,
            Self::Evidence => settings.spatial_evidence_target as f32,
            Self::Response => settings.display_response.as_secs_f32() * 1000.0,
            Self::Gain => display.intensity,
            Self::Reach => model.reach,
            Self::Bounces => model.bounces as f32,
        }
    }

    fn set(self, value: f32, settings: &mut VoxelGiSettings, display: &mut VoxelGiDisplay, panel: &mut PanelState) {
        match self {
            Self::GpuShare => settings.gpu_time_share = value,
            Self::CacheRadius => settings.cache_radius = value,
            Self::ViewReceivers => settings.view_receivers = (value * 1024.0) as u32,
            Self::CacheShare => settings.cache_share = value,
            Self::Spatial => settings.spatial_reconstruction = value,
            Self::Evidence => settings.spatial_evidence_target = value as u32,
            Self::Response => settings.display_response = Duration::from_secs_f32(value * 0.001),
            Self::Gain => display.intensity = value,
            // Changing the light model restarts every measurement: stage it.
            Self::Reach => panel.draft.get_or_insert(settings.transport).reach = value,
            Self::Bounces => panel.draft.get_or_insert(settings.transport).bounces = value as u32,
        }
    }
}

/// A checkbox bound to one presentation switch.
#[derive(Component, Clone, Copy, Default, PartialEq, Eq)]
enum Toggle {
    #[default]
    Enabled,
    Corners,
    VoxelSun,
    Irradiance,
}

impl Toggle {
    fn get(self, display: &VoxelGiDisplay) -> bool {
        match self {
            Self::Enabled => display.enabled,
            Self::Corners => display.corner_reconstruction,
            Self::VoxelSun => display.voxel_sun,
            Self::Irradiance => display.debug_view != GiDebugView::None,
        }
    }

    fn set(self, value: bool, display: &mut VoxelGiDisplay) {
        match self {
            Self::Enabled => display.enabled = value,
            Self::Corners => display.corner_reconstruction = value,
            Self::VoxelSun => display.voxel_sun = value,
            Self::Irradiance => {
                display.debug_view = if value { GiDebugView::Irradiance } else { GiDebugView::None }
            }
        }
    }
}

#[derive(Component, Clone, Copy, Default, PartialEq, Eq)]
enum Action {
    #[default]
    TogglePanel,
    Quality(VoxelGiQuality),
    ApplyModel,
    DiscardModel,
    Copy,
    Paste,
}

#[derive(Resource)]
struct PanelOpen(bool);

#[derive(Resource, Default)]
struct PanelState {
    /// Staged light model (reach, bounces), applied explicitly.
    draft: Option<super::TransportSettings>,
    paste: Option<ClipboardRead>,
}

#[derive(Component, Clone, Default)]
struct HudText;
#[derive(Component, Clone, Default)]
struct Panel;
#[derive(Component, Clone, Default)]
struct StatusText;
/// Names the preset the current settings match.
#[derive(Component, Clone, Copy, Default)]
struct QualityLabel;

fn heading(text: &'static str) -> impl Scene {
    bsn! {
        Node { margin: UiRect::top(px(8)) }
        Children [ (Text(text) TextFont { font_size: FontSize::Px(13.0) } TextColor(Color::srgb(0.65, 0.7, 0.8))) ]
    }
}

fn knob_row(knob: Knob, settings: &VoxelGiSettings) -> impl Scene {
    let (min, max, step, precision) = knob.range();
    let value = knob.get(settings, &VoxelGiDisplay::default(), &PanelState::default());
    bsn! {
        Node { flex_direction: FlexDirection::Row, align_items: AlignItems::Center, column_gap: px(8) }
        Children [
            (Node { width: px(170) } Children [ (Text({knob.label()}) TextFont { font_size: FontSize::Px(13.0) }) ]),
            (
                @FeathersSlider { @value: value, @min: min, @max: max }
                Node { width: px(180) }
                SliderStep(step)
                SliderPrecision(precision)
                template_value(knob)
            ),
        ]
    }
}

fn toggle_row(toggle: Toggle, caption: &'static str) -> impl Scene {
    bsn! {
        @FeathersCheckbox
        template_value(toggle)
        Children [ (Text(caption) TextFont { font_size: FontSize::Px(13.0) }) ]
    }
}

fn button(action: Action, caption: &'static str) -> impl Scene {
    bsn! {
        @FeathersButton
        template_value(action)
        Children [ (Text(caption) TextFont { font_size: FontSize::Px(13.0) }) ]
    }
}

fn spawn_ui(mut commands: Commands, settings: Res<VoxelGiSettings>, open: Res<PanelOpen>) {
    let s = &*settings;
    let panel_display = if open.0 { Display::Flex } else { Display::None };
    commands.spawn_scene(bsn! {
        Node {
            position_type: PositionType::Absolute,
            right: px(12),
            top: px(12),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: UiRect::all(px(10)),
            border_radius: BorderRadius::all(px(5)),
        }
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.72))
        Children [
            (Text("") TextFont { font_size: FontSize::Px(15.0) } HudText),
            button(Action::TogglePanel, "Lighting controls"),
        ]
    });
    commands.spawn_scene(bsn! {
        Node {
            display: panel_display,
            position_type: PositionType::Absolute,
            left: px(12),
            top: px(80),
            width: px(400),
            flex_direction: FlexDirection::Column,
            row_gap: px(6),
            padding: UiRect::all(px(12)),
            border_radius: BorderRadius::all(px(6)),
        }
        BackgroundColor(Color::srgba(0.08, 0.09, 0.11, 0.94))
        Panel
        Children [
            heading("Quality preset (performance knobs only)"),
            (
                Node { flex_direction: FlexDirection::Row, column_gap: px(6) }
                Children [
                    button(Action::Quality(VoxelGiQuality::Low), "Low"),
                    button(Action::Quality(VoxelGiQuality::Default), "Default"),
                    button(Action::Quality(VoxelGiQuality::High), "High"),
                ]
            ),
            (Text("") TextFont { font_size: FontSize::Px(12.0) } QualityLabel),
            heading("Performance"),
            knob_row(Knob::GpuShare, s),
            knob_row(Knob::CacheRadius, s),
            knob_row(Knob::ViewReceivers, s),
            knob_row(Knob::CacheShare, s),
            heading("Look"),
            knob_row(Knob::Spatial, s),
            knob_row(Knob::Evidence, s),
            knob_row(Knob::Response, s),
            knob_row(Knob::Gain, s),
            heading("Light model (Apply restarts measurements)"),
            knob_row(Knob::Reach, s),
            knob_row(Knob::Bounces, s),
            (
                Node { flex_direction: FlexDirection::Row, column_gap: px(6) }
                Children [
                    button(Action::ApplyModel, "Apply model"),
                    button(Action::DiscardModel, "Discard"),
                ]
            ),
            heading("Display"),
            toggle_row(Toggle::Enabled, "Indirect light (F4)"),
            toggle_row(Toggle::Corners, "Corner smoothing (F5)"),
            toggle_row(Toggle::VoxelSun, "Voxel sun (F7)"),
            toggle_row(Toggle::Irradiance, "Irradiance view (F6)"),
            heading("Settings as RON"),
            (
                Node { flex_direction: FlexDirection::Row, column_gap: px(6) }
                Children [
                    button(Action::Copy, "Copy"),
                    button(Action::Paste, "Paste"),
                ]
            ),
            (Text("") TextFont { font_size: FontSize::Px(12.0) } StatusText),
        ]
    });
}

fn on_slider(
    change: On<ValueChange<f32>>,
    knobs: Query<&Knob>,
    mut settings: ResMut<VoxelGiSettings>,
    mut display: ResMut<VoxelGiDisplay>,
    mut panel: ResMut<PanelState>,
) {
    if let Ok(knob) = knobs.get(change.source) {
        knob.set(change.value, &mut settings, &mut display, &mut panel);
    }
}

fn on_checkbox(change: On<ValueChange<bool>>, toggles: Query<&Toggle>, mut display: ResMut<VoxelGiDisplay>) {
    if let Ok(toggle) = toggles.get(change.source) {
        toggle.set(change.value, &mut display);
    }
}

fn on_button(
    activate: On<Activate>,
    actions: Query<&Action>,
    mut settings: ResMut<VoxelGiSettings>,
    mut panel: ResMut<PanelState>,
    mut panels: Query<&mut Node, With<Panel>>,
    mut status: Query<&mut Text, With<StatusText>>,
    mut clipboard: ResMut<Clipboard>,
) {
    let Ok(action) = actions.get(activate.entity) else {
        return;
    };
    let mut report = |message: String| {
        for mut text in &mut status {
            text.0 = message.clone();
        }
    };
    match *action {
        Action::TogglePanel => {
            for mut node in &mut panels {
                node.display = if node.display == Display::None { Display::Flex } else { Display::None };
            }
        }
        Action::Quality(quality) => {
            let preset = VoxelGiSettings::quality(quality);
            settings.gpu_time_share = preset.gpu_time_share;
            settings.cache_radius = preset.cache_radius;
            settings.view_receivers = preset.view_receivers;
        }
        Action::ApplyModel => {
            if let Some(draft) = panel.draft.take() {
                settings.transport = draft.normalized();
            }
        }
        Action::DiscardModel => panel.draft = None,
        Action::Copy => report(match clipboard.set_text(settings.to_ron()) {
            Ok(()) => "Copied settings.".into(),
            Err(error) => format!("Copy failed: {error:?}"),
        }),
        Action::Paste => panel.paste = Some(clipboard.fetch_text()),
    }
}

/// Clipboard reads can complete later (web); apply when they do.
fn finish_paste(
    mut panel: ResMut<PanelState>,
    mut settings: ResMut<VoxelGiSettings>,
    mut status: Query<&mut Text, With<StatusText>>,
) {
    let Some(read) = panel.paste.as_mut() else {
        return;
    };
    let Some(result) = read.poll_result() else {
        return;
    };
    panel.paste = None;
    let message = match result.map_err(|error| format!("{error:?}")).and_then(|text| VoxelGiSettings::from_ron(&text)) {
        Ok(pasted) => {
            *settings = pasted;
            panel.draft = None;
            "Applied pasted settings.".to_string()
        }
        Err(error) => format!("Paste failed: {error}"),
    };
    for mut text in &mut status {
        text.0 = message.clone();
    }
}

/// Keep widgets in step with settings changed elsewhere (presets, keys, paste).
fn sync_controls(
    mut commands: Commands,
    settings: Res<VoxelGiSettings>,
    display: Res<VoxelGiDisplay>,
    panel: Res<PanelState>,
    sliders: Query<(Entity, &Knob, &SliderValue)>,
    toggles: Query<(Entity, &Toggle, Has<Checked>)>,
    mut labels: Query<&mut Text, With<QualityLabel>>,
) {
    if !settings.is_changed() && !display.is_changed() && !panel.is_changed() {
        return;
    }
    for (entity, knob, value) in &sliders {
        let current = knob.get(&settings, &display, &panel);
        if value.0 != current {
            commands.entity(entity).insert(SliderValue(current));
        }
    }
    for (entity, toggle, checked) in &toggles {
        let want = toggle.get(&display);
        if want && !checked {
            commands.entity(entity).insert(Checked);
        } else if !want && checked {
            commands.entity(entity).remove::<Checked>();
        }
    }
    let label = match settings.matching_quality() {
        Some(quality) => format!("Current: {quality:?}"),
        None => "Current: custom".into(),
    };
    let label = if panel.draft.is_some() { format!("{label} (light model change staged)") } else { label };
    for mut text in &mut labels {
        text.0 = label.clone();
    }
}

fn update_hud(
    diagnostics: Res<DiagnosticsStore>,
    stats: Res<VoxelGiStats>,
    time: Res<Time<Real>>,
    mut next_update: Local<f64>,
    mut text: Query<&mut Text, With<HudText>>,
) {
    if time.elapsed_secs_f64() < *next_update {
        return;
    }
    *next_update = time.elapsed_secs_f64() + 0.25;
    let fps = diagnostics
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed());
    for mut text in &mut text {
        text.0 = hud_text(&stats, fps);
    }
}

fn hud_text(stats: &VoxelGiStats, fps: Option<f64>) -> String {
    let fps = fps
        .filter(|fps| fps.is_finite() && *fps >= 0.0)
        .map_or_else(|| "n/a".into(), |fps| format!("{fps:.0}"));
    if stats.surface_probes == 0 {
        return format!(
            "FPS:       {fps}\nCoverage:  n/a\nBootstrap: n/a\nSettled:   n/a{}",
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
            .map_or_else(|| "n/a".into(), |value| format!("{value:.2} ms/frame"))
    };
    let view = stats.view_gpu;
    let view_text = if view == super::ViewReceipts::default() {
        "n/a".into()
    } else {
        format!(
            "{:.2} ms (trace {:.2}, present {:.2}, discover {:.2})",
            view.trace_ms + view.presentation_ms + view.discovery_ms,
            view.trace_ms,
            view.presentation_ms,
            view.discovery_ms
        )
    };
    format!(
        "\nGI main:   {}\nCache GPU: {}\nView GPU:  {}",
        cost(stats.main_thread_ms_per_frame),
        cost(stats.gpu_ms_per_frame),
        view_text
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
    fn widget_events_reach_settings_and_presets_keep_the_look() {
        let mut app = App::new();
        app.init_resource::<VoxelGiSettings>()
            .init_resource::<VoxelGiDisplay>()
            .init_resource::<PanelState>()
            .init_resource::<Clipboard>()
            .add_observer(on_slider)
            .add_observer(on_checkbox)
            .add_observer(on_button);
        let world = app.world_mut();
        let slider = world.spawn(Knob::Spatial).id();
        world.trigger(ValueChange { source: slider, value: 0.25_f32, is_final: true });
        assert_eq!(world.resource::<VoxelGiSettings>().spatial_reconstruction, 0.25);
        let checkbox = world.spawn(Toggle::VoxelSun).id();
        world.trigger(ValueChange { source: checkbox, value: false, is_final: true });
        assert!(!world.resource::<VoxelGiDisplay>().voxel_sun);
        let high = world.spawn(Action::Quality(VoxelGiQuality::High)).id();
        world.trigger(Activate { entity: high });
        let settings = world.resource::<VoxelGiSettings>().clone();
        assert_eq!(settings.matching_quality(), Some(VoxelGiQuality::High));
        assert_eq!(settings.spatial_reconstruction, 0.25, "presets leave look settings alone");
        let reach = world.spawn(Knob::Reach).id();
        world.trigger(ValueChange { source: reach, value: 64.0_f32, is_final: true });
        assert_eq!(world.resource::<VoxelGiSettings>().transport.reach, 768.0, "staged, not applied");
        let apply = world.spawn(Action::ApplyModel).id();
        world.trigger(Activate { entity: apply });
        assert_eq!(world.resource::<VoxelGiSettings>().transport.reach, 64.0);
    }

    #[test]
    fn knobs_and_toggles_write_back_what_they_read() {
        let knobs = [
            Knob::GpuShare, Knob::CacheRadius, Knob::ViewReceivers, Knob::CacheShare, Knob::Spatial,
            Knob::Evidence, Knob::Response, Knob::Gain, Knob::Reach, Knob::Bounces,
        ];
        let (mut settings, mut display, mut panel) =
            (VoxelGiSettings::default(), VoxelGiDisplay::default(), PanelState::default());
        for knob in knobs {
            let (min, max, _, _) = knob.range();
            let value = knob.get(&settings, &display, &panel);
            assert!((min..=max).contains(&value), "{} default {value} outside {min}..{max}", knob.label());
            knob.set(max, &mut settings, &mut display, &mut panel);
            assert!((knob.get(&settings, &display, &panel) - max).abs() < 1e-3, "{}", knob.label());
        }
        // The light model is staged, never applied by dragging.
        assert_eq!(settings.transport, VoxelGiSettings::default().transport);
        assert_eq!(panel.draft.unwrap().bounces, 3);
        for toggle in [Toggle::Enabled, Toggle::Corners, Toggle::VoxelSun, Toggle::Irradiance] {
            let before = toggle.get(&display);
            toggle.set(!before, &mut display);
            assert_eq!(toggle.get(&display), !before);
        }
    }

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
            view_gpu: crate::voxel_gi::ViewReceipts { trace_ms: 1.0, presentation_ms: 0.5, discovery_ms: 0.25 },
            ..default()
        };
        assert_eq!(
            hud_text(&stats, Some(59.7)),
            "FPS:       60\nCoverage:  75.0%\nBootstrap: 32.5%\nSettled:   12.5%\nGI main:   1.25 ms/frame\nCache GPU: 0.38 ms/frame\nView GPU:  1.75 ms (trace 1.00, present 0.50, discover 0.25)"
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
            "FPS:       n/a\nCoverage:  n/a\nBootstrap: n/a\nSettled:   n/a\nGI main:   n/a\nCache GPU: n/a\nView GPU:  n/a"
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
            "\nGI main:   0.00 ms/frame\nCache GPU: n/a\nView GPU:  n/a"
        );
    }
}
