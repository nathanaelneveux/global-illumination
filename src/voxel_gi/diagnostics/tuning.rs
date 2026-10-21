//! Demo tuning surface. The reusable solver owns settings and invalidation;
//! this adapter owns widgets, session comparisons and opt-in JSON import/export.
use super::*;
use crate::voxel_gi::{TransportSettings, VoxelGiSettings, render::GpuTransportSettings};

#[cfg(test)]
#[path = "tuning_capture.rs"]
mod capture;

#[derive(Resource, Default)]
pub(super) struct TuningPanel {
    pub open: bool,
    draft: Option<TransportSettings>,
    saved: [Option<Preset>; 2],
    json: String,
    status: String,
}

#[derive(Clone)]
struct Preset {
    gi: VoxelGiSettings,
    display: DisplaySettings,
}

/// Only presentation, never handles, capacities, hash masks or cache generations.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DisplaySettings {
    enabled: bool,
    corners: bool,
    cached_sun: bool,
    irradiance: bool,
    intensity: f32,
}

impl DisplaySettings {
    fn read(settings: &GpuTransportSettings) -> Self {
        Self {
            enabled: settings.enabled != 0,
            corners: settings.reconstruction != 0,
            cached_sun: settings.voxel_sky != 0,
            irradiance: settings.debug_irradiance != 0,
            intensity: settings.intensity,
        }
    }

    fn write(self, settings: &mut GpuTransportSettings) {
        settings.enabled = u32::from(self.enabled);
        settings.reconstruction = u32::from(self.corners);
        settings.voxel_sky = u32::from(self.cached_sun);
        settings.debug_irradiance = u32::from(self.irradiance);
        settings.intensity = self.intensity;
    }
}

pub(super) fn load_launch_settings(app: &mut App) {
    let Some(path) = std::env::var_os("VOXEL_GI_SETTINGS") else {
        return;
    };
    let settings = std::fs::read_to_string(&path)
        .map_err(|error| error.to_string())
        .and_then(|json| parse_settings(&json))
        .unwrap_or_else(|error| panic!("Cannot load VOXEL_GI_SETTINGS={path:?}: {error}"));
    // Fail visibly on invalid files rather than benchmarking a silent fallback.
    app.insert_resource(settings);
}

fn parse_settings(json: &str) -> Result<VoxelGiSettings, String> {
    let mut settings: VoxelGiSettings = serde_json::from_str(json).map_err(|e| e.to_string())?;
    settings.transport = settings.transport.normalized();
    settings.samples_per_dispatch = settings.batch_samples();
    settings.simulation_sampling_importance = settings.sampling_importance().infrared;
    settings.trace_budget_ms = settings.budget_ms();
    settings.max_probes_per_dispatch = settings.batch_limit();
    settings.stable_min_samples = settings.stable_samples();
    settings.stable_relative_error = settings.stable_error();
    settings.stable_audit_interval = settings.audit_interval();
    settings.display_response = settings
        .display_response
        .clamp(Duration::from_millis(10), Duration::from_millis(500));
    settings.display_transition_frames = settings.display_transition_frames.min(32);
    settings.scene_publish_budget = settings.scene_publish_budget.min(Duration::from_millis(8));
    settings.display_upload_interval = settings
        .display_upload_interval
        .clamp(Duration::from_millis(16), Duration::from_millis(500));
    Ok(settings)
}

pub(super) fn draw_panel(
    mut contexts: EguiContexts,
    mut panel: ResMut<TuningPanel>,
    mut settings: ResMut<VoxelGiSettings>,
    handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    cursor: Single<&CursorOptions, With<PrimaryWindow>>,
) {
    if !panel.open {
        return;
    }
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    let Some(handle) = handle else {
        return;
    };
    let Some(material) = materials.get(&handle.handle) else {
        return;
    };
    let mut next = settings.clone();
    let mut display = DisplaySettings::read(&material.extension.settings);
    let mut open = panel.open;
    egui::Window::new("Lighting controls")
        .open(&mut open)
        .default_pos([16.0, 80.0])
        .default_width(440.0)
        .default_height(700.0)
        .vscroll(true)
        .show(context, |ui| {
            ui.add_enabled_ui(cursor.visible, |ui| {
                contents(ui, &mut panel, &mut next, &mut display)
            });
        });
    panel.open = open;
    // Idle panels must not mark resources/assets changed every frame.
    if *settings != next {
        *settings = next;
    }
    if display != DisplaySettings::read(&material.extension.settings) {
        display.write(
            &mut materials
                .get_mut(&handle.handle)
                .unwrap()
                .extension
                .settings,
        );
    }
}

fn contents(
    ui: &mut egui::Ui,
    panel: &mut TuningPanel,
    settings: &mut VoxelGiSettings,
    display: &mut DisplaySettings,
) {
    ui.label("Esc resumes play. HUD costs remain visible while tuning.");
    ui.label("Change one lever at a time; coverage is not convergence.");
    ui.horizontal(|ui| {
        if ui.button("Reset defaults").clicked() {
            *settings = VoxelGiSettings::default();
            *display = DisplaySettings::read(&GpuTransportSettings::default());
            panel.draft = Some(settings.transport);
            panel.status = "Defaults restored. Model changes refine progressively.".into();
        }
        for index in 0..2 {
            let name = if index == 0 { "A" } else { "B" };
            if ui.button(format!("Save {name}")).on_hover_text("Save applied settings and presentation in this session, not pending sliders or radiance history.").clicked() {
                panel.saved[index] = Some(Preset { gi: settings.clone(), display: *display });
                panel.status = format!("Saved {name} (this session only).");
            }
            if ui.add_enabled(panel.saved[index].is_some(), egui::Button::new(format!("Load {name}"))).clicked() {
                let preset = panel.saved[index].as_ref().unwrap();
                *settings = preset.gi.clone();
                *display = preset.display;
                panel.draft = Some(settings.transport);
                panel.status = format!("Loaded {name}. A changed model needs fresh refinement, not an instant image swap.");
            }
        }
    });
    if !panel.status.is_empty() {
        ui.small(&panel.status);
    }
    ui.separator();
    egui::CollapsingHeader::new("Light model — Apply to restart measurements")
        .default_open(true).show(ui, |ui| {
            let draft = panel.draft.get_or_insert(settings.transport);
            ui.add(egui::Slider::new(&mut draft.reach, 1.0..=768.0).logarithmic(true).integer().text("Reach (voxels)"))
                .on_hover_text("Each diffuse ray / voxel-emitter connection. Not total path length or Minecraft light-level falloff. Short cutoffs can remove distant indirect light and environment, making large interiors darker.");
            ui.horizontal_wrapped(|ui| {
                for reach in [16.0, 32.0, 64.0, 128.0, 256.0, 768.0] {
                    ui.selectable_value(&mut draft.reach, reach, format!("{reach:.0}"));
                }
            });
            ui.add(egui::Slider::new(&mut draft.bounces, 1..=3).text("Diffuse surface hits"));
            ui.small("Sun visibility stays at 768 voxels; analytic point lights keep their own ranges. All loaded receivers stay eligible.");
            ui.horizontal(|ui| {
                if ui.add_enabled(*draft != settings.transport, egui::Button::new("Apply model")).clicked() {
                    settings.transport = draft.normalized();
                    *draft = settings.transport;
                    panel.status = "Model applied. Old measurements restart; displayed history fades toward the new result.".into();
                }
                if ui.button("Discard draft").clicked() { *draft = settings.transport; }
                ui.small(format!("Active: {:.0} voxels / {} hits", settings.transport.reach, settings.transport.bounces));
            });
        });
    egui::CollapsingHeader::new("Work & refinement — live, keeps measurements")
        .default_open(true).show(ui, |ui| {
            ui.add(egui::Slider::new(&mut settings.trace_budget_ms, 0.25..=16.0).logarithmic(true).text("Batch target (ms)"))
                .on_hover_text("Adaptive compute target per completed batch, not a hard per-frame GPU cap. Includes sky work; asynchronous queue timing varies.");
            ui.add(egui::Slider::new(&mut settings.samples_per_dispatch, 1..=64).text("Paths / face / batch"))
                .on_hover_text("Cooperative lane count depends on the backend. Larger batches refresh fewer faces per dispatch. Measure settling, frame time and data age together.");
            ui.add(egui::Slider::new(&mut settings.simulation_sampling_importance, 0.0..=1.0).text("IR / radiation sampling importance"))
                .on_hover_text("RGB importance stays one. Zero (default) samples visible emitters only; IR/ionizing-only sources are then found by paths that hit them. Changes uncertainty, not physical energy or retained history.");
            ui.checkbox(&mut settings.adaptive_updates, "Let settled faces sleep between audits");
            ui.add(egui::Slider::new(&mut settings.stable_relative_error, 0.01..=0.5).logarithmic(true).text("Settle error tolerance"))
                .on_hover_text("All-band heuristic, not guaranteed convergence. Lower is stricter; already settled faces reassess on their next audit.");
            ui.add(egui::Slider::new(&mut settings.stable_min_samples, 32..=256).text("Minimum samples"))
                .on_hover_text("Dark faces still require at least 256. Four stable batches and displayed agreement are also required.");
            duration_slider(ui, &mut settings.stable_audit_interval, 100.0..=30000.0, "Audit interval (ms)");
            ui.collapsing("Advanced work limits", |ui| {
                ui.add(egui::Slider::new(&mut settings.max_probes_per_dispatch, 64..=8192).logarithmic(true).text("Receiver batch ceiling"));
                duration_slider(ui, &mut settings.scene_publish_budget, 0.0..=8.0, "Scene publication (ms)");
                ui.small("Publication is page-atomic: one page can exceed the target. Zero still permits one page per frame.");
                duration_slider(ui, &mut settings.display_upload_interval, 16.0..=500.0, "Display upload interval (ms)");
            });
        });
    egui::CollapsingHeader::new("Presentation — live, does not alter transport")
        .default_open(true).show(ui, |ui| {
            duration_slider(ui, &mut settings.display_response, 10.0..=500.0, "Temporal response (ms)");
            ui.add(egui::Slider::new(&mut settings.display_transition_frames, 0..=32).text("Publication transition (frames)"))
                .on_hover_text("Blend completed resident RGB targets on every rendered frame. Zero is instantaneous publication. Does not change measurements or delay simulation delivery.");
            ui.small("Larger = slower/smoother. Evidence and noise limits can dominate this time constant; it is not a settling deadline.");
            ui.add(egui::Slider::new(&mut display.intensity, 0.0..=3.0).text("Indirect display gain"));
            ui.horizontal(|ui| {
                ui.checkbox(&mut display.enabled, "Indirect (F4)");
                ui.checkbox(&mut display.corners, "Corners (F5)");
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut display.cached_sun, "Cached sun (F7)");
                ui.checkbox(&mut display.irradiance, "Irradiance view (F6)");
            });
        });
    ui.collapsing("Export / import transport & work settings", |ui| {
        ui.small("JSON excludes presentation toggles/gain. Save copied text to a file and launch with VOXEL_GI_SETTINGS=path.json. A/B slots above include presentation but last only this session.");
        ui.horizontal(|ui| {
            if ui.button("Copy applied JSON").clicked() {
                panel.json = serde_json::to_string_pretty(settings).expect("finite GI settings");
                ui.ctx().copy_text(panel.json.clone());
                panel.status = "Copied applied transport/work settings (not the model draft).".into();
            }
            if ui.button("Apply pasted JSON").clicked() {
                match parse_settings(&panel.json) {
                    Ok(next) => {
                        *settings = next;
                        panel.draft = Some(settings.transport);
                        panel.status = "Imported settings; values clamped to supported ranges.".into();
                    }
                    Err(error) => panel.status = format!("Not applied: {error}"),
                }
            }
        });
        ui.add(egui::TextEdit::multiline(&mut panel.json).code_editor().desired_rows(6));
    });
}

fn duration_slider(
    ui: &mut egui::Ui,
    duration: &mut Duration,
    range: std::ops::RangeInclusive<f32>,
    label: &str,
) {
    let mut ms = duration.as_secs_f32() * 1000.0;
    if ui
        .add(egui::Slider::new(&mut ms, range).text(label))
        .changed()
    {
        *duration = Duration::from_secs_f32(ms * 0.001);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tuning_clicks_stage_apply_compare_and_reset_without_live_cache_state() {
        let context = egui::Context::default();
        let mut panel = TuningPanel::default();
        let mut settings = VoxelGiSettings::default();
        let mut display = DisplaySettings::read(&GpuTransportSettings::default());
        let mut frame = |events| {
            context.run_ui(
                egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(1000.0, 1100.0),
                    )),
                    events,
                    ..default()
                },
                |root| {
                    egui::Window::new("Lighting controls")
                        .default_width(440.0)
                        .default_height(700.0)
                        .show(root.ctx(), |ui| {
                            contents(ui, &mut panel, &mut settings, &mut display)
                        });
                },
            )
        };
        let mut click = |label: &str| {
            frame(vec![]);
            let output = frame(vec![]);
            let position = output
                .shapes
                .iter()
                .find_map(|shape| {
                    if let egui::epaint::Shape::Text(text) = &shape.shape
                        && text.galley.job.text == label
                    {
                        Some(text.pos + text.galley.size() * 0.5)
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| panic!("Missing widget {label}"));
            frame(vec![
                egui::Event::PointerMoved(position),
                egui::Event::PointerButton {
                    pos: position,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: default(),
                },
            ]);
            frame(vec![egui::Event::PointerButton {
                pos: position,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: default(),
            }]);
        };
        click("16");
        // Save A before Apply: the draft must not leak into stored settings.
        click("Save A");
        click("Apply model");
        click("Save B");
        click("Load A");
        click("Save A");
        click("Load B");
        click("Reset defaults");
        click("Indirect (F4)");
        assert_eq!(settings, VoxelGiSettings::default());
        assert_eq!(panel.saved[0].as_ref().unwrap().gi.transport.reach, 768.0);
        assert_eq!(panel.saved[1].as_ref().unwrap().gi.transport.reach, 16.0);
        assert_eq!(panel.draft, Some(TransportSettings::default()));
        assert!(!display.enabled);
        assert!(display.corners && display.cached_sun);
    }

    #[test]
    fn tuning_presets_never_replace_cache_addressing() {
        let mut live = GpuTransportSettings {
            hash_mask: 8191,
            probe_capacity: 5000,
            ..default()
        };
        let preset = DisplaySettings {
            corners: false,
            intensity: 0.5,
            ..DisplaySettings::read(&live)
        };
        preset.write(&mut live);
        assert_eq!(live.hash_mask, 8191);
        assert_eq!(live.probe_capacity, 5000);
        assert_eq!(DisplaySettings::read(&live), preset);
        DisplaySettings::read(&GpuTransportSettings::default()).write(&mut live);
        assert_eq!(live.hash_mask, 8191);
        assert_eq!(live.probe_capacity, 5000);
    }

    #[test]
    fn tuning_import_clamps_and_idle_panel_does_not_change_settings() {
        let imported =
            parse_settings(r#"{"transport":{"reach":16,"bounces":0},"samples_per_dispatch":999}"#)
                .unwrap();
        assert_eq!(
            imported.transport,
            TransportSettings {
                reach: 16.0,
                bounces: 1
            }
        );
        assert_eq!(imported.samples_per_dispatch, 64);
        for paths in [1, 8, 16, 32, 64] {
            let imported =
                parse_settings(&format!("{{\"samples_per_dispatch\":{paths}}}")).unwrap();
            assert_eq!(imported.batch_samples(), paths);
        }
        assert!(parse_settings("broken").is_err());
        let defaults = VoxelGiSettings::default();
        assert_eq!(
            parse_settings(&serde_json::to_string(&defaults).unwrap()).unwrap(),
            defaults
        );
        let context = egui::Context::default();
        let mut panel = TuningPanel::default();
        let mut settings = defaults.clone();
        let mut display = DisplaySettings::read(&GpuTransportSettings::default());
        for _ in 0..2 {
            let output = context.run_ui(egui::RawInput::default(), |root| {
                egui::Window::new("Lighting controls").show(root.ctx(), |ui| {
                    contents(ui, &mut panel, &mut settings, &mut display)
                });
            });
            assert!(!output.shapes.is_empty());
        }
        assert_eq!(settings, defaults);
        assert_eq!(
            display,
            DisplaySettings::read(&GpuTransportSettings::default())
        );
    }
}
