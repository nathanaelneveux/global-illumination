//! The lab's voxel GI wiring: launch settings, the debug overlay's
//! configuration, scripted validation and the gameplay action buttons.
use crate::player::{BombAtAim, PlaceEmissiveBlock};
use voxel_gi_debug::VoxelGiDebugPlugin;
use voxel_gi::VoxelGiSettings;
#[cfg(debug_assertions)]
use voxel_gi::VoxelGiSystems;
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
            crate::validation::install_gpu_pacing(app);
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
        .and_then(|text| voxel_gi::VoxelGiSettings::from_ron(&text))
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

/// `VOXEL_GI_LAB_GI=0` is a host-cost control: the same world, streaming and
/// validation workload without GI (no GI plugins, BVW's standard material).
pub fn gi_enabled() -> bool {
    std::env::var("VOXEL_GI_LAB_GI").as_deref() != Ok("0")
}

/// The GI resources the lab and validation read, left at their defaults so
/// the control runs the same harness. Nothing updates them.
pub fn init_gi_off_resources(app: &mut App) {
    app.init_resource::<voxel_gi::VoxelGiStats>()
        .init_resource::<voxel_gi::VoxelGiDisplay>()
        .init_resource::<voxel_gi::VoxelGiEnvironment>()
        .init_resource::<voxel_gi::VoxelMaterialTransportTable>()
        .init_resource::<VoxelGiSettings>();
}

#[cfg(test)]
mod tests {
    use voxel_gi::VoxelGiSettings;

    fn config(text: &str) -> VoxelGiSettings {
        VoxelGiSettings::from_ron(text).unwrap()
    }

    #[test]
    fn lab_configs_parse_and_keep_their_meaning() {
        let settings = VoxelGiSettings::default();
        assert_eq!(config(include_str!("../config/gi-display-spatial.ron")), settings);
        let control = config(include_str!("../config/gi-display-unfiltered.ron"));
        assert_eq!(control.spatial_reconstruction, 0.0);
        assert_eq!(control.transport, settings.transport);
        let cacheless = config(include_str!("../config/gi-cacheless.ron"));
        assert_eq!(cacheless.cache_radius, 0.0);
        assert_eq!(cacheless.transport, settings.transport);
        assert_eq!(config(include_str!("../config/gi-cache-32.ron")).cache_radius, 32.0);
        let near_only = config(include_str!("../config/gi-far-tier-off.ron"));
        assert_eq!(near_only.far_tier.voxel_pixels, 0.0);
        assert_eq!(near_only.transport, settings.transport);
        for text in [include_str!("../config/gi-low.ron"), include_str!("../config/gi-high.ron")] {
            assert!(config(text).matching_quality().is_some());
        }
    }
}
