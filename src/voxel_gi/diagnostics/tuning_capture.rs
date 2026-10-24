//! Opt-in UI-only image check, independent of screen lock/focus. Not gameplay
//! quality or frame-pacing evidence. Uses the production widgets and Bevy capture.
use super::*;
use bevy::camera::RenderTarget;
use bevy::render::{
    pipelined_rendering::PipelinedRenderingPlugin,
    render_resource::{TextureFormat, TextureUsages},
    view::screenshot::{Screenshot, ScreenshotCaptured},
};
use bevy::window::WindowPlugin;
use bevy_inspector_egui::bevy_egui::{EguiPlugin, PrimaryEguiContext};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

#[test]
#[ignore = "native offscreen UI capture; no OS window or gameplay timings"]
fn tuning_panel_offscreen_capture() {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<bevy::log::LogPlugin>()
            .disable::<bevy::audio::AudioPlugin>()
            .disable::<bevy::gilrs::GilrsPlugin>()
            .disable::<PipelinedRenderingPlugin>()
            .set(WindowPlugin {
                primary_window: None,
                ..default()
            }),
    )
    .add_plugins(EguiPlugin::default())
    .add_systems(EguiPrimaryContextPass, capture_widgets);
    let mut image = Image::new_target_texture(1000, 1100, TextureFormat::Rgba8UnormSrgb, None);
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    app.world_mut().spawn((
        Camera2d,
        PrimaryEguiContext,
        RenderTarget::Image(target.clone().into()),
    ));
    app.finish();
    app.cleanup();
    let started = Instant::now();
    let complete = Arc::new(AtomicBool::new(false));
    let mut requested = false;
    while !complete.load(Ordering::SeqCst) {
        assert!(
            started.elapsed() < Duration::from_secs(30),
            "UI capture timed out"
        );
        if !requested && started.elapsed() > Duration::from_secs(2) {
            requested = true;
            let complete = complete.clone();
            app.world_mut()
                .spawn(Screenshot::image(target.clone()))
                .observe(move |capture: On<ScreenshotCaptured>| {
                    let image = capture.image.clone().try_into_dynamic().unwrap();
                    assert_eq!((image.width(), image.height()), (1000, 1100));
                    std::fs::create_dir_all("target").unwrap();
                    image.save("target/gi-tuning-panel.png").unwrap();
                    complete.store(true, Ordering::SeqCst);
                });
        }
        app.update();
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn capture_widgets(
    mut contexts: EguiContexts,
    mut panel: Local<TuningPanel>,
    mut settings: Local<VoxelGiSettings>,
) {
    let context = contexts.ctx_mut().unwrap();
    let mut display = DisplaySettings::read(&crate::voxel_gi::VoxelGiDisplay::default());
    egui::Window::new("Lighting controls")
        .default_pos([16.0, 16.0])
        .default_width(440.0)
        .default_height(700.0)
        .vscroll(true)
        .show(context, |ui| {
            contents(ui, &mut panel, &mut settings, &mut display)
        });
}
