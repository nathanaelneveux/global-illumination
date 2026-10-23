//! Opt-in scripted workloads. Gameplay keeps the preferred presentation;
//! comparisons and toggle stress have distinct report identities.
mod background;
mod cave;
mod day;
mod host_timing;
mod handoff;
mod house;
mod memory;
mod metrics;
mod profiles;
mod reference;
pub(super) mod reference_oracle;
mod response;
mod structures;
mod tunnel;
pub(super) mod temporal;

use super::render::VoxelTransportMaterial;
use super::world::FACE_OFFSETS;
use super::{
    EnergyEmitter, EnergySpectrum, MaterialTransport, VoxelEnergyField, VoxelGiDirty,
    VoxelGiSettings, VoxelGiStats, VoxelMaterialTransportTable,
};
use crate::assets::{PrototypeConfig, TemplateAssets};
use crate::{player::Player, terrain::PrototypeWorld};
use bevy::camera::{RenderTarget, ShadowLodOrigin};
use bevy::prelude::*;
use bevy::render::render_resource::{TextureFormat, TextureUsages};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured, save_to_disk};
use bevy_voxel_world::prelude::*;
use bevy_voxel_world::rendering::VoxelWorldMaterialHandle;
use metrics::Metrics;
use profiles::Profile;
use serde_json::json;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Resource, Default)]
pub(super) struct CaptureEvidence {
    output: Option<PathBuf>,
    render_target: Option<Handle<Image>>,
    completed: usize,
    failed: bool,
    windows: Vec<serde_json::Value>,
    last_capture: Option<Instant>,
}

const OFFSCREEN_WIDTH: u32 = 2560;
const OFFSCREEN_HEIGHT: u32 = 1440;

pub(super) fn install_frame_timing(app: &mut App) {
    if std::env::var_os("VOXEL_GI_VALIDATE_FRAME_TIMINGS").is_some() {
        host_timing::install(app);
    }
}

pub(super) fn setup_capture_target(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut evidence: ResMut<CaptureEvidence>,
    cameras: Query<Entity, With<Camera3d>>,
) {
    if !crate::headless_validation() {
        return;
    }
    let mut image = Image::new_target_texture(
        OFFSCREEN_WIDTH,
        OFFSCREEN_HEIGHT,
        TextureFormat::Rgba8UnormSrgb,
        None,
    );
    image.texture_descriptor.usage |= TextureUsages::COPY_SRC;
    let target = images.add(image);
    for camera in &cameras {
        // An image target has no implicit window camera. Make the same camera
        // the explicit point/spot-shadow LOD origin, rather than relying on
        // Bevy's warning-producing fallback to an arbitrary camera.
        commands
            .entity(camera)
            .insert((RenderTarget::Image(target.clone().into()), ShadowLodOrigin));
    }
    evidence.render_target = Some(target);
}

impl CaptureEvidence {
    fn fail(&mut self, reason: &str) {
        self.failed = true;
        if let Some(output) = &self.output {
            let status = json!({"status": "failed", "reason": reason});
            if let Err(error) = std::fs::write(output.join("status.json"), status.to_string()) {
                error!(%error, "could not record validation failure");
            }
        }
    }
}

#[derive(Default)]
pub(super) struct ValidationRun {
    startup: Option<Instant>,
    started: Option<Instant>,
    profile: Profile,
    motion_diagnostic: u32,
    motion_static: bool,
    /// Walk forward at this many voxels per second instead of turning
    /// (VOXEL_GI_VALIDATE_MOTION_WALK; consecutive-frame bursts).
    motion_walk: Option<f32>,
    metrics: Metrics,
    captures: usize,
    edits: u32,
    edit_attempts: u32,
    moves: u32,
    mode_changes: u32,
    light: Option<Entity>,
    center: Vec3,
    removed: Option<(IVec3, WorldVoxel<u8>)>,
    emissive: Option<(IVec3, WorldVoxel<u8>, MaterialTransport)>,
    emitter_started: bool,
    response: response::ResponseRun,
    tunnel: tunnel::TunnelRun,
    house: house::HouseRun,
    temporal: temporal::TemporalRun,
    cave_camera_initialized: bool,
    structures: structures::StructuresRun,
    background: background::BackgroundRun,
    cave: cave::CaveObservations,
    day: day::DayRun,
    memory: memory::MemorySampler,
    max_pages: usize,
    capture_pause_until: f32,
    done: bool,
    reference: reference::ReferenceRun,
}

pub(super) fn run_validation(
    mut run: Local<ValidationRun>,
    mut evidence: ResMut<CaptureEvidence>,
    time: Res<Time<Real>>,
    gi: (
        Res<VoxelGiStats>,
        Res<super::VoxelGiRuntime>,
        Option<Res<host_timing::HostTiming>>,
        Option<Res<super::gpu::GpuTransportDevice>>,
    ),
    settings: Res<VoxelGiSettings>,
    field: Res<VoxelEnergyField>,
    mut dirty: ResMut<VoxelGiDirty>,
    mut world: VoxelWorld<PrototypeWorld>,
    mut player_state: Single<&mut Transform, (With<Player>, With<bevy_ahoy::CharacterLook>)>,
    mut lights: Query<&mut Transform, Without<Player>>,
    cameras: Query<Entity, With<Camera3d>>,
    mut commands: Commands,
    handle: Option<Res<VoxelWorldMaterialHandle<VoxelTransportMaterial>>>,
    mut materials: ResMut<Assets<VoxelTransportMaterial>>,
    mut transport_materials: ResMut<VoxelMaterialTransportTable>,
    asset_context: (
        Res<TemplateAssets>,
        Res<AssetServer>,
        Res<Assets<PrototypeConfig>>,
        Res<PrototypeWorld>,
    ),
) {
    if run.done {
        run.reference.poll(&mut commands, &mut evidence);
        return;
    }
    if std::env::var_os("VOXEL_GI_VALIDATE").is_none() || evidence.failed {
        return;
    }
    let (stats, runtime, host_timing, device) = gi;
    let (assets, asset_server, configs, generator) = asset_context;
    if run.startup.is_none() {
        // Shared Cargo build directories can reuse another worktree's package
        // artifact. The optional expected root is supplied by the run command;
        // this compile-time identity must agree before accepting any evidence.
        if let Some(expected) = std::env::var_os("VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT") {
            let expected = std::fs::canonicalize(expected);
            let compiled = std::fs::canonicalize(env!("CARGO_MANIFEST_DIR"));
            if expected.is_err() || expected.ok() != compiled.ok() {
                error!(
                    compiled_root = env!("CARGO_MANIFEST_DIR"),
                    "Validation binary is from the wrong worktree; rebuild the application package."
                );
                run.done = true;
                commands.write_message(AppExit::error());
                return;
            }
        }
        let profile =
            std::env::var("VOXEL_GI_VALIDATE_PROFILE").unwrap_or_else(|_| "gameplay".into());
        let profile = match Profile::parse(&profile) {
            Ok(profile) => profile,
            Err(error) => {
                error!(%error);
                run.done = true;
                commands.write_message(AppExit::error());
                return;
            }
        };
        if std::env::var_os("VOXEL_GI_VALIDATE_SKY").is_some()
            || std::env::var_os("VOXEL_GI_VALIDATE_TOGGLES").is_some()
            || std::env::var_os("VOXEL_GI_SMOKE_EDIT").is_some()
            || std::env::var_os("VOXEL_GI_SMOKE_MOVE").is_some()
        {
            error!(
                "Use VOXEL_GI_VALIDATE_PROFILE=gameplay|sources|compare|stress; remove legacy SKY/TOGGLES and SMOKE flags."
            );
            run.done = true;
            commands.write_message(AppExit::error());
            return;
        }
        run.profile = profile;
        if matches!(profile, Profile::Motion | Profile::Tunnel | Profile::Streaming) {
            run.motion_static = std::env::var_os("VOXEL_GI_VALIDATE_MOTION_STATIC").is_some();
            run.motion_walk = profiles::motion_walk();
            let component = std::env::var("VOXEL_GI_VALIDATE_LIGHT_COMPONENT").ok();
            run.motion_diagnostic = match profiles::motion_diagnostic(
                std::env::var_os("VOXEL_GI_VALIDATE_VIEW_COVERAGE").is_some(), component.as_deref(),
            ) {
                Ok(mode) => mode,
                Err(message) => {
                    error!(message);
                    run.done = true;
                    commands.write_message(AppExit::error());
                    return;
                }
            };
        }
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let output = PathBuf::from(format!(
            "target/gi-validation/{}-{stamp}-{}",
            profile.name(),
            std::process::id()
        ));
        let initialized = std::fs::create_dir_all("target/gi-validation")
            .and_then(|_| std::fs::create_dir(&output))
            .and_then(|_| {
                std::fs::write(
                    output.join("status.json"),
                    json!({"status": "running", "profile": profile.name(), "schema_version": 3,
                        "capture_mode": if evidence.render_target.is_some() { "offscreen" } else { "window" }})
                        .to_string(),
                )
            });
        if let Err(error) = initialized {
            error!(%error, "could not initialize validation artifacts");
            run.done = true;
            commands.write_message(AppExit::error());
            return;
        }
        info!(profile = profile.name(), output = %output.display(), "GI validation initialized");
        evidence.output = Some(output);
        run.memory.start();
        run.startup = Some(Instant::now());
    }
    if run.started.is_none() {
        let ids = [
            assets.voxel_texture.id().untyped(),
            assets.prototype_config.id().untyped(),
        ];
        let failed = ids.iter().any(|id| {
            matches!(
                asset_server.load_state(*id),
                bevy::asset::LoadState::Failed(_)
            )
        });
        let ready = ids
            .iter()
            .all(|id| asset_server.is_loaded_with_dependencies(*id));
        if failed || (!ready && run.startup.unwrap().elapsed().as_secs() >= 15) {
            let reason = "Required terrain assets unavailable; use cargo run or BEVY_ASSET_ROOT=<project directory>.";
            error!(reason);
            evidence.fail(reason);
            commands.write_message(AppExit::error());
            return;
        }
        if !ready || handle.is_none() {
            return;
        }
        run.started = Some(Instant::now());
        info!("GI validation terrain assets verified");
    }
    let elapsed = run.started.unwrap().elapsed().as_secs_f32();
    let profile = run.profile;
    let player = &mut *player_state;
    if crate::cave_scene::background_terrain_requested() {
        run.background
            .tick(elapsed, &evidence, &mut commands, &mut lights);
    }
    if profile == Profile::Structures {
        run.structures
            .tick(elapsed, player, &mut commands, &cameras, &mut lights);
    }
    if matches!(
        profile,
        Profile::Cave
            | Profile::Canyon
            | Profile::DayCycle
            | Profile::Hallway
            | Profile::LiveCave
            | Profile::LiveHallway
    ) && !run.cave_camera_initialized
    {
        let (eye, target, fov) = if profile == Profile::LiveHallway {
            // The user's view: beside the lamp, looking down the dark end.
            let eye = crate::structures::hallway::SOURCE.as_vec3() + Vec3::new(2.5, 1.62, 0.5);
            (eye, eye + Vec3::new(64., -1., 0.), 65.)
        } else if profile == Profile::LiveCave {
            (Vec3::new(47., -11.4, -18.), Vec3::new(76., -17., -34.), 65.)
        } else if profile == Profile::Hallway {
            crate::structures::hallway::overview()
        } else if matches!(profile, Profile::Canyon | Profile::DayCycle) {
            (crate::canyon_scene::EYE, crate::canyon_scene::TARGET, 65.)
        } else {
            let (eye, target) = cave::view();
            (eye, target, 65.)
        };
        player.translation = eye - Vec3::Y * 1.62;
        for camera in &cameras {
            commands
                .entity(camera)
                .remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>()
                .insert(Transform::from_translation(eye).looking_at(target, Vec3::Y))
                .insert(Projection::Perspective(PerspectiveProjection {
                    fov: fov.to_radians(),
                    ..default()
                }));
        }
        run.cave_camera_initialized = true;
    }
    if handoff::enabled(profile) {
        handoff::tick(elapsed, player, &cameras, &mut lights);
    }
    if matches!(
        profile,
        Profile::Cave | Profile::Structures | Profile::LiveCave
    ) {
        run.cave.record(elapsed, &field);
    }
    if profile == Profile::DayCycle {
        run.day.tick(elapsed, &mut commands, &field, &stats, &runtime);
    }
    if matches!(
        profile,
        Profile::House
            | Profile::HouseEdits
            | Profile::Cave
            | Profile::Structures
            | Profile::LiveCave
            | Profile::LiveHallway
    ) {
        run.temporal
            .record(elapsed, profile, &field, &runtime);
    }
    if matches!(profile, Profile::House | Profile::HouseEdits) {
        run.house.tick(
            elapsed,
            profile == Profile::HouseEdits,
            &mut world,
            &mut dirty,
            &field,
            &runtime,
            &mut commands,
            &cameras,
            &mut lights,
        );
    }
    if matches!(profile, Profile::Response | Profile::Edits) {
        run.response.tick(
            elapsed,
            profile == Profile::Edits,
            &mut world,
            &mut dirty,
            &field,
            &mut transport_materials,
            &mut commands,
            &cameras,
            &mut lights,
        );
    }
    if profile == Profile::Tunnel {
        run.tunnel.tick(elapsed, &mut world, &mut dirty, &mut commands, &cameras, &mut lights);
    }
    if profile == Profile::Streaming {
        // Replay an absolute path, independent of asynchronous startup physics
        // and controller synchronization. Mutating the body Transform alone can
        // be overwritten by the KCC and silently turn this into a static test.
        let distance = if profiles::turn_burst() { 0.0 } else { 4.32 * (elapsed - 15.0).clamp(0.0, 15.0) };
        let eye = generator.player_spawn_position() + Vec3::Y * 1.62 + Vec3::Z * distance;
        let yaw = if elapsed >= 15.0 { std::f32::consts::PI } else { 0.0 };
        let transform = Transform::from_translation(eye).with_rotation(Quat::from_rotation_y(yaw));
        for camera in &cameras {
            commands.entity(camera)
                .remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>()
                .insert(transform);
        }
        if (15.0..30.0).contains(&elapsed) { run.moves += 1; }
    }
    if profile == Profile::Motion {
        // Same eye and two repeatable orientations, separated by 0.15 degrees.
        // Exact returns distinguish selection flicker from camera parallax.
        // Walk mode instead moves steadily forward along the view from 15 s.
        let moved = !run.motion_static && run.motion_walk.is_none()
            && (15.0..30.0).contains(&elapsed) && elapsed as u32 % 2 == 1;
        let yaw = if moved { 0.15f32.to_radians() } else { 0.0 };
        // Use the generator's fixed spawn, not the physics body's position
        // during asynchronous startup (which can still be falling).
        let rotation = Quat::from_rotation_y(std::f32::consts::PI + yaw) * Quat::from_rotation_x(-0.12);
        let walked = run.motion_walk.map_or(0.0, |speed| speed * (elapsed - 15.0).clamp(0.0, 15.0));
        let eye = generator.player_spawn_position() + Vec3::Y * 1.62 + rotation * Vec3::NEG_Z * walked;
        let transform = Transform::from_translation(eye).with_rotation(rotation);
        for camera in &cameras {
            commands.entity(camera)
                .remove::<bevy_ahoy::prelude::CharacterControllerCameraOf>()
                .insert(transform);
        }
    }
    if elapsed < 1.0 {
        run.center = player.translation;
    }
    if let Some(handle) = handle.as_ref()
        && let Some(material) = materials.get(&handle.handle)
    {
        let mut next = material.extension.settings;
        profile.configure(elapsed, &mut next);
        if matches!(profile, Profile::Motion | Profile::Tunnel | Profile::Streaming) {
            next.debug_irradiance = run.motion_diagnostic;
        }
        if next != material.extension.settings {
            materials
                .get_mut(&handle.handle)
                .unwrap()
                .extension
                .settings = next;
            run.mode_changes += 1;
        }
    }
    let phase = profile.phase(elapsed);
    if let Some(&(at, name)) = profile.captures().get(run.captures)
        && elapsed >= at
    {
        let output = evidence
            .output
            .as_ref()
            .unwrap()
            .join(format!("{name}.png"));
        commands
            .spawn(
                evidence
                    .render_target
                    .as_ref()
                    .map_or_else(Screenshot::primary_window, |image| {
                        Screenshot::image(image.clone())
                    }),
            )
            .observe(check_capture)
            .observe(save_to_disk(output));
        if matches!(profile, Profile::Response | Profile::Edits) {
            run.response
                .captures
                .push(json!({"name": name, "scheduled_s": at, "requested_s": elapsed}));
        }
        if matches!(profile, Profile::House | Profile::HouseEdits) {
            run.house
                .captures
                .push(json!({"name": name, "scheduled_s": at, "requested_s": elapsed}));
        }
        run.captures += 1;
        run.capture_pause_until = elapsed + 0.25;
    }
    let capture_in_flight = evidence.completed < run.captures
        || evidence
            .last_capture
            .is_some_and(|at| at.elapsed().as_secs_f32() < 0.25)
        || elapsed < run.capture_pause_until;
    let phase_stats = run.metrics.phases.entry(phase).or_default();
    if capture_in_flight {
        phase_stats.excluded_capture_frames += 1;
    } else {
        phase_stats.frames_ms.push(time.delta_secs() * 1000.0);
    }
    run.metrics.record_frame(
        elapsed,
        phase,
        time.delta_secs() * 1000.0,
        capture_in_flight,
        &stats,
        host_timing.as_ref().map(|timing| timing.completed_ms),
    );
    run.metrics.poll_edit(&field);
    run.metrics
        .record_work(elapsed, phase, &stats, &runtime, &field);
    run.metrics
        .sample(&field, player.translation, elapsed, phase);
    run.max_pages = run.max_pages.max(stats.loaded_pages);

    if profile.sources() && (25.0..65.0).contains(&elapsed) {
        if !run.emitter_started {
            let position = IVec3::new(0, 12, -18);
            run.emissive = Some((
                position,
                world.get_voxel(position),
                transport_materials.get(250),
            ));
            transport_materials.set(
                250,
                MaterialTransport {
                    emission: EnergySpectrum::new(Vec3::new(12.0, 3.0, 0.3), 4.0, 1.0),
                    ..default()
                },
            );
            dirty.set_voxel(&mut world, position, WorldVoxel::Solid(250));
            run.emitter_started = true;
        }
        let center = run.center;
        let entity = *run.light.get_or_insert_with(|| {
            commands
                .spawn((
                    PointLight {
                        color: Color::srgb(1.0, 0.2, 0.08),
                        intensity: 3000.0,
                        range: 24.0,
                        ..default()
                    },
                    EnergyEmitter {
                        spectrum: EnergySpectrum::new(Vec3::ZERO, 12.0, 3.0),
                        range: 24.0,
                    },
                    Transform::from_translation(center + Vec3::Y * 4.0),
                ))
                .id()
        });
        if let Ok(mut transform) = lights.get_mut(entity) {
            transform.translation = center
                + Vec3::new(
                    (elapsed * 1.2).cos() * 5.0,
                    4.0,
                    (elapsed * 1.2).sin() * 5.0,
                );
        }
    }
    if profile.edits()
        && (25.0..45.0).contains(&elapsed)
        && ((elapsed - 25.0) * 4.0) as u32 > run.edit_attempts
    {
        let edit = if let Some((voxel, material)) = run.removed.take() {
            Some((voxel, material, false))
        } else {
            let eye = player.translation;
            let target = field
                .probes
                .keys()
                .copied()
                .filter(|voxel| (9.0..144.0).contains(&voxel.as_vec3().distance_squared(eye)))
                .min_by_key(|voxel| {
                    (
                        (((voxel.x as i64 * 73_856_093) ^ (voxel.z as i64 * 19_349_663))
                            + run.edits as i64 * 83_492_791)
                            .unsigned_abs(),
                        voxel.x,
                        voxel.y,
                        voxel.z,
                    )
                });
            target.and_then(|voxel| {
                let material = world.get_voxel(voxel);
                if !matches!(material, WorldVoxel::Solid(_)) {
                    return None;
                }
                run.removed = Some((voxel, material));
                Some((voxel, WorldVoxel::Air, true))
            })
        };
        if let Some((voxel, material, removing)) = edit {
            let targets = edit_receivers(voxel, removing, |position| world.get_voxel(position));
            run.metrics.edit(&field, targets);
            dirty.set_voxel(&mut world, voxel, material);
            run.edits += 1;
        }
        run.edit_attempts += 1;
    }
    if matches!(
        profile,
        Profile::Gameplay | Profile::Sources | Profile::Stress
    ) && (45.0..65.0).contains(&elapsed)
        && ((elapsed - 45.0) / 4.0) as u32 >= run.moves
    {
        player.translation += Vec3::new(24.0, 2.0, 0.0);
        run.moves += 1;
    }
    if elapsed >= 45.0
        && let Some((voxel, material)) = run.removed.take()
    {
        let targets = edit_receivers(voxel, false, |position| world.get_voxel(position));
        run.metrics.edit(&field, targets);
        dirty.set_voxel(&mut world, voxel, material);
        run.edits += 1;
    }
    if elapsed >= 65.0 {
        if let Some((position, voxel, transport)) = run.emissive.take() {
            dirty.set_voxel(&mut world, position, voxel);
            transport_materials.set(250, transport);
        }
        if let Some(entity) = run.light.take() {
            commands.entity(entity).despawn();
        }
    }
    if elapsed < profile.duration() {
        return;
    }
    let captures_saved = profile.captures().iter().all(|(_, name)| {
        evidence
            .output
            .as_ref()
            .unwrap()
            .join(format!("{name}.png"))
            .metadata()
            .is_ok_and(|metadata| metadata.len() > 0)
    });
    if evidence.completed != profile.captures().len() || !captures_saved {
        if elapsed < profile.duration() + 5.0 {
            return;
        }
        evidence.fail("Not all required screenshots completed and saved.");
        commands.write_message(AppExit::error());
        return;
    }
    let config = configs.get(&assets.prototype_config);
    if let Err(error) = run.temporal.save(evidence.output.as_ref().unwrap()) {
        evidence.fail(&format!("Could not save temporal sequence: {error}"));
        commands.write_message(AppExit::error());
        return;
    }
    if matches!(profile, Profile::Cave | Profile::Structures | Profile::LiveCave)
        && let Err(error) = std::fs::write(evidence.output.as_ref().unwrap().join("cave.json"),
            json!({"samples": run.cave.samples, "note": "Fixed wall receivers; all five measured bands, RGB display, history and measurement age. No ground-truth/convergence claim."}).to_string()) {
        evidence.fail(&format!("Could not save cave sequence: {error}"));
        commands.write_message(AppExit::error());
        return;
    }
    if profile == Profile::DayCycle
        && let Err(error) = std::fs::write(evidence.output.as_ref().unwrap().join("day.json"),
            json!({"settings": run.day.settings(), "samples": run.day.samples,
                "note": "Canyon, fixed camera, sun rotating about world Z. Sample counts are the CPU statistics mirror (current only with gi-mirror-full.json). Paired captures one second apart show flicker."}).to_string()) {
        evidence.fail(&format!("Could not save day sequence: {error}"));
        commands.write_message(AppExit::error());
        return;
    }
    if matches!(profile, Profile::House | Profile::HouseEdits)
        && let Err(error) = std::fs::write(evidence.output.as_ref().unwrap().join("house.json"),
            json!({"samples": run.house.samples, "events": run.house.events, "captures": run.house.captures,
                "note": "Prototype house, fixed eye, opposite ceiling corners at 30/50/70s. House-edits opens a skylight at 90s, west door at 120s, east door plus third-corner turn at 150s. Fixed warm-up is not certified convergence. Normal terrain streaming and adaptive scheduling remain enabled. Not a windowed FPS benchmark."}).to_string()) {
            evidence.fail(&format!("Could not save house sequence: {error}"));
            commands.write_message(AppExit::error());
            return;
    }
    if matches!(profile, Profile::Response | Profile::Edits)
        && let Err(error) = std::fs::write(evidence.output.as_ref().unwrap().join("response.json"),
            json!({"samples": run.response.samples, "captures": run.response.captures,
                "note": "Fixed-camera synthetic room. Capture names are scheduled offsets; requested_s records actual timing. Not a gameplay/FPS benchmark."}).to_string()) {
            evidence.fail(&format!("Could not save response sequence: {error}"));
            commands.write_message(AppExit::error());
            return;
    }
    if profile == Profile::Tunnel
        && let Err(error) = std::fs::write(evidence.output.as_ref().unwrap().join("tunnel.json"),
            json!({"events": run.tunnel.events,
                "note": "One-wide two-high tunnel, 20 depths, upper block then lower 250ms later. Capture offsets follow upper edit; 400ms captures are 150ms after lower edit. Fixed warmup, not certified convergence."}).to_string()) {
        evidence.fail(&format!("Could not save tunnel sequence: {error}"));
        commands.write_message(AppExit::error());
        return;
    }
    let (memory, footprint) = run.memory.finish();
    if let Some(text) = footprint {
        // Category breakdown at the end of the workload (IOAccelerator is GPU).
        let _ = std::fs::write(evidence.output.as_ref().unwrap().join("footprint.txt"), text);
    }
    let report = json!({
        "schema_version": 3, "status": "completed", "profile": profile.name(),
        "capture_mode": if evidence.render_target.is_some() { "offscreen" } else { "window" },
        "timing_domain": if evidence.render_target.is_some() { "offscreen app cadence; 60Hz ScheduleRunner; no swapchain/vsync" } else { "windowed frame pacing" },
        "performance_baseline": profile == Profile::Gameplay && evidence.render_target.is_none(),
        "build": "development", "backend": stats.backend,
        "compiled_manifest_dir": env!("CARGO_MANIFEST_DIR"),
        "scene": match crate::scene_presets::ScenePreset::current() {
            crate::scene_presets::ScenePreset::Cave => "many-emitter cave",
            crate::scene_presets::ScenePreset::Canyon => "daylight canyon",
            crate::scene_presets::ScenePreset::Terrain => "terrain",
        },
        "house_only_control": crate::structures::showcase::validation_house_only(),
        "cache_handoff": handoff::enabled(run.profile),
        "background_terrain": {"enabled": crate::cave_scene::background_terrain_requested(),
            "moves": run.background.moves, "speed_voxels_per_second": 4.32,
            "note": "Inactive BVW loader walks from (-96,40,0) toward -Z. Does not render or alter GI view priority. Loaded terrain participates in transport."},
        "presentation": "shared-corner sun + corner-reconstructed indirect; preferred modes except compare/stress",
        "view_diagnostic": run.motion_diagnostic,
        "motion_turn_degrees": if run.motion_static || run.motion_walk.is_some() { 0.0 } else { 0.15 },
        "motion_walk_voxels_per_second": run.motion_walk,
        "spatial_retention": super::retention::POLICY_NAME,
        "settings": {"transport": settings.transport, "cache_radius": settings.cache_radius,
            "cache_share": settings.cache_share,
            "samples_per_dispatch": settings.batch_samples(),
            "simulation_sampling_importance": settings.sampling_importance().infrared,
            "spatial_reconstruction": settings.spatial_reconstruction,
            "spatial_evidence_target": settings.spatial_evidence_target,
            "temporal_trace_scope": "world-cache receivers only; use captures for presentation",
            "trace_budget_ms": settings.trace_budget_ms, "gpu_time_share": settings.gpu_time_share,
            "max_probes_per_dispatch": settings.max_probes_per_dispatch,
            "scene_publish_budget_ms": settings.scene_publish_budget.as_secs_f32() * 1000.0,
            "adaptive_updates": settings.adaptive_updates, "display_response_ms": settings.display_response.as_secs_f32() * 1000.0,
            "measured_retention": settings.measured_retention,
            "stable_min_samples": settings.stable_min_samples, "stable_relative_error": settings.stable_relative_error,
            "stable_audit_interval_ms": settings.stable_audit_interval.as_secs_f32() * 1000.0},
        "terrain": config.map(|c| json!({"seed": c.world_seed, "period": c.terrain_period, "ambient_occlusion": c.ambient_occlusion})),
        "captures": evidence.completed, "capture_windows": evidence.windows,
        "edits": run.edits, "edit_attempts": run.edit_attempts, "moves": run.moves, "mode_changes": run.mode_changes,
        "max_loaded_pages": run.max_pages, "coverage": stats.interactive_coverage, "sampled_mean_history": stats.mean_history_samples,
        "sky_rays_scheduled": stats.sky_rays_scheduled, "sky_cached_corners": stats.sky_cached_corners,
        "transport_batches": stats.transport_batches, "sky_only_batches": stats.sky_only_batches,
        "traced_paths": stats.traced_paths, "packed_geometry_bytes": stats.packed_geometry_bytes,
        "last_selection_ms": stats.last_selection_ms, "last_merge_ms": stats.last_merge_ms,
        "measurements": run.metrics.report(),
        "memory": memory,
        "measurement_notes": "Field observations sample at most 1024 voxels every 250 ms; repeated observations are not independent. Display steps are not image-space noise or error. Edit latency is first fresh measurement of selected adjacent receivers, not visual convergence. Capture/readback plus 250 ms cooldown frames are excluded. Offscreen timings are not windowed gameplay FPS and must not be compared against a windowed baseline. Completed is workload integrity, not a quality/performance pass."
    });
    let output = evidence.output.as_ref().unwrap();
    let saved = std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .and_then(|_| {
        std::fs::write(
            output.join("status.json"),
            json!({"status": "completed", "profile": profile.name(), "schema_version": 3,
                "capture_mode": if evidence.render_target.is_some() { "offscreen" } else { "window" }})
                .to_string(),
        )
    });
    run.done = true;
    if let Err(error) = saved {
        error!(%error, "could not save validation report");
        evidence.fail("Report publication failed.");
        commands.write_message(AppExit::error());
    } else {
        info!(output = %output.display(), "GI validation workload complete (review quality/performance separately)");
        if reference::ReferenceRun::requested() {
            if let Err(error) = run.reference.start(
                &runtime,
                device.as_deref(),
                &settings,
                profile,
                output.clone(),
            ) {
                evidence.fail(&error);
                commands.write_message(AppExit::error());
            }
        } else if std::env::var_os("VOXEL_GI_VALIDATE_EXIT").is_some() {
            commands.write_message(AppExit::Success);
        }
    }
}

fn edit_receivers(
    voxel: IVec3,
    removing: bool,
    lookup: impl Fn(IVec3) -> WorldVoxel<u8>,
) -> Vec<(IVec3, usize)> {
    FACE_OFFSETS
        .iter()
        .enumerate()
        .filter_map(|(face, offset)| {
            let neighbor = voxel + *offset;
            match (removing, lookup(neighbor)) {
                (true, WorldVoxel::Solid(_)) => Some((neighbor, face ^ 1)),
                (false, WorldVoxel::Air) => Some((voxel, face)),
                _ => None,
            }
        })
        .collect()
}

fn check_capture(
    capture: On<ScreenshotCaptured>,
    mut evidence: ResMut<CaptureEvidence>,
    mut commands: Commands,
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    cameras: Query<(&Camera, &GlobalTransform, &RenderTarget)>,
) {
    let varied = capture
        .image
        .clone()
        .try_into_dynamic()
        .is_ok_and(|image| has_rgb_variation(image.to_rgb8().as_raw()));
    let offscreen = evidence.render_target.is_some();
    let mut target_valid = false;
    if let Some(target) = &evidence.render_target {
        target_valid = capture.image.texture_descriptor.size.width == OFFSCREEN_WIDTH
            && capture.image.texture_descriptor.size.height == OFFSCREEN_HEIGHT
            && cameras.iter().any(|(camera, _, render_target)| {
                camera.is_active && render_target.as_image() == Some(target)
            });
        evidence.windows.push(json!({"capture_mode": "offscreen", "width": OFFSCREEN_WIDTH, "height": OFFSCREEN_HEIGHT,
            "present_mode": null, "focused": null, "visible": null, "target_valid": target_valid, "varied": varied}));
        info!(target_valid, varied, "GI offscreen validation capture");
    }
    for window in &windows {
        if !offscreen {
            target_valid |= window.visible
                && window.focused
                && window.physical_width() > 0
                && window.physical_height() > 0;
        }
        evidence.windows.push(json!({"focused": window.focused, "visible": window.visible,
            "width": window.physical_width(), "height": window.physical_height(), "present_mode": format!("{:?}", window.present_mode), "varied": varied}));
        info!(
            focused = window.focused,
            visible = window.visible,
            varied,
            "GI validation capture"
        );
    }
    for (camera, transform, _) in &cameras {
        info!(active = camera.is_active, position = ?transform.translation(), forward = ?transform.forward(), "GI validation camera");
    }
    evidence.completed += 1;
    evidence.last_capture = Some(Instant::now());
    if !varied || !target_valid {
        let reason = if offscreen {
            "Offscreen capture is uniform/black, has unexpected dimensions, or has no active camera targeting its image."
        } else {
            "Capture is uniform/black or game is not foreground/visible; use VOXEL_GI_VALIDATE_HEADLESS=1 for screen-independent captures."
        };
        error!(reason);
        evidence.fail(reason);
        commands.write_message(AppExit::error());
    }
}

fn has_rgb_variation(rgb: &[u8]) -> bool {
    rgb.len() >= 3 && rgb.chunks_exact(3).any(|pixel| pixel != &rgb[..3])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn uniform_clear_or_black_frames_are_not_gameplay_evidence() {
        assert!(!has_rgb_variation(&[]));
        assert!(!has_rgb_variation(&[0; 12]));
        assert!(!has_rgb_variation(&[100, 180, 240, 100, 180, 240]));
        assert!(has_rgb_variation(&[100, 180, 240, 100, 179, 240]));
    }

    #[test]
    fn edit_receivers_follow_new_faces_not_the_removed_face() {
        assert_eq!(
            edit_receivers(IVec3::ZERO, true, |v| if v == IVec3::NEG_Y {
                WorldVoxel::Solid(1)
            } else {
                WorldVoxel::Air
            }),
            vec![(IVec3::NEG_Y, 2)]
        );
        assert_eq!(
            edit_receivers(IVec3::ZERO, false, |v| if v == IVec3::Y {
                WorldVoxel::Air
            } else {
                WorldVoxel::Solid(1)
            }),
            vec![(IVec3::ZERO, 2)]
        );
    }
}
