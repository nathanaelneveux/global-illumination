//! Opt-in, read-only normal-gameplay receiver history. F8 pins the aimed face;
//! F9 saves it. Never requests samples, freezes the world, or moves a camera.
use super::*;
use crate::{
    terrain::PrototypeWorld,
    voxel_gi::{
        VoxelGiSettings,
        render::{GpuProbePatchQueue, GpuTransportCache},
        validation::temporal::TemporalRun,
        world::FACE_OFFSETS,
    },
};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy_voxel_world::prelude::{VoxelWorld, WorldVoxel};
use serde::Serialize;
use std::path::PathBuf;

#[derive(Serialize, PartialEq)]
struct Context {
    generation: u64,
    transport_revision: u64,
    source_revision: u64,
    sky_revision: u64,
}

#[derive(Serialize)]
struct Pose {
    at_s: f32,
    position: [f32; 3],
    forward: [f32; 3],
}

struct Recording {
    run: TemporalRun,
    output: PathBuf,
    voxel: IVec3,
    face: usize,
    started: Instant,
    poses: Vec<Pose>,
    contexts: Vec<(f32, Context)>,
    settings: Vec<(f32, VoxelGiSettings)>,
    next_pose: f32,
}

impl Recording {
    fn new(voxel: IVec3, face: usize) -> Self {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self {
            run: TemporalRun::pinned(voxel, face),
            output: PathBuf::from(format!(
                "target/gi-face-watch/{stamp}-{}",
                std::process::id()
            )),
            voxel,
            face,
            started: Instant::now(),
            poses: Vec::new(),
            contexts: Vec::new(),
            settings: Vec::new(),
            next_pose: 0.0,
        }
    }

    fn save(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.output)?;
        self.run.save(&self.output)?;
        let data = serde_json::json!({"schema_version":1,"voxel":self.voxel.to_array(),"face":self.face,
            "compiled_manifest_dir":env!("CARGO_MANIFEST_DIR"),"poses":self.poses,
            "contexts":self.contexts,"settings":self.settings,
            "note":"Normal gameplay, read-only pin. No extra rays, forced refresh, scene freeze or reference bake. Temporal mirror is the latest completed/queued CPU snapshot, not a direct read of the currently rendered GPU frame. Pinning cannot reconstruct history before the pin."});
        serde_json::to_writer(
            std::io::BufWriter::new(std::fs::File::create(self.output.join("watch.json"))?),
            &data,
        )?;
        info!(output=%self.output.display(), "GI face watch saved");
        Ok(())
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        if let Err(error) = self.save() {
            error!(%error, "GI face watch could not be saved");
        }
    }
}

#[derive(Resource, Default)]
struct FaceWatch {
    recording: Option<Recording>,
    launch: Option<(IVec3, usize)>,
}

fn parse_site(value: &str) -> Result<(IVec3, usize), String> {
    let parts: Vec<i32> = value
        .split(',')
        .map(|v| v.trim().parse::<i32>())
        .collect::<Result<_, _>>()
        .map_err(|e| e.to_string())?;
    if parts.len() != 4 || !(0..6).contains(&parts[3]) {
        return Err("Expected x,y,z,face with face in 0..5".into());
    }
    Ok((IVec3::new(parts[0], parts[1], parts[2]), parts[3] as usize))
}

pub(super) fn install(app: &mut App) {
    let launch = std::env::var("VOXEL_GI_WATCH_FACE")
        .ok()
        .map(|v| parse_site(&v).expect("Invalid VOXEL_GI_WATCH_FACE"));
    app.insert_resource(FaceWatch {
        launch,
        ..default()
    })
    .add_systems(
        Update,
        (
            pin.before(record)
                .run_if(|keys: Res<ButtonInput<KeyCode>>| keys.just_pressed(KeyCode::F8)),
            record
                .after(VoxelGiSystems)
                .run_if(|watch: Res<FaceWatch>| {
                    watch.recording.is_some() || watch.launch.is_some()
                }),
        ),
    );
}

fn pin(
    mut commands: Commands,
    mut watch: ResMut<FaceWatch>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    world: VoxelWorld<PrototypeWorld>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let Some((_, camera)) = cameras.iter().find(|(c, _)| c.is_active) else {
        return;
    };
    let Some(hit) =
        world.raycast(
            Ray3d::new(camera.translation(), camera.forward()),
            &|(_, v)| matches!(v, WorldVoxel::Solid(_)),
        )
    else {
        return;
    };
    let Some(normal) = hit.normal else {
        return;
    };
    let Some(face) = FACE_OFFSETS.iter().position(|n| n.as_vec3() == normal) else {
        return;
    };
    let voxel = hit.position.as_ivec3();
    let recording = Recording::new(voxel, face);
    capture(&mut commands, &windows, recording.output.join("pinned.png"));
    watch.launch = None;
    watch.recording = Some(recording);
    info!(
        ?voxel,
        face, "GI face watch pinned (F9 saves; exit also saves)"
    );
}

fn capture(commands: &mut Commands, windows: &Query<&Window, With<PrimaryWindow>>, path: PathBuf) {
    if !windows
        .iter()
        .any(|w| w.focused && w.width() > 0.0 && w.height() > 0.0)
    {
        return;
    }
    commands
        .spawn(Screenshot::primary_window())
        .observe(move |event: On<ScreenshotCaptured>| {
            let result = (|| {
                let image = event
                    .image
                    .clone()
                    .try_into_dynamic()
                    .map_err(|e| format!("{e:?}"))?;
                let rgb = image.to_rgb8();
                let (lo, hi) = rgb
                    .as_raw()
                    .iter()
                    .fold((255u8, 0u8), |(lo, hi), &v| (lo.min(v), hi.max(v)));
                if rgb.width() == 0 || rgb.height() == 0 || hi <= lo {
                    return Err("Blank face-watch capture".to_string());
                }
                std::fs::create_dir_all(path.parent().unwrap()).map_err(|e| e.to_string())?;
                image.save(&path).map_err(|e| e.to_string())
            })();
            if let Err(error) = result {
                error!(%error, "GI face watch capture rejected");
            }
        });
}

fn record(
    mut commands: Commands,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut watch: ResMut<FaceWatch>,
    cameras: Query<(&Camera, &GlobalTransform), With<Camera3d>>,
    field: Res<VoxelEnergyField>,
    runtime: Res<VoxelGiRuntime>,
    settings: Res<VoxelGiSettings>,
    cache: Res<GpuTransportCache>,
    queue: Res<GpuProbePatchQueue>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    let camera = cameras.iter().find(|(c, _)| c.is_active).map(|(_, t)| t);
    if let Some((voxel, face)) = watch.launch.take() {
        watch.recording = Some(Recording::new(voxel, face));
        info!(
            ?voxel,
            face, "GI face watch pinned (F9 saves; exit also saves)"
        );
    }
    let Some(watch) = watch.recording.as_mut() else {
        return;
    };
    let at_s = watch.started.elapsed().as_secs_f32();
    watch
        .run
        .record_pinned(at_s, &field, &runtime, &cache, &queue);
    let context = Context {
        generation: runtime.generation,
        transport_revision: field.transport_revision,
        source_revision: runtime.source_revision,
        sky_revision: runtime.sky_revision,
    };
    if watch.contexts.last().is_none_or(|(_, old)| *old != context) {
        watch.contexts.push((at_s, context));
    }
    if watch
        .settings
        .last()
        .is_none_or(|(_, old)| old != &*settings)
    {
        watch.settings.push((at_s, settings.clone()));
    }
    if at_s >= watch.next_pose {
        if let Some(camera) = camera {
            watch.poses.push(Pose {
                at_s,
                position: camera.translation().to_array(),
                forward: camera.forward().to_array(),
            });
        }
        watch.next_pose = at_s + 0.25;
    }
    if keyboard.just_pressed(KeyCode::F9) {
        if let Err(error) = watch.save() {
            error!(%error, "GI face watch save failed");
        }
        capture(&mut commands, &windows, watch.output.join("saved.png"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn watch_coordinates_reject_missing_and_invalid_faces() {
        assert_eq!(
            parse_site("-54,-12,-20,2").unwrap(),
            (IVec3::new(-54, -12, -20), 2)
        );
        for value in ["1,2,3", "1,2,3,6", "1,2,3,-1", "1,2,x,2", "1,2,3,2,0"] {
            assert!(parse_site(value).is_err());
        }
    }
}
