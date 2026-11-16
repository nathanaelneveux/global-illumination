//! A minimal voxel GI host with no voxel engine: one static 32³ page (a
//! room with a lamp and an open side), drawn as one cube per exposed voxel
//! with `VoxelGiStandardMaterial`.
//!
//! `cargo run -p bevy_voxel_gi --example minimal`
//! (`VOXEL_GI_EXAMPLE_FRAMES=n` exits after n frames; with
//! `VOXEL_GI_EXAMPLE_SCREENSHOT=path` the camera renders offscreen and the
//! image is saved first, so the check works with the screen asleep.)
use bevy::prelude::*;
use bevy_voxel_gi::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Block {
    Stone,
    Lamp,
}

/// The room: a floor, two walls and a ceiling, open towards +x and +z, a
/// pillar, and a lamp on the floor. Page (0, 0, 0) only; everything else
/// is air.
fn block(p: IVec3) -> Option<Block> {
    if p.cmplt(IVec3::ZERO).any() || p.cmpge(IVec3::splat(32)).any() {
        return None;
    }
    let inside = p.x < 24 && p.z < 24;
    let shell = p.y == 0 || (inside && (p.x == 0 || p.z == 0 || p.y == 12));
    let pillar = (10..12).contains(&p.x) && (10..12).contains(&p.z) && p.y < 12;
    if p == IVec3::new(6, 1, 6) {
        return Some(Block::Lamp);
    }
    (shell || pillar).then_some(Block::Stone)
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, VoxelGiPlugins))
        .add_systems(Startup, setup)
        .add_systems(Update, exit_after_frames)
        .run();
}

#[derive(Resource)]
struct Offscreen(Handle<Image>);

fn setup(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    mut input: ResMut<VoxelGiInput>,
    mut gi_materials: ResMut<VoxelGiMaterials>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VoxelGiStandardMaterial>>,
) {
    // Optics per block type; identical optics would share a material.
    let stone = MaterialTransport {
        diffuse: Vec3::splat(0.6),
        ..default()
    };
    let lamp = MaterialTransport {
        diffuse: Vec3::splat(0.8),
        emission: Vec3::new(40.0, 28.0, 14.0),
    };
    let ids = [
        gi_materials.insert(stone).unwrap(),
        gi_materials.insert(lamp).unwrap(),
    ];
    // GI samples 32³ pages from the source on worker threads; a closure is
    // a source. Outside the room everything is air.
    commands.insert_resource(VoxelGiSource::new(move |page: IVec3| {
        dense_page(page, |p| block(p).map(|b| ids[b as usize]))
    }));
    input.page_loaded(IVec3::ZERO);

    let cube = meshes.add(Cuboid::from_length(1.0));
    let stone = materials.add(standard_material(StandardMaterial {
        base_color: Color::srgb(0.6, 0.6, 0.6),
        perceptual_roughness: 0.9,
        ..default()
    }));
    let lamp = materials.add(standard_material(StandardMaterial {
        base_color: Color::srgb(1.0, 0.8, 0.5),
        ..default()
    }));
    for z in 0..32 {
        for y in 0..13 {
            for x in 0..32 {
                let p = IVec3::new(x, y, z);
                let Some(kind) = block(p) else { continue };
                let exposed = [
                    IVec3::X,
                    IVec3::NEG_X,
                    IVec3::Y,
                    IVec3::NEG_Y,
                    IVec3::Z,
                    IVec3::NEG_Z,
                ]
                .into_iter()
                .any(|n| block(p + n).is_none());
                if exposed {
                    let material = if kind == Block::Lamp {
                        lamp.clone()
                    } else {
                        stone.clone()
                    };
                    commands.spawn((
                        Mesh3d(cube.clone()),
                        MeshMaterial3d(material),
                        Transform::from_translation(p.as_vec3() + 0.5),
                    ));
                }
            }
        }
    }
    // GI traces the sun's visibility itself; Bevy's shadow maps are unused.
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(-0.4, -0.8, -0.5), Vec3::Y),
    ));
    // `VoxelGiCamera` marks the cameras GI lights (it adds the depth and
    // normal prepasses GI reads).
    let camera = commands
        .spawn((
            Camera3d::default(),
            VoxelGiCamera,
            Transform::from_xyz(34.0, 9.0, 34.0).looking_at(Vec3::new(8.0, 3.0, 8.0), Vec3::Y),
        ))
        .id();
    if std::env::var_os("VOXEL_GI_EXAMPLE_SCREENSHOT").is_some() {
        let mut image = Image::new_target_texture(
            1280,
            720,
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb,
            None,
        );
        image.texture_descriptor.usage |= bevy::render::render_resource::TextureUsages::COPY_SRC;
        let target = images.add(image);
        commands
            .entity(camera)
            .insert(bevy::camera::RenderTarget::Image(target.clone().into()));
        commands.insert_resource(Offscreen(target));
    }
}

fn exit_after_frames(
    mut commands: Commands,
    offscreen: Option<Res<Offscreen>>,
    mut frames: Local<u32>,
    mut exit: MessageWriter<AppExit>,
) {
    use bevy::render::view::screenshot::{Screenshot, save_to_disk};
    *frames += 1;
    let Some(limit) = std::env::var("VOXEL_GI_EXAMPLE_FRAMES")
        .ok()
        .and_then(|n| n.parse::<u32>().ok())
    else {
        return;
    };
    if *frames + 30 == limit
        && let (Ok(path), Some(offscreen)) =
            (std::env::var("VOXEL_GI_EXAMPLE_SCREENSHOT"), offscreen)
    {
        commands
            .spawn(Screenshot::image(offscreen.0.clone()))
            .observe(save_to_disk(path));
    }
    if *frames >= limit {
        exit.write(AppExit::Success);
    }
}
