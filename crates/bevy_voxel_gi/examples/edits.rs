//! Edits: the minimal example's room with a window in its back wall that
//! opens and closes every two seconds. Each change is one `edit_voxel` per
//! voxel; GI relights the room from the new opening while the rest of its
//! light history stays.
//!
//! `cargo run -p bevy_voxel_gi --example edits`
use bevy::prelude::*;
use bevy_voxel_gi::{
    MaterialPalette, MaterialTransport, PageSample, VoxelGiInput, VoxelGiMaterialData,
    VoxelGiPlugin, VoxelGiSource, VoxelGiStandardMaterial, VoxelGiStandardMaterialPlugin,
    VoxelMaterialTransportTable, VoxelSource, dense_page, standard_material,
};
use std::sync::Arc;

/// The room: a floor, two walls and a ceiling, open towards +x and +z.
/// The window is part of the x = 0 wall at the start.
fn stone(p: IVec3) -> bool {
    if p.cmplt(IVec3::ZERO).any() || p.cmpge(IVec3::splat(32)).any() {
        return false;
    }
    let inside = p.x < 24 && p.z < 24;
    p.y == 0 || (inside && (p.x == 0 || p.z == 0 || p.y == 12))
}

/// A 3 x 4 opening in the x = 0 wall.
fn opening_voxels() -> impl Iterator<Item = IVec3> {
    (3..7).flat_map(|y| (9..12).map(move |z| IVec3::new(0, y, z)))
}

/// GI's view of the room as it was built. Edits reach GI through
/// `VoxelGiInput::edit_voxel` and stay applied on top of this source, so
/// it never needs rebuilding.
struct Room {
    stone: u8,
}

impl VoxelSource for Room {
    fn sample_page(&self, page: IVec3, _resample: bool) -> PageSample {
        dense_page(page, |p| stone(p).then_some(self.stone))
    }
}

#[derive(Resource)]
struct Opening {
    open: bool,
    timer: Timer,
    stone: u8,
    cube: Handle<Mesh>,
    material: Handle<VoxelGiStandardMaterial>,
    /// The window's cube entities while it is closed.
    cubes: Vec<Entity>,
}

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, VoxelGiPlugin, VoxelGiStandardMaterialPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, toggle_window)
        .run();
}

fn setup(
    mut commands: Commands,
    mut input: ResMut<VoxelGiInput>,
    mut source: ResMut<VoxelGiSource>,
    mut table: ResMut<VoxelMaterialTransportTable>,
    gi_bindings: Res<VoxelGiMaterialData>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VoxelGiStandardMaterial>>,
) {
    let mut palette = MaterialPalette::new();
    let optics = MaterialTransport {
        diffuse: Vec3::splat(0.6),
        emission: Vec3::ZERO,
    };
    let stone_id = palette.insert("stone", optics).unwrap() as u8;
    *table = palette.table().clone();
    source.0 = Some(Arc::new(Room { stone: stone_id }));
    input.page_loaded(IVec3::ZERO);

    let cube = meshes.add(Cuboid::from_length(1.0));
    let material = materials.add(standard_material(
        StandardMaterial {
            base_color: Color::srgb(0.6, 0.6, 0.6),
            perceptual_roughness: 0.9,
            ..default()
        },
        &gi_bindings,
    ));
    let mut cubes = Vec::new();
    for z in 0..32 {
        for y in 0..13 {
            for x in 0..32 {
                let p = IVec3::new(x, y, z);
                let exposed = IVec3::AXES
                    .iter()
                    .flat_map(|&n| [n, -n])
                    .any(|n| !stone(p + n));
                if stone(p) && exposed {
                    let entity = commands
                        .spawn((
                            Mesh3d(cube.clone()),
                            MeshMaterial3d(material.clone()),
                            Transform::from_translation(p.as_vec3() + 0.5),
                        ))
                        .id();
                    if opening_voxels().any(|w| w == p) {
                        cubes.push(entity);
                    }
                }
            }
        }
    }
    commands.insert_resource(Opening {
        open: false,
        timer: Timer::from_seconds(2.0, TimerMode::Repeating),
        stone: stone_id,
        cube,
        material,
        cubes,
    });

    // A low sun from -x, so the window throws a patch of light on the floor.
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(0.8, -0.5, 0.2), Vec3::Y),
    ));
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(30.0, 9.0, 30.0).looking_at(Vec3::new(4.0, 3.0, 8.0), Vec3::Y),
    ));
}

/// Opens or closes the window: the meshes change, and GI is told exactly
/// which voxels did.
fn toggle_window(
    time: Res<Time>,
    mut opening: ResMut<Opening>,
    mut input: ResMut<VoxelGiInput>,
    mut commands: Commands,
) {
    if !opening.timer.tick(time.delta()).just_finished() {
        return;
    }
    opening.open = !opening.open;
    for p in opening_voxels() {
        // `None` is air; `Some(id)` a solid voxel of that material.
        let material = (!opening.open).then_some(opening.stone);
        input.edit_voxel(p, material);
    }
    if opening.open {
        for entity in opening.cubes.drain(..) {
            commands.entity(entity).despawn();
        }
    } else {
        let (cube, material) = (opening.cube.clone(), opening.material.clone());
        let cubes = opening_voxels()
            .map(|p| {
                commands
                    .spawn((
                        Mesh3d(cube.clone()),
                        MeshMaterial3d(material.clone()),
                        Transform::from_translation(p.as_vec3() + 0.5),
                    ))
                    .id()
            })
            .collect();
        opening.cubes = cubes;
    }
}
