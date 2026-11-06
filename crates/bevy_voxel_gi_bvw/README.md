# bevy_voxel_gi_bvw

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi_bvw.svg)](https://crates.io/crates/bevy_voxel_gi_bvw)
[![docs.rs](https://docs.rs/bevy_voxel_gi_bvw/badge.svg)](https://docs.rs/bevy_voxel_gi_bvw)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi_bvw.svg)

Connects [`bevy_voxel_world`](https://crates.io/crates/bevy_voxel_world) 0.17
to [`bevy_voxel_gi`](https://docs.rs/bevy_voxel_gi): chunk events become GI page events, chunk
data (including its stored levels of detail) becomes GI's `VoxelSource`, and
`VoxelTransportMaterial` is BVW's texture-array terrain material lit by GI.

```toml
[dependencies]
bevy_voxel_gi = "0.1"
bevy_voxel_gi_bvw = "0.1"
bevy_voxel_world = "0.17"
```

Any `VoxelWorldConfig` with `u8` materials works; a BVW chunk is exactly one
GI page (32³), and chunk spawn, update and despawn become page events.

## Quick start

```rust,no_run
use bevy::prelude::*;
use bevy_voxel_gi::{VoxelGiMaterialData, VoxelGiPlugin};
use bevy_voxel_gi_bvw::{
    GiFallbackVoxel, VoxelGiBvwPlugin, material_template, prepare_texture_array,
    texture_array_ready,
};
use bevy_voxel_world::prelude::*;

#[derive(Resource, Clone, Default)]
struct MyWorld;

impl VoxelWorldConfig for MyWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();
}

// Where no chunk is loaded, GI asks your world for voxels; by default
// that space is air. Return your generator's voxel here if you have one.
impl GiFallbackVoxel for MyWorld {}

/// Block textures stacked vertically in one image, one layer per block.
#[derive(Resource)]
struct Blocks(Handle<Image>);
const LAYERS: u32 = 4;

fn main() {
    let mut app = App::new();
    app.add_plugins((DefaultPlugins, VoxelGiPlugin));
    // BVW's GI-lit material needs GI's binding data, so build it after
    // VoxelGiPlugin is added.
    let blocks = app.world().resource::<AssetServer>().load("blocks.png");
    let gi_bindings = app.world().resource::<VoxelGiMaterialData>().clone();
    app.add_plugins((
        VoxelWorldPlugin::with_config(MyWorld)
            .with_material(material_template(blocks.clone(), &gi_bindings)),
        // Chunk events become GI page events; chunks become GI's voxels.
        VoxelGiBvwPlugin::<MyWorld>::default(),
    ))
    .insert_resource(Blocks(blocks))
    .add_systems(Startup, setup)
    .add_systems(Update, prepare_blocks)
    .run();
}

fn setup(mut commands: Commands) {
    // BVW streams chunks around this camera; GI lights what it sees.
    commands.spawn((
        Camera3d::default(),
        VoxelWorldCamera::<MyWorld>::default(),
        Transform::from_xyz(0.0, 40.0, 60.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    // GI traces the sun's visibility itself: no shadow maps.
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(-0.4, -0.8, -0.5), Vec3::Y),
    ));
}

/// Turns the loaded strip into a mipmapped texture array, once.
fn prepare_blocks(blocks: Res<Blocks>, mut images: ResMut<Assets<Image>>) {
    let Some(mut image) = images.get_mut(&blocks.0) else {
        return;
    };
    // Check before writing: a mutable write re-uploads the texture.
    if !texture_array_ready(&image, LAYERS) {
        prepare_texture_array(&mut image, LAYERS).expect("a vertical strip of layers");
    }
}
```

Edit voxels with `BvwVoxelEdits::set_voxel` (on `VoxelGiInput`), which
writes BVW and tells GI the exact change in one call.

## Block optics

GI receives BVW's `u8` material indices unchanged, so set each block's
optics in `VoxelMaterialTransportTable` at its BVW index. Don't use
`bevy_voxel_gi::MaterialPalette` here: it assigns its own sequential IDs and
merges blocks with identical optics, so its IDs need not match BVW's, and a
block could take another block's emission. GI treats every solid BVW voxel
as opaque.

```rust
use bevy::prelude::*;
use bevy_voxel_gi::{MaterialTransport, VoxelMaterialTransportTable};

const STONE: u8 = 0;
const LAMP: u8 = 3;

fn block_optics(mut table: ResMut<VoxelMaterialTransportTable>) {
    table.set(STONE, MaterialTransport { diffuse: Vec3::splat(0.4), ..default() });
    table.set(
        LAMP,
        MaterialTransport { diffuse: Vec3::splat(0.8), emission: Vec3::new(4.0, 2.4, 1.0) },
    );
}
```

LoD chunks are read at their stored stride; colliders and full resolution
belong to your near LoD bands, as in BVW. Licensed under MIT or Apache-2.0.
