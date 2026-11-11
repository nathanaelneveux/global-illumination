# bevy_voxel_gi_bvw

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi_bvw.svg)](https://crates.io/crates/bevy_voxel_gi_bvw)
[![docs.rs](https://docs.rs/bevy_voxel_gi_bvw/badge.svg)](https://docs.rs/bevy_voxel_gi_bvw)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi_bvw.svg)

Add bounced light to a `bevy_voxel_world` terrain. The connector reads your
chunks, forwards their changes to GI and supplies a material for your
textured meshes. A glowing block uses the same material index for its
texture and its lighting properties.

## Quick start

Use Bevy 0.20, `bevy_voxel_world` 0.19 and matching versions of the two GI
crates.

```toml
[dependencies]
bevy = "0.20"
bevy_voxel_gi = "0.2"
bevy_voxel_gi_bvw = "0.2"
bevy_voxel_world = "0.19"
```

If you already have a world, keep its `VoxelWorldConfig` and voxel generator.
Implement `GiFallbackVoxel` for the config, then add `VoxelGiPlugin` and
`VoxelGiBvwPlugin` **before** the world plugin. The connector registers the
material type that BVW needs when it starts. Pass a prepared texture array
to `material_template` and give the result to `with_material`.

Starting from an empty app? Copy this example into `src/main.rs` and run
`cargo run`. It creates a stone platform with a warm lamp and makes its own
texture.

<details>
<summary>Copy the complete main.rs for a lit BVW platform</summary>

```rust,no_run
use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use bevy_voxel_gi::{
    MaterialTransport, VoxelGiMaterialData, VoxelGiPlugin, VoxelMaterialTransportTable,
};
use bevy_voxel_gi_bvw::{
    GiFallbackVoxel, VoxelGiBvwPlugin, material_template, prepare_texture_array,
};
use bevy_voxel_world::prelude::*;

const STONE: u8 = 0;
const LAMP: u8 = 1;

#[derive(Resource, Clone, Default)]
struct MyWorld;

impl VoxelWorldConfig for MyWorld {
    type MaterialIndex = u8;
    type ChunkUserBundle = ();

    // Keep chunks behind the camera: GI traces them.
    fn chunk_despawn_strategy(&self) -> ChunkDespawnStrategy {
        ChunkDespawnStrategy::FarAway
    }

    fn voxel_lookup_delegate(&self) -> VoxelLookupDelegate<u8> {
        Box::new(|_, _, _| {
            Box::new(|p, _| {
                if p == IVec3::new(7, 0, 7) {
                    WorldVoxel::Solid(LAMP)
                } else if p.y == -1 && (0..16).contains(&p.x) && (0..16).contains(&p.z) {
                    WorldVoxel::Solid(STONE)
                } else {
                    WorldVoxel::Air
                }
            })
        })
    }
}

// Loaded chunks supply the platform. Treat positions outside them as air.
impl GiFallbackVoxel for MyWorld {}

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins,
        VoxelGiPlugin,
        VoxelGiBvwPlugin::<MyWorld>::default(),
    ));

    let mut image = Image::new_fill(
        Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[156, 156, 156, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    prepare_texture_array(&mut image, 1).unwrap();
    let texture = app.world_mut().resource_mut::<Assets<Image>>().add(image);
    let material = material_template(texture, app.world().resource::<VoxelGiMaterialData>());

    let mut optics = app.world_mut().resource_mut::<VoxelMaterialTransportTable>();
    optics.set(STONE, MaterialTransport { diffuse: Vec3::splat(0.5), ..default() });
    optics.set(
        LAMP,
        MaterialTransport { diffuse: Vec3::splat(0.8), emission: Vec3::new(4.0, 2.4, 1.0) },
    );

    app.add_plugins(VoxelWorldPlugin::with_config(MyWorld).with_material(material))
        .add_systems(Startup, setup)
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        VoxelWorldCamera::<MyWorld>::default(),
        Transform::from_xyz(13.0, 7.0, 17.0).looking_at(Vec3::new(7.5, 0.0, 7.5), Vec3::Y),
    ));
}
```

</details>

## Textures and light

BVW's texture mapper picks each block's top, side and bottom layers. To
build the texture array:

1. Stack equally sized tiles vertically in one image.
2. Give `material_template` that image's handle.
3. Once the image loads, call `prepare_texture_array` on it, once. Check
   `texture_array_ready` first, reading the image with `Assets::get`: a
   mutable borrow re-uploads it to the GPU.

Set each block's optical properties in `VoxelMaterialTransportTable` at
its BVW material index. If BVW calls your lamp material `1`, set `emission`
on slot `1`. Skip `MaterialPalette` here: it assigns its own IDs. Every
solid BVW voxel is opaque to GI.

## Edits and levels of detail

Call `BvwVoxelEdits::set_voxel` on `VoxelGiInput` to write a voxel to BVW and
report the exact change to GI together. For example, dig a block by passing
`WorldVoxel::Air`.

```rust
use bevy::prelude::*;
use bevy_voxel_gi::VoxelGiInput;
use bevy_voxel_gi_bvw::BvwVoxelEdits;
use bevy_voxel_world::prelude::*;

fn dig<C: VoxelWorldConfig<MaterialIndex = u8>>(
    gi: &mut VoxelGiInput,
    world: &mut VoxelWorld<C>,
    position: IVec3,
) {
    gi.set_voxel(world, position, WorldVoxel::Air);
}
```

GI lights what BVW has loaded, including geometry behind the camera that
blocks or bounces light. BVW's default unloads chunks out of view beyond
`min_despawn_distance` chunks of a camera. A room then relights when you
turn back to it, since its walls and the terrain around it left GI's world.
Return `ChunkDespawnStrategy::FarAway` from `chunk_despawn_strategy` to keep
everything within the spawning distance loaded. Before bevy_voxel_world
0.19 the protected radius was effectively 32 times larger, so worlds that
relied on it need `FarAway` to light as they did.

A level-0 BVW chunk is one 32³ GI page. A chunk at a coarser level of
detail spans several pages, and GI traces it as its mesh shows it, one
voxel per sample block. While BVW swaps detail levels, GI follows the finer
chunk until the coarser one has fully replaced it. Chunks above level 5
(`MAX_GI_LOD`) are left out of GI's world.

Within a loaded page, where the chunk has no voxel, the connector asks
`GiFallbackVoxel`; the default answers air. Chunks that `chunk_fill_hint`
keeps from spawning never load, so GI treats them as air. That is right
for empty chunks, but light passes through a chunk hinted solid.

See the [core guide](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi/README.md)
for platform setup, budgets and troubleshooting.

## License

MIT or Apache-2.0, at your option.
