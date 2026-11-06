# bevy_voxel_gi

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi.svg)](https://crates.io/crates/bevy_voxel_gi)
[![docs.rs](https://docs.rs/bevy_voxel_gi/badge.svg)](https://docs.rs/bevy_voxel_gi)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi.svg)

Add bounced sunlight, sky and lamp light to an axis-aligned voxel world.
Supply your voxels and their materials, and GI traces the light on the GPU.

Use **Bevy 0.19** and **Rust 1.95 or newer** with a
[native GPU backend](#platforms).

## Quick start

If your app uses `bevy_voxel_world`, start with the
[BVW connector guide](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi_bvw/README.md).
For your own voxel storage, add the core crate to your app.

```toml
[dependencies]
bevy = "0.19"
bevy_voxel_gi = "0.1"
```

To see a working room first, run the example from a
[repository checkout](https://github.com/nathanaelneveux/global-illumination#try-the-examples).

```sh
cargo run --locked -p bevy_voxel_gi --example minimal
```

Want to give the room a colder light? Swap the red and blue values in the
lamp's `MaterialTransport::emission` and run it again. Look at the nearby
stone to see the difference.
[Room example source](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi/examples/minimal.rs)

## Light your own world

Start with a layer of stone and give GI the same surface you draw.

1. **Give blocks optical properties.** A `MaterialTransport` describes
   diffuse reflectance and emission. Use `MaterialPalette` to assign
   transport IDs and copy its table into `VoxelMaterialTransportTable`.
2. **Provide a voxel snapshot.** Put your `VoxelSource` in `VoxelGiSource`.
   Use `dense_page` to return a material per voxel and `None` for air.
   Worker threads read this snapshot.
3. **Declare the loaded pages.** Call `VoxelGiInput::page_loaded` for each
   32³ page. Page `(1, 0, 0)` starts at voxel `(32, 0, 0)`.
4. **Draw the same surfaces.** Use `standard_material` to make a GI-lit
   Bevy material. Draw the mesh on the same boundaries as the source.
   Here the stone ends at `y = 4`, so the plane goes there too.

<details>
<summary>Copy the complete main.rs for a sunlit plane</summary>

```rust,no_run
use bevy::prelude::*;
use bevy_voxel_gi::{
    MaterialPalette, MaterialTransport, PageSample, VoxelGiInput, VoxelGiMaterialData,
    VoxelGiPlugin, VoxelGiSource, VoxelGiStandardMaterial, VoxelGiStandardMaterialPlugin,
    VoxelMaterialTransportTable, VoxelSource, dense_page, standard_material,
};
use std::sync::Arc;

fn main() {
    App::new()
        // GI itself, and Bevy's `StandardMaterial` lit by GI.
        .add_plugins((DefaultPlugins, VoxelGiPlugin, VoxelGiStandardMaterialPlugin))
        .add_systems(Startup, setup)
        .run();
}

/// GI reads your world through a `VoxelSource`, one 32³ page at a time, on
/// worker threads: give it an immutable snapshot of your voxel data.
struct Flat {
    stone: u8,
}

impl VoxelSource for Flat {
    // `resample` asks for authoritative data after a change GI wasn't told
    // the details of; a source with one representation can ignore it.
    fn sample_page(&self, page: IVec3, _resample: bool) -> PageSample {
        // A transport material per voxel; `None` is air.
        dense_page(page, |voxel| (voxel.y < 4).then_some(self.stone))
    }
}

fn setup(
    mut commands: Commands,
    mut palette_table: ResMut<VoxelMaterialTransportTable>,
    mut source: ResMut<VoxelGiSource>,
    mut input: ResMut<VoxelGiInput>,
    gi_bindings: Res<VoxelGiMaterialData>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<VoxelGiStandardMaterial>>,
) {
    // 1. Optics. The palette gives each distinct optics a transport ID
    //    (blocks with identical optics share one); GI reads the table.
    let mut palette = MaterialPalette::new();
    let stone = MaterialTransport {
        diffuse: Vec3::splat(0.6),
        emission: Vec3::ZERO, // a lamp would set this
    };
    let stone = palette.insert("stone", stone).unwrap() as u8;
    *palette_table = palette.table().clone();

    // 2. Voxels: the source GI samples from.
    source.0 = Some(Arc::new(Flat { stone }));

    // 3. Pages: the 32³ pages GI knows about are its whole world. Tell it
    //    as they load and unload (`page_unloaded`); report edits with
    //    `edit_voxel` so it relights only what changed.
    for x in -2..2 {
        for z in -2..2 {
            input.page_loaded(IVec3::new(x, 0, z));
        }
    }

    // 4. Meshes. GI lights a fragment from the voxel face it lies on, so
    //    meshes sit on voxel faces: here, the top of the stone at y = 4.
    //    `gi_bindings` carries GI's GPU data; the plugin keeps it current.
    let ground = materials.add(standard_material(
        StandardMaterial {
            base_color: Color::srgb(0.6, 0.6, 0.6),
            ..default()
        },
        &gi_bindings,
    ));
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(128.0, 128.0))),
        MeshMaterial3d(ground),
        Transform::from_xyz(0.0, 4.0, 0.0),
    ));

    // Directional lights become GI's suns; GI traces their visibility
    // itself, so turn their shadow maps off.
    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: false,
            ..default()
        },
        Transform::default().looking_to(Vec3::new(-0.4, -0.8, -0.5), Vec3::Y),
    ));
    // Each camera gets the depth and normal prepasses GI needs.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(20.0, 12.0, 20.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}
```

</details>

Run `cargo run` in your app. You should see a gray plane lit from above.

### Reporting edits

Tell GI whenever the world changes:

- **You know the block's new material.** Call `edit_voxel`, and update your
  mesh to match.
- **A page changed and you don't know exactly how.** Call `page_changed`.
  GI resamples the page from your source, so the source must already hold
  the change.
- **A chunk unloaded.** Call `page_unloaded`.

To watch an edit change the light, run the
[`edits` example](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi/examples/edits.rs):
a window opens and closes every two seconds. The
[input API](https://docs.rs/bevy_voxel_gi/latest/bevy_voxel_gi/struct.VoxelGiInput.html)
covers exact edit tracking and explicit resampling.

### Levels of detail

For a distant chunk, return a coarse `PageSample` and mesh the same blocks.
With a stride of 4, each 4³ block has one material.

## How lighting follows the camera

**Visible faces.** Each camera's depth and normal prepasses tell GI which
voxel faces to light. GI keeps each face's history as the camera moves and
blends nearby samples while a face is still noisy.

**Off-screen faces.** The world cache keeps light on faces within 64 voxels
of a camera by default. A ray can reuse that light to build
up multiple bounces. The cache also keeps it available when you turn back.

**Distant faces.** When a voxel projects to less than 8 pixels across by
default, GI switches to a screen-space estimate. This lets it light distant
terrain with a cost tied to the view.

## Lighting your own material

Wrap your material in `ExtendedMaterial<YourMaterial, VoxelGiBindings>`.
Add `VoxelGiMaterialPlugin::<_>::extended()` so the bindings follow GI's
buffer updates. Reserve binding 104 and bindings 106–111 in the material
bind group for GI.

Choose forward rendering on the base material. For `StandardMaterial`, set
`opaque_render_method` to `OpaqueRendererMethod::Forward`. The extension
enables the prepass and shadow passes. Add `NotShadowCaster` to any entity
that should skip Bevy's shadow pass.

Add the indirect and emissive terms in your forward fragment shader.

```wgsl
#import bevy_voxel_gi::presentation::{indirect_light, emission}

let gi = indirect_light(in.position.xy, in.world_position.xyz, normal);
color += albedo * gi.light + emission(in.world_position.xyz, normal);
```

### Shader reference

Pass the world position on the voxel face and its axis-aligned normal.
`pixel` is `@builtin(position).xy` in framebuffer pixels. For a ray-traced
primary renderer, use the hit position and normal and supply compatible
depth and normal prepasses.

<details>
<summary>Every presentation function</summary>

| Function | Result and use |
| --- | --- |
| `indirect_light(pixel, position, normal)` | Multiply `.light` by diffuse albedo. `.found` reports sample support. `.coverage` identifies the source of that support. |
| `sunlight(position, normal, footprint)` | Direct sun in transport units. Multiply by diffuse albedo × 10,000 / π and apply it in place of Bevy's directional term. The footprint is the largest component of `fwidth(position)` in world voxels. |
| `emission(position, normal)` | Add the voxel's emitted RGB once. |
| `face_in_geometry(position, normal)` | True for an exposed face in the published scene. Provide host lighting for other surfaces. |
| `sun_count()`, `sun_direction(i)`, `sun_color(i)` | Read the same suns for your own visibility test. Index from 0 to count − 1. Direction points toward the sun. Color is in transport units. |
| `enabled()`, `voxel_sky()`, `debug_mode()` | Read the indirect toggle, the GI sunlight toggle and the diagnostic mode. |
| `sunlight_reference(position, normal)` | Trace the face's own four sun corners for a diagnostic comparison. |

</details>

The stock shader shows how to combine these terms with Bevy's PBR.
[Stock WGSL material](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi/shaders/voxel_gi_standard.wgsl)

## Platforms

Use a native Metal, Vulkan or DX12 backend. The view tier needs more storage
buffers per shader stage than WebGPU allows. GI uses compute shaders, so it
can run on devices without dedicated ray-tracing hardware.

Vulkan is the simplest starting point on Windows. On the tested AMD RX 7900
XT, wgpu selected Vulkan by default. DX12 needs the DXC shader compiler to
build the compute pipelines.
[Windows test report](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/docs/history/windows-first-run.md)

### DX12

Enable Bevy's `statically-linked-dxc` feature, or ship `dxcompiler.dll` and
`dxil.dll` beside your executable. Select that path explicitly as shown
below so the app finds DXC from any working directory.

```rust,no_run
use bevy::prelude::*;
use bevy::render::{
    RenderPlugin,
    settings::{Dx12Compiler, WgpuSettings},
};

let dxc = std::env::current_exe().unwrap().with_file_name("dxcompiler.dll");
App::new().add_plugins(DefaultPlugins.set(RenderPlugin {
    render_creation: WgpuSettings {
        dx12_shader_compiler: Dx12Compiler::DynamicDxc {
            dxc_path: dxc.to_string_lossy().into_owned(),
        },
        ..default()
    }
    .into(),
    ..default()
}));
```

## Tuning

Start with `VoxelGiSettings::quality(VoxelGiQuality::Low)` if GI needs a
smaller budget. Insert the settings resource before adding `VoxelGiPlugin`,
or change it while the app runs. Use `VoxelGiSettings::from_ron` to load
settings saved by the optional
[debug panel](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi_debug/README.md).

| If you want to | Adjust | What you trade |
| --- | --- | --- |
| Spend less GPU time tracing | Lower `gpu_time_share` | Lighting takes longer to update. The default budget is 20%, shared by visible faces and the world cache. |
| Keep fewer faces lit off screen | Lower `cache_radius` | Less light history when you turn back. The default is 64 voxels. Set 0 to disable the cache. |
| Reduce discovery and presentation cost | Lower `view_receivers` | The camera tracks fewer faces independently. |
| Use screen-space lighting farther away | Lower `far_tier.voxel_pixels` | More faces use the per-face path. The default is 8 pixels across a full-resolution voxel. Set 0 to disable this tier. |

`VoxelGiStats` reports progress and costs. Its `view_gpu` receipt separates
tracing from the cost of finding and presenting visible faces. Use
`geometry_pending` to tell whether GI is still preparing the scene.

## Performance

On an Apple M4 at 2560×1440, finding and presenting visible faces takes
3.89 ms a frame with the far tier on (a headless run of the lab's default
world). Tracing comes on top, within the GPU budget you set.
[Measurements, with a far-tier-off comparison](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/docs/history/far-per-pixel.md#update-distance-cheaper-classify-handoffs-flicker)

## Troubleshooting

For a blank scene, check the render log and run the
[room example](#quick-start) on the same backend. In your app, confirm an
active 3D camera, declared pages and a GI-lit material.

| What you see | What to check |
| --- | --- |
| A chunk is dark as it appears | Check `VoxelGiStats::geometry_pending`. The mesh can arrive before its GI geometry. Once ready, check that the page is declared and its mesh matches the voxel source. |
| Lighting ignores an edit | Send the voxel edit along with the remesh. For resampling, make sure the current source contains the change. [Reporting edits](#reporting-edits) |
| A glowing texture casts no light | Set the block's `MaterialTransport::emission` and check the source's material ID. For BVW, use the original block index in the transport table. [Block optics](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi_bvw/README.md#textures-and-light) |
| A custom material misses GI | Check the forward pass, reserved bindings and binding update plugin. [Material setup](#lighting-your-own-material) |
| Sun shadows look wrong | Check `VoxelGiDisplay::voxel_sun`. With GI sunlight, apply the GI sun once. With Bevy sunlight, enable the directional light's shadow maps to get shadows. |
| Distant faces flicker as you move | Compare the same view with `far_tier.voxel_pixels = 0` to isolate the far tier. [Tuning](#tuning) |
| GPU cost is high | Read the discovery, presentation and trace timings separately. Lower the receiver cap for discovery and presentation or the GPU share for tracing. [Tuning](#tuning) |
| DX12 fails to start | Use DXC or Vulkan. A first DXC run can spend several seconds compiling shaders. [DX12 setup](#dx12) |

The **Coverage** debug view distinguishes support from brightness. Green is
a face receiver, cyan is the far tier, yellow is reconstructed support
for a missing sample, and red means no indirect support.
[Debug plugin](https://github.com/nathanaelneveux/global-illumination/blob/HEAD/crates/bevy_voxel_gi_debug/README.md)

## Limitations

- GI computes diffuse reflection from opaque blocks. It treats transparent
  blocks as air. Specular GI and colored transmission are outside this model.
- Distant indirect light can flicker more in motion than nearby lighting.
- When you edit a block or move a light, the affected lighting takes time
  to settle. The rate depends on scene complexity, retained history and
  the tracing budget.
- Scene builds scale with the number of declared pages. Give GI the pages
  you want rays to traverse, including nearby off-screen occluders and
  emitters.

## Credits

- [NAADF: Globally Illuminated Voxel Worlds Accelerated with Nested
  Axis-Aligned Distance Fields](https://www.cg.tuwien.ac.at/research/publications/2026/ulschmid-2026-naadf/)
  ([code](https://github.com/cg-tuwien/NAADF)): the acceleration structure
  and the per-pixel lighting of small faces. This is an independent compute
  implementation, not a port of NAADF's renderer.
- [vxray](https://github.com/Nelarius/vxray) and its
  [development log](https://nelari.us/post/voxel-ray-tracing/): voxel
  traversal.
- [sobol_burley](https://github.com/cessen/sobol_burley) by Nathan Vegdahl:
  the hash-based Owen scramble in the path sampler (MIT,
  `licenses/sobol_burley-MIT.txt`).
- [PBRT](https://pbr-book.org/4ed/Shapes/Managing_Rounding_Error#SpawnRayTo):
  ray spawning.

## License

Licensed under MIT or Apache-2.0, at your option. The adapted `sobol_burley`
code keeps its MIT notice.
