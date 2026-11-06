# bevy_voxel_gi

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi.svg)](https://crates.io/crates/bevy_voxel_gi)
[![docs.rs](https://docs.rs/bevy_voxel_gi/badge.svg)](https://docs.rs/bevy_voxel_gi)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi.svg)

Real-time diffuse global illumination for voxel worlds in
[Bevy](https://bevyengine.org) 0.19: sunlight, sky, emissive voxels and
multi-bounce indirect light, traced against the voxels themselves on ordinary
GPU compute. No ray-tracing hardware is needed, so it runs on Apple Silicon and
integrated GPUs.

![A cave lit by emissive voxels and a skylight](docs/images/cave.jpg)
![A daylight canyon: sunlit walls bounce warm light into the shaded side](docs/images/canyon.jpg)

| Crate | What it is |
| --- | --- |
| [`bevy_voxel_gi`](crates/bevy_voxel_gi) | GI itself. Start with its [README](crates/bevy_voxel_gi/README.md): host setup, tuning, performance, limitations |
| [`bevy_voxel_gi_bvw`](crates/bevy_voxel_gi_bvw) | Connector for [`bevy_voxel_world`](https://github.com/splashdust/bevy_voxel_world): chunk events, voxel source, the lit mesh material |
| [`bevy_voxel_gi_debug`](crates/bevy_voxel_gi_debug) | Optional HUD and tuning panel |

## How it works

- **No ray-tracing hardware.** Compute shaders trace the voxels themselves,
  so it runs on Apple Silicon and integrated GPUs. On an Apple M4 at 1440p,
  finding and presenting visible faces costs about 4 ms a frame, and tracing
  fills a GPU time share you set (20% of the frame by default).
- **Visible faces** are discovered from each camera's depth and normal prepass
  and traced every frame within that share, with histories that follow each
  face as the camera moves.
- **A world cache** keeps faces within 64 voxels of the cameras lit off screen
  and trades history with visible faces, so turning around, revisiting and
  editing out of view stay warm. A ray that reaches a cached face ends there
  with its stored light, so bounces add up over time
  ([surface cache](docs/surface-cache.md)).
- **Faces small on screen** (a voxel under 8 pixels across) are lit per pixel
  instead, so distant terrain costs follow the screen, not the world.
- **Edits** tell GI which voxels changed, and it drops history near each
  change in proportion to its measured effect: digging a tunnel relights the
  tunnel without resetting the rest of the scene, and nothing is guessed from
  frame-to-frame differences.
- **Emissive blocks are lights.** A light tree chooses which of many emitters
  to sample.
- **Presentation** pools coplanar neighbours by their measured noise until each
  face has enough samples of its own, then blends over time. Sunlight is
  sampled at shared voxel corners.
- **Your engine, your meshes.** GI reads voxels through a trait and lights your
  materials. It never meshes or owns the world, accepts coarse level-of-detail
  pages, and works under ray-traced renderers as well as meshes.

## Getting started

```sh
cargo add bevy_voxel_gi
```

```rust,ignore
App::new()
    .add_plugins((DefaultPlugins, VoxelGiPlugin, VoxelGiStandardMaterialPlugin))
    .run();
```

Then describe how your blocks reflect and emit light, hand GI a read-only
view of your voxels, say which 32³ pages exist, and draw with a GI-lit
material. The [quick start](crates/bevy_voxel_gi/README.md#quick-start) does
all four in one file. Using `bevy_voxel_world`?
[`bevy_voxel_gi_bvw`](crates/bevy_voxel_gi_bvw) does the wiring for you.

| Bevy | bevy_voxel_gi |
| --- | --- |
| 0.19 | 0.1 |

From a checkout, three things to run:

```sh
cargo run -p bevy_voxel_gi --example minimal   # a small complete host
cargo run -p bevy_voxel_gi --example edits     # a window opening and closing
cargo run --locked                             # the lab: a streaming world to explore
```

## Platforms

Native GPUs only; WebGPU allows too few storage buffers per stage.

- **Metal:** developed and measured on an Apple M4.
- **Vulkan and DX12:** tested on an AMD RX 7900 XT under Windows 11
  ([first run](docs/history/windows-first-run.md)). Vulkan needs no setup;
  DX12 needs the DXC shader compiler and started slower there (see the
  [`bevy_voxel_gi` README](crates/bevy_voxel_gi/README.md)).

[Testing on another GPU](docs/testing-other-gpus.md) explains how to try it on
yours and what to send back.

## Limitations

- Opaque, diffuse light only: transparent blocks are air to GI, and there's
  no specular or coloured transmission.
- Native GPUs only, not WebGPU.
- In motion, distant lighting flickers a little more than nearby lighting.
- In very large worlds, new geometry and edits take longer to reach GI.

The [changelog](CHANGELOG.md#known-limitations) has the details.

## Try the lab

The root package is the development app: a streaming
[`bevy_voxel_world`](https://github.com/splashdust/bevy_voxel_world) world
lit by GI, forked from `prototype_voxel_template`, plus the validation harness
behind every measurement in `docs/`.

```sh
cargo run --locked                         # the default world
VOXEL_GI_SCENE=cave cargo run --locked     # an isolated many-emitter cave
VOXEL_GI_SCENE=canyon cargo run --locked   # high rims and long daylight sightlines
```

The default world has a house, a hallway west of it, and a cave whose lit
entrance is near **(20, 11, -18)**: follow the pale stone path to the right of
the house. The cave descends to a lava rim.

| Input | Action |
| --- | --- |
| WASD / mouse | Move / look |
| Space / Ctrl / Shift | Jump / crouch / sprint |
| Space twice | Toggle creative flight: WASD, Space up, Ctrl down, Shift 3× faster |
| Left / right click | Remove / place a voxel |
| E | Place a warm emitting voxel at the aimed-at face |
| B | Remove a radius-5 sphere at the aimed-at voxel |
| F3 | Mesh debug overlay |
| F4 | Toggle indirect GI |
| F5 | Flat faces / corner interpolation |
| F6 | Cycle diagnostic views (irradiance, tier coverage, indirect, sun, exact sun, face validity) |
| F7 | GI's corner sun / Bevy's per-pixel sun |
| Esc | Release the cursor and open the inspector and HUD buttons |

**Lighting controls** (press Esc, then the HUD button) has quality presets,
look and light-model settings, and the display switches. **Copy** and **Paste**
move settings as RON. Launch with saved settings using
`VOXEL_GI_SETTINGS=path.ron cargo run --locked`. `config/` holds the presets
(`gi-low.ron`, `gi-high.ron`) and the comparison controls the validation
docs use.

The top-right HUD shows FPS, GI's main-thread and GPU time, and world-cache
progress. Coverage means a face has at least one sample, not that it has
converged. CPU and GPU work overlap, so don't add their times.

## Documentation

- [API reference](https://docs.rs/bevy_voxel_gi) and the
  [changelog](CHANGELOG.md).
- [Architecture](docs/architecture.md): how GI is built and its limits.
- [Research and decisions](docs/research.md): what was adopted from NAADF,
  vxray, DDGI and Solari, what differs, and what's planned after 0.1.
- [Validation](docs/validation.md): which check to run for which change, the
  profiles, and the evidence rules.
- [History](docs/history/README.md): past studies, rejected alternatives and
  the measurements behind today's defaults.

## Contributing

Bug reports and GPU reports are welcome as issues; [testing on another
GPU](docs/testing-other-gpus.md) says what to send. For pull requests:

- One concern per PR, with the problem, its cause and the fix.
- Run `cargo fmt --all`, Clippy with `-D warnings` and the relevant tests;
  CI checks them. Changes to GPU code need the native GPU tests
  (`--include-ignored`) on real hardware.
- Performance or lighting claims need evidence: before and after runs of the
  same profile, as [validation](docs/validation.md) describes. Faster frames
  that show staler or noisier lighting aren't an improvement.
- [`AGENTS.md`](AGENTS.md) is the project's working agreement, for people and
  coding agents alike; [the release guide](docs/releasing.md) covers
  versions and publishing.

## License

The crates are `MIT OR Apache-2.0` ([MIT](LICENSE-MIT.txt),
[Apache](LICENSE-Apache-2.0.txt)); `bevy_voxel_gi` keeps the MIT notice of the
`sobol_burley` code it adapts. The lab app (the root package) is
`CC0-1.0 OR MIT OR Apache-2.0` ([CC0](LICENSE-CC0-1.0.txt)).
