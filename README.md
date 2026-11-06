# bevy_voxel_gi

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi.svg)](https://crates.io/crates/bevy_voxel_gi)
[![docs.rs](https://docs.rs/bevy_voxel_gi/badge.svg)](https://docs.rs/bevy_voxel_gi)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi.svg)

Light a voxel world with sunlight, sky and glowing blocks. Light bounces
off the voxels, so a lamp can warm the walls of a cave and sunlight can
reach surfaces in shadow. The lighting updates as you build and dig.

`bevy_voxel_gi` supports axis-aligned voxel worlds in Bevy 0.19. It traces
rays in compute shaders on native GPUs.

## Getting started

- **See it in action.** Run the [examples](#try-the-examples) to see a small
  room light up, or [explore the lab](#try-the-lab) to dig and place lights.
- **Light your own voxel engine.** Follow the
  [core quick start](crates/bevy_voxel_gi/README.md#quick-start) to supply voxels
  and light your meshes.
- **Light a `bevy_voxel_world` project.** Use the
  [BVW connector](crates/bevy_voxel_gi_bvw/README.md#quick-start) for chunk
  streaming, block edits and a lit terrain material.

Add the [debug plugin](crates/bevy_voxel_gi_debug/README.md) to see lighting
costs and tune the look while your app runs.

## Try the examples

Clone this repository and run the room example from the checkout. Use
[Rust 1.95 or newer](https://rustup.rs) and install
[Bevy’s platform dependencies](https://bevy.org/learn/quick-start/getting-started/setup/#installing-os-dependencies).

```sh
git clone https://github.com/nathanaelneveux/global-illumination
cd global-illumination
cargo run --locked -p bevy_voxel_gi --example minimal
```

Look for the orange lamp on the floor and the warm light on the nearby
stone. Then try the edits example. A window opens and closes every two
seconds, and a patch of sunlight appears and disappears on the floor.

```sh
cargo run --locked -p bevy_voxel_gi --example edits
```

## How it works

A camera's depth and normal prepasses tell GI which voxel faces are visible.
The tracer samples those faces within a GPU time budget and reuses their
light as the camera moves. A world cache keeps lighting around the camera
available off screen. When a ray hits a cached face, it can use the light
stored there. This lets light build up over multiple bounces.

An edit tells GI where the world changed. Faces near the change use more
fresh samples, while farther faces can keep more of their history. Small
faces in the distance use screen-space lighting. Read the
[architecture guide](docs/architecture.md) for the full path from a voxel
edit to a lit fragment.

![Three views of the same room, with dim walls, warm walls, then dim walls again](docs/images/response.jpg)

*The room before placing an off-screen lamp, 0.2 seconds after placing it,
and 0.3 seconds after removing it. Captured with the lab's offscreen
`response` profile on an Apple M4.*

## Platforms

Voxel GI uses native Metal, Vulkan or DX12 and needs GPU compute with more
storage buffers than WebGPU allows. It was tested on Apple M4 with Metal
and on an AMD RX 7900 XT with Vulkan and DX12. On Windows, start with
Vulkan. DX12 requires the DXC shader compiler. See
[platform setup](crates/bevy_voxel_gi/README.md#platforms) and the
[GPU testing guide](docs/testing-other-gpus.md).

## Try the lab

The lab is the streaming voxel world in this repository. Start it from the
checkout with `cargo run --locked`. From spawn, follow the pale stone path
to the right of the house. The cave entrance is near **(20, 11, −18)** and
leads down to a lava rim. Try placing a light against the wall with **E**,
then remove it with **left click** and watch the nearby surfaces.

![Green ceiling lights and orange lava light tint the walls of a cave. Sunlight enters through a hole above the spire.](docs/images/cave.jpg)

To go straight to a test scene, use one of these commands on macOS or Linux.
In PowerShell set `$env:VOXEL_GI_SCENE = "cave"` or `"canyon"` before running
`cargo run --locked`.

```sh
VOXEL_GI_SCENE=cave cargo run --locked
VOXEL_GI_SCENE=canyon cargo run --locked
```

The cave mixes a skylight with many colored emitters. The
[daylight canyon](docs/images/canyon.jpg) shows sunlit walls bouncing warm
light into shade across a much larger space.

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
progress. Coverage counts faces with at least one sample. CPU and GPU work
overlap, so read their timings separately.

## Documentation

- **Integrate and tune** with the [core guide](crates/bevy_voxel_gi/README.md)
  and [API reference](https://docs.rs/bevy_voxel_gi).
- **Understand the renderer** in the [architecture](docs/architecture.md) and
  [surface cache study](docs/surface-cache.md). See
  [research and decisions](docs/research.md) for the papers and tradeoffs.
- **Measure a change** using the [validation guide](docs/validation.md).
  The [historical studies](docs/history/README.md) record the measurements
  behind the defaults.
- **Check compatibility and known limitations** in the
  [changelog](CHANGELOG.md#known-limitations).

## Contributing

Open a GitHub issue for bugs and GPU reports. The
[GPU testing guide](docs/testing-other-gpus.md) lists what to include.
For a pull request, explain the problem and the change, run formatting,
Clippy and the checks relevant to the diff. Use the
[validation guide](docs/validation.md#choose-the-relevant-check) to pick
tests and collect paired evidence for performance or lighting changes.
[AGENTS.md](AGENTS.md) records the development workflow, and the
[release guide](docs/releasing.md) covers publishing.

## License

The crates are `MIT OR Apache-2.0` ([MIT](LICENSE-MIT.txt),
[Apache](LICENSE-Apache-2.0.txt)); `bevy_voxel_gi` keeps the MIT notice of the
`sobol_burley` code it adapts. The lab app (the root package) is
`CC0-1.0 OR MIT OR Apache-2.0` ([CC0](LICENSE-CC0-1.0.txt)).
