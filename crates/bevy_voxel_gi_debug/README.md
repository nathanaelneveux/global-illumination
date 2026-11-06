# bevy_voxel_gi_debug

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi_debug.svg)](https://crates.io/crates/bevy_voxel_gi_debug)
[![docs.rs](https://docs.rs/bevy_voxel_gi_debug/badge.svg)](https://docs.rs/bevy_voxel_gi_debug)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi_debug.svg)

See how much your voxel lighting costs and change its settings while your
app runs. The debug plugin adds a HUD, a lighting panel and keyboard
shortcuts to an app that already uses
[`bevy_voxel_gi`](https://docs.rs/bevy_voxel_gi).

## Quick start

Use the same version as the core crate.

```toml
[dependencies]
bevy_voxel_gi_debug = "0.1"
```

Add the plugin after GI. Include Bevy's frame diagnostics for the FPS
reading. If your app already adds `FrameTimeDiagnosticsPlugin`, reuse it.

```rust
use bevy::{diagnostic::FrameTimeDiagnosticsPlugin, prelude::*};
use bevy_voxel_gi_debug::VoxelGiDebugPlugin;

fn add_lighting_controls(app: &mut App) {
    app.add_plugins((
        FrameTimeDiagnosticsPlugin::default(),
        VoxelGiDebugPlugin::default(),
    ));
}
```

Open **Lighting controls** in the HUD to choose a quality preset or adjust
individual settings. Use **Copy** to save the current settings as RON and
**Paste** to try them in another run.

| Key | Compare |
| --- | --- |
| F4 | Indirect GI on or off |
| F5 | Flat face shading or corner interpolation |
| F6 | Diagnostic views such as indirect light or face validity |
| F7 | GI corner sunlight or Bevy per-pixel sunlight |

The HUD reports main-thread, cache GPU and view GPU time separately.
Coverage is the share of cached faces with at least one sample. Bootstrap
tracks progress toward eight samples per face, and Settled tracks the
cache's stability heuristic. To configure the HUD, shortcuts or periodic
terminal stats, see the
[plugin fields](https://docs.rs/bevy_voxel_gi_debug/latest/bevy_voxel_gi_debug/struct.VoxelGiDebugPlugin.html).

## License

MIT or Apache-2.0, at your option.
