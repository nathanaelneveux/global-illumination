# bevy_voxel_gi_debug

[![crates.io](https://img.shields.io/crates/v/bevy_voxel_gi_debug.svg)](https://crates.io/crates/bevy_voxel_gi_debug)
[![docs.rs](https://docs.rs/bevy_voxel_gi_debug/badge.svg)](https://docs.rs/bevy_voxel_gi_debug)
[![CI](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml/badge.svg?branch=voxel-global-illumination)](https://github.com/nathanaelneveux/global-illumination/actions/workflows/ci.yml)
![License: MIT or Apache-2.0](https://img.shields.io/crates/l/bevy_voxel_gi_debug.svg)

An optional HUD and tuning panel for [`bevy_voxel_gi`](https://docs.rs/bevy_voxel_gi): frame rate,
coverage, GI main-thread and GPU time, and live controls for
`VoxelGiSettings` and `VoxelGiDisplay` (including the diagnostic views), built on GI's public API with Bevy UI
and feathers widgets, plus keyboard presentation toggles and periodic
terminal stats. Nothing runs unless `VoxelGiDebugPlugin` is added. Licensed under MIT or Apache-2.0.

```toml
[dependencies]
bevy_voxel_gi = "0.1"
bevy_voxel_gi_debug = "0.1"
```
