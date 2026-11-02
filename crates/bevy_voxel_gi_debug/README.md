# bevy_voxel_gi_debug

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
