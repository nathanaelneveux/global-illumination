# Changelog

All notable changes to `voxel_gi`, `voxel_gi_bvw` and `voxel_gi_debug`.
The crates share one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- First release, for Bevy 0.19 on native GPUs (Metal, Vulkan, DX12).
- `voxel_gi`: diffuse global illumination for axis-aligned voxel worlds:
  sunlight, sky, emissive voxels and multi-bounce indirect light traced
  against the voxels on the GPU.
  - Three lighting tiers: per-camera view receivers for visible faces, a
    persistent world cache (`cache_radius`, default 64) that keeps faces
    near the cameras lit off screen, and a per-pixel far tier for faces
    small on screen (`far_tier.voxel_pixels`, default 8).
  - Host input: 32³ pages declared through `VoxelGiInput`, voxels read through
    `VoxelSource` at full or coarse stride, `MaterialPalette` for mapping
    block types to transport materials, and `VoxelGiEmitter` for lights with
    no Bevy counterpart. Declared pages bound GI's interest.
  - Presentation: the `voxel_gi::presentation` WGSL module and
    `VoxelGiBindings` for lighting your own material, or the stock
    `VoxelGiStandardMaterial`.
  - `VoxelGiStats` and `ViewReceipts` for work, GPU time and readiness
    (`geometry_pending`).
  - Degrades instead of panicking: no timestamp queries (fixed path budget),
    receiver hash overflow (counted), failed view allocation (view tier
    skipped for that view), no device (idles).
  - `examples/minimal.rs`.
- `voxel_gi_bvw`: `bevy_voxel_world` 0.17 connector: chunk events become page
  events, chunks (including LOD chunks at their stride) are the voxel
  source, and `VoxelTransportMaterial` lights the chunk meshes.
- `voxel_gi_debug`: an opt-in HUD, tuning panel, presentation toggles and
  terminal stats.

### Known limitations

- Opaque diffuse transport only: no colored transmission, no specular GI.
- The far tier's indirect light flickers somewhat more in motion than the
  per-face tiers'.
- Not WebGPU: the view tier needs more storage buffers per stage than
  WebGPU allows.
