# Changelog

All notable changes to `bevy_voxel_gi`, `bevy_voxel_gi_bvw` and `bevy_voxel_gi_debug`.
The crates share one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.1] - 2026-11-06

### Fixed

- Scene builds always sample edited pages first. A queue of 768 pages or
  fewer was not sorted, so the sampling time budget could defer a player's
  edit behind streaming pages.
- `bevy_voxel_gi_bvw`: a full resample of a downgraded chunk (BVW keeps its
  finer data while drawing a coarse mesh) stays at the drawn resolution
  instead of tracing the retained full-resolution data. Chunks whose data is
  itself coarse keep the authoritative edit overlay.

### Documentation

- The crate README's quick start is a complete, commented example; custom
  materials and ray-traced hosts have their own section.
- Windows guidance states what was measured (both backends pass on an RX
  7900 XT; DX12 needs DXC and reached GI's first frame in 6.4 s against
  0.3 s on Vulkan) instead of recommending Vulkan.
- Both READMEs open with one "How it works" section that also says what
  sets the approach apart, so crates.io readers see it too.
- Every public item is documented (`missing_docs` is now enforced), and the
  docs.rs landing page adds a linked API map and an FAQ of common pitfalls.
- `bevy_voxel_gi_bvw`: a compile-checked quick start covering plugin order,
  the block texture array, the camera and the sun.
- A second example, `edits`: a window opening and closing with `edit_voxel`.
- Badges, an examples list and a contributing section.

## [0.1.0] - 2026-11-02

### Added

- First release, for Bevy 0.19 on native GPUs (Metal, Vulkan, DX12). On
  Windows, Vulkan is recommended; DX12 needs the DXC shader compiler (FXC
  rejects GI's compute shaders).
- `bevy_voxel_gi`: diffuse global illumination for axis-aligned voxel worlds:
  sunlight, sky, emissive voxels and multi-bounce indirect light traced
  against the voxels on the GPU.
  - Three lighting tiers: per-camera view receivers for visible faces, a
    persistent world cache (`cache_radius`, default 64) that keeps faces
    near the cameras lit off screen, and a per-pixel far tier for faces
    small on screen (`far_tier.voxel_pixels`, default 8).
  - Host input: 32³ pages declared through `VoxelGiInput`, voxels read through
    `VoxelSource` at full or coarse stride, `MaterialPalette` for mapping
    block types to transport materials (with keyless fallback optics), and `VoxelGiEmitter` for lights with
    no Bevy counterpart. Declared pages bound GI's interest.
  - Presentation: the `bevy_voxel_gi::presentation` WGSL module and
    `VoxelGiBindings` for lighting your own material (meshes or a
    ray-traced primary), or the stock `VoxelGiStandardMaterial`. GI lights
    forward-rendered materials only; the stock materials force forward.
    `face_in_geometry` and GI's sun list (`sun_count`, `sun_direction`,
    `sun_color`) let hosts light faces GI does not cover.
  - `VoxelGiStats` and `ViewReceipts` for work, GPU time and readiness
    (`geometry_pending`).
  - Degrades instead of panicking: GI pipelines that fail to build (GI idles
    with one warning per tier), no timestamp queries (fixed path budget),
    receiver hash overflow (counted), failed view allocation (view tier
    skipped for that view), no device (idles). A GI-lit material still needs
    DXC on DX12: Bevy builds it, and FXC's failure there exits the app.
  - `examples/minimal.rs`.
- `bevy_voxel_gi_bvw`: `bevy_voxel_world` 0.17 connector: chunk events become page
  events, chunks (including LOD chunks at their stride) are the voxel
  source, and `VoxelTransportMaterial` lights the chunk meshes. BVW's
  material indices reach GI unchanged: set block optics in
  `VoxelMaterialTransportTable` at those indices.
  `prepare_texture_array` builds the block texture array with a full mip
  chain (sRGB averaged in linear light) and 8x anisotropic filtering.
- `bevy_voxel_gi_debug`: an opt-in HUD, tuning panel, presentation toggles and
  terminal stats.

### API stability

- 0.x releases may break the API. Each release targets one Bevy minor version
  (0.1: Bevy 0.19); material shaders override Bevy PBR internals that move
  between Bevy versions.
- `SchedulerStats` and the `testing` module (hidden) are diagnostics and test
  hooks, not stable API. GPU uniform layouts are internal; build GI's material
  bindings with `VoxelGiBindings::from(&VoxelGiMaterialData)`.

### Known limitations

- Opaque diffuse transport only: no colored transmission, no specular GI.
- The far tier's indirect light flickers somewhat more in motion than the
  per-face tiers'.
- Not WebGPU: the view tier needs more storage buffers per stage than
  WebGPU allows.
- Scene builds cost time in proportion to the whole loaded world: about a
  second or more at 100,000 pages (`VOXEL_GI_LAB_FAR=64` in the lab). Far
  geometry fills in over seconds there, and faces a host just remeshed (a
  chunk changing LoD) can show dark until GI's next build lands.
- Authoritative BVW resampling can reintroduce retained fine geometry into a
  downgraded chunk's coarse GI page. Normal strided ingestion follows the mesh;
  the authoritative-resampling path still needs the same consistency fix.
- The scene sampling time budget can defer a nearby edit behind streaming in
  queues of 768 pages or fewer, which do not currently receive priority sorting.
- Measured on an Apple M4 (Metal) and an AMD RX 7900 XT (Vulkan, DX12 with
  DXC) only.
