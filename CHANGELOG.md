# Changelog

All notable changes to `bevy_voxel_gi`, `bevy_voxel_gi_bvw` and `bevy_voxel_gi_debug`.
The crates share one version. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed

- Bevy 0.20 (wgpu 30), `bevy_voxel_world` 0.19 and Rust 1.97.1.
- Shaders are WESL: Bevy 0.20 removed naga_oil. `bevy_voxel_gi::presentation`
  keeps its import path. Host materials can no longer override Bevy's sun;
  they call `bevy_voxel_gi::pbr::apply_pbr_lighting(pbr_input, footprint)`
  instead of Bevy's, a copy of Bevy's function with GI's sun in place of
  the first directional light while GI's sky is on.
- An API pass for approachability (breaking; see "Migrating from 0.1" below):
  - `GiVoxel` replaces raw material IDs: `GiVoxel::AIR` or
    `GiVoxel::solid(0..=255)`, in `PageSample`, `dense_page`, `coarse_page`
    and `edit_voxel`. Page lookups return `GiVoxel` or `Option<GiVoxel>`.
  - `VoxelGiMaterials` (formerly `VoxelMaterialTransportTable`) assigns
    materials itself: `insert(optics)` shares a material between identical
    optics. `MaterialPalette` and its table copy are gone; `PaletteFull` is
    `MaterialsFull`.
  - `VoxelGiSource::new(source)` and `set`; the field is private. A closure
    `Fn(IVec3) -> PageSample` is a `VoxelSource`. `sample_page` loses its
    `resample` flag: sources that cache override `resample_page` instead.
  - `standard_material(base)` and the BVW connector's
    `material_template(texture)` no longer take `VoxelGiMaterialData`;
    custom materials start from `VoxelGiBindings::default()`, and
    `VoxelGiMaterialPlugin` fills the bindings in when a material is added
    or replaced, before it is first drawn.
  - `VoxelGiPlugins` adds GI and the GI-lit `StandardMaterial`, and
    `bevy_voxel_gi::prelude` holds the types most hosts use.
  - `VoxelGiSettings { cache_radius: 32.0, ..default() }` compiles outside
    the crate.
  - `VoxelGiCamera` marks the cameras GI lights, and brings the depth and
    normal prepasses it needs. Other 3D cameras cost GI nothing (before,
    every active `Camera3d` got the view tier and world cache). GI warns
    once when only unmarked cameras are active.
- `bevy_voxel_gi_bvw`: BVW 0.19's levels of detail. A level-L chunk spans
  (2^L)³ GI pages, traced at its sample spacing; while BVW swaps levels, GI
  follows the finer chunk until the coarser one has replaced it. Levels
  above `MAX_GI_LOD` (5) are not traced. bevy_voxel_world 0.19 protects
  a 32 times smaller radius from out-of-view despawning than 0.17 did; use
  `ChunkDespawnStrategy::FarAway` to keep the lighting of 0.1.
- `bevy_voxel_gi_bvw`: GI traces what BVW draws. A page a published chunk
  covers is that chunk as meshed, whether sampled or resampled
  (`mark_voxel`, `resample_page`, `mark_all_loaded`); 0.1 resampled through
  BVW's edit map, at full resolution even inside coarse chunks. Writes made
  with BVW's own `set_voxel` reach GI when BVW republishes the chunk;
  `BvwVoxelEdits::set_voxel` still reports its exact edit at once.

### Fixed

- World-cache residency visits the pages around the cameras instead of every
  loaded page each frame: the lab's main-thread Update schedule fell from
  1.75 to 0.13 ms a frame in a 160,000-page world.
- Scene builds cost the change instead of the whole loaded world: the
  region map is shared between builds and the top-level empty-space field is
  updated where regions change. In a 110,000-page world an edit that fills
  sky took about 2 s to build and now takes about 70 ms; streaming far
  terrain no longer shows faces black while a build catches up.
- Scene builds sample and publish pages nearest any active `VoxelGiCamera`
  first. They followed the first marked camera, active or not, so under a
  streaming backlog an inactive camera's surroundings went before the
  player's.
- GI-lit materials show emission through cameras without `VoxelGiCamera`
  (a minimap, a portal) and through GI cameras without receivers that frame,
  as `VoxelGiCamera` documents. They showed none.
- `bevy_voxel_gi_bvw`: with default input tracking, a chunk BVW republishes
  or swaps for another level is sampled at its stride, as drawn. It was
  resampled through BVW's edit map at full resolution, 32,768 lookups a
  page (about 134 million for a level-4 chunk).
- `bevy_voxel_gi_bvw`: GI traces a chunk at its meshing shape
  (`chunk_meshing_shape`), as BVW's mesher resamples it, instead of at its
  stored data shape. A chunk stored finer than it is drawn cast shadows and
  bounced light from detail its mesh does not show.

### Added

- `PBR_SHADER` and `insert_voxel_gi_shaders`, which registers GI's shader
  modules for composing materials outside an app.
- `VoxelGiDemand`: where GI needs geometry, for hosts that choose what to
  load. It ranks pages by distance from the cameras and along each sun's
  shadow column, so a streaming host can keep what GI's rays reach.
- `VoxelGiInput::resample_page`: drop a page's kept edits and sample its new
  source, without the urgent relighting `mark_voxel` gives an edit.
- `bevy_voxel_gi::presentation::diagnostic_view`, the `DEBUG_*` mode constants
  and `SUN_LAMBERT_SCALE`: one diagnostic display for every host material.
  The stock `StandardMaterial` now shows every diagnostic view, not only
  coverage.

### Known limitations

- Opaque diffuse transport only: no colored transmission, no specular GI.
- The far tier's indirect light flickers somewhat more in motion than the
  per-face tiers'.
- Not WebGPU: the view tier needs more storage buffers per stage than
  WebGPU allows.
- A change of the loaded world's bounds still copies the top-level field and
  repacks every region header, and the GPU starts a new buffer generation.
  Other scene builds cost the change.
- `VoxelGiDemand`'s ranking is reasoned, not measured, and the BVW connector
  cannot honor it: bevy_voxel_world has no way to keep chunks a consumer
  asks for.
- `bevy_voxel_gi_bvw`: GI follows BVW's own mesher; a host's meshing
  function can draw something else.
- 0.2 is measured on an Apple M4 (Metal). 0.1 was also tested on an AMD
  RX 7900 XT (Vulkan, DX12 with DXC).

### Migrating from 0.1

| 0.1 | 0.2 |
| --- | --- |
| `add_plugins((VoxelGiPlugin, VoxelGiStandardMaterialPlugin))` | `add_plugins(VoxelGiPlugins)` (both still exist on their own) |
| `use bevy_voxel_gi::{...}` | `use bevy_voxel_gi::prelude::*` |
| Every active `Camera3d` is lit | Add `VoxelGiCamera` to the cameras GI lights |
| `source.0 = Some(Arc::new(MySource))` | `source.set(MySource)` or `commands.insert_resource(VoxelGiSource::new(MySource))` |
| `fn sample_page(&self, page, resample: bool)` | `fn sample_page(&self, page)`; override `resample_page` if `resample` mattered |
| `MaterialPalette::new()`, `insert(key, optics)`, `*table = palette.table().clone()` | `materials.insert(optics)` on `ResMut<VoxelGiMaterials>`; keep your own key map if you need one |
| `palette.optics_id(optics)` | `materials.insert(optics)` |
| `insert_transparent(key)` | `GiVoxel::AIR` |
| `VoxelMaterialTransportTable` | `VoxelGiMaterials` (`get`, `set` unchanged) |
| `u16` IDs, `AIR`, `id as u8` | `GiVoxel`, `GiVoxel::AIR`, `GiVoxel::solid(id)` |
| `PageSample::full(Vec<u16>)` | `PageSample::full(Vec<GiVoxel>)` (no copy) |
| `page.material(local)` | `page.voxel(local)` |
| `dense_page(page, \|v\| Some(id_u8))` | `dense_page(page, \|v\| Some(gi_voxel))` or return a `GiVoxel` |
| `edit_voxel(v, Some(id))`, `edit_voxel(v, None)` | `edit_voxel(v, GiVoxel::solid(id))`, `edit_voxel(v, GiVoxel::AIR)` |
| `standard_material(base, &data)` | `standard_material(base)` |
| `material_template(texture, &data)` (BVW) | `material_template(texture)` |
| `VoxelGiBindings::from(&data)` | `VoxelGiBindings::default()` (the plugin fills it in) |
| `VoxelGiSettings { cache_radius: 32.0, ..default() }` did not compile | It does |

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

- The repository README opens in a lit stone room, routes readers by goal
  (your own engine, `bevy_voxel_world`, live tuning) and points to runnable
  examples with what to look for.
- Core guide: a copyable quick start, a which-call-when guide to reporting
  edits, custom materials with a shader function reference, a tuning table,
  and troubleshooting by symptom.
- `bevy_voxel_gi_bvw`: a copyable quick start for a lit platform, and steps
  for building the block texture array.
- Every public item is documented (`missing_docs` is now enforced); the
  docs.rs page adds a linked API map.
- Windows guidance reports what was tested (both backends pass on an RX
  7900 XT; DX12 needs DXC) instead of an unqualified recommendation.
- A second example, `edits`: a window opening and closing with `edit_voxel`.
- A captioned strip of light responding to an edit, badges and a
  contributing section.

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
