# Packaging plan: voxel GI crate + BVW connector

Draft, 2026-10-23, for review before any code moves. Base: `voxel-global-illumination`
at `36c9cea` (view-owned faces, two-way world cache radius 64, shared-corner sun).

## Target shape

One Cargo workspace in this repository:

| Crate | Contents | Depends on |
| --- | --- | --- |
| `voxel_gi` | Transport scene (16³ pages), GPU path tracer and world cache, view tier (prepass discovery, shared receiver pool, two-way cache, presentation), shared-corner sun, settings, WGSL shading functions | `bevy` |
| `voxel_gi_bvw` | `bevy_voxel_world` adapter: chunk spawn/despawn/edit events to pages and dirty voxels, material ID to optics table, material attached to BVW meshes | `voxel_gi`, `bevy_voxel_world` |
| `gi_lab` (this app) | Terrain scenes, player, crosshair, validation profiles, diagnostics HUD/tuning, `tools/`, `config/` | both |

The core crate's input surface is small and storage-agnostic:

- **Geometry:** insert/replace/remove a 16³ page of material IDs; `set_voxel` /
  `mark_voxel` edits (today's `VoxelGiDirty`).
- **Materials:** an optics table (albedo, emission) per material ID.
- **Lights:** sun/sky from Bevy directional lights (today's `sources.rs`).
- **Cameras:** a marker component for cameras that own view receivers.
- **Settings:** `cache_radius` (the performance knob), GPU share, reach/bounces.

## Decision 1: how a game's material gets the lighting

Today `voxel_transport.wgsl` is one `ExtendedMaterial` that also contains the demo's
texture-array sampling (`mat_array_texture`, `tex_idx`). A library cannot ship that.

- **A (recommended).** The core crate exports a WGSL module (`voxel_gi::lighting`)
  with `indirect(position, normal)`, `sun(position, normal)` and `emission(voxel)`,
  plus the bind-group layout. Any material imports it. The crate also ships
  `VoxelGiMaterial = ExtendedMaterial<StandardMaterial, VoxelGiExtension>` for users
  with plain colors. The lab's texture-array material becomes an example of a
  custom material.
- **B.** Ship one fixed material with a pluggable albedo hook. Less flexible and
  harder to keep in step with Bevy's PBR changes.

The directional-light and shadow overrides (`lighting::directional_light`,
`fetch_directional_shadow`) are part of what A must expose as well.

## Decision 2: what is out of scope

- **Infrared/ionizing bands.** `direction.md` stage 3 already plans band removal.
  Visible RGB only would remove `EnergySpectrum`'s extra bands from CPU records,
  GPU state and shaders. Recommended unless StarWrought will query heat or
  radiation from GI.
- **CPU energy field and gameplay queries** (`VoxelEnergyField`, the CPU statistics
  mirror). Today only diagnostics, dispatch and publication read it; the GPU state
  is the authority. Recommended: keep only what the world cache's scheduling
  needs, and make a light-level query API a later, explicit feature.
- **CPU path tracer / oracle** (`tracer.rs`, `sampling.rs`, emission tree,
  `reconstruction.rs`, `correctness_audit`, `research.rs`). Dead in release
  (compiler, below); used only by tests and validation. Move to the lab's test
  suite or drop.

## Audit method

Every system must contribute to the default frame or move out. Three passes:

1. **Compiler:** release build with the dead-code allowances removed (done,
   below).
2. **Default-path trace:** for each Update system, worker job and GPU dispatch at
   default settings, record what it writes and who reads it, down to the frame.
   Anything whose output no frame reads is a candidate.
3. **Delete and measure:** remove a candidate; it stays removed if these match
   the baseline: headless streaming turn burst (black frames), motion walk
   (black flashes), static pose (combined, sun vs `sun-reference`, indirect),
   tunnel and response (black-then-lit, FLIP), cave/streaming GPU receipts,
   native view/cache/coverage tests. These are the gates used on 2026-10-23.

### Pass 1 result: dead in a release build

32 warnings, all CPU-oracle code: `tracer.rs` (`trace_work_batch`,
`trace_sky_batch`, `trace_face`, `trace_path`, emitter sampling, ...),
`sampling.rs` (`sample_1d`, `hemisphere`, ...), `emission.rs`/`emission/tree.rs`
selection, `gpu.rs::trace_compact`, `readback::wait`, and three `world.rs`
helpers. About 1,000 lines.

### Preliminary inventory (to verify in pass 2)

GI code today: 31.5k lines of Rust (5.6k in test files, 4.0k validation and
research, 1.2k diagnostics) and 3.3k of WGSL.

| Area | Files | Proposed home | Note |
| --- | --- | --- | --- |
| Scene and traversal | `world.rs`, `world/delta.rs`, `naadf.rs`, `gpu_scene.rs`, `gpu/geometry.rs`, `voxel_traversal.wgsl` | core | |
| GPU tracer, world cache | `gpu.rs`, `gpu/resident.rs`, `gpu/face_index.rs`, `gpu/readback.rs`, `gpu/scatter.rs`, `voxel_trace/path/accumulate/retention.wgsl` | core | `gpu.rs` and `resident.rs` (5.3k lines) likely still carry world-display leftovers; record 8 is unused |
| Scheduling and publication | `dispatch.rs`, `work.rs`, `scheduling.rs`, `publication.rs`, `statistics.rs`, `telemetry.rs` | core | CPU event jobs and the stats mirror need pass 2 |
| Retention | `retention.rs`, `geometry.rs` | core | Edit-flux retention for the world cache; check its value against a no-retention run |
| Sun | `sky.rs`, `sky_cache.rs` | core | The view tier traces its own corners; is the 1k-line CPU sky cache still needed for world-cache faces? |
| Sources, materials | `sources.rs`, `spectrum.rs`, `emission.rs` | core | Shrinks with band removal |
| View tier | `gpu/view.rs`, `gpu/view/runtime.rs`, `gpu/view/timing.rs`, `voxel_view_*.wgsl` | core | Timing receipts: optional feature |
| Rendering | `render.rs`, `voxel_transport.wgsl` | core, split per decision 1 | |
| Display history | `history.rs`, `FaceDisplay` in `SurfaceProbe` | likely delete | CPU display seeds from the removed world display; verify nothing reads them |
| CPU field | `field.rs` | per decision 2 | |
| CPU oracle | `tracer.rs`, `sampling.rs`, `emission/tree.rs`, `reconstruction.rs`, `gpu/correctness_audit*`, `research.rs` | lab tests or delete | Dead in release |
| BVW adapter | `bvw.rs` | connector | Includes residency (`update_receiver_residency`), which belongs in core |
| Diagnostics | `diagnostics*.rs` | lab | HUD, tuning UI, face watch |
| Validation | `validation*.rs`, `tools/`, `config/` | lab | |

## Staging

1. **Audit in place.** Passes 2 and 3 on the current crate, one small commit per
   removal, each with its gate evidence. The app keeps working throughout.
2. **Band removal** (if decided), measured like any other change.
3. **Workspace split.** Move files into `crates/voxel_gi`, `crates/voxel_gi_bvw`
   and the lab without behavior changes; one commit, gates rerun.
4. **Public API.** Narrow `pub` items, the material split (decision 1), plugin
   configuration, docs and a minimal example (a few voxels, a sun, one camera).
5. **Publish readiness.** Licence, README, Bevy/BVW version pins, CI for native
   tests that need a GPU (Metal locally; decide what runs headless elsewhere).

## Questions for the user

1. Material integration: A (WGSL functions + a default material) or B?
2. Drop infrared/ionizing bands and the CPU energy field?
3. Crate names, and the workspace in this repository or a new one?
