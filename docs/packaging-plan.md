# Packaging plan: voxel GI crate + BVW connector

2026-10-23. Base: `voxel-global-illumination` at `36c9cea` (view-owned faces,
two-way world cache radius 64, shared-corner sun). Reviewed with the user the
same day; decisions are recorded inline.

## Requirements

- **No performance loss.** Every stage is compared against `36c9cea` on the same
  machine: windowed uncapped frame pacing (`VOXEL_GI_VALIDATE_NO_VSYNC=1`), GPU
  receipts (trace, presentation, discovery, pool) and GI main-thread time, plus
  the visual gates below. A refactor that costs frame time is not done.
- **Drop-in for asteroids-redux.** Adding the two crates to
  `~/Developer/asteroids-redux` must just work. That project uses the user's BVW
  fork (`perf/optimization-series-preview`), its own
  `ExtendedMaterial<StandardMaterial, AsteroidVoxelExtension>` (fade
  parameters), BVW chunk LOD (`lod_samples`), a sun with no sky, and a
  WebGPU/Safari build (`web_lab`). planet_voxel_warp_redux is a later StarWrought
  goal, out of scope for now.
- **Per-face or per-pixel lighting.** Today lighting is resolved per voxel face
  (corner-interpolated). Other users may want per-pixel lighting; the crate must
  offer it as an option (see "Lighting resolution").
- **Tuning that matters.** Keep the knobs that scale between the MacBook Neo and
  GPUs better than the M4; drop research-only switches.
- **RON, not JSON**, for settings and presets (done in `4f3e0f5`; validation
  reports stay JSON for the Python tools).

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

## Decision 1: how a game's material gets the lighting (decided: A, 0.20-ready)

Today `voxel_transport.wgsl` is one `ExtendedMaterial` that also contains the demo's
texture-array sampling (`mat_array_texture`, `tex_idx`). A library cannot ship that.

The core crate exports GI lighting as plain shader functions (`indirect`, `sun`,
`emission`) plus their bind-group layout, and ships a default
`ExtendedMaterial<StandardMaterial, …>` for users with plain colors. The lab's
texture-array material and asteroids-redux's fade material call the same functions.

**Bevy 0.20 constraint.** 0.20 replaces naga_oil with WESL: `#import` becomes
`import …;`, `#ifdef` becomes `@if(…)`, `#{MATERIAL_BIND_GROUP}` becomes
`constants::MATERIAL_BIND_GROUP`, and shaders are renamed `.wesl`. Plain WGSL
with no directives keeps working. The 0.20 release notes and migration guide
describe no replacement for naga_oil's `override fn`, which our material uses
today to replace `lighting::directional_light` and
`shadows::fetch_directional_shadow` (the corner sun takes the place of Bevy's
sun and shadow map). So:

- No `override fn` anywhere. The material computes the sun term itself and keeps
  the GI sun out of Bevy's directional-light loop. How to exclude it per mesh
  without overrides (light render layers, a GI-owned sun component, or calling
  Bevy's point/spot functions directly) is the first spike, built against 0.20
  in a branch before the API is fixed.
- The compute shaders (`voxel_trace`, `voxel_path`, `voxel_view_*`,
  `voxel_traversal`) are composed from plain WGSL by our own Rust, so the WESL
  switch does not affect them. Keep them directive-free.
- Package on 0.19 (BVW, avian and asteroids-redux are on 0.19), with the
  material module written so the 0.20 port is the mechanical import/def rewrite.

Sources: [Bevy 0.20 release notes](https://bevy.org/news/bevy-0-20/),
[0.19 to 0.20 migration guide](https://bevy.org/learn/migration-guides/0-19-to-0-20/).

## Decision 2: what is out of scope (decided)

GI does no simulation, and simulation has no place in a general package.

- **Infrared/ionizing bands: remove.** Visible RGB only, in CPU records, GPU
  state and shaders (`direction.md` stage 3).
- **CPU energy field and gameplay queries: remove** (`VoxelEnergyField`, the CPU
  statistics mirror), keeping only what world-cache scheduling needs.
- **CPU path tracer / oracle** (`tracer.rs`, `sampling.rs`, emission tree,
  `reconstruction.rs`, `correctness_audit`, `research.rs`). Dead in release
  (compiler, below); used only by tests and validation. Move to the lab's test
  suite or drop.

## Decision 3: layout (decided)

A workspace in this repository, as in "Target shape". Crate names can still change
before publishing.

## Asteroids-redux drop-in

What the connector and core must handle for it, beyond the lab:

- **BVW version.** The connector builds against the fork branch it uses; keep the
  BVW surface it touches small (chunk events, voxel reads, mesh entities).
- **Its own material.** Covered by decision 1: its extension calls the GI functions.
- **Chunk LOD.** GI geometry is 1-voxel pages. Decide per LOD: feed only LOD 0
  chunks to the transport scene and shade coarser chunks view-only against the
  geometry that exists, or downsample. Today's code assumes every chunk is LOD 0.
- **Space lighting.** Sun only, no sky: sky radiance zero and the ambient term
  explicit, and nothing assumes terrain below a horizon.
- **WebGPU.** View-tier binding indices reach 18 in one group (the pool is at
  17/18), and WebGPU's default limit is 8 storage buffers per shader stage.
  Count the storage buffers each entry point actually uses before deciding. Timestamp queries are optional there.
  Either merge buffers to fit, or document native-only and fall back to a
  cheaper path on web. Decide before the bind-group layout becomes public API.

## Lighting resolution

- **Per face (current).** Receivers are voxel faces; sun and indirect are
  interpolated across the face from shared corners. Cheap, stable, voxel look.
- **Per pixel (to add).** Two parts:
  - Sun: trace visibility from the fragment's own position, not the face
    corners (today's `sun-reference` diagnostic traces the four corners per
    fragment, which is the cost model to start from).
  - Indirect: either finer receivers (sub-face cells) or per-pixel sampling with
    temporal and spatial reuse, NAADF-style (`direction.md`). This is a
    separate design and cost study after packaging, not a flag.
- A setting selects the mode; the per-face path must not pay for the per-pixel one.

## Tuning surface

Keep public, with RON presets for the Neo, the M4 and larger GPUs:

- `cache_radius` (world cache size, the main memory/cost knob);
- `gpu_time_share` and `trace_budget_ms` (GI's GPU share and batch size);
- `samples_per_dispatch`;
- view receiver density (today the fixed 4-pixel grid capped by
  `MAX_PIXELS = 65536`; at 1440p cells are already ~7.5 px, so better GPUs can
  afford a finer grid);
- `spatial_reconstruction` and `spatial_evidence_target`;
- `transport.reach` and `transport.bounces`.

Candidates to drop or make internal: `adaptive_updates` (A/B only),
`statistics_mirror_per_batch` and `cpu_statistics` (with the CPU field),
`simulation_sampling_importance` (with the bands), `measured_retention` (keep
only if pass 3 shows a visible benefit), `stable_*` and `display_response` unless
a preset needs them.

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
   removal, each with its gate evidence and the performance comparison. The app
   keeps working throughout.
2. **Band removal and CPU field removal**, measured like any other change.
3. **Workspace split.** Move files into `crates/voxel_gi`, `crates/voxel_gi_bvw`
   and the lab without behavior changes; one commit, gates rerun.
4. **Public API.** Narrow `pub` items, the material split (decision 1, after
   the 0.20 spike), plugin configuration and tuning presets, docs and a minimal
   example (a few voxels, a sun, one camera).
5. **Asteroids-redux integration.** Drop both crates in; resolve LOD, space
   lighting and the WebGPU decision; it must work without changes to GI.
6. **Per-pixel lighting option** (design and cost study first).
7. **Publish readiness.** Licence, README, Bevy/BVW version pins, CI for native
   tests that need a GPU (Metal locally; decide what runs headless elsewhere).

## Open questions

1. Per-pixel lighting: is the interpretation above (sub-face sun and indirect,
   selectable per project) what other users would want?
2. Asteroids-redux on the web: is the WebGPU build a target for GI, or native only?
