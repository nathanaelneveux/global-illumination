# Voxel GI Lab

Hybrid near/far diffuse lighting for the complete loaded voxel world,
forked from `prototype_voxel_template`. Bevy 0.19.1 and ordinary GPU compute;
hardware ray-query support is not required. BVW supplies terrain and meshes, while
GI owns an independent sparse voxel acceleration structure and energy field.

The preferred look is shared-corner direct sunlight, corner-reconstructed
indirect lighting, textures on, and baked AO off. Indirect transport still starts
at exposed voxel-face centers. Transport and virtual voxel corners do not
depend on greedy-mesh vertices. Near histories live in world space; far face
identities and paths live in a bounded view cache, discovered from depth/normals.

## Start here

- [Architecture](docs/architecture.md): data model, estimator, ownership,
  invalidation, presentation, simulation separation, and integration limits.
- [Structures experiment](docs/structures.md): shared templates, placements,
  chunk-local generation, and the nearby cave route.
- [Research and decisions](docs/research.md): what we adopted, what differs from
  NAADF/vxray/DDGI/Solari, and the next speed experiments.
- [Validation](docs/validation.md): focused checks, visual workloads, evidence
  requirements, and the shared build cache.
- [Cave reference](docs/history/cave-reference.md): complete chamber geometry audit,
  high-sample rendered target, independent estimator checks and image comparison.
- [October 8 audit](docs/history/audit.md): findings, fixes, retained boundaries, and
  verification for this cleanup.
- [October 10 scene-history audit](docs/history/scene-history-audit.md): removal of the
  emitter-triggered global reset and receiver material-swap reset, completed
  policy controls, response tradeoffs and next performance targets.
- [Streaming and cave performance](docs/history/streaming-performance.md): batching,
  publication and upload wins, attribution, and remaining cave noise/frame tails.
- [Current performance experiments](docs/history/experiments.md): measured winners,
  isolated alternatives, and the limits of each result.
- [Edit-directed history experiment](docs/history/history-edit-guided.md): paired edit
  and sunlight-change diagnostics, measured benefits, and why they are not yet
  a gameplay policy replacement.
- [Many-emitter research decisions](docs/history/research-2026-10-09.md): light-tree win,
  cave workload, residency contract, and rejected/deferred correction strategies.
- [Historical experiments](docs/history/experiments-2026-10.md): previous measurements,
  rejected experiments, and their limitations. This is not the current design.

```sh
cargo run --locked
```

Every visible face is view-owned: each camera discovers faces after its
depth/normal prepass and traces them with the shared estimator, with exact-face
reprojection and resumable paths. A persistent world cache within **64 voxels**
of active cameras (`cache_radius`; page bounds, 32-voxel retirement hysteresis)
terminates secondary rays and exchanges RGB history with visible faces, so
reveals, returns and offscreen edits stay warm. `config/gi-cache-32.ron` and
`config/gi-cacheless.ron` trade that cache for memory. Geometry remains
available across all loaded terrain. The hybrid near/far and all-near strategies
were removed after [the 2026-10-23 comparison](docs/history/option-comparison-2026-10-23.md);
see also [view-first](docs/history/view-first-transport.md) and
[two-way cache](docs/history/two-way-cache.md) history.

GI is a client visual system, not the heat/radiation simulation. Emitter sampling
and roulette prioritize RGB (nonvisual importance is zero). Legacy multi-band
records remain, but there is no gameplay simulation-query API. Shared view/cache history is RGB-only. A future
CPU-capable authoritative simulation will own heat/radiation and publish material
changes back to the visual world; see [the split](docs/direction.md).

With surface-cache transport, a path that hits a measured face ends there
with that face's cached energy, which gives infinite bounces. See
[surface cache](docs/surface-cache.md). Path reuse and source guiding were
removed on this branch: neither was qualified with the cache. Their history
lives in [path reuse](docs/history/resident-path-reuse.md) and
[hallway outliers](docs/history/hallway-path-outliers.md).
The game always uses resident GPU accumulation; the synchronous and CPU
runtime backends were removed on this branch (last present at `c68a924`). The
CPU tracer remains as a test oracle.

The view tier's spatial RGB filter uses a **1024 evidence target**, the value
chosen with the user-calibrated FLIP gate for the former resident display (see
[surface cache](docs/surface-cache.md#display-evidence-target)); it has not been
re-qualified for the view filter. `config/gi-display-evidence-2048.ron` raises
it. It filters presentation only, never physical samples. Use
`VOXEL_GI_SETTINGS=config/gi-display-unfiltered.ron` for the unfiltered
control; sampler accuracy must be qualified separately from this reconstruction.

The many-emitter cave is now in the regular world. Walk toward the house, then
follow the pale stone path to its right to the lit entrance near **(20, 11, -18)**.
The passage and interior stairs lead down to the lava rim.

For the original isolated many-emitter benchmark scene:

```sh
VOXEL_GI_SCENE=cave cargo run --locked
```

This finite test chamber contains a lava floor with IR, a blue-lit central spire,
green ceiling crystals with ionizing emission, wall torches, and a roof skylight.
It shares geometry/materials with the compute studies; it is not a calibrated
thermal/radiation simulation or a finished art scene. E places a torch and right
click places rock in this isolated palette. Omit the variable to explore the
regular world, with both the house and nearby cave.

The top-right HUD reports FPS, world-cache Coverage, Bootstrap and Settled, GI
main-thread time, world-cache GPU time and the visible-face (view) GPU time split
into trace, presentation and discovery. Coverage means at least one sample;
Bootstrap means progress toward eight samples; Settled means passing a
revisitable stability heuristic. None means convergence. CPU and GPU work
overlap, so their displayed costs must not be added.

## Controls

| Input | Action |
| --- | --- |
| WASD / mouse | Move / look |
| Space / Ctrl / Shift | Jump / crouch / sprint |
| Left / right click | Remove / place a voxel |
| E / Add emitting block button | Place a warm emitting voxel at the aimed-at face |
| B / Bomb button | Immediately remove a radius-5 sphere centered on the aimed-at voxel; left-click remains single-block digging. |
| Lighting controls button | Quality presets, performance and look sliders, staged light model, display switches, RON copy/paste |
| F3 | Mesh debug overlay |
| F4 | Toggle indirect GI display |
| F5 | Flat faces / virtual-corner interpolation |
| F6 | Irradiance diagnostic view |
| F7 | Cached corner sun / Bevy per-pixel sun comparison |
| Esc | Inspector and cursor release/capture |

Aim before releasing the cursor to use the emitter button. Movement/look pauses
while the cursor is released. Display toggles do not invalidate transport.

### Visual quality tuning

Press **Esc**, then **Lighting controls** in the HUD (Bevy UI, feathers widgets).

- **Quality preset** (Low / Default / High) sets only the performance knobs:
  GI GPU share (0.12 / 0.2 / 0.3), world cache radius (32 / 64 / 96 voxels) and
  visible receivers per camera (32k / 64k / 128k). Receivers set the view grid's
  cell size and are the only lever on the fixed presentation/discovery cost,
  which the GPU share does not cover; the HUD's View GPU line shows it.
- **Look**: spatial smoothing and its target, temporal response, indirect gain.
- **Light model**: reach and bounces are staged until **Apply model**, which
  restarts measurements. The first use of a depth/range combination compiles a
  cached GPU variant on the worker.
- **Display**: the F4-F7 switches.

Keep an eye on FPS, GI cost, dark-corner quality and response after edits, not
just coverage. `config/gi-low.ron` and `config/gi-high.ron` start the lab at
the other presets.

Reach limits each diffuse ray and voxel-emitter connection, not total path length
or Minecraft-style block-by-block light attenuation. A shorter reach may discard
useful distant light, and is not guaranteed to run faster: empty space is already
cheap to traverse, while rejected distant emitter samples still cost work. Sun
visibility keeps its independent 768-voxel horizon; analytic point lights retain
their source ranges. All loaded receivers remain eligible, including IR/radiation.

The opt-in `VOXEL_GI_SCENE=canyon` fixture tests high rims and long daylight
sightlines. See the [reach/depth study and invalidation constraints](docs/history/transport-horizon.md)
for controlled comparisons; neither shorter rays nor its live captures alone
establish a safe history-invalidation radius.

The [distance/history study](docs/history/distance-history.md) measures actual streamed
geometry changes and retained-history tails. Remote retention substantially
reduces repeated cave noise, but a distant bounced-sun shadow rejects a universal
384-block cutoff.
The [sun-aware follow-up](docs/history/sun-aware-retention.md) retains the remote-cave win
without changing the measured baseline shadow/edit responses, and calibrates
history curves against measured sampler quality. The
[runtime follow-up](docs/history/runtime-retention.md) introduced page-local history,
incremental sun-change evidence and worker-side retention queries. These remain
the default architecture; the original global-journal control was removed on
the surface-cache branch (last present at `8cab848`). Repeated composed-cave tests show substantially less
corner oscillation with comparable frame pacing, not a complete convergence or
locked-60-FPS solution.

The [distance-band follow-up](docs/history/distance-band-retention.md) preserves distance
information when those page journals overflow and is now the only summary.
Eight exact events remain; band storage is allocated only on first overflow.
Lighting thresholds, sampling and BVW are unchanged. The summary's measured
late-stability gain does not establish faster startup or qualified windowed FPS.

The [near-history calibration](docs/history/near-history-calibration.md) separately
measures post-edit paths to sustained FLIP acceptance and repeated-cap noise.
It uses common native sample tapes and actual material captures; replay-clock
seconds are not gameplay latency. The inherited near 48/16 defaults are not
assumed analytically valid; the completed replacement decision is below.
The [cross-scene follow-up](docs/history/history-cross-scene.md) tightens crossing
observations and adds actual house skylight, doorway and emitter images.

Those studies found that higher floors help small edits but can retain too much
stale source brightness; a cave-only blanket replacement was not promoted.
The [distance and visible-sunlight follow-up](docs/history/history-cross-scene.md#distance-attribution-and-a-visible-sunlight-patch)
adds a floor-facing 3x3 skylight and regional FLIP measurements. Its proposed
16-block neighborhood was an intermediate experiment, not the final equation.

The [controlled hallway study](docs/history/hallway-history-study.md) supports equal-sized
floor measurements from the adjacent block out to 64 blocks, with literal retained
weights below 16, matched source/occluder controls and native material captures.
The hallway is also visitable west of the house; see the [route](docs/structures.md).

The [whole-range history decision](docs/history/whole-range-history.md) is now integrated:
`min(existing_weight, round((distance / 12)^4))` over the **entire** distance
range. There is no 48-block handoff, inherited far plateau, or fixed 278/384
cutoff. Both coefficients were free; crossing 48/256 is a measured selection,
not a required anchor. It replaces the earlier near-only recommendation.
Native comparisons support a better speed/noise balance, with explicit
brightening and diffuse-roof regressions in the report; it is not a universal
speedup. Keep the current 65,536 ceiling: the earlier
[static/cap decision](docs/history/history-decision.md) did not qualify a permanent
2,048 cap. **A normal `cargo run --locked` uses the whole-range curve** for
geometry and changed sunlight, with per-receiver queries still on the tracing worker.
On the surface-cache branch it is the fallback for changes of unknown flux;
measured retention ([research](docs/retention-research.md)) is the default. The
previous calibrated runtime is preserved at `5381c17`. See the
[integration record](docs/history/whole-range-history.md#runtime-integration).

**Copy** puts the applied settings on the clipboard as RON (display switches
and gain are not settings); **Paste** applies RON from the clipboard. Save it to
a file and launch with `VOXEL_GI_SETTINGS=path.ron cargo run --locked`. Omitted
fields use defaults; unknown fields or malformed files are errors, and numeric
values are clamped to supported ranges.

## Current behavior

By default, sixteen new paths per selected face update RGB, infrared, and ionizing channels.
Every loaded exposed surface remains eligible, including offscreen receivers.
Bounded asynchronous work prioritizes nearby changes while reserving a world
sweep for distant surfaces. Finite work budgets mean lighting changes are
progressive, not instantaneous.

Display smoothing is temporal: each face blends new indirect RGB into its own
previous displayed value when a worker result is accepted. The fragment shader
then reconstructs connected coplanar corner values and interpolates across each
unit face. Those are separate temporal and spatial operations. Direct-sun
visibility is measured at shared corners and is not temporally filtered.

The latest display filter deliberately favors stable transitions over minimum
latency. A 75 ms response setting is not a promise to reach final brightness in
75 ms: sample uncertainty and available tracing work can make it much slower.
Increasing raw history is not our default answer to noise.

Terrain configuration is hot-reloaded from
[default.prototype.ron](assets/config/default.prototype.ron). Quality/work limits live in
`VoxelGiSettings`; background transport energy in `VoxelGiEnvironment`; optical
properties in `VoxelMaterialTransportTable`. Source adaptation and texture assets
have explicit owners rather than being encoded into the tracer.

## Scope

This is a tested prototype engine with library-style internal boundaries, not a
published storage-independent crate or an unlimited-world renderer. It supports
opaque, axis-aligned voxels, diffuse bounces, directional/point sources and
emissive voxel materials. The [architecture limits](docs/architecture.md#integration-contract-and-limits)
are part of the design, not promises of implemented planet warps or physical
radiation simulation.

Infrared/ionizing fields are unfiltered transport estimates in normalized units,
not temperature or dose. Gameplay must check freshness and provide absorption,
heat capacity, conduction, cooling and material-state rules. Radiological
penetration and scattering are not modeled by this diffuse opaque solver.

## License

`CC0-1.0 OR MIT OR Apache-2.0`: [CC0](LICENSE-CC0-1.0.txt),
[MIT](LICENSE-MIT.txt), [Apache](LICENSE-Apache-2.0.txt).
