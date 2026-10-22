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
- [Cave reference](docs/cave-reference.md): complete chamber geometry audit,
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

The playable near/far default keeps world-space receivers within **64 voxels**
of active cameras (page bounds, with 32-voxel retirement hysteresis). Far faces
use the same transport and corner sunlight, with exact-face reprojection and
resumable paths. Geometry remains available across all loaded terrain. Use
`VOXEL_GI_SETTINGS=config/gi-all-near.json cargo run --locked` for the former
all-near control. See [runtime evidence](docs/history/far-view-runtime.md).

Experimental view-first controls are available without changing that default:
`VOXEL_GI_SETTINGS=config/gi-view-cached.json cargo run --locked` puts all visible
RGB on the view tier and keeps a 32-voxel bounce cache with two-way face-history
handoff (`view_cache_exchange`);
`config/gi-view-cacheless.json` removes that cache. The
[comparison and limitations](docs/history/view-first-transport.md) include cave,
house turns, emitter response and memory. These are not promoted defaults;
the [two-way cache qualification](docs/history/two-way-cache.md) substantially
improves reveal/return history. Windowed frame pacing still needs qualification.

GI is a client visual system, not the heat/radiation simulation. Emitter sampling
and roulette prioritize RGB (nonvisual importance is zero). Legacy multi-band
records and the `config/gi-equal-band.json` research control remain, but there is
no gameplay simulation-query API. Shared view/cache history is RGB-only. A future
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

Early spatial RGB reconstruction defaults to a **1024 evidence target** on this
branch, with its full adaptive footprint and no extra donor cap. The cache
estimator passes the user-calibrated FLIP gate at 1024 raw paths, half of the
old 2048. That target also keeps smooth-wall noise inside the preferred envelope
at every prefix; see [surface cache](docs/surface-cache.md#display-evidence-target).
`config/gi-display-evidence-2048.json` restores the previous target.
It filters presentation only, never physical samples or simulation data.
Use `VOXEL_GI_SETTINGS=config/gi-display-unfiltered.json` for the unfiltered
control; sampler accuracy must be qualified separately from this reconstruction.
Total-light-aware refinement remains opt-in. See [controls, architecture and image
evidence](docs/history/early-spatial-reconstruction.md). `gi-display-evidence-2048-cap32/64/128.json`
presets explicitly trade maximum candidate neighbors for cost. The 64-neighbor
option improves measured cave throughput, but loses four-sample smoothness and
has not qualified as a universal edit-response improvement.

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

The top-right HUD reports FPS and near-cache Coverage, Bootstrap, Settled, and
worker GI CPU/GPU cost. Far view costs are measured separately in runtime logs.
Coverage means at least one sample; Bootstrap means progress toward eight samples;
Settled means passing a revisitable stability heuristic. None means convergence.
CPU and GPU work overlap, so their displayed costs must not be added.

## Controls

| Input | Action |
| --- | --- |
| WASD / mouse | Move / look |
| Space / Ctrl / Shift | Jump / crouch / sprint |
| Left / right click | Remove / place a voxel |
| E / Add emitting block button | Place a warm emitting voxel at the aimed-at face |
| B / Bomb button | Immediately remove a radius-5 sphere centered on the aimed-at voxel; left-click remains single-block digging. |
| Lighting controls… button | Quality/work sliders, A/B settings, defaults reset and JSON export |
| F3 | Mesh debug overlay |
| F4 | Toggle indirect GI display |
| F5 | Flat faces / virtual-corner interpolation |
| F6 | Irradiance diagnostic view |
| F7 | Cached corner sun / Bevy per-pixel sun comparison |
| Esc | Inspector and cursor release/capture |

Aim before releasing the cursor to use the emitter button. Movement/look pauses
while the cursor is released. Display toggles do not invalidate transport.

### Visual quality tuning

Press **Esc**, then **Lighting controls…** in the HUD. The light-model section
stages changes until **Apply model**; work/refinement and presentation sliders
apply live. Defaults remain 768-voxel reach, three diffuse surface hits, sixteen paths
per face per batch, a 4 ms batch target and 75 ms display response.

Start by saving defaults to **A**, then compare reach **16 / 32 / 64 / 128 / 256**
with bounce depth and budget unchanged. Save a candidate to **B**. These restore
settings, not cached images: a changed model needs fresh refinement. Keep an eye
on FPS, GI cost, dark-corner quality and response after edits, not just coverage.
The first use of a depth/range-policy combination also compiles a cached GPU
variant on the worker; subsequent comparisons reuse it.

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

**Copy applied JSON** exports transport/work settings (not display toggles/gain).
Paste JSON into the same panel to import, or save it to a file and launch with
`VOXEL_GI_SETTINGS=path.json cargo run --locked`. Omitted fields use defaults;
unknown fields or malformed files are errors, and numeric values are clamped to
supported ranges. A/B slots are session-only and include presentation.

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
