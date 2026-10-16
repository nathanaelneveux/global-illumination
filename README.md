# Voxel GI Lab

Persistent, world-space diffuse lighting for the complete loaded voxel world,
forked from `prototype_voxel_template`. Bevy 0.19.1 and ordinary GPU compute;
hardware ray-query support is not required. BVW supplies terrain and meshes, while
GI owns an independent sparse voxel acceleration structure and energy field.

The preferred look is shared-corner direct sunlight, corner-reconstructed
indirect lighting, textures on, and baked AO off. Indirect transport still starts
at exposed voxel-face centers. Neither transport nor virtual voxel corners
depends on greedy-mesh vertices or screen resolution.

## Start here

- [Architecture](docs/architecture.md): data model, estimator, ownership,
  invalidation, presentation, simulation API, and integration limits.
- [Structures experiment](docs/structures.md): shared templates, placements,
  chunk-local generation, and the nearby cave route.
- [Research and decisions](docs/research.md): what we adopted, what differs from
  NAADF/vxray/DDGI/Solari, and the next speed experiments.
- [Validation](docs/validation.md): focused checks, visual workloads, evidence
  requirements, and the shared build cache.
- [Cave reference](docs/cave-reference.md): complete chamber geometry audit,
  high-sample rendered target, independent estimator checks and image comparison.
- [October 8 audit](docs/audit.md): findings, fixes, retained boundaries, and
  verification for this cleanup.
- [October 10 scene-history audit](docs/scene-history-audit.md): removal of the
  emitter-triggered global reset and receiver material-swap reset, completed
  policy controls, response tradeoffs and next performance targets.
- [Streaming and cave performance](docs/streaming-performance.md): batching,
  publication and upload wins, attribution, and remaining cave noise/frame tails.
- [Current performance experiments](docs/experiments.md): measured winners,
  isolated alternatives, and the limits of each result.
- [Edit-directed history experiment](docs/history-edit-guided.md): paired edit
  and sunlight-change diagnostics, measured benefits, and why they are not yet
  a gameplay policy replacement.
- [Many-emitter research decisions](docs/research-2026-10-09.md): light-tree win,
  cave workload, residency contract, and rejected/deferred correction strategies.
- [Historical experiments](docs/experiments-2026-10.md): previous measurements,
  rejected experiments, and their limitations. This is not the current design.

```sh
cargo run --locked
```

The default uses GPU-resident accumulation with asynchronous simulation snapshots
and RGB-prioritized emitter sampling (RGB importance 1, IR/radiation 0.125).
All five bands keep receiving unfiltered physical estimates; simulation bands
may be noisier. This sampling priority is independent of path reuse and applies
to every backend. `config/gi-equal-band.json` selects equal-band importance;
it does not disable guiding. Explicit values in settings files remain respected.

Source-informed path guiding also defaults on. It addresses rare high-energy
indirect paths that left repeatable bright faces in the long hallway. It changes
sampling directions and MIS PDFs, not energy, history retention, or spatial
smoothing. A small GPU preparation pass adds no CPU readback. The measured
normal-world trade-off is about 16% fewer raw paths at the same compute budget;
the live hallway's 30-second raw error beats the old three-minute result.
Use `VOXEL_GI_SETTINGS=config/gi-guiding-control.json cargo run --locked` for
the unguided control. See [diagnosis, evidence and limits](docs/hallway-path-outliers.md).

The qualified path-reuse experiment is still opt-in:

```sh
VOXEL_GI_BACKEND=reuse VOXEL_GI_SETTINGS=config/gi-guiding-control.json cargo run --locked
```

Reuse combined with RGB priority reaches the cave's calibrated visual threshold
at 1280 rather than the earlier 2048-sample target. The independent
RGB-prioritized control first passes at 2048 in the subsequent matched spatial
study. Reuse costs more per batch and is not a universal throughput win. See
[measurements and tradeoffs](docs/resident-path-reuse.md).
Those historical results used guiding off; the combined reuse/guiding mode has
not been qualified. Reproducing older equal-band results requires both
`simulation_sampling_importance: 1.0` and `emitter_guiding: false` in the preset.
`VOXEL_GI_BACKEND=sync` keeps the synchronous backend available; add the equal-band
preset to reproduce its historical sampling priority too.

Early spatial RGB reconstruction now defaults to the user-selected **2048
evidence target with its full adaptive footprint**, without an extra donor cap.
It filters presentation only, never physical samples or simulation data.
Use `VOXEL_GI_SETTINGS=config/gi-display-unfiltered.json` for the unfiltered
control; sampler accuracy must be qualified separately from this reconstruction.
Total-light-aware refinement remains opt-in. See [controls, architecture and image
evidence](docs/early-spatial-reconstruction.md). `gi-display-evidence-2048-cap32/64/128.json`
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

The top-right HUD reports FPS, Coverage, Bootstrap, Settled, and GI CPU/GPU cost.
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
sightlines. See the [reach/depth study and invalidation constraints](docs/transport-horizon.md)
for controlled comparisons; neither shorter rays nor its live captures alone
establish a safe history-invalidation radius.

The [distance/history study](docs/distance-history.md) measures actual streamed
geometry changes and retained-history tails. Remote retention substantially
reduces repeated cave noise, but a distant bounced-sun shadow rejects a universal
384-block cutoff.
The [sun-aware follow-up](docs/sun-aware-retention.md) retains the remote-cave win
without changing the measured baseline shadow/edit responses, and calibrates
history curves against measured sampler quality. The
[runtime follow-up](docs/runtime-retention.md) introduced page-local history,
incremental sun-change evidence and worker-side retention queries. These remain
the default architecture; use `VOXEL_GI_RETENTION=baseline cargo run --locked`
for the original global-journal control. Repeated composed-cave tests show substantially less
corner oscillation with comparable frame pacing, not a complete convergence or
locked-60-FPS solution.

The [distance-band follow-up](docs/distance-band-retention.md) preserves distance
information when those page journals overflow and is now the default. Use
`VOXEL_GI_RETENTION_SUMMARY=floors cargo run --locked` for the previous summary.
Eight exact events remain; band storage is allocated only on first overflow.
Lighting thresholds, sampling and BVW are unchanged. The summary's measured
late-stability gain does not establish faster startup or qualified windowed FPS.

The [near-history calibration](docs/near-history-calibration.md) separately
measures post-edit paths to sustained FLIP acceptance and repeated-cap noise.
It uses common native sample tapes and actual material captures; replay-clock
seconds are not gameplay latency. The inherited near 48/16 defaults are not
assumed analytically valid; the completed replacement decision is below.
The [cross-scene follow-up](docs/history-cross-scene.md) tightens crossing
observations and adds actual house skylight, doorway and emitter images.

Those studies found that higher floors help small edits but can retain too much
stale source brightness; a cave-only blanket replacement was not promoted.
The [distance and visible-sunlight follow-up](docs/history-cross-scene.md#distance-attribution-and-a-visible-sunlight-patch)
adds a floor-facing 3x3 skylight and regional FLIP measurements. Its proposed
16-block neighborhood was an intermediate experiment, not the final equation.

The [controlled hallway study](docs/hallway-history-study.md) supports equal-sized
floor measurements from the adjacent block out to 64 blocks, with literal retained
weights below 16, matched source/occluder controls and native material captures.
The hallway is also visitable west of the house; see the [route](docs/structures.md).

The [whole-range history decision](docs/whole-range-history.md) is now integrated:
`min(existing_weight, round((distance / 12)^4))` over the **entire** distance
range. There is no 48-block handoff, inherited far plateau, or fixed 278/384
cutoff. Both coefficients were free; crossing 48/256 is a measured selection,
not a required anchor. It replaces the earlier near-only recommendation.
Native comparisons support a better speed/noise balance, with explicit
brightening and diffuse-roof regressions in the report; it is not a universal
speedup. Keep the current 65,536 ceiling: the earlier
[static/cap decision](docs/history-decision.md) did not qualify a permanent
2,048 cap. **A normal `cargo run --locked` uses the whole-range curve** for
geometry and changed sunlight, with per-receiver queries still on the tracing worker.
`VOXEL_GI_RETENTION=whole-range` selects it explicitly; `calibrated` is a
compatibility alias for the new curve, not the old formula. The previous
calibrated runtime is preserved at `5381c17`. See the
[integration record](docs/whole-range-history.md#runtime-integration).

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
