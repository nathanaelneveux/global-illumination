# Direction: near/far lighting and the simulation split

Decided with the user on 2026-10-21, while moving the base spec from an M4 Mac
mini to the MacBook Neo (A18 Pro: 5-core GPU, 8 GB shared memory, 60 GB/s) and
the A19 Apple TV (same class). This is the plan, not the current
implementation; [architecture](architecture.md) describes what exists.

## Why change

GI today keeps a world-space lighting cache for every exposed face in the
loaded world. Memory and the fixed per-batch GPU cost (ranking ~800k slots,
applying events) scale with world size, not with what is on screen.

Per-face sizes: ~240 B of GPU state (state 144 + metadata 48 + rollback
checkpoint 48), a 936 B CPU `SurfaceProbe` and a 256 B display probe per
probe, and probes are almost one per face. Tracked faces at the end of each
profile (M4, 2026-10-21) and the share within a radius of the player:

| Profile | Faces | Probes | <=64 | <=128 | <=192 | <=256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gameplay | 791k | 732k | 4% | 14% | 25% | 41% |
| live-hallway | 1.19M | 1.10M | 2.5% | 9% | 20% | 36% |
| day-cycle (canyon) | 528k | 437k | 7% | 22% | 41% | 67% |
| cave | 1.05M | 980k | 4.5% | 15% | 30% | 55% |

So GI's per-face state is roughly 0.6-1.5 GB (CPU probe records alone
0.4-1 GB, likely most of the 831 MB MALLOC_SMALL measured on the response
profile), and a 128-voxel world-space radius keeps 9-22% of it.

NAADF's renderer reaches long draw distances because its lighting is view
driven: it stores path samples per pixel with temporal and spatial reuse, so
lighting cost scales with the screen, while its nested distance fields keep
traversal cheap at any distance. Our world-space cache is view independent
(stable when turning, persistent off screen, editable), which is what we want
near the player, but it pays for every loaded face.

## Lighting: world-space near, view-driven far

- **Near (world space).** The current per-face cache, residency limited to a
  small radius around each local player (target: about the simulation
  distance, a few chunks). Edits, retention, the surface cache, corner sun and
  the day cycle work as now.
- **Far (view driven).** Path samples per pixel for surfaces beyond the
  world-space radius, with temporal reprojection and spatial reuse in the
  manner of NAADF, using the same transport rules (materials, sun, sky,
  emitters, reach) so the look matches at the boundary and at any distance.
  Far paths may end on near-cache faces.
- **Why this order of preference.** A different far lighting model (e.g. a
  Minecraft-style skylight) would need the near radius to stay large to hide
  the change of look. Matching rules let the world-space radius shrink, which
  is where the memory and fixed GPU cost go.
- **Budget.** Both tiers share `gpu_time_share`. Fixed per-batch cost shrinks
  with the near radius, the cost that a halved memory bandwidth hits hardest.

Open questions: blending across the radius (dithered handoff or a band where
both run); disocclusion and ghosting in the far tier when turning; its
per-pixel memory at 4K.

Staging:

1. **Receiver radius: done** (`receiver_radius`, default 0 = unlimited).
   Geometry (the transport scene) stays all loaded pages, since rays and the
   far tier traverse it; receivers (CPU probes, display probes, resident
   slots, sun corners) exist only for pages within the radius of any camera
   (page nearest point), kept until 32 voxels beyond it. Pages crossing it are
   rescanned or retired through the page publication path, without a
   geometry rebuild, and retired pages release their resident slots and sun
   corners. Measured (M4, headless gameplay): peak footprint 3.14 GB
   unlimited, 1.17 GB at 128, 1.0 GB at 64; GI main-thread p99 6.4 -> 3.2 /
   2.1 ms; more paths traced (fewer slots to rank). Faces beyond the radius
   now get view-owned indirect transport and shared-corner sunlight (F2 below).
   The radius remains opt-in while gameplay qualification is completed.
2. **Far tier**, view driven, as above; then the radius shrinks. In stages:
   - **F0, shared traversal: done.** `trace_domain` and its helpers moved to
     `voxel_traversal.wgsl`, included textually by the transport compute
     shader and the terrain material (globals `params`, `directory`,
     `regions`, `cells` declared by each), so near and far traverse the same
     geometry with the same code.
   - **F1, far sun in the material: superseded by shared-corner visibility.** The GI worker publishes its geometry
     buffers to the terrain material as coherent allocation generations.
     Far view owners trace the same four voxel-corner visibility rays as near
     receivers. The material interpolates these corners; view coverage holes
     evaluate the same corners directly, rather than changing to hard shadows.
     Geometry transactions and receiver retirement also run without near
     receivers or ray allowance. Native lifetime/maintenance tests cover these
     contracts. This per-fragment visibility pass is not yet charged against
     the shared compute budget; do not promote a smaller default radius on
     its strength alone.
   - **F2, far bounce: live integration implemented.**
     `voxel_path.wgsl` is the shared estimator; `voxel_view_trace.wgsl` writes
     individual raw paths from explicit surface positions. Matched prefixes
     pass against the near consumer in the hallway, skylight and cave, including
     shortened reach and frozen near-cache termination. Each 128-byte diagnostic
     record retains receiver identity, direction/PDF, first-hit geometry, RNG
     replay key, separate emitter/continuation energy, direct energy and known
     status. Black paths count; unknown and inactive paths do not. This is not
     a reservoir or compressed production layout. Live view state retains these
     records alongside raw moments and unfinished continuations.
     A second entry point advances uncached tails by at most eight hits per
     dispatch, saving throughput, accumulated energy, ray, MIS state and bounce
     index in 128 bytes of **in-flight scratch**. Paused energy is not published
     as a sample. Without the qualification-only depth limit, paths finish by
     cache hit, escape, roulette, zero throughput, reach, or explicit unknown.
     One/three/eight-hit partitions match the unsplit estimator; generation and
     receiver changes restart incompatible continuations. The live per-view
     queue uses three-hit waves, exact voxel-face deduplication/reprojection,
     shared change retention and a 25% partition of the existing ray budget.
     Depth/normal prepasses discover receivers at reduced resolution; the
     material reads geometry-aware reconstructed results. Spatial filtering is
     presentation only: shifted path reconnection/resampling remains future
     work, not something implemented implicitly by the filter. See the
     [runtime evidence](history/far-view-runtime.md).
   - **F3, shrink the radius** to about the simulation distance.
3. **Band removal and a slim CPU record**, with stability recalibrated.

F2 supplies indirect light beyond the radius. Camera movement, handoff and
frame pacing determine default-radius promotion, not kernel correctness alone.

### F2 design constraints from NAADF

The user's view-space direction is deliberate: retain the world-space near
cache, but make far receiver/sample storage proportional to the view, not to
loaded terrain. Geometry remains world-space and shared by both tiers.

The upstream implementation retains individual path records, reprojects them
into screen buckets, then reconnects candidates with geometry/PDF checks and
visibility rays. It is **not just a temporally blended screen texture**. See
[sample refinement](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/render/versions/base/renderSampleRefine.fx)
and [spatial resampling](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/render/versions/base/renderSpatialResampling.fx).

Build the next stage around these contracts:

- First-hit data belongs to a view and frame: world position/voxel face,
  material, normal, depth and near-cache ownership. Resizing, camera cuts,
  disocclusion and edits must reject incompatible reuse explicitly.
- Far tracing uses the same material/source/reach rules as near transport;
  it may terminate on valid near-cache measurements. It must not recursively
  treat a filtered screen result as an independent path measurement.
  Matching a finite-path kernel is necessary but not sufficient: the near
  cache's converged multi-bounce solution must also be matched when far paths
  cannot reach a cached face. The analytic furnace is the gate for that tail.
  On M4/Metal, 98,304 raw paths per depth in the existing RGB furnace
  (reflectances 0.30/0.55/0.70) measured fractions of the infinite-bounce
  solution of **0.9925/0.9094/0.7608** at three hits,
  **1.0006/1.0009/0.9992** at 16, and **1.0006/1.0010/1.0013** at 32.
  No unknown paths occurred. This is an analytic transport check, not a
  gameplay/FLIP qualification or a universal 16-hit cutoff. Far paths need
  budgeted continuation of uncached tails; reusing a permanently truncated
  estimate would retain its darker fixed point. Deeper sampling reserves
  distinct dimensions so it cannot alias the near prefix's roulette choices.
  The resumable version, with three hits per dispatch and no path-depth cutoff,
  measured **1.0003/0.9999/0.9983** over 49,152 raw paths; every path finished
  within ten waves in this fixture. That is a correctness result, not a chosen
  runtime cadence or a promise that every scene finishes in ten waves.
- Temporal path reuse needs origin, direction/hit geometry and proposal
  information, including correct accounting for dark samples. Color-only
  history cannot later be reinterpreted as a resampling reservoir.
- Spatial reconnection needs visibility and density/Jacobian handling;
  normal/depth rejection alone does not establish correctness. Keep raw-path
  tests separate from presentation smoothing.
- Near/far handoff uses actual published cache validity, with an explicit
  transition policy; a radius test alone is insufficient during streaming.
- A view-local dispatch budget shares the existing GPU-time allowance. Measure
  disocclusion response, edit freshness, memory and frame tails before shrinking
  the default radius. No full-frame CPU readback or main-thread GPU wait.
  Pending paths must retain their original receivers across ordinary camera
  motion; a new frame is not a new generation. Reproject completed records into
  the current view. Replacing pending receiver lists every frame would keep
  restarting longer paths and bias the moving view dark. Likewise, do not treat
  the early-completed subset of a cohort as representative of all its paths:
  completion time is correlated with path depth. Queue ownership and publication
  must explicitly handle that before the kernel becomes a gameplay tier.

### Near-kernel regression checkpoint

At `81c6b72` plus the resumable-kernel diff, compared against the pre-experiment
`5a638df` composed shader on Apple M4/Metal, dev profile, no viewport (isolated
compute), unchanged source/material settings: 30 alternating pairs passed
five-band mean/moment and exact count/direct/sky parity. GPU p50/p95 ms:

| Workload | Baseline | Resumable shared kernel |
| --- | ---: | ---: |
| 1,024 open-floor faces × 4 paths | 0.053 / 0.059 | 0.053 / 0.058 |
| 8,100 faces in nine emissive rooms × 4 paths | 1.798 / 3.484 | 1.798 / 3.498 |

No near-kernel performance change is claimed by that historical checkpoint.
The subsequent live integration is documented above; gameplay evidence must be
associated with its own commit and settings, not this isolated kernel timing.

## Simulation: deterministic direct transport, not GI

Game rules from the user: walls block radiation; materials exposed to enough
infrared for long enough melt and become emitters (a planet near its star may
melt its whole surface); simulation only runs a few chunks around players.

GI should not be the source of truth for these:

- **Authority.** A block melting is a world edit the server must decide. GI
  runs on each client's GPU, is randomly sampled, and depends on that client's
  camera and residency; a dedicated server may have no GPU. StarWrought will
  support all three multiplayer models (each client authoritative for its own
  area, a player hosting, dedicated servers), so simulation cannot depend on
  any one machine running GI near every player. Within a single player's
  64-128 voxels, GI-carried bands would cost little (a few percent of compute,
  ~15% of face state) and include bounce; the deciding factor is authority,
  not resources.
- **Value.** Melting is driven by direct exposure (the star, a nearby hot
  source). Bounce light is a small, costly, client-specific correction.

The simulation field instead uses direct, deterministic transport inside the
simulation distance, sharing geometry and traversal (voxel world, material
table, NAADF), not the GI solver:

- **Star infrared** = the corner cache's sun visibility x N.L x the star's
  infrared strength: no extra rays. The CPU corner tracer
  (`sample_sky_corner`) makes it server capable.
- **Emitters and radiation sources**: a visibility ray from each face to each
  source in range, 1/d^2, walls blocking, on a slow tick (a few Hz).
- **Exposure** accumulates in the occupied-cell simulation (per block dose);
  a molten block becomes an emitter, which GI already handles as an edit.
- **IR goggles** display the deterministic infrared field.

GI's display path becomes RGB only. The transport code stays band-generic so
bands can return. Done (6cc4407): emitter sampling (`simulation_sampling_
importance` 0, was 1/8) and Russian roulette follow the visible bands.
Remaining, best folded into the near-residency work that redesigns face state:
removing the bands from GPU state, checkpoints, readback and the CPU probe
record (~32 B per face and ~144 B per probe; roughly 50 MB in the canyon, ~2%
of the measured footprint). Scheduling stability must be recalibrated at the
same time: judged on visible bands alone at today's tolerance, cave faces
rested too early (flicker p95 roughly doubled, settle p90 +4 s), because
IR noise had been acting as a stricter visible convergence test.

## Emitters

The surface cache changes which emitters need explicit light sampling:

- **Small bright emitters** (a torch voxel) still need it: an unaided path hits
  a one-voxel emitter 10 voxels away about once per thousand directions, and
  faces next to it are the most visible light in the scene. Faces one hop away
  already gather it from the cache.
- **Large or dense emitters** (lava, molten terrain, glowing walls) are found
  by paths directly. They should not enter the emitter tree: a molten planet
  surface would make it huge and expensive to rebuild exactly when it matters.

Plan: keep the tree for small emitters, exclude dense emissive regions above a
size/density threshold, and shorten emitter reach (32-64 voxels, Minecraft-
like) so the tree only holds emitters near the world-space region. Open
question, to settle by measurement: the cave and response profiles with the
tree disabled at equal GPU time (FLIP and settle time).

## Other levers, in expected order of value

1. Near-only world-space residency (above).
2. Convergence judged against visible luminance (direct + indirect): sunlit
   faces need few samples, compute moves to shade, caves and dusk.
3. Separate emitter reach (short) from bounce/sky reach (~128).
4. Storage precision: half-precision or shared-exponent moments would roughly
   halve GPU face state.
5. A slimmer CPU record: done (CPU statistics optional, off by default;
   0.4-0.9 GB less footprint per profile on the M4).
6. A lower display evidence target (donor lists cost ~1.35 ms/frame of worker
   time) and fewer explicit bounces (the cache ends most paths at the first hit).

## Measurement

The Neo results from `tools/base_spec_suite.sh` (windowed frame pacing,
headless lighting, process memory) set the budget for both the Neo and the
A19 Apple TV. tvOS adds app memory limits, storage limits and Rust/Bevy
platform maturity questions (Rust's tvOS target is tier 3 to our knowledge),
to check separately from GI.
