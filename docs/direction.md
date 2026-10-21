# Direction: near/far lighting and the simulation split

Decided with the user on 2026-10-21, while moving the base spec from an M4 Mac
mini to the MacBook Neo (A18 Pro: 5-core GPU, 8 GB shared memory, 60 GB/s) and
the A19 Apple TV (same class). This is the plan, not the current
implementation; [architecture](architecture.md) describes what exists.

## Why change

GI today keeps a world-space lighting cache for every exposed face in the
loaded world. Memory and the fixed per-batch GPU cost (ranking ~800k slots,
applying events) scale with world size, not with what is on screen.

Rough sizes (canyon, ~570k faces; struct sizes, not a profile): ~240 B of GPU
state per face (state 144 + metadata 48 + rollback checkpoint 48), a 936 B CPU
`SurfaceProbe` per probe and a 256 B display probe. GI is very roughly
0.4-0.5 GB of the 2.6 GB process footprint measured on the response profile
(M4, dev build); the rest is world data, meshes, textures and the engine.

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

## Simulation: deterministic direct transport, not GI

Game rules from the user: walls block radiation; materials exposed to enough
infrared for long enough melt and become emitters (a planet near its star may
melt its whole surface); simulation only runs a few chunks around players.

GI should not be the source of truth for these:

- **Authority.** A block melting is a world edit the server must decide. GI
  runs on each client's GPU, is randomly sampled, and depends on that client's
  camera and residency; a dedicated server may have no GPU.
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
5. A slimmer CPU statistics mirror (the GPU is authoritative).
6. A lower display evidence target (donor lists cost ~1.35 ms/frame of worker
   time) and fewer explicit bounces (the cache ends most paths at the first hit).

## Measurement

The Neo results from `tools/base_spec_suite.sh` (windowed frame pacing,
headless lighting, process memory) set the budget for both the Neo and the
A19 Apple TV. tvOS adds app memory limits, storage limits and Rust/Bevy
platform maturity questions (Rust's tvOS target is tier 3 to our knowledge),
to check separately from GI.
