# NAADF comparison for distant geometry

Why NAADF renders an 8256²-voxel world "without using LOD" while our GI
falls over at 48 streamed chunks. Source: `cg-tuwien/NAADF` at `3ea3bb6`
(MIT; read-only copy in `~/Developer/reference/NAADF`), compared with this
workspace at `f28df62`. Nothing here is copied code.

## Where our far-distance cost is

Lab streaming profile, M4, headless 1440p, `VOXEL_GI_LAB_FAR=48` (48k pages,
distance LOD) against the `VOXEL_GI_LAB_GI=0` control, one run each:

| | Frame p50/p95/p99 ms | Peak memory |
| --- | --- | --- |
| GI off, far 48 | 21.6 / 23.9 / 25.9 | 725 MB |
| GI on, far 48 | 31.3 / 51.7 / 59.7 | 5,816 MB |

BVW streaming that far costs nothing measurable. All of it is GI's. The
packed GPU geometry is only 31 MB, so the 5 GB is CPU-side GI state; the
largest suspect is the per-voxel `materials` and `voxel_aadf` arrays every
mixed `NaadfRegion` keeps (about 32 KB per mixed 16³ region). Not yet split
by component.

## What NAADF does differently

| | NAADF | Ours |
| --- | --- | --- |
| World residency | Whole fixed-size world on the GPU; generated there in segments | Host voxels dense-sampled per page on the CPU through `VoxelSource` |
| Construction | GPU: 4³ blocks hashed and deduplicated, voxel/block bounds in group-shared memory (`chunkCalc.fx`); chunk bounds start at zero and are refined over frames under a dispatch budget (`boundsCalc.fx`, `WorldBoundHandler`) | CPU workers build each region's full NAADF; the complete top level is built before a scene publishes; packed and uploaded afterwards |
| Storage | Bounds live inside the cell word at every level; voxels 2 bytes, only for unique non-uniform blocks | CPU keeps per-voxel material and 6-byte AADF arrays per mixed region; GPU packing is compact |
| Top level | Dense chunk texture, one load per step | Hashed region directory, up to 32 probes per region change |
| Primary visibility | Ray traced into the structure (`renderFirstHit.fx`): every visible pixel is in it | Host meshes rasterized into a depth/normal prepass; faces looked up afterwards |
| Ray budget | Steps, not distance: primary 120, GI 100, sun 120 / 80, visibility 60; exhausted is a miss | 2,048 steps, 768-voxel reach, exhausted is "unknown" |
| Per-pixel work | ≤1 GI path per pixel per frame, 0.25 once TAA history exists (`rayQueueCalc.fx`); one soft-sun ray | Bounded receivers (65k) and budgets; the material's fallback can trace 4 corner sun rays per fragment |
| LOD | None | BVW LOD meshes; LOD hosts have no full-resolution far voxels |

What NAADF does not have: streaming, a host engine's meshes and voxel
store, or a persistent world-space cache. Its world is fixed and complete.

## What we are missing

1. **Discovery from the structure, not from host meshes.** NAADF's primary
   hits come from the same voxels the GI traces. Ours come from whatever the
   host rasterized, so LOD meshes, foreign meshes and materials without a
   prepass produce faces GI cannot light, at high cost. Tracing discovery
   rays into GI's geometry (at the view grid's resolution) would make face
   identity independent of host meshes and drop the prepass requirement.
2. **GPU construction from compact uploads.** NAADF never builds on the CPU.
   For a BVW host the voxels are on the CPU, so an upload is unavoidable, but
   it can be the host's compact chunk arrays, with bounds built on the GPU
   and no CPU per-voxel AADF copies.
3. **A dense, camera-relative top level.** A toroidal chunk grid around the
   camera gives NAADF's O(1) lookup for a streaming world.
4. **Bounded steps.** Their budgets make far rays cheap and bounded; ours
   allow 2,048 steps and treat exhaustion as unknown.

None of this is LOD. Full resolution everywhere is what NAADF shows is
affordable, and our 31 MB of packed geometry for 48k pages agrees. The
exception is a BVW LOD host (asteroids-redux): it never generated
full-resolution far voxels, so GI must either run the host generator at full
resolution or accept coarse far pages.

## Experiments (2026-10-24)

Each measured on the far lab (`VOXEL_GI_LAB_FAR=48`, LOD bands 0/2/4/8,
paced, M4 headless 1440p) against main run next to it, then gated on the
default lab when promising. Per-stage view times come from `far_stages_ms`.

| Idea (source) | Result | Verdict |
| --- | --- | --- |
| Top-level empty-space field kept on for large worlds (NAADF chunk bounds) | p50 24.5 -> 23.2 ms, GI main p95 1.5 -> 0.3 ms | adopted `32697f5` |
| Traced discovery past 0.1 world units per pixel (NAADF primary rays) | uncovered 3.5k -> 0.48k, distant LOD patches gone, about +0.9 ms far, none in default lab | adopted `2c19b71` |
| ...with a per-tile workgroup identity cache | discovery +1.3 ms even in the cave (occupancy) | rejected |
| ...with per-invocation register identity cache | far discovery 5.3 -> 3.7 ms but no frame change; default streaming discovery +0.3 ms | rejected |
| Corner sun refreshed every 4th frame for retained faces (NAADF sun accumulates) | reprojection far 1.20 -> 0.82 ms, +33% completed paths far | adopted `f8ed4c7` |
| Per-pixel sun past 0.5 or 0.25 world units per pixel (user idea) | p50 +0.6 / +1.7 ms: faces a few pixels wide already borrow corners from owned neighbours cheaply | rejected |
| Faces with >= 256 samples traced every 4th frame (NAADF 0.25 spp with history) | no settling gain; slower edit response (remove-300ms FLIP 0.096 -> 0.121) | rejected |
| Path and shadow rays capped at 128 steps (NAADF budgets) | known/completed unchanged (99.4%); rays rarely need more here | not needed now |

Not yet tried: a camera-relative dense top level (the bounding-box field
still switches off past 255 regions per axis), GPU construction, capping
discovery rounds per tile per frame (discovery is now the largest far
stage, 4-5 ms), and the reconstruction cost at receiver saturation (0.5 ms
at 48 chunks, 8-12 ms in the 200-chunk lod test).
