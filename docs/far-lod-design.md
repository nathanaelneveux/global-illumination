# Far LOD: coarse GI pages

> Historical design and qualification record from 2026-10-24 onward. The proposal below is followed by implementation updates; it is not a list of current missing features. Current coarse-page and receiver contracts are in [architecture](architecture.md#host-input-contract).

Proposal, 2026-10-24. Evidence from the `lod` profile (modeled on
`bevy_voxel_world`'s `noise_terrain_lod`: 200 chunks, LOD strides 1-32, a
drifting and turning high camera), run on the BVW performance fork, M4,
paced headless 1440p.

## What the punishing test shows

| Flight phase | p50 / p95 / p99 ms |
| --- | --- |
| GI off | 21.2 / 105 / 654 |
| GI on | 83.4 / 435 / 1304 |

View-tier GPU stages with GI on (ms per frame): discovery 14-21,
reconstruction 8-12, reprojection 4-5, trace 4-6 (default lab: all of
presentation is about 2.3, discovery about 1.2).

- **Ingestion cannot keep up.** GI samples every chunk at full resolution
  through the host generator; in 40 s it ingested 21k pages while most of the
  visible terrain is still absent from GI's scene. BVW never generated
  full-resolution voxels for those chunks; it holds them at their LOD stride.
- **The receiver pool saturates.** Distant faces are about one pixel, so
  discovery fills all ~64k receivers; with a moving camera about 24k of them
  are never sampled before they churn. Discovery recomputes each pixel's
  identity several times per frame, and reconstruction and reprojection cost
  per receiver.
- **Traced discovery alone does not help** (branch
  `experiment/traced-discovery`): with no GI geometry for most distant
  terrain, its rays find nothing and cost 5-8 ms more.
- The dark contour bands on the terrain are the low sun's shadows from
  one-voxel terrace steps, present with and without traced discovery.

NAADF renders full resolution everywhere because it owns a fixed, GPU-built
world. A BVW LOD host has no full-resolution data far away, so GI has to use
what the host stores.

## Proposal

1. **Pages carry a scale.** The BVW connector forwards each chunk's stored
   data at its own stride (1, 2, 4, ... 32). A coarse page is a smaller dense
   grid whose cells are stride-sized voxels. Near chunks stay stride 1.
2. **Regions are self-describing in scale.** A region header stores its
   scale; its bricks, cells and empty-space bounds are in its own units.
   Traversal reads the scale when it enters a region, so rays cross mixed
   resolutions without a separate structure. LOD changes re-ingest the page
   at the new stride (BVW already re-meshes then).
3. **Receivers are keyed by (cell, face, scale).** A coarse cell's face is
   what the host's LOD mesh rasterizes, so raster identity and GI identity
   agree again (no mismatch, no traced discovery for BVW LOD), and coarse
   faces cover several pixels, so far tiles hold fewer distinct receivers.
4. **Discovery computes each pixel's identity once per frame**, then the tile
   passes reuse it.
5. **Sun and indirect at distance come from coarse geometry**, consistent
   with what the player sees (the LOD meshes).

Expected effects: ingestion bounded by host-stored data; distant faces lit
from geometry that exists; fewer, larger far receivers; lower discovery and
per-receiver cost. Each step is measured on `lod` (with the fork) and gated
on the default lab.

## Open questions

- Near/far boundary: should GI also coarsen chunks that BVW keeps at full
  resolution but that are far away (memory/ingest budget), or follow BVW's
  LOD exactly?
- LOD transitions: re-ingest immediately, or keep the finer page until the
  coarse one is ready (no gap in GI geometry)?
- Separately: GI on the BVW fork shows edit-timing regressions in the default
  gate (asteroids-integration item 9); asteroids needs that fixed too.

## Update: traced discovery on the 48-chunk baseline

`VOXEL_GI_LAB_FAR=48` (LOD bands 0/2/4/8), paced, M4 headless 1440p, two
alternating pairs each, branch `experiment/traced-discovery`:

| Variant | p50 ms | Discovery ms | Uncovered receivers |
| --- | --- | --- | --- |
| main (top-level field on, `32697f5`) | 22.7-23.7 | 0.9-1.2 | 3.5k |
| traced, first version | 30.5 | 5.2 | 2.4k |
| + identities cached per tile pixel | 28.7 | 3.5-4.0 | 2.5k |
| + failed lookup on loaded geometry is no receiver | 24.9-25.7 | 3.2-3.3 | 0.55k |

It removes the dark blocky patches LOD meshes left on distant hills. In the
default lab it costs discovery +1.3-1.7 ms (streaming and cave) with static
captures identical, because the fallback also fires on ordinary grazing
and boundary pixels. Not adopted yet; next try limits it to distant pixels.

The default-gate run of that branch crashed once in a GI worker: wgpu
"Mismatched pop_error_scope call: error scopes must be popped in reverse
order" (tunnel profile). Unrelated to the shader change; GI pushes
validation scopes when it creates pipelines on worker threads
(`gpu.rs` quality variants). Fixed 2026-10-24: wgpu without its `std`
feature shares one scope stack across threads; the workspace enables it
(`error_scopes_are_thread_local`).

Raising the top-level field's size cap (it was off above 262,144 region
cells) took the baseline from 24.5 to 23.2 ms p50 on its own (GI off:
20.5 ms).

## Update: strided pages

Implemented step 1 of the proposal in a simpler form: pages carry a stride
(`PageSample`), and GI expands them on the worker to its dense page rather
than making regions self-describing in scale. Coarse cells line up with GI's
4³ bricks from stride 4, so their regions store one material per brick (128
bytes) and the GPU packs them without payload; traversal is unchanged.
Receivers stay unit faces (step 3 not done), and discovery still recomputes
identities (step 4 not done). The connector reads BVW's stored LOD samples
directly. Retention binning for each scene build moved to the scene worker:
cheap coarse pages made full 768-page builds arrive several times a second,
and binning their ~5k change bounds took 12-15 ms of main thread each.

Receipts (M4, paced headless 1440p; baseline 29bd138, full-resolution
synthesis through the generator; two runs each):

| Far 48 (bands 1/2/4/8) | Baseline | Strided |
| --- | --- | --- |
| Frame p50 | 23.6, 24.3 ms | 22.6, 22.0 ms |
| Frame p95 | 25.7, 26.6 ms | 26.1, 26.0 ms |
| Frame p99 | 26.8, 27.5 ms | 28.1, 28.1 ms |
| GI main p95 | 0.23 ms | 0.22, 0.21 ms |
| View discovery / presentation | 2.9 / 5.1 ms | 1.2 / 3.5 ms |
| Uncovered faces | 651, 673 | 680, 732 |
| Peak memory | 1491, 1518 MB | 1450, 1388 MB |

Discovery is cheaper because coarse LOD meshes now match GI's blocks. The
punishing `lod` flight (strides 1-32, one run each, noisy): uncovered 35.0k
-> 27.2k, discovery 20.3 -> 9.1 ms, presentation 31.4 -> 18.7 ms, p95/p99
231/861 -> 186/746 ms, peak 1891 -> 1657 MB; p50 84.8 -> 98.3 ms is
unattributed in one run of a host-bound scene. Default gates and house are
within their spreads. A first measurement round was void: an empty-chunk
shortcut wrote material 0 (a solid) for air (`PageSample::air` now exists
so hosts need not know GI's air value).

## Update: coarse receivers

Step 3: a coarse block's face is one view receiver. Regions carry their
page's stride (header w bits 28-30); discovery keys a face by the block's
minimum voxel with its surface at the block face centre, receiver identity
compares surfaces (unit and block faces at one voxel differ), witnesses and
corner sun span the block, and the material's far lookups and corner
reconstruction step by blocks. Spatial reach (view pooling, pool coverage,
the material's hole fill) is sixteen blocks rather than sixteen voxels: at
a fixed 16 voxels new stride-8+ faces found no donors and showed black for
a frame (walk flashes at the horizon in a first version). Block faces stay
view-only; the world cache and edit priors keep unit faces.

Receipts (M4, paced headless 1440p; baseline 277dcc1):

| Far 48 (bands 1/2/4/8) | Baseline (2 runs) | Coarse receivers |
| --- | --- | --- |
| Active far receivers | 13.2k, 13.1k | 10.7k, 10.2k, 10.0k |
| Mean receiver history | 207, 222 frames | 245, 237, 233 |
| Frame p50 | 22.3, 22.1 ms | 21.5, 21.9, 21.6 ms |
| Frame p95 / p99 | 25.6 / 27.7, 25.5 / 27.6 | 25.8 / 27.8, 26.5 / 28.4, 26.3 / 28.7 |
| Presentation | 3.55, 3.49 ms | 3.32, 3.32, 3.16 ms |
| Discovery | 1.14, 1.19 ms | 1.39, 1.42, 1.27 ms |

The third coarse run is the final code (the first two predate the block
reach). The `lod` flight (two runs each) is still pool-bound by near unit
faces: active receivers and exhaustion are unchanged, mean history 10.5 ->
13.7 frames, p50 97.8 -> 86.2 ms, p95 209 -> 232 ms, p99 745 -> 763 ms,
noisy; its captures look the same. Default gates: static, tunnel and
response within spread; walk flashes over five walks each, baseline 1132-5471
(median 1459) and candidate 1024-4743 (median 2909), are indistinguishable
(rank-sum 12 of 12.5), and the baseline shows the same late turn tail in one
of two runs. Streaming discovery costs about 0.15 ms more (block lookups;
paired runs 1.54 vs 1.39 ms) at equal presentation and frame times. House
settling matches.
