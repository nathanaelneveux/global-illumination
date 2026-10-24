# Far LOD: coarse GI pages

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
