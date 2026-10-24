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
