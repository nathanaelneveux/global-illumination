# Surface radiance cache experiment

Branch `gi-surface-cache`, based on `0790c12`. If this approach does not hold up,
the whole branch is abandoned. Systems that conflict with it are removed here
instead of being kept side by side.

## Change

Each resident face already stores its measured incident energy: emitter NEE plus
the continued path, averaged. When a path hits a resident face with at least 16
samples, it now ends at that hit with

```text
emission * MIS + reflectance * (direct at the hit + cached incident energy of the hit face)
```

The cached value already contains that face's NEE and continuation, so nothing is
counted twice. Bounces accumulate through the cache over time, which gives
infinite-bounce transport. Most paths become one cosine ray plus the receiver's
emitter shadow ray and the hit's analytic-light shadow rays. Uncached hits still
continue explicitly, up to three hits. That covers cold start, and leaves the
synchronous and CPU comparison backends unchanged.

- `gpu/face_index.rs`: GPU `(voxel, face) -> slot` table, maintained alongside the
  worker's slot map. It uses open addressing with backward-shift deletion and
  uploads only changed entries.
- Rays that reach unloaded regions are excluded from the sample count, instead of
  being averaged in as black. Kernels report the known count; history advances by it.

## Evidence

Apple M4, headless 2560x1440, background loading, default settings
(including the 2048-evidence display reconstruction unless noted).
`tools/live_site_quality.py` reports self-consistency metrics on the fixed
receivers, not accuracy against a reference:

- neighbour difference: median relative luminance difference between adjacent
  floor faces
- settle: when a face stays within 15% of its own late mean
- flicker: relative standard deviation over the last third of the run

`live-hallway` (182 s, 237 floor faces). Baseline `0790c12` vs `5d5016b`:

| Metric | Baseline | Cache | Baseline, unfiltered | Cache, unfiltered |
| --- | ---: | ---: | ---: | ---: |
| Neighbour difference @15 s | .210 | .141 | .237 | .150 |
| Neighbour difference @180 s | .206 | .118 | .188 | .115 |
| Settle p50 / p90 (s) | 57.5 / 126.1 | 11.5 / 94.5 | 49.2 / 127.2 | 9.8 / 87.1 |
| Late flicker p50 / p95 | .025 / .094 | .007 / .048 | .020 / .102 | .007 / .050 |
| Traced paths | 464M | 565M | | |
| Frame p95 (offscreen) | 24.2 ms | 24.5 ms | | |

The unfiltered pair uses `config/gi-display-unfiltered.json` and repeats the
default-display result. Median floor luminance is 1.9x higher with the cache:
light carried down the corridor by bounces beyond the third was previously
truncated.

`cave` (62 s, normal-world cave with background loading, 36 wall faces):

| Metric | Baseline | Cache |
| --- | ---: | ---: |
| Settle p50 / p90 (s) | 7.0 / 10.8 | 8.2 / 13.2 |
| Late flicker p50 / p95 | .0052 / .0128 | .0029 / .0060 |
| Traced paths | 174M | 211M |
| Frame p95 / p99 (offscreen) | 21.6 / 28.4 ms | 23.5 / 27.9 ms |

The cave images are visually equivalent; the cache is slightly brighter. One run
per arm for the cave, and two pairs for the hallway. Offscreen cadence is not
windowed FPS.

Energy: the native test `cached_furnace_converges_to_infinite_bounce_series`
traces all 1,176 interior faces of a closed emissive furnace. Every band converges
to `E / (1 - r)` within 0.03% using a 256-sample test history.

## Rejected: 256-sample gameplay history

A short history would bound stale feedback without edit journals. In
`live-hallway` it made late flicker p50/p95 .064/.449 and median settle 137 s:
dark corridor faces still have heavy-tailed samples. Gameplay keeps the
65,536 ceiling and the existing geometry/source retention caps.

## Known limits

- With the long ceiling, early under-bounced cache values stay in the running
  mean. In the furnace, the r=0.7 band is at 95.5% after 400 batches. In ordinary
  scenes (reflectance 0.3-0.6) this is a few percent and fades, and edits cap
  nearby histories anyway.
- The hit uses its face-centre cached energy, not a value interpolated across
  the face.
- Path reuse, guiding, the synchronous backend and the CPU reference do not use
  the cache. They are candidates for removal on this branch.
- The display stack (temporal filter, 2048-evidence reconstruction, transition)
  is unchanged. Retuning it to the lower noise is the next presentation step.
