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

## Display evidence target

The user-calibrated cave gate (`tools/quality_profiles/cave-user-2048.json`,
FLIP 1.7 at 67 px/degree) is mean FLIP <= .0274 and worst 32px tile <= .0730.
It was set where the old estimator's raw output looked indistinguishable from
the reference (2,048 paths). The display evidence target followed that count.

Frozen-cache sweep, run `cave-1792278749460839000-9510` on `53539d2`.
`VOXEL_GI_VALIDATE_CACHE_SWEEP=1`, headless cave with background loading,
15,163 chamber faces, 131,072-path reference replicas. The replicas agree to
p99 0.2% and max 0.5% RGB, with FLIP .0024 between them. Worst of four windows:

| Paths | Cache raw FLIP / worst tile | Old raw FLIP / worst tile |
| ---: | --- | --- |
| 256 | .0327 / .1037 | |
| 512 | .0230 / .0743 (tile fails by 2%) | |
| 1024 | .0163 / .0535 **pass** | .0367 / .1045 |
| 2048 | .0117 / .0342 **pass** | .0274 / .0730 (accepted) |

The cache output reaches the gate at half the paths. At equal paths, evidence
reconstruction never improved FLIP fidelity: blurring moves shading detail, as
it did for the old estimator. Its job is early smoothness. Smooth-wall noise uses
the regions, metric and limit of `cave-spatial-noise-2048.json` (the preferred
2048 look), measured against this run's own reference
(`tools/cache_wall_noise.py`). Normalized worst over 4..2048 paths, where <= 1
is the preferred smoothness: raw 9.49 (within the envelope from 256 paths),
target 256 1.36, 512 1.04, **1024 0.93**.

1024 meets both calibrations: the raw count that passes FLIP, and
smoothness inside the envelope at every prefix. It is the branch default.
Its footprint is about 1/sqrt(2) of 2048's, so less detail is smeared.
`config/gi-display-evidence-2048.json` restores the old target.
`tools/cache_flip_gate.py` reproduces the FLIP table.

## Rejected: 256-sample gameplay history

A short history would bound stale feedback without edit journals. In
`live-hallway` it made late flicker p50/p95 .064/.449 and median settle 137 s:
dark corridor faces still have heavy-tailed samples. Gameplay keeps the
65,536 ceiling and the existing geometry/source retention caps.

## History ceiling and recency

Question: is there a cap between 256 and 65,536 that works better? Two knobs
(`history_limit`, `history_recency`) were tested on `d08d72a`.

Native furnace (`cached_furnace_lag_by_history_policy`, r=0.7 band, fraction of
the fixed point after 25 / 400 batches of 16 paths):

| Policy | 25 | 400 |
| --- | ---: | ---: |
| 65,536, recency 1 | .900 | .956 |
| 65,536, recency 1.5 | .937 | .982 |
| 65,536, recency 2 | .961 | .993 |
| 8,192, recency 1 | .900 | .956 |
| 2,048, recency 1 | .900 | .967 |

`live-hallway`, one run each, same setup as above:

| Policy | Settle p50 / p90 (s) | Late flicker p50 / p95 |
| --- | --- | --- |
| 65,536, recency 1 (control) | 9.0 / 91.2 | .0068 / .0457 |
| 65,536, recency 2 | 20.1 / 125.5 | .0126 / .0863 |
| 2,048 | 10.4 / 109.3 | .0157 / .0815 |
| 8,192 | 11.0 / 85.6 | .0069 / .0486 |

A cap only matters once faces reach it. At 2,048 it binds within a few minutes
and roughly doubles flicker. At 8,192 it rarely binds in a session, so it matches
the control and changes nothing. Recency removes the furnace lag, but in gameplay
retention caps keep counts low, and there k=2 discards most of a face's history.
Flicker doubles. **Decision: keep 65,536 and recency 1.** The knobs stay for
scene-specific tuning; the validation report now records them.

## Edit response and the rejected change detector

`response` profile (emitter add at ~20 s, removal at ~25 s, 7 tracked faces,
one run each, background loading). Settle means staying within 10% of the
change, as measured by `tools/response_timing.py`:

| Build | Add p50 / p90 | Remove p50 / p90 |
| --- | --- | --- |
| Baseline `0790c12` | 1.58 / 3.68 s | 1.15 / 2.96 s |
| Cache `d53331f` | 1.51 / 1.78 s | 0.93 / 0.93 s |
| Cache + change detector (z = 4) | 1.77 / 2.28 s | 1.04 / 1.57 s |

The detector (`9102402`, reverted in `9fb1b30`) cut prior weight when a batch
disagreed with the retained mean by more than z standard errors. It did not
improve edits, because retention caps already handle them. In `live-hallway` it
raised flicker p50/p95 to .064/.348, the same as a short history: the first
rare bright path after a quiet stretch looks like a change. Distance-curve
retention stays. A loaded chunk is not a free pass either: rays that left the
loaded area counted as sky, and the new terrain contradicts them. The existing
curve already leaves faces more than ~190 blocks from a change untouched.

## Known limits

- With the long ceiling, early under-bounced cache values stay in the running
  mean. In the furnace, the r=0.7 band is at 95.5% after 400 batches. In ordinary
  scenes (reflectance 0.3-0.6) this is a few percent and fades, and edits cap
  nearby histories anyway.
- The hit uses its face-centre cached energy, not a value interpolated across
  the face.
- Path reuse, guiding, the synchronous backend and the CPU reference do not use
  the cache. They are candidates for removal on this branch.
- The temporal filter and transition are unchanged. The evidence target is now
  1024 (above).
