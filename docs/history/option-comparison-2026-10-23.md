# Receiver ownership comparison, 2026-10-23

Question: which of the playable receiver strategies to keep before cleanup.
The user picked by feel in gameplay: all three new options are much better than
the all-near baseline, +cache edges out the others in settling, and hybrid goes
black for a moment when a page first enters the near radius.

Build `3c88915` (`codex/edit-face-bootstrap`); windowed uncapped runs add only
the validation present-mode switch (`VOXEL_GI_VALIDATE_NO_VSYNC`). M4 Mac mini,
16 GB, macOS, Metal, rustc 1.95.0, dev profile (opt-level 1, deps 3),
2560×1440, background terrain, preferred corner look. Configs:

| Name | Settings |
| --- | --- |
| all-near | `config/gi-all-near.json` (receiver radius 0, the old baseline) |
| hybrid | defaults (near world cache 64, far view tier) |
| cacheless | `config/gi-view-cacheless.json` |
| +cache 32 | `config/gi-view-cached.json` |
| +cache 48 / 64 | as +cache 32 with `cache_radius` 48 / 64, share and exchange unchanged |

Scripts, logs and the run manifest are in `target/bench-1023/` (`bench.sh`,
`pacing.py`, `gpu.py`, `quality.py`, `manifest.txt`).

## Windowed frame pacing

Vsync (Bevy's default Fifo) runs, two interleaved rounds each: every option
except all-near hold 60 Hz (0.03-0.3% of frames over 20 ms); all-near misses
0.6-0.9% with 42-56 ms spikes. That says nothing about headroom, so the matrix
was repeated uncapped (`AutoNoVsync`, latency 3), two interleaved rounds.
Intervals after 5 s, capture frames excluded; GI = main-thread GI system bodies.

| Profile / config | p50 ms | p95 ms | p99 ms | GI p95 / max ms | Peak MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| gameplay all-near | 6.4-6.5 | 9.2-16.7 | 17.7-17.9 | 3.2-3.8 / 13-16 | 2970-3300 |
| gameplay hybrid | 8.2-8.3 | 12.5-12.6 | 16.7-17.9 | 0.55 / 8.9-11.3 | 1250-1270 |
| gameplay cacheless | 8.4 | 10.9-15.0 | 16.8-18.1 | 0.01 / 0.2 | 890-930 |
| gameplay +cache 32 | 8.1-8.2 | 11.0-11.1 | 15.3-16.7 | 0.33 / 2.1-2.9 | 1000-1010 |
| gameplay +cache 48 | 8.2 | 11.2 | 15.2-15.8 | 0.45 / 3.2-3.3 | 1040 |
| gameplay +cache 64 | 8.1 | 11.2-11.3 | 16.8-16.9 | 0.50 / 3.8-4.1 | 1050 |
| streaming all-near | 7.1 | 10.7 | 18.2-18.3 | 5.3 / 16-18 | 3320-3380 |
| streaming hybrid | 10.7-10.8 | 14.5 | 18.0-18.5 | 0.6 / 4.7-5.4 | 1150-1170 |
| streaming cacheless | 9.7 | 11.5 | 17.0-18.4 | 0.01 / 0.3 | 880-900 |
| streaming +cache 32 | 9.5 | 12.2-12.4 | 15.5-17.3 | 0.34 / 1.3-1.5 | 950-960 |
| streaming +cache 48 | 9.5 | 12.5-16.5 | 16.1-19.1 | 0.43 / 1.3 | 975-990 |
| streaming +cache 64 | 9.5 | 12.6-12.7 | 15.6-15.9 | 0.51 / 2.0-2.6 | 1010 |

All-near has the lowest median because it has no view tier (no prepass,
discovery or view presentation), but the worst tails, main-thread cost and
memory. Every view configuration pays a fixed ~1.5-2 ms median for the view
tier. Hybrid is slowest while streaming. The +cache radii are indistinguishable
in frame time on the M4; 64 adds ~0.17 ms main-thread GI p95 and ~50 MB.

View-tier GPU receipts (median, windowed uncapped round 1, ms per frame):

| Config | Trace | Presentation | Discovery | Pool prepare + scan |
| --- | ---: | ---: | ---: | ---: |
| hybrid gameplay / streaming | 0.29 / 0.69 | 0.60 / 1.64 | 0.33 / 0.68 | 0.33 / 0.58 |
| cacheless | 1.54 / 2.11 | 0.74 / 1.73 | 0.38 / 0.63 | 0.32 / 0.68 |
| +cache 32 | 1.19 / 1.46 | 0.79 / 1.84 | 0.38 / 0.66 | 0.31 / 0.67 |
| +cache 48 | 1.19 / 1.41 | 0.76 / 1.80 | 0.36 / 0.67 | 0.31 / 0.65 |
| +cache 64 | 1.20 / 1.42 | 0.76 / 1.76 | 0.37 / 0.65 | 0.32 / 0.68 |

Trace is budgeted (it scales with frame time). Presentation, discovery and pool
are not: 1.4 ms stationary, 3.0-3.2 ms moving. On a GPU with half the M4's
throughput that is the cost to watch. Headless cave history per visible face
after one minute: hybrid 1.7k, cacheless 7.9k, +cache 32/48/64 10.2k/11.3k/15.6k.

## Visible behavior (headless captures)

Early capture against the same run's later capture: mean FLIP of remaining
visible change, not accuracy. One run per config.

| Workload | all-near | hybrid | cacheless | +cache 32 | +cache 48 | +cache 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| house turned 400 ms | .0077 | .0087 | .0469 | .0087 | .0111 | .0103 |
| house return 400 ms | .0049 | .0075 | .0736 | .0034 | .0037 | .0057 |
| emitter added 200 ms | .698 | .724 | .135 | .306 | .315 | .155 |
| emitter removed 300 ms | .721 | .723 | .227 | .403 | .118 | .197 |
| emitter removed 600 ms | .491 | .481 | .024 | .023 | .014 | .016 |
| cave 15 s / 30 s | .030 / .018 | .015 / .010 | .018 / .013 | .010 / .006 | .010 / .006 | .008 / .005 |
| tunnel depth 1 / 2 / 4 | .39 / .29 / .11 | .39 / .41 / .45 | .07 / .06 / .04 | .15 / .10 / .02 | .15 / .08 / .03 | .37 / .09 / .02 |

No configuration had black-then-lit tunnel pixels at 400 ms (hybrid had 78 at
depth 8). All-near, hybrid and +cache 64 at depth 1 show a bright-then-dimmer
transient (0.2-3.7M pixels): new tunnel faces inside the world cache start too
bright and darken. View-owned edits do not. Cacheless loses everything turned
away from (house return .074). +cache 64 settles the cave best and responds
to emitter additions fastest.

## Hybrid's near-entry black

Code inspection, not yet a capture: `gpu_probe` publishes every exposed face
with weight `max(confidence, 1/32)`, so a newly admitted near probe passes the
material's `near_valid` check before it has any indirect data, and its sun
corners are still unknown (no sun). The view result is dropped at admission.
+cache's two-way exchange imports the view history instead.

## Decision input

Hybrid has no measured advantage over +cache: slower when streaming, more
main-thread cost and memory, slower edit response, much less history per
visible face. There is nothing to port back. +cache 32-64 cost the same frame
time on the M4; radius is a quality/memory knob (64 for large caves).
Windowed Neo runs remain necessary before choosing the shipped default radius.

## Re-check, 2026-10-24

After traced discovery, amortized corners, MSE pooling and GPU sunlight
detection (`5e6afb4`, headless, one run each; see
[the review](../review-2026-10-24.md)): 64 still matches 32 on house reveal
(.0087 vs .0089 turned) and 96 on cave settling (.0076 vs .0073 at 15 s);
cacheless loses revealed history (.058 turned, .069 return). Kept at 64.
