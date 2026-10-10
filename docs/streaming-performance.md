# Cave stability and streaming cost — October 10, 2026

The accepted scene-history work through `b051274` is merged into the regular
`voxel-global-illumination` branch. Keep its history and presentation policy fixed
while removing publication overhead. Neither the cave's residual noise nor the
always-above-60-FPS target is solved by this checkpoint.

## Attribution and display publication

`b8aa8d5` adds opt-in per-frame GI stage observations and event-complete cave
corner traces. `9f3d4c6` corrects a receiver hidden by the normal-world staircase;
all 36 receiver faces / 144 reconstructed corners now exist. Observe them with:

```sh
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_PROFILE=cave VOXEL_GI_VALIDATE_BACKGROUND=1 \
VOXEL_GI_VALIDATE_TEMPORAL=1 VOXEL_GI_VALIDATE_FRAME_TIMINGS=1 \
VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD" VOXEL_GI_VALIDATE_EXIT=1 \
RUST_LOG=warn cargo run --locked
```

`measurements.frame_timeline` records the ending frame interval and the **previous**
Update's seven GI system-body times. The mapping is in `gi_main_stage_names`.
Pipeline overlap means interval minus GI time is **not** a BVW-only measurement.

An exploratory 12-second macOS `sample` trace found mirror-lock waits during GI
display publication and substantial per-word probe encoding. The render thread
called `Queue::write_buffer` while owning that same lock. The profiler attach
itself caused a ~1.5-second pause: run `cave-1791643762924840000-67414` is only
diagnostic evidence, not a timing baseline or complete receiver stability pass.

The display change snapshots both sparse buffer transactions atomically into
reusable render-local scratch, then releases the mirror lock before driver calls.
Main-world changes arriving meanwhile form the next transaction. Destination
validation, retries, generation ownership, and atomic hash/probe publication are
preserved. Encoding uses one packed little-endian copy rather than 52 tiny copies;
no unsafe layout assumption or new dependency is introduced.

### Preliminary paired run

Apple M4 / 10-core GPU / Metal, macOS 26.6.2, Rust 1.95.0, Bevy 0.19.1, BVW 0.17,
development profile/shared Cargo intermediates. Offscreen 2560×1440, 60-Hz
ScheduleRunner, textures on, AO off, corner sunlight/reconstructed indirect.
4 paths/batch, 3 bounces, reach 768, 4-ms trace target, 8,192 receivers,
75-ms display response / 50-ms upload, adaptive defaults. No simultaneous build,
profiler, or other benchmark. Both runs completed with three valid nonblank
captures, valid assets/target, clean render logs; final images inspected.

- Baseline: clean `9f3d4c6`, `cave-1791644108566432000-67653`.
- Candidate: `9f3d4c6` plus **only the render.rs change in this document's commit**,
  `cave-1791644257373830000-67861`. Later emission-cache changes were not in this
  binary. Build succeeded before launch; source edits during the run were not built.
- Logs: `target/cave-attribution-clean-baseline.log`,
  `target/cave-display-snapshot-candidate.log`; runs under `target/gi-validation/`.

Settling interval 15–62 seconds, capture/readback cooldown excluded:

| Metric | Baseline | Display candidate |
| --- | ---: | ---: |
| GI display publication p95 / p99 | 15.79 / 22.13 ms | 6.42 / 6.70 ms |
| Sum of GI system-body times p95 / p99 | 21.42 / 28.48 ms | 11.05 / 12.93 ms |
| Offscreen app interval p50 / p95 / p99 | 19.50 / 38.48 / 50.45 ms | 20.84 / 29.40 / 49.65 ms |
| Near measurement age p95 / p99 | 657 / 1,763 ms | 584 / 1,402 ms |
| Total fresh paths | 79.86 M | 83.40 M |
| Loaded pages at last observation | 1,131 | 1,131 |

This supports retaining the publication optimization, **not** a 60-FPS claim.
Overall p99 barely moved and median app interval worsened in this one pair. Do
not attribute the remaining tail to BVW without further stage/driver evidence.
No budget reduction, debounce, longer history, or lighting freeze was used.

Focused verification: 15 render tests and the two explicit native buffer tests
passed (generation/material preparation and reported out-of-bounds retry).
The new test also proves that uploads release the lock, use immutable paired
bytes, and preserve a newer transaction across a failed destination retry.
Dev build passed. Strict all-target Clippy passed on the subsequent tree including
the emission-cache candidate; it does not establish live timing for that candidate.

## Cave noise remains an open issue

Event-complete 40–60-second corner traces: all 144 corners present, minimum 84
new paths at every receiver in both runs. Compressed-RGB excess-variation p95
was 0.1376 baseline / 0.1416 candidate (not an improvement claim). Many receivers
lost statistical weight 14–16 times in those 20 seconds, ending near 316–324
samples. The background loader continues to advance geometry revisions, and the
general distant-history cap is 256. Coverage/bootstrap being 100% therefore does
not mean the cave has accumulated a stable long history.

This suggests repeated conservative invalidation contributes to the noise; it
does not prove it is the only cause. Next isolate steady-scene sampling from
streaming, then evaluate dependency-aware retention without weakening response
to real distant source/occluder changes. Simply increasing the retained cap has
already shown a slow-extinction tradeoff in the scene-history audit. The house's
0.1 variation gate is not a calibrated universal cave quality threshold.
