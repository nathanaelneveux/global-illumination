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

## Emission-distribution dependency reuse

Follow-up to `0cd067e`: compare immutable region identities and emission material
masks before rebuilding the emitter PDF/tree. A changed region with an emitter
in either scene, or an emitter in any of its six neighboring regions, still
rebuilds conservatively. This includes nonemitting occluders that expose/hide
faces across region/page boundaries. Material-table identity changes still
rebuild. Unrelated geometry reuses the distribution Arc, avoiding the GPU emitter
buffer upload too. All visibility rays use the latest scene. This is a cache
dependency optimization, **not** an emitter-specific lighting-history policy.

The existing isolated `policy_emission_cache_rebuild_cost` benchmark now compares
reuse with a full build on the same scenes: 576 regions, 4,585 emitting faces,
32 alternating distant nonemitter edits. Median preparation: **0.02054 ms** with
reuse versus **1.523 ms** for a full build. Same-scene identity fast path:
0.000041 ms. These are worker-side CPU timings, not whole-GI or frame speedups.
Log: `target/emission-dependencies-benchmark.log`.

Five focused emission tests passed, including IR-only emitters, changed optical
tables, emitter add/remove/unload, unrelated page residency, and adjacent-page
occlusion/unload. Dev build passed; earlier all-target Clippy covered production
changes and four tests (the fifth adjacent-page test was added afterwards).

Integration run `cave-1791644516436002000-68041`: `0cd067e` plus the emission.rs and
history_audit.rs changes in this section's commit. Same settings/machine/command
as above, no concurrent compiler or profiler; three valid nonblank captures,
clean log `target/cave-emission-dependencies.log`. Paths 84.28 M, near age
p95/p99 453/1,392 ms, display p99 7.00 ms, app interval p95/p99 29.72/47.17 ms.
Corner p95 variation 0.1440, all 144 present, minimum 80 fresh paths. This keeps
the publication improvement and transport progress; the small timing differences
against the preceding run are not evidence for an additional gameplay speedup.
Persistent cave noise and the residual frame tail remain open.

### Steady-scene diagnostic control

Run `cave-1791644600829123000-68163` used the same binary as the preceding
integration (subsequently committed as `02babc2`), with only
`VOXEL_GI_VALIDATE_BACKGROUND=0`. The existing isolated-cave fixture is then
selected: 76 loaded pages / 15,232 receivers, instead of normal-world terrain.
This is **not** a matched frame-performance comparison; the fixture also differs
in camera placement, exterior geometry, and host lighting. No gameplay behavior
was disabled or changed to produce a claimed winner.

At 40–60 seconds the transport revision remained 3, history drops were zero,
all 144 corners were present and each receiver accepted at least 36 new paths.
Corner excess-variation median/p95 was **0.00762 / 0.01124**, versus roughly
0.106 / 0.144 in the streaming cave. Worst listed receivers ended with 508–912
history weight, instead of repeatedly returning near 256. Three valid nonblank
captures and a clean log (`target/cave-static-control.log`); final image inspected.

The sampler can become substantially quieter in a steady chamber while still
accepting new paths. Repeated streaming invalidation and scheduling contention
are therefore priority targets, ahead of adding another smoothing layer. This
control does not isolate their individual contributions or prove an identical
normal-world cave would have the same variation without streaming.

### Windowed limitation and final checks

An ordinary `streaming` workload was attempted at clean `02babc2`, with frame
timings enabled, `VOXEL_GI_VALIDATE_HEADLESS=0`, preferred defaults, and no
background-loader override. It produced a black 2560×1440 first capture and the
validator stopped it: `streaming-1791644707468869000-68246`,
`target/streaming-windowed-check.log`. **Discard this run as gameplay evidence.**
No screensaver/security setting was changed and no interactive FPS claim follows.
A valid foreground windowed run is still required to assess the 60-FPS target.

Final `cargo clippy --locked --all-targets -- -D warnings` passed at `02babc2`
plus documentation-only changes (`target/streaming-perf-final-clippy.log`).
Focused tests/native checks above remain the verification for unchanged code;
the full suite and optional toggle-stress workload were not rerun.
