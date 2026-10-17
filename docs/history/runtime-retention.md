# Runtime retention experiment

Historical calibrated-curve measurements follow. Current builds use the
[whole-range runtime policy](whole-range-history.md#runtime-integration).
The `calibrated` launch spelling now aliases that policy; use `5381c17` to
reproduce the previous calibrated implementation, not current default flags.

Baseline: `986683a`. Branch: `codex/gi-runtime-retention`. **Keep the candidate
as this branch's default:** the final paired windowed qualification retains a
clear lighting-stability win with comparable frame pacing and sampling freshness.
Use `VOXEL_GI_RETENTION=baseline` for the existing 16/256, eight-global-event
control, or `calibrated` to select the new policy explicitly. No BVW, sampler,
presentation or work-budget changes. This decision is not a locked-60-FPS or
complete cave-convergence claim.

Follow-up: [history-cut attribution](retention-attribution.md) measures the
remaining reductions and frozen-scene target motion without changing thresholds.
Its [distance-band follow-up](distance-band-retention.md) is now the default on
`voxel-global-illumination`, replacing the original overflow floors described
below. Use `VOXEL_GI_RETENTION_SUMMARY=floors` to reproduce this study's summary.

The native curve uses the fixed `cave-user-2048` appearance goal, Q=2048,
baseline work anchor 2048, fresh response work Q/6 and stale-error allowance .03.
Caps round upward to integer weights (16/256/316/1986 at distances
48/128/192/256). Beyond 48/sqrt(.03), the heuristic retains history. This is
not a hard light range or a physical upper bound on remote changes.

Geometry bounds are binned on the scene worker. Per-receiver-page journals hold
eight spatial events; older local events fold into independently timestamped
16/256 conservative floors. Unrelated distant streaming never evicts local
evidence. Retired receiver pages release their metadata. Queries consume all
intervening events, including late sunlight evidence. At `9293cbb`, selection
only checks the newest page stamp; immutable copy-on-write page snapshots travel
to the tracing worker, which evaluates exact per-receiver caps. Publication
clones at most eight small event headers when a worker still holds a page, not
its bounds arrays. Conservative page-edge wakeups may add an audit, never a
stronger cap. There is no main-thread world/probe scan for the change signal.

Existing CPU/GPU shared-corner results contribute sunlight-change evidence.
Known equal values (including black) contribute none; either transition direction
counts. Unknown/new samples are conservative evidence, not black or unchanged.
Points coalesce into 16-voxel bucket bounds and bin on the tracing worker. No new
rays are added. Partial cache coverage is not claimed complete: as-yet unqueried
corners can deliver evidence later. Until then geometric evidence and normal
freshness/audit sampling remain active. This latency must be qualified, especially
for distant shadows; the visible directional cache is not a detector for every
point-light, diffuse-sky or nonvisible-band dependency.

Spatial evidence has its own revision. A selected job carries that stamp through
publication, so later evidence still wakes it. Stale scene results cannot publish
sky display values, but their possible changes remain conservative retention
evidence. All faces consume a cap together, including unselected faces; means,
moments, sequences and display histories are preserved. Source/model replacement
continues using its existing policy. No debounce or radiance clamping is added.

## Qualification — October 13, 2026

Implementation `00c329e`; exact-cap scan shortcuts `b448140`; worker-side query
ownership `9293cbb`. These are local, buildable checkpoints. The control at
`00c329e` keeps the pre-experiment policy, with the new (unused) per-probe stamp
present in both arms. The sampling estimator, work budgets, presentation and
BVW are identical across policies.

Apple M4 integrated GPU, 16 GiB, Metal, macOS 26.6.2; Rust 1.95,
Bevy 0.19.1 / registry BVW 0.17. Development profile: application opt-level 1,
dependencies 3. 2560×1440, corner sun/reconstructed indirect, textures on, AO off.
16 paths/face, 4-ms adaptive trace target, 8,192 receiver cap, 768 reach/3 bounces,
75-ms display response, 50-ms upload, 2-ms scene publication. All runs use the
normal terrain plus nearby cave structure, moving nonrendering background loader
at 4.32 voxels/s, 182-second workload and event-complete 36-face/144-corner
observations. No reference-only residency loader or full-cave bake. No concurrent
compiler, analysis workload or second native process during timed captures.

### Accuracy and stability (headless; not FPS)

Both at native `00c329e` (only the later Python analyzer was untracked):

- Control: `target/gi-validation/cave-1791925532295999000-61021`.
- Candidate: `target/gi-validation/cave-1791925755993216000-61155`.

Each run computes its **own** exact final-scene reference after timing/captures:
two disjoint 65,536-path replicas, including connected corner neighborhoods.
Reference disagreement is .604% corner p99 (1.30% face-center p99). The two
runs' mean corner references differ by only .072% p99, despite modest residency
differences (maximum 1,544 versus 1,594 pages).

| Headless measure | Control | Candidate |
| --- | ---: | ---: |
| Corner p99 error, 60 s | 10.00% | 3.38% |
| Corner p99 error, 120 s | 6.44% | 3.52% |
| Corner p99 error, 180 s | 8.23% | 4.18% |
| 60–120 s p95 excess corner variation | .23468 | .06699 |
| 120–180 s p95 excess corner variation | .19924 | .11649 |
| 120–180 s worst checkpoint corner p99 error | 11.53% | 8.82% |
| Total traced paths | 400.47 million | 431.51 million |
| Median retained weight at 180 s | 272 | 888 |
| Median path sequence at 180 s | 4,216 | 4,608 |

Earlier residency differs from the final-scene oracle; error is not time-varying
ground truth. Excess variation removes monotone drift in compressed RGB and
measures world-space reconstructed corner oscillation, not FLIP or screenshot
flicker. All sites are present, and fresh sampling continues. These sparse error
improvements do **not** establish a whole-image 2,048-path quality pass: the
accepted fixed corner p99 target is 3.266%, and the late candidate still exceeds
it. No IR/ionizing accuracy qualification is inferred from RGB.

### Windowed control and first candidate

Native `00c329e`, five focused/visible/nonblank captures per run, Fifo:

- Control: `target/gi-validation/cave-1791926119753586000-61286`.
- Candidate: `target/gi-validation/cave-1791926337934717000-61353`.

| Windowed measure | Control | Candidate |
| --- | ---: | ---: |
| Settling frame p50 / p95 / p99, ms | 16.75 / 20.54 / 22.88 | 16.79 / 20.89 / 23.64 |
| Settling maximum, ms | 58.56 | 48.03 |
| Settling frames >33.3 ms / >50 ms | 3 / 1 | 4 / 0 |
| 30–60 s p95 excess corner variation | .24647 | .10628 |
| 60–120 s p95 excess corner variation | .41229 | .10417 |
| 120–180 s p95 excess corner variation | .29128 | .17202 |
| Total traced paths | 608.95 million | 618.34 million |
| Near / far measurement-age p99, s | 1.258 / 34.574 | 1.138 / 30.174 |

The candidate improves real windowed stability, not by freezing measurements.
It is not free: selection p99 rises from 2.13 to 2.59 ms and frame p99 is
.76 ms worse in this pair. Neither arm is an always-60-FPS pass. Startup has
roughly 614–616-ms worst intervals in both; capture/cooldown frames are excluded
from the reported settling distribution. No stage attribution was enabled.

The exact-cap scan cleanup at `b448140` was separately checked in
`target/gi-validation/cave-1791926736752451000-61510`: selection p99 2.36 ms,
frame p99 23.99 ms, 629.75 million paths. Corner variation is .08470/.09377/.21340
over the same three intervals. This repeats the lighting win but does not prove
a frame-pacing win (62.03-ms worst settling interval). It motivated moving the
remaining receiver-distance queries to the worker in `9293cbb`.

### Final worker-side pair and decision

Both use native `9293cbb`; `113e94c` only adds documentation/analysis. Candidate
ran first, then the control was repeated to check run-to-run pacing variation:

- Candidate: `target/gi-validation/cave-1791927228047255000-61792`.
- Repeated control: `target/gi-validation/cave-1791927488055810000-61980`.

| Final windowed measure | Repeated control | Worker-side candidate |
| --- | ---: | ---: |
| Settling frame p50 / p95 / p99, ms | 16.89 / 21.76 / 24.71 | 16.79 / 21.22 / 24.07 |
| Settling maximum, ms | 48.85 | 58.68 |
| Settling frames >33.3 ms / >50 ms | 2 / 0 | 3 / 1 |
| Selection p50 / p99, ms | 1.07 / 2.16 | 1.01 / 2.02 |
| Publication p50 / p99, ms | .97 / 2.36 | .96 / 2.31 |
| 30–60 s p95 excess corner variation | .20561 | .11291 |
| 60–120 s p95 excess corner variation | .41255 | .08879 |
| 120–180 s p95 excess corner variation | .29214 | .19716 |
| Minimum fresh paths, those three intervals | 400 / 752 / 592 | 400 / 736 / 592 |
| Total traced paths | 571.34 million | 557.43 million |
| Near / far measurement-age p99, s | 1.303 / 35.255 | 1.151 / 32.155 |
| Maximum resident pages | 1,573 | 1,587 |

All 144 observed corners are continuously present in each evaluated interval.
P95 oscillation is **1.82× / 4.65× / 1.48× lower**; median oscillation is
2.04× / 5.40× / 1.75× lower. Total paths are 2.44% lower, not identical, while
the slowest observed receivers continue comparable fresh sampling. This small
work difference cannot be advertised as a transport speedup. Near/far freshness
is not sacrificed to freeze the picture. No new sampling or smoothing is used.

The earlier control p99 was 22.88 ms and the repeated control 24.71 ms, bracketing
the final candidate's 24.07 ms. Treat pacing as **comparable, not faster**. Its
single worst settling frame is worse than this repeated control, so there is no
blanket tail/non-regression guarantee. Startup remains expensive (611/664-ms
maxima), and neither policy is always 60 FPS. The additional *game-thread
receiver-query* cost is removed; worker and bounded publication costs still exist.

Decision: retain and enable the policy on this experimental branch, keeping the
baseline switch and parent baseline intact. Across headless accuracy controls and
repeated real-window runs, the stability benefit is unambiguous. Late truncations
and residual cave error remain: do not infer the cause of every remaining cut
without event attribution, increase the accepted quality limits, or call coverage
convergence. The next step is to separate real changed-light dependencies from
conservative/unknown evidence before weakening more retention rules.

### Reproduction and integrity

Use `VOXEL_GI_VALIDATE=1`, `VOXEL_GI_VALIDATE_PROFILE=cave`,
`VOXEL_GI_VALIDATE_BACKGROUND=1`, `VOXEL_GI_VALIDATE_CAVE_LONG=1`,
`VOXEL_GI_VALIDATE_TEMPORAL=1`, `VOXEL_GI_VALIDATE_EXIT=1`,
`VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD"`, `RUST_LOG=info`, and
`cargo run --locked --offline`. Always explicitly select `VOXEL_GI_RETENTION=calibrated`
or `VOXEL_GI_RETENTION=baseline` for reproduction: the default switched only after
these comparisons (historical controls above ran with the variable absent).
Headless accuracy runs additionally set `VOXEL_GI_VALIDATE_HEADLESS=1` and
`VOXEL_GI_VALIDATE_REFERENCE=1`; never mix those cadence timings with windowed FPS.
Analyze run directories with `tools/runtime_retention_analysis.py`; it rejects
missing/invalid capture evidence and keeps windowed-only results reference-free.
Raw captures, logs and JSON remain under ignored `target/`.
The seven-run aggregate is `target/runtime-retention-all-qualified.json`, SHA-256
`67e32dd87461d6b7fa55f68bb1c97857babd03ba425bbbdaeb8dfdb12da9fa82`.

Focused native checks cover ceil rounding, full-scan equivalence, local overflow,
unrelated streaming, unknown/known sky transitions, partial-face publication,
late-sun wakeups, immutable snapshot lifetime and stronger source caps. The M4
GPU/CPU test measured RMS 0 on 504 × 64-path faces and also passed emissive-room
parity. Strict production Clippy passes. No routine toggle stress/full suite was
run. Render logs have no validation errors or GPU fallback; normal Metal
bindless-fallback and window-destroy warnings are not hidden.
