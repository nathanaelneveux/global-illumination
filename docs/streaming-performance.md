# Cave stability and streaming cost — October 10, 2026

## Retained changes

- Sixteen paths per selected face, still a 4-ms adaptive transport target. Two
  controlled cave pairs show 53–57% more paths and 33–36% less corner variation,
  with lower GI CPU cost. Initial coverage/individual refresh can be slower.
- Fast trusted-key hashing in scene construction and publication. A lookup-only
  paired benchmark is 3.3× faster; do not extrapolate that to whole frames.
- Reusable sparse-upload staging, after testing it with the new default. The
  integrated streaming/cave pair brings offscreen p99 to about 30 ms from
  38/47 ms, with continued lighting progress.
- Bounded private display-cache catch-up and worker-built, directly mapped
  buffers on supported integrated GPUs. On this M4, the measured 69–88 ms
  main-world publication bursts fall below 8 ms. Repeated cave worst intervals
  fall from 196 ms to 38–46 ms; a separate turn spike remains. Details below.

The scene-history policy and preferred corner appearance are unchanged. These
are **not** windowed FPS measurements or proof of convergence/60-FPS compliance.
Each experiment and its tradeoffs are recorded below; failed alternatives stay
on local branches, not in the production path.

The [BVW branch comparison](#bvw-optimization-branch-comparison) below reduces
observed worst intervals, but changes terrain residency and lighting freshness.
Its voxel-write lock fix alone does not remove the remaining hitch. GI display
publication is addressed by the retained experiment below without changing BVW.

## Bounded display publication

### Attribution and implementation

The instrumented baseline `5e863bc` records a 68.98-ms display update: 66.34 ms
is topology catch-up, not installation itself. A narrower diagnostic on that
commit plus `target/topology-detail-instrumentation.diff` measures 75.16 ms:
29.41 ms planning, **0.002 ms waiting for the mirror lock**, 10.83 ms committing
108,108 additions/10,452 removals, and 34.91 ms updating 205,773 changed values.
The async snapshot accumulated a world-scale delta that was applied in one frame.

Retained checkpoints, in order:

- `48c5d91`: reconcile a private generation in bounded batches before publishing
  its material/buffers atomically. Reserve lookup-map growth on the worker too.
- `658d520`: journal actual changes, not unchanged pending live-upload debt;
  cancel catch-up work already captured by the incremental snapshot. The first
  pilot's unnecessary tail extended streaming preparation to 23.8 seconds; this
  correction ended it around 20.8 seconds.
- `e9ee0d2`: worker-allocated buffers remain privately mapped through catch-up.
  Publish the exact allocations, not duplicate Bevy initializations.
- `4ce0984`: add `MAP_WRITE` when the device already enables
  `MAPPABLE_PRIMARY_BUFFERS`. Local wgpu 29.0.4 source shows that without it,
  unmap queues a full-size staging copy (`BufferMapState::Init`); Bevy 0.19.1
  enables the primary-mapping feature on supported integrated GPUs, but normally
  excludes discrete GPUs. This removes that deferred copy on the M4. Neither
  renderer feature negotiation nor the fallback's memory policy is changed.

The previously rejected worker-buffer experiment (`0dd5a03`) did not bound the
CPU catch-up and still used full staging initialization. It remains a valid
rejected result; simply moving allocation to a worker was not sufficient.

No sampling, history, smoothing, transport/upload budgets, draw distance, BVW
dependency or preferred-look setting changes are included. Catch-up has a
1.5-ms soft CPU target, 512-key batches and an 8,192-key scanned-entry cap.
Old lighting continues updating. Newly added topology can wait longer for the
atomic replacement: the final streaming run prepared from about 16.83 to
21.5 seconds, versus the baseline's large installation near 18.93 seconds.
This is a documented latency tradeoff, not faster convergence. All final
candidate runs drain private/topology and first-sample debt.

### Controlled measurements

Apple M4, 10-core GPU, 16 GiB; macOS 26.6.2; Rust 1.95; Bevy 0.19.1 and registry
BVW 0.17. Development build (root opt-level 1, dependencies 3, dynamic Bevy),
2560×1440 offscreen, corner sun/reconstructed indirect, textures on/AO off.
Sixteen paths/selected face, 4-ms adaptive trace target, 8,192 receivers,
three bounces, reach 768, 75-ms display response/50-ms upload. No concurrent
build or profiler. This was a live desktop; the existing Chorio engine and OS
services were not stopped. Offscreen intervals do not establish windowed FPS.

`VOXEL_GI_VALIDATE=1`, `VOXEL_GI_VALIDATE_HEADLESS=1`,
`VOXEL_GI_VALIDATE_FRAME_TIMINGS=1`, `VOXEL_GI_VALIDATE_EXIT=1`,
`VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD"`, `RUST_LOG=warn`, `cargo run --locked`.
Use `PROFILE=streaming` via `VOXEL_GI_VALIDATE_PROFILE`; cave also uses
`VOXEL_GI_VALIDATE_BACKGROUND=1` and `VOXEL_GI_VALIDATE_TEMPORAL=1`.
Rows use 15 seconds onward, excluding capture-associated intervals, including
the streaming turn. Reported system-body maxima are independent of app intervals.

| Runtime | Profile | Interval p95 / p99 / max (ms) | Display max (ms) | Total paths (M) | Final pages / probes |
| --- | --- | --- | --- | --- | --- |
| `5e863bc` baseline | streaming | 26.75 / 31.07 / 162.70 | 68.98 | 86.78 | 604 / 452,561 |
| `658d520` CPU catch-up | streaming | 26.43 / 28.80 / 72.75 | 6.49 | 92.41 | 600 / 440,933 |
| `e9ee0d2` worker/staging | streaming | 26.25 / 29.37 / 89.95 | 7.36 | 91.17 | 603 / 432,860 |
| `4ce0984` primary mapping | streaming | 26.23 / 28.71 / 52.75 | 6.90 | 90.78 | 611 / 442,792 |
| `0ddc304` prior control | cave/background | 26.98 / 30.76 / 195.93 | 87.90 | 136.53 | 1,124 / 1,011,501 |
| `e9ee0d2` worker/staging | cave/background | 26.69 / 30.11 / 103.44 | 6.81 | 127.84 | 1,091 / 1,023,462 |
| `4ce0984` primary mapping | cave/background | 27.20 / 30.52 / 38.17 | 5.85 | 128.14 | 1,073 / 1,016,652 |
| `4ce0984` repeat | cave/background | 26.30 / 30.40 / 45.99 | 7.50 | 125.59 | 1,098 / 976,668 |

All runtimes above are clean commits. The prior cave control differs from
`5e863bc` only by documentation and opt-in substage instrumentation; the solver,
renderer and dependency baseline are unchanged. Geometry/async scheduling are
not identical between runs: do not turn small throughput/percentile differences
into a precise speedup claim. The targeted long publication bursts are gone in
these runs, without reducing the workload to the optimized BVW branch's smaller
resident set. Cave paths remain 6–8% below the prior control, not a demonstrated
throughput win; near-age p95 is 980–986 ms versus 912 ms. Streaming near-age p95
is 1,279 ms versus 2,025 ms and total paths are slightly higher.

The final streaming maximum occurs at the turn (15.11 seconds), not display
replacement; the remainder after 15.5 seconds peaks at 32.42 ms. The worker/staging
pilot also peaks at 32.07 ms after the turn, but retained a 103-ms cave swap.
Do not subtract CPU timings from pipelined intervals to attribute the turn to
BVW. **Remaining turn/ordinary frame costs and 60-FPS compliance are not solved.**
Other backends without primary mapping retain the staging-copy limitation.

### Freshness, quality and verification

Both final cave runs have all 144 corner traces present during 40–60 seconds,
with at least 128/208 new paths per site. P95 excess variation is 0.0764/0.0840
versus 0.0999 in the prior control. This is no evidence of a noise regression,
not a new estimator or ground-truth accuracy win. Coverage/bootstrap finish at
100%; a few ingestion pages remain from continuous background loading, while
scene publication, private display catch-up and first-sample debt are empty.

The background `response` run at `e9ee0d2` retains 14 valid captures, no final
display backlog, and emitter-removal red residual below 10% after 2.39/2.61 s
at the two fixed ceiling receivers; final residual is 2.85%/3.54%. The prior
control was 2.38 s at both. Response evidence is reused for the unchanged
estimator/publication logic in `4ce0984`; that commit only changes native buffer
usage. It is not represented as a second response run.

Focused verification: 16 render tests, three change-journal tests and 22 field/
publication integration tests; native generation/material and out-of-bounds
retry regressions; development build, strict all-target Clippy, format/diff
checks. Native checks cover exact prepared-buffer reuse, byte parity after
private catch-up, actual GPU submission after unmap, growth/shrink/equal-size
generations and CPU fallback. No routine toggle stress or full suite was run.
All cited candidate captures pass dimensions/assets/nonblank checks and logs are
clean; streaming, early/final cave and recovered edit-room images were inspected.

Exact ignored artifacts under `target/gi-validation/`:

- `streaming-1791652631596739000-74333`: baseline `5e863bc`.
- `streaming-1791652741257912000-74385`: diagnostic-only topology detail diff.
- `streaming-1791653299637589000-74758`, `cave-1791653365515950000-74806`:
  first bounded pilot `48c5d91` (superseded catch-up bookkeeping).
- `streaming-1791653554514279000-74971`: `658d520`.
- `streaming-1791653952312124000-75201`, `cave-1791654015710649000-75239`,
  `response-1791654091313993000-75302`: `e9ee0d2`.
- `streaming-1791654375936014000-75514`, `cave-1791654272299960000-75443`,
  `cave-1791654449505670000-75550`: final `4ce0984`.
- `cave-1791651580682425000-73424`: prior control `0ddc304` from the BVW comparison.

Logs use `target/{streaming,cave,response}-display-*.log`; focused test/build logs
use `target/display-*.log`. See [display-cache lifetime](architecture.md#display-cache-lifetime)
for the maintained ownership and publication contracts.

## Follow-up experiments

Rejected dependency representation: `codex/path-footprint-retention` at
`545244d` preserves the implementation and evidence. It stores a 14-plane union
of retained paths and checks geometry changes against it. Early rays through
incomplete residency permanently inflate the union, even after their measured
weight becomes tiny. Per-region change bounds did not rescue it. Native energy
and metadata parity passed, but cave p95 variation remained 0.1381 with 16
history drops. Do not promote it or approximate away valid distant dependencies.

`9b3aa9d` replaces the optional five-second full-world diagnostic scan with a
bounded 1,024-receiver estimate. Exact coverage/freshness counters are unchanged.
Validation enabled the old scan even with warning-only logs; several largest
spikes coincided with it. Normal play already disabled it. This is a measurement
overhead fix, not a claim that ordinary-play stalls are solved.

Batch-size pilot on `9b3aa9d`, same cave/background/offscreen settings as below:

| Metric | 4 paths/face | 16 paths/face |
| --- | ---: | ---: |
| Total paths | 83.97 M | 128.53 M |
| Corner excess variation p95, 40–60 s | 0.1479 | 0.0939 |
| Minimum new paths at measured sites, 40–60 s | 88 | 144 |
| GI main-body sum p95 | 11.22 ms | 7.30 ms |
| Offscreen app interval p50 / p95 / p99 | 20.81 / 29.64 / 45.22 ms | 21.62 / 26.97 / 46.61 ms |
| Near measurement age p50 / p95 / p99 | 106 / 545 / 1,813 ms | 349 / 958 / 1,437 ms |

Runs `cave-1791647202747113000-69643` and
`cave-1791647269860800000-69758`; the latter uses launch JSON
`{"samples_per_dispatch":16}`. The test-only accessor annotation in `02c3023`
was edited during the latter run but not built. All 144 corners present and all
three captures valid in each. No history, display, scene policy or work-budget
change. Promising amortization, but individual refresh cadence slows and overall
p99 does not improve: check response and repeat before choosing the default.

BVW in-flight limit pilot: clean runtime `02c3023` versus only overriding
`max_active_chunk_threads` to 8. Streaming runs
`streaming-1791647431713115000-69830` and
`streaming-1791647504446268000-69917`: streaming p95/p99/max intervals
29.61/40.85/172.58 ms versus 29.48/38.44/198.13 ms; 33.85/33.99 M paths,
605/621 final pages with no remaining publication backlog. Not a demonstrated
stall fix. Same cap with `CARGO_PROFILE_DEV_OPT_LEVEL=3`,
`streaming-1791647586359960000-70001`, gives 28.37/38.71/175.86 ms and 34.41 M
paths: no useful overall win. Retain normal build settings and do not promote
the arbitrary job limit. Captures valid and render logs clean in all three.

Worker-prepared display buffers remain only on `codex/worker-display-buffers`
(`0dd5a03`). Native identity/lifetime tests passed, but its streaming pilot still
had a 153-ms interval; it does not justify extra renderer integration on main.

## Repeated batch-size result

Runtime `1208e0f`, same settings/hardware as the pilot: 16-path repeat
`cave-1791648562239088000-70820` and 4-path repeat
`cave-1791648681119103000-70924`. Source-only hashing edits were made while these
ran but were not built into either binary. The 4-path binary was launched directly
with explicit asset/library roots. An earlier missing-asset launch was rejected
by the guard and is **not** evidence. Both valid runs have 144/144 corner traces,
three valid captures and clean logs. The 16-path final image was inspected.

| Metric | 4 paths | 16 paths |
| --- | ---: | ---: |
| Paths | 81.76 M | 128.29 M |
| Corner excess variation p95, 40–60 s | 0.1331 | 0.0891 |
| Minimum fresh corner paths | 80 | 176 |
| GI main-body p95 / p99 | 10.85 / 13.56 ms | 7.25 / 9.69 ms |
| Complete Main schedules p95 / p99 | 12.90 / 15.72 ms | 9.51 / 12.03 ms |
| Offscreen interval p95 / p99 | 29.51 / 47.86 ms | 26.74 / 46.26 ms |

The larger batch gives 53–57% more total paths and 33–36% less corner variation
across the two pairs, with about one third less GI CPU-tail cost. It does not
eliminate noise or the renderer/runner tail. Geometry streaming is stochastic;
repeat endpoints were 1,110/1,144 pages, with no publication backlog.

Response countercheck: 4-path runtime `dc7b3bd`,
`response-1791648380639395000-70616`, versus 16-path runtime on the experimental
worker-buffer branch, `response-1791647864864104000-70278`. This is a countercheck,
not a controlled renderer-speed comparison. Using the scene-history audit's
pre-add/lit red reference at `(9,38,-9)` and `(10,38,-9)`, last excursions above
10% after removal were 1.97/2.47 s versus 2.03/2.03 s. Final residuals were
0.026/0.050 versus 0.033/0.034. All 14 captures valid in each; clean logs.

Adopt 16 paths as the default, keeping the 4-ms controller target, range, depth,
history and display response unchanged. It is not uniformly better at first-ray
coverage: around five seconds the repeats covered 67%/43% of roughly 470k/459k
already-published probes (4/16 paths); around ten seconds both were effectively
covered, with no publication backlog. Lower batch sizes remain a quality/work
knob for hosts that prefer faster individual refresh over refinement throughput.

## Trusted spatial-key hashing

Use Bevy's existing fast maps consistently for scene regions, packed bricks,
receiver-page indices and publication scratch. The field's receiver and sky
indices already use them. Keys are engine-owned coordinates/material words, not
untrusted strings; no traversal, history or allocation representation changes.
The opt-in `scene_lookup_hash_benchmark` alternates ten measured pairs over
524,288 identical hit/miss lookups (including negative keys): standard-map median
6.132 ms, Bevy-map 1.860 ms, identical checksums. This is a **lookup** speedup,
not a claim that the whole frame is 3.3× faster. Seven construction tests, five
packed-scene tests and the integration contracts cover the changed indices.

## Upload experiment and interaction with batching

`codex/display-upload-belt` at `a95ba97` reuses wgpu staging chunks for sparse
display writes. Fifteen render tests, native generation/material and retry tests,
build and strict Clippy passed. Four-path streaming on `6b914f3`, same settings:

| Run suffix | Interval p95 / p99 / max (ms) | Frames >33.3 / >50 ms | Paths |
| --- | ---: | ---: | ---: |
| `1791648859087822000-71144` baseline | 29.49 / 38.35 / 142.95 | 18 / 4 | 34.48 M |
| `1791648979960457000-71263` belt | 27.97 / 30.66 / 208.45 | 6 / 3 | 33.60 M |
| `1791649032981354000-71291` belt repeat | 28.06 / 29.98 / 196.53 | 4 / 3 | 35.07 M |
| `1791649165438900000-71422` baseline repeat | 29.14 / 41.07 / 179.35 | 18 / 4 | 33.79 M |

All captures valid, logs clean, no publication backlog; 601–608 final pages.
The frequent tail improves, but the largest stall worsens. This four-path pilot
was kept separate rather than hiding the tradeoff behind p99.
The default-setting edits were not compiled into the baseline repeat.

### Integrated 16-path baseline

Runtime `c7f3e85`, normal offscreen settings and background loading, no launch
override. Clean render logs and valid assets/captures in all three:

- `response-1791649304852133000-71571`: 67.70 M paths, 14 valid captures;
  last red-residual excursions above 10% at 2.375 s for both floor receivers,
  final residuals 0.0441/0.0419. Recovered image inspected. Within the observed
  four-path response range, not an extinction speedup claim.
- `streaming-1791649353580803000-71613`: 94.71 M paths, 607 pages and no
  publication backlog. Interval p95/p99/max 27.33/38.13/192.35 ms; GI body
  p95/p99 8.55/11.79 ms. Near age p50/p95/p99 236/2,080/2,814 ms.
- `cave-1791649402838945000-71654`: 133.74 M paths; 144/144 corners, minimum
  160 new paths, p95 excess variation 0.0956. GI body p95/p99 7.49/9.84 ms;
  interval p95/p99 26.78/46.71 ms. A 244.77-ms outlier at 36.45 s remains;
  **the rare stall is not solved**. Final 1,132 pages with no publication backlog.

The staging implementation is byte-for-byte the previously verified `a95ba97`
render file, now tested with the 16-path default (`c7f3e85` plus only that file).
`streaming-1791649547119667000-71790`: 94.77 M paths, 604 pages, no backlog;
interval p95/p99/max 26.65/29.75/162.49 ms and 4 frames over 33.3 ms versus 17.
Near age p50/p95/p99 245/2,081/2,969 ms: comparable, not improved freshness.
Frequent-tail reduction now repeats across both batch sizes without reducing
transport throughput. Worst-case timing remains variable; do not claim a hard
latency bound or windowed 60 FPS from this evidence.

The combined cave check, `cave-1791649650795561000-71840`, has 129.89 M paths,
144/144 corners and minimum 192 fresh paths at the sites. Corner p95 variation
is 0.08535, GI body p95/p99 7.02/8.94 ms, and interval p95/p99/max
26.61/29.76/44.43 ms. Final 1,136 pages / 1.015 M probes, no publication backlog;
all captures valid, final image inspected, render log clean. Its slightly lower
total paths than the 133.74 M control is within the observed run variability;
no fresh-site starvation or added history/filtering produced the lower tail.

Retain staging with the new default: frequent-tail improvement is consistent
across the four-path repeats and both integrated workloads. The earlier
worst-frame tradeoff remains documented; a single cave run without a huge
outlier does not establish a bound. The render file is identical to `a95ba97`,
so its focused/native lifetime tests remain valid; the integration build and
strict all-target Clippy were checked again. No toggle stress was needed.

The accepted scene-history work through `b051274` is merged into the regular
`voxel-global-illumination` branch. Keep its history and presentation policy fixed
while removing publication overhead. Neither the cave's residual noise nor the
always-above-60-FPS target is solved by this checkpoint.

## BVW optimization branch comparison

### Decision and attribution

The complete optimization branch is promising for streaming tails, but is not
promoted as an equivalent-work stall fix. Leave the default dependency and GI
code unchanged while investigating display publication and the residency tradeoff.
The optimized checkout's uncommitted work was neither modified nor benchmarked.

Stock BVW 0.17 acquires the modified-voxel write lock every PreUpdate, even with
an empty edit buffer, while generation workers retain read guards. The optimized
branch skips empty buffers and defers contended writes without dropping edits.
However, porting **only** that guard and nonblocking write onto stock BVW still
produces a 169-ms streaming interval. This rules out that particular wait as the
sole cause in this workload; it does not rule out every BVW lock or workload.

The same lock-only run records **88.66 ms in GI display publication**. Stock
repeats record 66.09 and 67.59 ms there; full-branch repeats record 7.55 and
8.11 ms maxima. PreUpdate maxima in these streaming windows are respectively
0.80 ms, 4.04/0.86 ms, and 2.51/2.17 ms. These are independent system/schedule
observations, not a subtraction from the pipelined frame interval. The specific
operation was not isolated at this checkpoint. The subsequent
[display investigation](#bounded-display-publication) isolates the synchronous
catch-up burst and retains its fix. A smaller loaded set avoiding a costly GI
cache operation remains a hypothesis about the BVW comparison, not proof that
every difference between its branches has been attributed.

### Controlled setup

Application runtime `0ddc304a94c827264333baf1a1038d60218f1385`, no GI/harness
source changes. Apple M4 / 10-core GPU / 16 GiB, macOS 26.6.2, Rust 1.95.0,
Bevy 0.19.1 / Metal. Same development profile, shared intermediates, offscreen
2560×1440 / 60-Hz runner, preferred corner look, textures on, AO off, 16 paths,
4-ms target, 8,192 receivers, 3 bounces, reach 768, display response/upload
75/50 ms. No concurrent builds/profilers during measurements.

Variants:

- Registry `bevy_voxel_world = "0.17"`, lockfile unchanged from `0ddc304`.
- Full local BVW `perf/integration-bevy-0.19` at
  `5aefb4f2615fa67251d083e5b1fec5cec95f3148`, selected with Cargo's CLI patch
  override; only Cargo.lock differs in the application checkout during a run.
- Lock-only BVW `codex/gi-write-lock-ablation` at
  `a7e694c63c170216a84a21fc3dc7e682dfe93666`, based on release `919a9bd`, via
  path override `target/bvw-write-lock-ablation`. Only the empty-buffer guard
  and `ModifiedVoxels::try_write` change. Discovery, generation and other locks
  remain stock. This is an attribution experiment, not a production fork.

The full-branch command uses the committed Git snapshot, not a path dependency
on the user's dirty working tree:

```sh
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_PROFILE=streaming VOXEL_GI_VALIDATE_FRAME_TIMINGS=1 \
VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD" VOXEL_GI_VALIDATE_EXIT=1 \
RUST_LOG=warn cargo run \
  --config 'patch.crates-io.bevy_voxel_world.git="file:///Users/nathanaelneveux/Developer/bevy_voxel_world"' \
  --config 'patch.crates-io.bevy_voxel_world.rev="5aefb4f2615fa67251d083e5b1fec5cec95f3148"'
```

This updates Cargo.lock for the experimental source. Subsequent runs with that
override use `--locked`; removing the override and running `cargo build --offline`
restores the registry source. At the end both manifest/lockfile match `0ddc304`
and the default build succeeds. The exact tested BVW commit could not be fetched
from GitHub into a fresh Cargo cache; do not substitute the published preview
branch silently or introduce an unresolvable default dependency. Nothing pushed.

### Streaming repeats

Turn at 15 s, walk until 30 s, finish at 35 s. Intervals below cover 15–35 s,
excluding screenshot/readback cooldowns. They are **not interactive FPS**.

| Variant | Interval p95 / p99 / max (ms) | Frames >33.3 / >50 ms | Total paths | Final pages / probes |
| --- | ---: | ---: | ---: | ---: |
| Registry A | 26.58 / 29.21 / 142.90 | 5 / 4 | 89.36 M | 606 / 450,483 |
| Full branch A | 27.01 / 28.97 / 75.53 | 2 / 1 | 89.79 M | 401 / 279,955 |
| Full branch B | 27.31 / 29.50 / 64.01 | 1 / 1 | 89.49 M | 398 / 277,262 |
| Write-lock-only | 26.59 / 29.53 / 169.36 | 3 / 3 | 91.25 M | 607 / 450,514 |
| Registry B | 26.61 / 29.62 / 140.37 | 3 / 3 | 90.42 M | 602 / 442,792 |

Final coverage/bootstrap are 100% with no publication/first-sample backlog in
all five. This is not a convergence claim. Residency is redistributed, not
simply uniformly reduced: just before the turn, registry A has 272 pages and
full A has 394; after the turn full A retains fewer pages. Near measurement-age
p50/p95 are 290/2,217 and 174/1,851 ms in the registry repeats versus 402/2,097
and 506/2,662 ms in the full repeats. The lower worst interval is not a uniform
freshness or p99 gain. Recovered images show intact nearby terrain; different
distant residency means this is not a matched-geometry comparison.

Run IDs under `target/gi-validation/`, in table order:

- `streaming-1791651528149466000-73326`
- `streaming-1791651691347756000-73547`
- `streaming-1791651811082409000-73628`
- `streaming-1791651901956730000-73722`
- `streaming-1791651956366522000-73766`

### Cave and edit counterchecks

Cave runs with background terrain and temporal capture enabled: registry
`cave-1791651580682425000-73424`, full branch
`cave-1791651737009569000-73583`. Intervals use 15–62 s excluding captures;
temporal measurements use 40–60 s at the same 144 corners.

| Metric | Registry | Full branch |
| --- | ---: | ---: |
| Interval p95 / p99 / max (ms) | 26.98 / 30.76 / 195.93 | 27.03 / 30.93 / 40.60 |
| Paths | 136.53 M | 113.85 M |
| Final pages / probes | 1,124 / 1,011,501 | 831 / 813,141 |
| Near age p50 / p95 (ms) | 332 / 912 | 401 / 1,777 |
| Corner excess variation p95 | 0.09988 | 0.08980 |
| Minimum fresh paths at corners | 128 | 160 |

All 144 sites are present throughout. The full branch finishes at 98.63%
coverage/bootstrap with 14,253 first-sample probes pending, versus 100% / zero
in this registry run. Both have continuing background work (1/7 dirty pages
respectively). No broader noise/convergence speedup is established by this pair.

Full-branch response run `response-1791652077041707000-73921` has background
loading, all three scheduled edits, 14 valid captures and 65.49 M paths. Using
the scene-history audit's independent pre-add/lit red reference, both floor
receivers' last >10% excursions after removal occur at 2.101 s, with final
residuals 0.0361/0.0388. This is within previous response measurements, not an
emitter-response speedup claim. The cavity and emitter removals reach the scene;
the report's generic edit counters do not track this fixture, so use its
`response.json` events/receiver histories, not the zero generic edit counts.

All eight workload logs are clean, assets/target dimensions valid and captures
nonblank (three each except response's fourteen). Recovered streaming, final
candidate cave, baseline cave and recovered room images were inspected. Logs
are `target/{streaming,cave,response}-bvw-*.log`. Only relevant workload builds
and these checks were run; no broad unit/native/toggle rerun is claimed for this
dependency experiment. The documentation checkpoint needs only diff review.

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
With frame timings enabled, `previous_host_schedule_ms` also brackets all seven
Main schedules, including their deferred commands and executor waits. Names are
in `host_schedule_names`; these are the last completed iteration, not the current
Update. Their sum excludes render extraction/pipeline waits after Main and the
headless runner's sleep. Bevy can feed `Time<Real>` from renderer timestamps
under pipelining, so the preceding Main observations do not exactly partition
the interval; even a negative remainder is possible. Compare their distributions
within the workload, not a subtraction attributed to BVW or GPU time.
The boundary schedules are absent from normal gameplay.

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
