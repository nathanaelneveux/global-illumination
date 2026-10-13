# Distance-preserving retention summary

Baseline: `fd7194d`, preserved on `codex/gi-retention-attribution`.
Implementation: `a194199`; analysis: `effb098`.
Experiment branch: `codex/gi-distance-band-summary`.
Cleanup `ff2b848` only gates report accessors to diagnostic builds and simplifies
the equivalent constant initializer; the measured retention behavior is unchanged.

**Decision:** integrate distance bands as the default on
`voxel-global-illumination`, following the user's acceptance of the repeated
late-stability gain and more faithful enforcement of the calibrated rules.
Keep eight exact events and retain `floors` as the comparison option. Two
reversed-order offscreen pairs support the improvement with unchanged thresholds.
The initial qualification kept it opt-in; this decision changes the default,
not the evidence. Startup regression was not causally established, and the
black/unfocused windowed capture remains excluded. No FPS or faster-startup claim.

The [attribution audit](retention-attribution.md) found that the eight-event
journal's overflow fallback, rather than measured cave target motion, dominated
remaining history loss. This experiment changes that compression, not the
retention curve, transport estimator, lighting range, presentation or BVW.

## Representation and ownership

Each existing 32³ receiver page still retains eight recent events with exact
change bounds. Older events enter a fixed-size summary. For each origin
(geometry or sunlight) and distance band, store the newest event revision.
An unconsumed revision means the band still contributes a cap; an already
consumed near event cannot contaminate a later distant event.

Distances are minimum AABB-to-AABB separation between the changed bounds and
the receiver page, **not center-to-center distance**. This is a lower bound on
every receiver's distance, so it cannot miss an affected face at a page edge.
The scene/tracing worker already computes these separations for change binning;
it now carries the minimum with each event. Publication does not rescan its
bounds when evicting it. Selection still checks only the newest page stamp;
the tracing worker evaluates caps against immutable copy-on-write snapshots.

There are 37 distance bands per origin: eight-voxel spacing with additional
exact squared-distance edges at 48²+1 and 76,800. These preserve the inherited
near-floor boundary and the current prior's no-cut boundary. Distances round
down, then the existing calibrated curve maps them to caps. We retain distances
and revisions, **not baked sample counts**, so recalibrating estimator work
against the same appearance target does not fossilize the old sample weights.
Changing the distance prior itself would require reviewing its bin boundaries
and influence radius too.

The band payload is 592 bytes/page (37 × 2 × 8), under 0.9 MiB for 1,500 pages,
plus allocation/header overhead and unchanged exact bounds. Since `194055a`,
it is allocated only when a page first overflows: no evictions means no band
allocation, with exactly the same caps as an all-zero summary. Retiring a page
releases its metadata. A worker-held snapshot can temporarily duplicate a page's
small summary on publication; there are no locks shared with that worker.

For identical events, receiver and consumed revision:

`old floor cap <= distance-band cap <= exact full-journal cap`.

Higher caps retain more statistical weight. This guarantees the new summary
does not respond *less* conservatively than the uncompressed existing policy.
It does not prove that the underlying distance prior bounds physical changes.
Page extent and band rounding still discard information; the diagnostic shadow
journal measures the resulting excess cuts. Smaller receiver subregions are
not implemented without evidence that this residual justifies their cost.

## Comparison and checks

Select `VOXEL_GI_RETENTION=calibrated` and
`VOXEL_GI_RETENTION_SUMMARY=floors|distance-bands`. Distance bands are the
integrated default; the recorded implementation checkpoint defaulted to floors.
The older
`VOXEL_GI_RETENTION=baseline` is a different, global-journal control; do not
confuse it with this experiment's calibrated per-page `floors` control.

Focused verification: 14 retention-filter tests, two geometry tests, six work
tests, the late-sun wake regression, strict production Clippy and a dev build.
Properties cover all consumed epochs, negative pages, face positions throughout
a page, both origins, monotonic/boundary behavior, immutable snapshots and
retirement. Six Python tests check summary identity, attribution accounting,
conservative exact-history caps and independent-oracle receiver ordering.

Qualification compares the same native build with either summary: the long
cave profile, moving background terrain, fixed preferred presentation and
unchanged budgets. Audit recordings collect six frozen scenes and two
65,536-path reference windows per scene after the timed workload. Live error
uses the final independent target; attribution separately measures target
movement between snapshots. Sparse corner RGB error is not whole-image FLIP,
and instrumented offscreen cadence is not windowed gameplay FPS.

## Native qualification — October 13, 2026

All four valid runs use native `a194199`, repository checkpoint `effb098`, with
only this follow-up's documentation edits during capture. No dirty native or
analyzer diff. Apple M4 integrated / 16 GiB / Metal, macOS 26.6.2; Rust 1.95,
Bevy 0.19.1 / registry BVW 0.17. Development app opt1 / dependencies opt3.
Offscreen 2560×1440, shared-corner direct sun / reconstructed indirect, textures
on, baked AO off. Sixteen paths/face, 4-ms adaptive trace target, 8,192 receiver
cap, reach 768 / three bounces, 75-ms display response, 50-ms upload and 2-ms
scene publication. Normal terrain plus cave structure; background loader
4.32 voxels/s; 182-second workload. No concurrent compiler, analyzer workload
or second native application during the timed portion.

Pair A ran control then candidate; pair B reversed that order to check the
ambiguous early result. Every valid run has all five nonblank, correctly sized
captures, valid camera/target/assets, clean render logs, complete 36-face /
144-corner observations and six independent reference checkpoints. The first
pair's 120-second captures and second pair's 180-second captures were inspected.

| Measure | A floors | A bands | B floors | B bands |
| --- | ---: | ---: | ---: | ---: |
| Actual cuts, 30–180 s | 106 | 91 | 68 | 73 |
| Statistical weight removed, 30–180 s | 105,304 | 58,151 | 85,053 | 50,345 |
| Excess removed by compression | 78,724 | 37,678 | 73,058 | 27,423 |
| Corner RGB p99 error, 30 s | 5.25% | 8.67% | 6.14% | 6.45% |
| Corner RGB p99 error, 60 s | 3.26% | 3.17% | 3.21% | 3.47% |
| Corner RGB p99 error, 120 s | 3.15% | 3.21% | 3.43% | 3.30% |
| Corner RGB p99 error, 180 s | 4.81% | 2.56% | 5.73% | 2.83% |
| Corner p95 excess variation, 120–180 s | .12491 | .05469 | .09145 | .06177 |
| Worst 1-s checkpoint corner p99, 120–180 s | 11.05% | 3.45% | 6.73% | 3.66% |
| Minimum new paths/site, 120–180 s | 416 | 384 | 400 | 336 |
| Total paths, millions | 414.48 | 369.96 | 370.37 | 369.25 |
| Maximum loaded pages | 1,536 | 1,602 | 1,564 | 1,546 |
| Median retained weight, 180 s | 504 | 1,170 | 688 | 1,279.5 |

The compression counterfactual evaluates the complete shadow journal at each
**observed** weight, not a hypothetical no-overflow trajectory. Candidate caps
never exceeded the exact journal's cap in any recorded publication. Excess
compression loss falls 52.1% / 62.5%; total weight loss falls 44.8% / 40.8%.
Late oscillation falls 56.2% / 32.5%; final error improves 1.88× / 2.03×.
There is still substantial excess loss from the conservative whole-page
approximation. This is a useful first summary, not a claim that compression is
lossless or that history invalidation is solved.

Both final candidate snapshots are below the fixed 3.266% raw corner p99 proxy
associated with the user's 2,048-path visual acceptance. Their worst late
checkpoints still exceed it, so neither is a sustained quality-threshold pass.
We did not run full-image FLIP or an IR/ionizing quality sweep here.

Reference-window corner p99 disagreement is approximately .604% in all runs.
Between-arm final targets differ by .073% / .016% p99. Within-run maximum
checkpoint target movement is .0040–.0484%. These are sparse RGB measurements,
not a confidence interval or proof that every intervening transient is harmless.

### Early response and cost limitations

The first candidate's 30-second error is worse, alongside 19% fewer median
paths at the watched faces (624 versus 768). In the repeat its 30-second error
is only slightly worse, but its 15-second error is 25.80% versus 14.52% for the
control. Residency, dispatch timing and early integrands differ across runs;
the frozen target series starts at 30 seconds. Do not conclude that history
alone caused or eliminated this early weakness. In particular, the candidate
does **not** establish faster initial settling.

Fresh measurements continue in every interval, but lower total path counts are
not a throughput win: the first pair differs by 10.7%, while the reversed pair
is within .3%. Offscreen settling selection p99 is 2.11/2.09 ms in A and
2.05/2.07 ms in B (floors/bands); merge p99 is 2.39/2.28 and 2.33/2.33 ms.
These are diagnostic component timings only. They show no large selection or
merge penalty, but do not qualify interactive frame pacing or a speedup.

The attempted windowed control, `cave-1791933739568020000-70281`, stopped at its
first capture: `focused=false`, `varied=false`. It contributes **no** gameplay
quality or timing evidence. The next qualification step is a valid focused
windowed floors/bands pair, with freshness and stability measured alongside
frame-time tails. The user subsequently accepted integration without treating
that missing check as a demonstrated regression. No screensaver/security
settings were changed, and no BVW changes were made.

## Integration and memory decision

Retain eight exact events. A native layout check reports a 40-byte development
event header, 24-byte bounds elements and 592-byte band payload. An event owns
a shared, variable-capacity bounds allocation; releasing it can save less or
more than the summary, and worker/diagnostic snapshots can delay that release.
Removing one event from the eight-slot deque does not shrink its allocation.
There is no fixed one-event-for-one-summary memory equivalence.

More importantly, seven exact events are a different accuracy tradeoff. The
regression test constructs a page spanning x=0..31 with a change at x=79:
the page is 48 away, but the receiver at x=0 is 79 away. Exact evidence permits
256 samples; moving that event into its summary one event earlier caps this
receiver to 16. This would reintroduce avoidable invalidation earlier. Lazy
summary allocation instead saves memory without changing any cap or epoch.

The two summary modes are identical before a page's first eviction. They can
differ during startup **after** eviction; the original audit already records
overflow cuts before 30 seconds. Thus the noisy startup measurements neither
prove a regression nor prove that startup is unaffected. This integration does
not change ray generation, trace budgets, the retention curve or the quality goal.

The journal is only lighting-change metadata, also used for waking receivers
with pending evidence. It is not geometry storage, an undo log or saved rays.
Simply dropping a newer far event because older near events exist is unsafe:
some receivers have consumed the near revisions and need the new revision.
Safe dominance pruning would require a **newer** event with an equal or stronger
cap for **every** receiver in the page, accounting for origin as well as distance.
Closest-to-page distance alone cannot prove that for exact events at different
locations. Within each summary band/origin we already retain only the newest
revision. Further dominance pruning is not part of this integration.

Integration verification reuses the four recorded native comparisons, rather
than repeating them for a default switch and behavior-equivalent allocation.
Focused tests cover pre-overflow parity, lazy allocation, immutable snapshots,
the seven-event counterexample, cap ordering, publication and late-sun wakeups.
No new native performance or memory-neutrality result is claimed.

### Artifact index and reproduction

All runs are under ignored `target/gi-validation/`:

| Arm | Run directory | Log under `target/` |
| --- | --- | --- |
| A floors | `cave-1791933274458887000-70103` | `distance-bands-floors-headless.log` |
| A bands | `cave-1791933509381577000-70214` | `distance-bands-candidate-headless.log` |
| B bands | `cave-1791933807128612000-70298` | `distance-bands-candidate-repeat.log` |
| B floors | `cave-1791934053417774000-70397` | `distance-bands-floors-repeat.log` |

Launch with `VOXEL_GI_VALIDATE=1`, `VOXEL_GI_VALIDATE_PROFILE=cave`,
`VOXEL_GI_VALIDATE_BACKGROUND=1`, `VOXEL_GI_VALIDATE_CAVE_LONG=1`,
`VOXEL_GI_VALIDATE_TEMPORAL=1`, `VOXEL_GI_VALIDATE_EXIT=1`,
`VOXEL_GI_VALIDATE_HEADLESS=1`, `VOXEL_GI_VALIDATE_RETENTION_AUDIT=1`,
`VOXEL_GI_RETENTION=calibrated`, the explicit summary arm and
`VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD" RUST_LOG=info cargo run --locked --offline`.
Wait for process exit, not merely `status.json`: the oracle runs after that
status is written. Analyze with `tools/retention_attribution.py RUN` and
`tools/runtime_retention_analysis.py RUN` using the bundled NumPy-enabled Python.
For the windowed pair, omit both `HEADLESS` and `RETENTION_AUDIT` variables.

Summary SHA-256, in arm order A floors, A bands, B floors, B bands:

```text
attribution
8a5a27e9763e886464224bb7da1e5127b26cb8fcf4f79e88d5e85b0b209862d0
555d818f00b5a3bcc7c37caf0565c4acb0c490c20f549e59002b4189fd5fe6ff
0066b2260c9d469498e6358035261135ef04f8480423ccfbb1199fd2d3bf9183
9f84570ddef1fd9cef2cad65fffb42ba039b0feee2d995abbfe135fc847c1770
live quality
0f7c55540ff27128afbd2a1d252d8117a601a4576607a07807892e955bc55a1b
43734179eba85da72752ebf6d4cea31dc654499e56ee2cd08d28ffd981409fbe
29bca988cbe5d0d142186340f7745b80ae315273b02f40796a97a55925f3f127
b0f4a4001d9a80820158a218c90abaa70f86ae2d83f051d7df5369927041b6de
```
