# Settling speed — October 10, 2026

## Batch-size and cached-bound decisions

Integration check at `10a8b7c` (runtime `3e09d56`): the focused native
cache/material-generation test passed. The default response workload
`response-1791858630010281000-41136` completed with 14 guarded offscreen
captures and clean render logs. Cavity-settled and emitter-removal +1 s images
were inspected; neither was black or missing assets. This verifies the
integrated response path, not a new independent-reference latency result or
interactive 60 FPS. No stress sweep was needed for these unchanged lifetimes.

The follow-up 6 ms + 500 ms display-response pilot used the existing knobs,
not a new history reset or filter. Runtime `3e09d56`, clean launch `10a8b7c`,
`cave-1791858395416839000-40989`: median/p95 settling 6.159/13.431 s, but only
140/144 sustain the final dwell. Error p95 at 15 s .05313; peak .09834/.28623.
Variation .11140/.09901 remains higher than quiet control .08529/.04889,
despite longer smoothing (fresh paths at least 304/288). Age p95 .710/17.549 s,
736.15 M paths, 99.43% coverage. CPU body 4.684/8.012/12.573/23.930 ms.
The median narrowly exceeds 2× control speed, but the noise/CPU tradeoff is
**not a qualified overall win**. Keep default 4 ms / 75 ms response. More work
at the same per-update statistical gate can increase visible variation; changing
only the time cap did not fully resolve that coupling. No interactive FPS claim.

A 4.5 ms pilot on `21c03a4` (runtime `3e09d56`),
`cave-1791858064391009000-40852`, did not justify a budget increase. Median/p95
settling 7.211/15.593 s, all 144 sustain, 15-second error .11618, peak
.10166/.30339. Variation .09429/.08119 (224/176 minimum new paths) is higher
than the 4 ms candidate. Near/far age .853/30.784 s, 462.01 M paths, 99.00%
coverage; GI CPU p50/p95/p99/max 3.429/6.430/9.123/15.114 ms. More refresh
does not necessarily mean earlier stable presentation. No default change.

On `3e09d56`, same quiet long cave/background/default 4 ms budget, 32 paths
(`cave-1791857166053389000-40017`) settle median/p95 in 8.347/22.690 s, with
15-second corner error .38504. Peak p95/max .09692/.22259 and variation
40–60/160–180 s .04320/.01896 improve, but near/far age rises to 3.709/71.837 s
and coverage falls to 88.86%. 273.82 M paths. CPU p95/p99/max
4.961/6.833/15.494 ms. Eight paths (`cave-1791857378380795000-40153`) settle
6.384/18.007 s, error .13222, but peak .10766/.33911 and variation
.12228/.08971 regress. Age .708/20.110 s, coverage 99.47%, 345.78 M paths;
CPU p95/p99/max 7.570/12.088/36.075 ms. Both runs sustain 143/144 corners,
have five valid captures and inspected 15-second images. Minimum fresh paths
in the two intervals: 32-path 128/64, eight-path 136/136. Later test-only
source edits were not compiled into the eight-path run. **Keep default 16**;
the expanded 1–64 control is for deliberate quality comparisons.

Separate test-only `codex/gi-gpu-light-bounds`, `c0ef49f`, calculates the cached
light radius on the GPU rather than CPU. Strict same-path five-band means,
moments and direct values now pass at 1/7/16/64/256 paths. But 64 alternating
native rounds/eight warm, 2,048 cave receivers × 16 paths, give M4/Metal kernel
median 3.48796→3.48204 ms: no useful speedup, even before charging tree-update
precomputation. Native study and strict Clippy pass; do not promote the extra
pass. The retained runtime does not contain this study's instrumentation.

## Quiet validation comparison (October 12)

Deadline-only snapshot capture `2f1ed03`, quiet long cave
`cave-1791856916784508000-39560`: median/p95 settling 7.075/18.294 s,
141/144 sustain; 15-second error .16298, peak .10724/.30448, variation
.06836/.04781 (192/96 minimum fresh paths). Near/far age p95 1.350/37.065 s;
369.53 M paths, 98.51% coverage. GI CPU body 2.529/5.902/8.631/16.567 ms.
Five valid captures, 15-second image inspected. Removing the extra count cap
did **not** establish faster replacement: preparation 139.253–144.449 s versus
138.706–143.412 s before. Most large capture slices already spent their time
allowance. Retain this as a time-budget consistency cleanup, not a measured
speedup or a reason to raise that allowance. Later batch-control source edits
were not compiled into this run.

Receiver-index partitioning `c49d86a`, quiet long cave
`cave-1791856639492812000-39306`: median/p95 corner settling 6.927/13.217 s,
141/144 sustain, 15-second p95 error .06840 (141/144 in tolerance), peak
.10198/.30433. Variation .07299/.05380, min fresh paths 160/128; near/far age
p95 1.248/36.511 s. 383.39 M paths, 98.83% final coverage. GI CPU body
p50/p95/p99/max 2.404/5.777/8.402/22.619 ms; worst body is trace publication,
not index growth. Five valid captures, 15-second image inspected. It supports
retaining smaller index growth, **not** attributing the full settling difference
to sharding or claiming stall elimination. Later source-only snapshot edits
were not compiled into this run. The isolated index study in the commit shows
12.65× smaller worst growth batches but 10.7% slower lookups.

Validation used to enable synchronous terminal statistics automatically, unlike
normal gameplay. `13db971` makes those explicitly opt-in and leaves the JSON
timelines enabled. Do not credit this harness correction as a gameplay speedup.
The speculative allocator-retirement branch did not remove the unexplained
page-return pauses and is not promoted. Logging was a confound, not a proven
explanation of every stall.

Matched quiet runs on M4/Metal/dev, 2560×1440 offscreen, long cave with background
terrain, default 4 ms/16-path transport, preferred corner look, no concurrent
compiler or native test:

| Metric | Control `c8e85fe` | Candidate `13db971` |
| --- | ---: | ---: |
| Sustained 10%-corner error median / p95 (s) | 12.582 / 21.426 | 7.726 / 16.857 |
| Corners sustaining through final dwell | 141 / 144 | 141 / 144 |
| Corner RGB error p95 at 15 s | .44124 | .14312 |
| Positive peak p95 / max | .09101 / .30090 | .10207 / .30441 |
| Variation p95 at 40–60 / 160–180 s | .08529 / .04889 | .07840 / .05248 |
| Minimum fresh paths in those intervals | 144 / 96 | 176 / 96 |
| Near / far measurement age p95 (s) | 3.112 / 59.659 | 1.423 / 38.530 |
| GI CPU body p50 / p95 / p99 / max (ms) | 2.014 / 6.116 / 8.127 / 34.274 | 2.560 / 5.836 / 8.654 / 16.642 |

Both launched clean and passed compiled-root/asset/capture guards, five captures
each; both 15-second images inspected. Control is `febcaf4` runtime plus the same
corner oracle and quiet harness; candidate runtime is `13024df`. Artifacts:
baseline worktree `cave-1791856242241706000-38923`, root
`cave-1791856014554688000-38671`; analysis/log prefixes `settling-quiet-*`.
Reference-pair disagreement p95 .472%. These establish useful early accuracy and
freshness progress (2.19× near-age, 3.08× less 15-second p95 error), **not 2×
settling** (1.63× median) or an interactive 60 FPS pass. CPU maxima are individual
observations, not a guaranteed bound. Small variation/overshoot differences
remain; do not describe all temporal behavior as identical.

An earlier 6 ms budget pilot on runtime `13024df`,
`cave-1791854655510236000-37067`, reached median settling 5.879 s but raised
40–60/160–180 s variation to .14658/.14022 and GI CPU maximum to 70.869 ms.
It is **not the default**: buying speed with substantially more variation did
not satisfy the target. The prior verbose 4 ms `13024df` run,
`cave-1791854197543590000-36653`, measured 7.015/16.173 s median/p95 settling,
15-second error .12518, variation .07020/.04125, CPU maximum 22.014 ms.
Its 30 s response countercheck `response-1791854482818420000-36945` changed
the two floor receivers' last >10% red residual after removal from control
2.719 s to 2.532 s. This is a display-response proxy, not independent accuracy
truth or a 2× edit-response result. Those runs predate the quiet harness.

## Latest streamed qualification (October 12)

Private worker catch-up `8d2d216`, run `cave-1791853243011741000-35728`:
first display publication about 5.748 s, sustained corner median/p95
7.119/18.379 s (141/144 sustain through the final dwell). Error p95 at 15 s
.25239; peak overshoot .10150/.30451. 367.61 M paths, final coverage 98.58%.
Main GI body p50/p95/p99/max 2.159/6.536/8.773/43.105 ms. Mixed overall
quality timing despite earlier publication; do not attribute run-to-run path
count variation solely to this change.

Bounded nonblocking live values `80e5263`, run
`cave-1791853582318240000-36063`: first publication about 2.897 s, corner
median/p95 6.984/15.428 s (141/144); 15-second error p95 .10531, peak
.10205/.30433. 371.42 M paths, coverage 99.00%. Corner excess variation
p95 .06672 over 40–60 s and .05548 over 160–180 s, minimum fresh paths
144/112. Near/far refresh age p95 1.208/35.816 s during settling. Main GI
body 2.400/5.884/8.411/22.055 ms; value stage p95/p99/max
1.527/1.536/6.885 ms. Soft slices cannot prevent OS preemption. Remaining
large spikes include atomic scene/topology publication, not just value uploads.
Both runs have five valid captures and clean render logs; no FPS claim.
They launched clean; later source-only edits were not compiled into them.

Exact unchanged-page check `13024df`: keep allocations and counters when
exposed records match, but still run explicit edit invalidation and refresh.
Focused 25 runtime/publication and two donor/identity tests, Clippy and build
pass. Isolated M4/dev 1,024-receiver fixture, 128 alternating pairs/16 warm:
full reconciliation 68.125 µs versus identity path 5.209 µs median (13.1× for
this stage only). Full streamed qualification is pending at this checkpoint.

The separate cached-radius pilot `c429924` is **not in this runtime**. It
uses a spare node word and passes the 3,872-choice forward/reverse PDF oracle,
but fails strict equal-path five-band parity (1.0340708→1.0420537). Timing
stopped at that check. Preserve the failed criterion and branch; do not call
this an exact or measured faster kernel. Evidence: `settling-light-radius*.log`.

Same M4/dev/2560×1440 offscreen long-cave/background settings as below;
five valid nonblank captures per run, clean render logs. Control is clean
`5c2d333` (runtime `febcaf4`, diagnostic corner oracle only), run
`cave-1791851270749646000-33719` in the baseline worktree. Its sustained
10%-corner-error settling median/p95 is 12.801/21.236 s.

Capacity-aware construction `c937f12`, run
`cave-1791852300606229000-34652`: median/p95 7.686/17.361 s; 15-second
corner error p95 .13366; positive overshoot p95/max .10179/.30453.
All 144 corners eventually sustain the tolerance. Main GI body
p50/p95/p99/max 2.500/7.394/9.611/83.761 ms: not a frame-pacing pass.

Measured-refinement eligibility `41ba201`, run
`cave-1791852636629598000-34987`: median/p95 7.881/15.397 s; 138/144
corners within tolerance at 15 seconds, error p95 .08730 versus control
.42926. Overshoot .10203/.30441, essentially unchanged from control
.10205/.30431. All 144 corners eventually sustain tolerance. First live
display generation appears around 7.94 s, so much of the early delay is
publication rather than unavailable measurements. This is useful progress,
**not an established 2× visual-settling or 60 FPS result**. Both runs launched
from clean commits; later source-only edits were not compiled into them.


Target: halve time to visually close-to-settled lighting without increasing
overshoot, starving other receivers, or blocking the player. Coverage is not
convergence. The starting runtime is `febcaf4`; the edit-guided experiment in
that commit is test-only, not the runtime policy.

## Baseline and isolated execution study

Apple M4, 16 GiB, Metal, Rust 1.95, Bevy 0.19.1, registry BVW 0.17. Dev profile
(root opt1, dependencies opt3), shared Cargo target. No concurrent compiler or
GPU workload during measurements. Initial study adds only test code.

Native lane-layout sweep: 2,048 receivers, 48 rotating-order batches (first
eight warmup), identical paths, all five means/second moments/direct values
checked with 17-receiver tails at 1/7/16/64 paths. GPU p50 milliseconds at 16
paths per face:

| Scene | 4 lanes (baseline) | 1 | 2 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open sky | .220 | .324 | .223 | .217 | .221 |
| Many-emitter cave | 3.978 | 6.070 | 4.050 | 3.301 | 3.480 |
| Skylight ceiling | .985 | 1.614 | 1.044 | .945 | .920 |

Eight lanes save about 17% cave kernel time, **not 2×**. At four paths they
waste lanes: cave 1.223 ms (four lanes) versus 1.874 ms (eight). A production
layout would need to respect batch size. This first sweep is not yet promotion
evidence for end-to-end speed.

Removing roulette at the default three-hit depth was independently tested with
four live sequence offsets, 128 batches of 16 paths, and independent 65,536-path
references. At 2,048 paths, pooled five-band relative p95 error changes:
skylight floor .138→.110, ceiling .0865→.0742, cave .0610→.0512. Cave GPU batch
time increases .951→1.031 ms. Ceiling worst bright excursion actually increases
4.388→4.720 relative to reference luminance, despite better aggregate error.
Analytic absorbing emitters are unchanged. **Not a universal winner** and not
promoted. PBRT explains both the variance cost of roulette and why terminated
GPU lanes may not save proportional execution time:
[Monte Carlo efficiency](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency),
[GPU execution](https://pbr-book.org/4ed/Wavefront_Rendering_on_GPUs/Mapping_Path_Tracing_to_the_GPU).

Baseline streamed cave: `cave-1791681216380638000-87350`, clean `febcaf4` at
build/launch, offscreen 2560×1440, long cave/background/temporal/reference/frame
timings enabled. Defaults: 16 paths, 4 ms worker target, 8,192-receiver ceiling,
2 ms scene publication, 75 ms display response, 50 ms display upload. Five valid
nonblank captures, clean render logs. Post-timing reference replicas: 262,144
paths each, relative RGB p95 disagreement .413%. At 30/60 seconds raw receiver
p95 error is 15.05%/9.62%; display 12.51%/9.80%. At 60 seconds 35/36 raw and
display receivers are within 10%. Corner excess variation p95 over 40–60 s is
.08230, with all 144 corners present and new paths continuing. Repeated history
cuts remain visible in the measurements.

After 15 s, excluding captures: offscreen app intervals p50/p95/p99/max
21.30/27.33/30.23/73.00 ms; GI main-system-body sum
1.68/5.60/7.75/18.03 ms. This is **not windowed FPS or a 60 FPS guarantee**.
288.72 M paths, 8,492 batches, up to 1,540 pages / 1.28 M probes.

Ignored evidence: `target/settling-{layout,roulette}-study.log`,
`target/settling-baseline-{build,cave}.log`,
`target/settling-baseline-{reference,temporal}-analysis.json`, and the run's
captures/JSON in `target/gi-validation/`. Reproduce isolated studies with
`cargo test --locked --offline settling_lane_layout_study -- --ignored --nocapture`
and `settling_roulette_study` respectively. Both passed; neither changes runtime.

## Local refinement-index invalidation

The nearby/regional index previously followed the renderer's global topology
generation. Any distant streaming change recopied the local receiver lists and
reset their fallback cursor. It now follows receiver-membership changes within
the two-page focus window only. Rebuilds look up at most 125 page coordinates,
instead of scanning every loaded page, and preserve the cursor unless the
player changes focus page. Face/material changes still invalidate lighting as
before; all-world fairness, history, rays and GPU display lifetime are unchanged.
All 23 runtime policy/publication tests passed, including new distant/local
addition/removal/movement coverage. This eliminates work by construction; an
end-to-end speed claim requires the streamed comparison below.

Streamed follow-up `cave-1791681993393562000-87849`, runtime `1088081` (only
test-only spatial-study edits made during the run), identical launch settings:
292.81 M paths versus 288.72 M baseline. All five captures valid, clean logs,
reference p95 disagreement .413%. Display p95 relative error at 30/60/180 s:
13.11%/12.73%/8.50%, versus baseline 12.51%/9.80%/6.66%. Sustained display
settling median is essentially unchanged (20.96 versus 20.86 s); p95 is worse
(166.15 versus 119.26 s). Corner variation p95 at 40–60 s falls .08230→.05785,
but streaming revisions differ, so this does not establish a general accuracy
gain. Both runs finish with 35/36 receivers sustaining 10% error.

Offscreen interval p50/p95/p99/max: 21.88/26.72/30.23/36.83 ms; GI body sum
2.11/6.16/7.86/15.63 ms. Maximum trace-dispatch body falls 14.37→9.42 ms,
but medians/tails do not support a broad speedup claim. Keep the local
invalidation for its bounded work and cursor-correctness improvement, **not as
the 2× settling result**. Remove the now-unused global topology counter; the
renderer already uses its independent dirty/lifetime bookkeeping.

## Spatial reconstruction pilot — not promoted

Test-only `settling_spatial_study` compares the existing connected four-face
corner reconstruction with an additional coplanar five-tap filter (center
weight four, edge neighbors one), both fixed and guided by per-channel sample
variance. Hard geometric boundaries exclude neighbors. This draws on the
variance/edge-stopping principle of [SVGF](https://cg.ivd.kit.edu/publications/2017/svgf/svgf_preprint.pdf),
not its screen-space implementation. Simulation values remain untouched.

M4/Metal/dev profile; four disjoint live sequence prefixes, 16-path batches
through 2,048 paths. Dense one-voxel neighborhoods around every third receiver
of the floor, ceiling and cave fixtures. Reference: 65,536 disjoint paths per
face, **unfiltered** connected corners. Reference submissions are bounded to
1,024 paths after the initial monolithic cave submission timed out; that failed
attempt is not accepted evidence. The complete bounded run passed.

At 256 paths, RGB relative corner-error p95 baseline/fixed/guided:
floor .369/.282/.320, ceiling .279/.216/.242, cave .0525/.1035/.0470.
The fixed filter's cave high-sample approximation error is 10.54% p95 (13.39%
worst): lower noise conceals real gradients near sources. Guided high-sample
deviation is only .051% p95, but its cave improvement is small and worst early
64-path error increases .193→.208. No filter cost or temporal startup is included
in this pilot; there is no 2× result and neither candidate is promoted.
Evidence: `target/settling-spatial-study.log`. Production remains unchanged.

## Fair refinement reservation

Experimental commit `0aba755` prevents a full streaming-coverage queue from
excluding adaptive refinement. A quarter of opportunities stays with the global
sweep, a quarter with initial coverage, and half with refinement; empty queues
lend their slots. Explicit edits still preempt all non-global slots. All 24
focused runtime/publication tests and the dev build passed.

Run `cave-1791846614020288000-29475`, same M4/profile/viewport/settings as baseline,
clean code at launch. Layout source edits made after the timed workload finished
were not compiled into this process. Five valid captures, native post-timing
reference. At 15 s, display p95 error improves .681→.283 and 28/36 receivers are
within 10% versus 13/36. Sustained raw settling median improves 27.53→12.80 s
(2.15×), but display median improves only 20.86→15.02 s and its p95 worsens
119.26→137.93 s. **Not yet a 2× visual-settling result.** At 60 s display p95
error is 9.38%; at 180 s 8.48%. Corner variation p95 at 40–60 s is .05670,
minimum 128 new paths; at 160–180 s .03772, minimum 64. No hidden pause in work.

306.34 M paths, final coverage 99.22%. After 15 s excluding captures: offscreen
interval p50/p95/p99/max 20.99/26.85/29.89/127.96 ms; GI body sum
1.98/6.16/8.45/24.11 ms. Large maxima remain; this is not a frame-pacing win.
Evidence: `target/settling-reservation-{tests,build,cave}.log` and
`target/settling-reservation-{reference,temporal}-analysis.json`.

## Batch-aware cooperative lanes

The execution-only candidate uses eight lanes for batches of at least 16 paths,
four for smaller batches. Quality/layout pipelines are bounded (3 depths × 2
reach modes × 2 layouts) and compiled on the worker. Samples, spectra, history,
ray lengths, bounce depth, scheduling budget and output buffer formats are
unchanged. Legacy test shaders retain their original layout. Automatic layout
switches invalidate auto-layout bindings correctly.

Confirmation at the same native fixture/work settings as the initial sweep:
cave four/eight-lane GPU p50 4.655/4.275 ms, worker 6.859/6.797 ms. Gains varied
between runs (roughly 4–17% kernel time), not a 17% guaranteed whole-engine gain.
Native automatic/fixed layout parity covers 17-face tails at 1/7/16/64 paths,
all five means/moments/direct values and continuing sequences. Standard and
emissive native CPU/GPU references report RMS 0.000000; resource reuse/growth
tests, formatting and strict all-target Clippy pass. End-to-end qualification
follows separately. Logs: `target/settling-layout-{confirm,pair,parity,lifetime,clippy}.log`.

## Corner-accuracy oracle

The offline reference now traces exposed coplanar neighbors as well as the fixed
center faces and records their connected-corner reconstruction at converged
equal confidence. This matches the preferred display's spatial definition;
face-center accuracy alone is not visual-corner accuracy. Unknown/missing
corners remain missing in the analysis, not a black measurement or a pass.

Schema 2 uses two disjoint **65,536-path** replicas per member face (previous
center-only reports used 262,144). Reference-replica disagreement is reported
for corners separately. The extra neighborhoods run only after timing/capture
completion. `python3 tools/reference_analysis.py RUN --corners` reports error,
sustained settling and positive luminance peaks relative to that independent
final-scene target. This is a presentation-space proxy, not tone-mapped pixel
error; early frames can have different residency from the final scene. No
last-frame or blurred-candidate target is substituted for truth.

Verification: native-independent reconstruction test (shared corners and
disconnected diagonals), three Python analysis tests (including missing spans
and known overshoot), fmt, strict all-target Clippy and dev build passed.

Combined runtime `c18380b`, run `cave-1791847590335826000-30282`: independent
corner replica disagreement p95 .472%; 140/144 corners sustain 10% error to
the end, median 11.68 s, p95 17.26 s. At 15 s 119/144 are within tolerance;
at 30 s all 144 are. Positive luminance peak p95 .1027, maximum .3045.
The diagnostic-only baseline backport `5c2d333` (runtime `febcaf4`) ran as
`cave-1791847874844864000-30513` in the temporal-baseline worktree. Its
143/144 sustained corners give median 15.89 s, p95 21.99 s and peak p95 .1021,
maximum .3045. However, its 15 s capture is effectively black and every
measured corner is absent then. **Reject that startup visual comparison**;
the later data alone does not establish a 2× improvement. Do not turn missing
lighting into a favorable baseline. Candidate 15 s capture is valid.

## Exact empty-donor culling

Page publication now skips display-prior searches when no receiver page can
exist within the donor algorithm's maximum three-cell coordinate reach.
Existing local receivers and all reachable border pages use the original
connectivity/measurement checks. This does not change history, seeds, rays or
work budgets. The search-bound test enumerates all six donor searches; page
tests cover boundaries, negative coordinates and existing local data.

Paired CPU test, M4/dev, `c18380b` plus this change, 1,024 newly loaded floor
faces beside a measured page, 40 alternating rounds (eight warmup): exact RGB,
sky and confidence parity; hash lookups 41,292→2,856; seed-stage median
233.625→33.000 µs (7.08×). This is **not a whole-frame or settling speedup**.
Both new regular tests, all history tests and the ignored parity/benchmark
passed. Evidence: `target/settling-donor-{tests,history,bench}.log`.

## Larger batches — not promoted

Equal 32,768-path native batches, M4/dev, five rotating modes, 48 rounds/eight
warmup (`12f887a` plus test-only study): cave GPU medians for 16/8, 32/8,
64/8, 32/16, 64/16 paths/lanes are 4.241/4.503/4.836/4.172/4.495 ms;
worker medians 6.670/6.606/6.627/6.500/6.578 ms. Ceiling and open sky are
also approximately tied. Larger batches sacrifice receiver refresh frequency
without a compelling throughput gain. Keep the 16-path default and limit.
The native study passed; log `target/settling-batch-study.log`.

## Cardinal light-node importance

`e8f2fd0` plus the shader reduction replaces six conditional normal projections
with signed-axis vector operations. It computes the same maximum cosine bound;
no proposal, PDF, path, history or work-budget change is intended. Absent normals
contribute -1 before adding the bounded angular spread.

M4/Metal/dev, 2,048 cave receivers × 16 paths, 64 alternating rounds (eight warm):
baseline/candidate GPU medians 4.362/3.665 ms; confirmation 3.075/2.509 ms.
Clock-dependent absolute times differ, but both pairs save 16–18% kernel time.
All five means/moments/direct values match at 1/7/16/64/256 paths; the separate
3,872-choice native selection/forward/reverse-PDF CPU oracle has zero differing
choices. Both native tests passed. This is not a whole-frame or settling claim.
Logs: `target/settling-light-importance.log` and
`target/settling-exact-importance-{confirm,pdf}.log`.

## Private-cache integration and facing-priority qualification

M4/Metal/dev, 2560×1440 headless, background streaming, preferred corner look,
16 paths and 4 ms transport target throughout. Every cave run below has valid
captures and its own disjoint native corner reference. App intervals are not FPS.

| Runtime / run | Sustained 10% corner error median / p95 | Within 10% at 15 s | Positive peak p95 / max |
| --- | --- | --- | --- |
| `c18380b` / `cave-1791847590335826000-30282` | 11.68 / 17.26 s | 119/144 | .1027 / .3045 |
| `3e2d459` / `cave-1791849222156685000-31687` | 9.08 / 18.27 s | 124/144 | .1044 / .3043 |
| `e3d1d42` / `cave-1791849592310651000-31991` | 8.47 / 13.65 s | 144/144 | .0975 / .2639 |

`3e2d459` is the private-cache synchronization candidate, not a different
estimator. First display publication advances from 11.81 to 9.08 s. Total paths
are approximately tied (375.13 versus 372.82 M). After 15 s excluding captures,
its offscreen interval p50/p95/p99/max is 20.93/27.00/30.00/127.89 ms; GI body
sum is 2.16/6.51/8.61/69.62 ms. The worst selection event is 63.63 ms. Thus the
isolated upload improvements are useful but do **not** establish stable frames.
A windowed streaming attempt (`streaming-1791849182476730000-31621`) correctly
rejected its unfocused black capture: no interactive 60 FPS evidence.

The separate `codex/gi-facing-refinement` branch prioritizes local receivers
with at least one camera-facing face, while tracing every due face once selected
and retaining global/coverage/urgent reservations. Although its early cave
accuracy is better, corner temporal variation p95 at 40–60 s rises to .1454
versus .0683 for `c18380b`; nearby refresh-age p95 becomes 36.81 s because hidden
faces lose opportunistic work. It is **not promoted** as a general default.
All 11 scheduling and 25 runtime-policy tests, Clippy and dev build passed.

Paired 30 s headless/background `response` runs (14 valid captures each):
facing `response-1791849801774352000-32121`, diagnostic-only `febcaf4` control
`response-1791850047067551000-32354` in the temporal-baseline worktree. Two
floor receivers' last red-light residual above 10% after emitter removal improves
2.719→1.774 s (1.53×), normalized using pre-add and pre-remove display means.
This is an event-response proxy, **not an independent accuracy reference**.
An earlier control launch failed the compiled-root guard and is excluded.
Logs/analyses use the `target/settling-private-*`, `settling-facing-*` and
`settling-control-response-verified.log` prefixes. No 2× overall claim follows.

## Exact-kernel integration and time-bounded catch-up

Same settings/hardware as above. `59d721c` ran clean at launch as
`cave-1791850540212693000-32968`; later documentation/test-only source edits were
not compiled into that process. All five captures are valid. Corner settling
median/p95 9.28/16.31 s; 128/144 within 10% at 15 s, p95 error .1296; peak
p95/max .1021/.3044. 386.95 M paths. Variation p95 at 40–60 and 160–180 s is
.07388/.05911, with at least 176/128 new paths. After 15 s excluding captures,
offscreen interval p50/p95/p99/max 20.88/26.88/30.22/89.29 ms, GI body
2.26/6.96/9.41/47.89 ms. First publication 9.28 s, replacement 6.55 s.

The isolated `0ebc3c8` experiment removes only the 8,192-record catch-up ceiling,
retaining the 1.5 ms deadline checked every 512 records. It ran clean at launch as
`cave-1791850901451970000-33399` (test-only study edits later, not compiled).
All five captures valid. Median/p95 9.52/15.87 s; 138/144 within tolerance at
15 s, p95 error .08886; peak .09887/.30451. 451.62 M paths. Variation
.09264/.06343, minimum fresh paths 192/96. Offscreen interval
20.54/27.31/31.19/68.93 ms, GI body 2.33/6.99/9.17/32.21 ms. First publication
9.68 s is not better; replacement 4.69 s is faster. Run-to-run throughput/temporal
differences exceed the narrow change's expected effect: do not attribute all of
them to removing the cap. This preserves a buildable experimental checkpoint,
not a demonstrated 2× result. Logs/analyses: `target/settling-{exact,time-budget}-*`.

## Power-MIS pilot — not promoted

Test-only study on `0ebc3c8` compares balance with exponent-two power weighting
for paired light/cosine techniques. The terminal one-sample mixture retains
balance, following [PBRT's MIS discussion](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency).
No history or presentation change. Four disjoint live sequences, 16-path batches
through 2,048 paths; analytic two-emitter truth or disjoint 65,536-path baseline
references. Candidate high-sample results are checked separately, never used to
move the target. Native references are split into bounded 1,024-path submissions.

On M4/Metal/dev, analytic emitter p95 error at 256 paths improves .00541→.00176;
peak p95/max .05328/.21219→.01547/.03240 at effectively equal GPU cost.
Both nonemissive skylight fixtures are identical. However, the many-emitter cave
barely improves at 256 paths (.19403→.19188) and early peak p95/max worsens
.55505/1.08611→.56092/1.11080. No general/cave win is established, so the runtime
estimator remains unchanged. Ignored native study passed;
`target/settling-power-mis.log`. This also refactors the previous roulette study's
harness; its historical numbers above are from the earlier harness, not a rerun.

## Conservative light support pilot — not promoted

`settling_light_support_study`, test-only diff on `28624e0`, uses the axis-aligned
receiver plane and emitting normal mask to exclude a child whose entire bounds
cannot connect. No energy threshold, visibility assumption or band-specific
rejection; other branches retain the power mixture. This is a restricted voxel
case of orientation-aware [many-light bounds](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf),
not a port of adaptive tree splitting.

Same fixed-work native fixtures and independent references as the power-MIS
pilot. Analytic CDF emitters and both skylight fixtures are unchanged. Cave
256-path p95 error only improves .19403→.19118; peak p95/max improves
.55505/1.08611→.53581/1.04392, but GPU batch median rises .74079→.77796 ms.
The extra bound tests do not pay for themselves here. Native study passed;
`target/settling-light-support.log`. Keep runtime proposals unchanged.

## Valid baseline repeat

Diagnostic-only `5c2d333` (runtime `febcaf4`), baseline worktree,
`cave-1791851270749646000-33719`: compiled-root guard passed and all five captures
are valid; the 15 s image was inspected. This replaces the earlier rejected
black-startup comparison, not its historical record. Same M4/dev/background/
2560×1440/default settings. Independent corner reference disagreement p95 .472%.

Sustained corner error median/p95 12.80/21.24 s, all 144 eventually within 10%;
at 15 s 103/144 within tolerance, p95 error .4293. Peak p95/max .10205/.30430.
Variation at 40–60/160–180 s .09921/.05821, min new paths 224/128.
343.42 M paths. After 15 s excluding captures, offscreen interval
21.81/26.39/30.08/89.03 ms and GI body 2.29/6.73/8.76/17.51 ms.
The candidates substantially reduce 15-second error, but the sustained settling
times are still only about 1.3–1.4× better and occasional CPU tails remain.
Evidence: `target/settling-control-repeat-*`; no 2× settling/60 FPS claim.

## Single-copy cache construction

`fd971e2` plus the construction diff encodes initial cache mirrors using the
already-tested sparse byte layout. It removes the full typed staging array,
cloned hash-entry staging array and duplicate encoded payloads from the native
build. CPU-only fallback still needs separate Bevy asset payloads; their creation
now explicitly stays on the worker along with native preparation. Publication
continues to receive a fully owned, private generation.

M4/dev CPU-only encoding benchmark, 262,144 live records, 393,216 allocated probe
slots and 1,048,576 hash entries, 24 alternating rounds/four warm: median
20.412→14.705 ms (1.39× this stage). Exact mirror/layout parity includes empty
snapshots, spare slots, tombstones, negative coordinates and packed sky words.
At this size, the removed allocations comprise an 81.79 MB typed probe array,
16.78 MB hash staging and 98.57 MB duplicate encoded payloads; these are allocation
sizes, not a measured 197 MB peak-RSS reduction (lifetimes overlap differently).
No GPU resource, lighting statistic, storage precision or work budget changes.

Verification correction: `0e3e79e`'s message overstated the final checks. The
17 preexisting regular render tests, new encoding parity/study, native generation
test and Clippy passed **before** the worker-completion type changed. The next
test build found one stale mock worker returning `GpuCacheBuild` instead of
`PreparedGpuCache`; the committed checkpoint therefore did not compile its tests.
The immediate follow-up updates that mock and reruns the relevant checks. Keep
this failure explicit rather than rewriting the checkpoint's history.
The corrective follow-up passed all 18 regular render tests, three journal tests,
the native generation/material/submission test, strict all-target Clippy and the
development build. These reruns include the new worker completion type.
Logs: `target/settling-cache-encoding-{tests,queue,parity,study,native,clippy}.log`.
End-to-end startup and frame effects are not inferred from this stage benchmark.

## Continuous-distance ranking pilot — not promoted

Isolated branch `codex/gi-distance-refinement`, `e982427`, applies soft inverse
square distance to opportunistic refinement only; reserved global/coverage and
urgent work stay unchanged. Eleven scheduling and 24 runtime tests, strict
Clippy and dev build pass. Streamed long cave
`cave-1791848585119270000-31043` has valid 15 s imagery, 141/144 corners
sustained, median 11.31 s and p95 22.36 s. At 15 s error p95 .359 versus .178
for `c18380b`; positive peak p95 .1021/max .3045 is essentially unchanged.
40–60 s corner variation .07327 versus .06831. This pilot also includes exact
donor culling, so it is not a clean isolation of that change. There is no
compelling settling win: keep the distance policy off the retained branch.

## Private mapped-view batching

Opening a native mapped view for each sparse write is unnecessary: replacement
buffers are private and remain mapped through catch-up. Open each whole-buffer
view once per bounded synchronization batch instead. All views still end before
installation/unmapping. The CPU fallback, atomic publication, buffer generation
and exact bytes are unchanged.

M4/dev, `4398697` plus this change: 512 scattered 208-byte writes, 48 alternating
rounds/eight warmup, exact byte checks after each mode. Per-range versus
per-batch median 30.750→1.875 µs. This is a **view-management microbenchmark**,
not a 16× engine claim. The native generation/material test passes with mapped
byte parity, growth/shrink/equal-size replacement, actual post-unmap submission
and CPU fallback. Logs `target/settling-private-{views,native}.log`.

Private catch-up now updates its own indices, mirrors and initial allocation
in place. Unlike live publication, it needs no rollback-safe preflight maps,
dirty upload sets or sorted/coalesced driver transactions. Removals precede
additions, so released slots remain reusable. Capacity/hash exhaustion poisons
the private generation; installation rejects it and the caller retains the
live generation while rebuilding. Live transactions are unchanged.

M4/dev, `c3be7bf` plus in-place change: paired native 512 changed values plus
32 additions/removals, 48 alternating rounds/eight warmup, 8,192 initial probes.
Previous transactional route (already using one mapped view per batch) versus
in-place: 79.500→42.958 µs median (1.85×). Every round compares slot/hash maps,
free slots, high water, all mirror and mapped bytes. Regular edit/duplicate/
exhaustion parity, all 17 regular render tests, native generation/material
lifetime, strict Clippy and dev build pass. Logs
`target/settling-private-inplace-{tests,study,render-tests,native,clippy,build}.log`.
End-to-end timing remains separate; the 1.5-ms catch-up budget is unchanged.
