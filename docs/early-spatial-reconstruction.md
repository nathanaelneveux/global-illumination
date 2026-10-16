# Early spatial reconstruction — October 15, 2026

## Purpose and controls

Reduce spatial noise early without changing the qualified temporal response,
history-retention curve or physical transport. Preserve the independent,
RGB-prioritized resident baseline at `codex/gi-spatial-baseline` (`c0e9be1`).
BVW, three-bounce/768-voxel transport, 75-ms temporal response, shared-corner sun,
corner-reconstructed indirect light, textures and baked-AO-off remain unchanged.

Three separate controls permit isolation, not three new defaults:

```sh
VOXEL_GI_SETTINGS=config/gi-adaptive-spatial.json cargo run
VOXEL_GI_SETTINGS=config/gi-visible-refinement.json cargo run
VOXEL_GI_BACKEND=reuse-wide cargo run
```

Both settings default off. `reuse-wide` is separate from the existing optional
paired `reuse` backend. The settings file uses normal default-filled settings,
so the other experiment remains disabled. No automatic source classification
or sunlit-face completion rule is added.

## Separation of responsibilities

1. **Physical statistics:** five-band means, moments and independent path
   sequence remain authoritative. Simulation continues receiving these values.
2. **Temporal RGB:** the existing nonspatial response uses those statistics and
   its own prior. Spatial output never feeds back into this history.
3. **Spatial RGB:** an optional bounded resolve combines compatible neighboring
   temporal values. The renderer reads this result; direct sunlight is still
   reconstructed separately from shared corners.
4. **Scheduling:** an independent opt-in policy ranks physical measurement
   uncertainty relative to total visible light. It does not consume the spatial
   result, inflate sample counts, mark faces settled, or bypass urgent edits.

The extra spatial output fits spare words in the existing 128-byte resident
record. CPU readbacks deliberately retain **pre-spatial temporal RGB**. Their
temporal traces cannot measure the final spatial presentation; use captures.
There is no screen-space history, TAA, debounce, corner transport or change to
IR/radiation values.

## Connected-surface reconstruction (`8f9eff4`)

Each selected resident face gathers four connected coplanar axial walks, each
up to four voxels long, and four immediate diagonals requiring an edge
connection. Missing faces stop a walk. Stable slot/incarnation pairs reject
retired donors. No normal-crossing or through-wall gather is permitted.

For each RGB channel, estimate standard error as
`sigma = sqrt(max(second_moment - mean², 0) / max(history, 1))`.
With `noise = max_channels(sigma / (.002 + abs(mean)))`, blend strength is
`strength * noise² / (noise² + .15²)`. Donor weights combine distance, a
variance-aware RGB compatibility test and donor history support. This is a
heuristic reconstruction, not a confidence interval: path correlation and
rare unseen sources can make observed variance misleading. Borrowing is
nonrecursive and never manufactures independent samples.

This is a world-space adaptation of variance-guided reconstruction ideas, not
a full [SVGF implementation](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/).
SVGF's hierarchical image-space filter is not implemented here. Our fixed
20-donor stencil is deliberately bounded; topology and raw/temporal separation
are more important than matching a screen-space implementation literally.

### Identical-prefix image evidence

Native `4c24108`, analysis `778f16b`; Apple M4/Metal, 16 GiB, development profile,
2560×1440 offscreen, RGB importance one/nonvisual .125. No concurrent compiler
or GPU workload. Analysis-only files were dirty during the capture run.
`cave-1792088574404361000-25554` contains 61 valid captures, complete geometry
audit, clean rendering and a qualified independent equal-band reference:
2×131072 paths, replica RGB p99 agreement .008163, all eight cosine checks pass.

Both reconstructed and unfiltered frames use **identical raw sample prefixes**,
four disjoint sequence windows and the ordinary material renderer. This isolates
spatial presentation; it does not include live neighborhood availability,
temporal response or adaptive scheduling. Worst across four windows:

| Paths | Unfiltered FLIP mean | Spatial FLIP mean | Unfiltered worst tile | Spatial worst tile |
| ---: | ---: | ---: | ---: | ---: |
| 16 | .161293 | .115862 | .509817 | .397465 |
| 64 | .096353 | .074893 | .293054 | .294684 |
| 256 | .054326 | .048520 | .174747 | .159564 |
| 512 | .042425 | .039253 | .120657 | .107136 |
| 1024 | .031226 | .030340 | .086496 | .081660 |
| 2048 | .024272 | .023629 | .067606 | .064411 |
| 4096 | .019053 | .018893 | .057628 | .055603 |

Mean error improves 28% at 16 and 22% at 64. The worst local tile is slightly
worse at 64. **Both first pass the unchanged user-calibrated FLIP gate at the tested 2048 checkpoint**,
not a demonstrated reduction in the required settled sample count. Coarse
64-pixel OKLab residual RMS improves .010132→.007635 at 16 and
.005330→.003850 at 64. This checks broad incorrect patches as well as grain;
smoothness alone is not accuracy. Fine one-pixel roughness is texture-dominated.

Reproduce the analysis with `tools/spatial_quality.py RUN`. Dependencies are
NumPy, Pillow and the official FLIP evaluator. Detailed results remain ignored
at `target/spatial-quality.json`.

Before trying stronger blending, `tools/spatial_field_sweep.py` reuses these
saved means/moments. Scales .075, .05 and .025 keep borrowing longer than .15.
They increase mid/late corner error: at 1024, worst-window corner p95 is .02636
at .15, .03189 at .075 and .04176 at .05, versus .02803 unfiltered. At 256,
.05 increases error to .08577 versus .06148 unfiltered. These are field errors,
not image FLIP, but they reject a simple "blend harder for longer" promotion
without another expensive render sweep. Smooth incorrect gradients remain a
real risk. Topology/constant/missing-donor tests cover the screening tool.

## Total-light-aware refinement (`84c0558`)

The host supplies `VoxelGiVisualWeights`, converting normalized direct and
indirect energy into the material's shading units. The demo accounts for camera
exposure and GI gain. The transport scheduler has no dependency on a camera or
specific material implementation.

Rank by RGB standard error divided by total visible direct+indirect magnitude
plus a floor, taking the maximum channel. Thus a strong red direct contribution
does not mask uncertain blue. Stale direct information is not trusted. Nonvisual
relative error retains positive weight using the sampling-importance setting;
simulation is not disconnected. Coverage, edit urgency, stability checks,
age/fairness rules and the all-loaded-world sweep remain unchanged.

The direct estimate is face-centered; it is not the exact corner shading or
post-tonemap derivative. This is an experimental ranking heuristic, not a
perceptual guarantee. Strong direct light can make an indirect error less visible,
but a binary "sunlit means complete" policy would discard important cases.

### Preliminary background-streaming cave runs

Native `84c0558`, same hardware/profile/view above, normal world residency and
background terrain loading. Later uncompiled wide-sharing source edits were
present during the candidate runs; the executable and loaded presentation
shader were unchanged. No concurrent compiler or GPU workload. All three runs
completed with valid assets/captures and clean render logs.

| Variant | Paths, M | Worker p95, ms | Offscreen frame p95 / p99, ms | Near / far age p95, ms |
| --- | ---: | ---: | ---: | ---: |
| Control | 186.36 | 2.018 | 25.122 / 29.246 | 1185 / 17845 |
| Spatial only | 177.14 | 4.419 | 24.101 / 30.780 | 814 / 19348 |
| Visible ranking only | 183.15 | 2.196 | 24.191 / 28.785 | 826 / 17424 |

These are single preliminary runs, **not interactive FPS or a 60-FPS pass**.
The spatial worker builds/uploads 176 bytes per selected face, with up to 20
stable-index lookups; its ~5% transport loss needs to be weighed against image
quality. No extra persistent per-face allocation is required.

At 15s, FLIP mean versus the frozen reference is .121389 control, .120539
spatial, .123962 visible ranking; at 60s it is .116650, .116197, .116577.
The reference run uses additional chamber residency, so these are comparative
errors, **not a certified live-scene convergence test**. The large shared
residual must not be attributed to these candidates. Small differences do not
establish an end-to-end win. Spatial gains are strongest at the very early
prefixes; visible ranking needs more evidence before promotion.

Runs: `cave-1792089195239319000-26085`, `cave-1792089307410473000-26176`,
`cave-1792089370019437000-26175`. Analysis: `target/spatial-runtime-quality.json`.

### House opening edits

Native `0755465`, documentation-only dirty files, same offscreen settings,
normal background terrain. Both 180-second `house-edits` runs completed with
nine valid captures, all three openings applied, and clean logs. Control:
`house-edits-1792090001202260000-26667`; spatial:
`house-edits-1792090182323994000-26762`. Transport is 420.99M versus 401.35M paths
(4.7% fewer); skylight-phase worker p95 is 2.139 versus 4.065 ms. Offscreen frame
p95 is 24.341 versus 24.191 ms, p99 27.375 versus 28.846 ms: mixed tails, not an
interactive frame-rate claim.

Using each control 29-second image as a **shared observed tail**, not certified
truth, 400-ms FLIP mean is .05078→.05814 for skylight, .03277→.03988 for west
door, and .03024→.02632 for east door plus turn. The candidate/control tails
also differ (.01855–.02485 mean FLIP). This does not establish either a general
edit-response win or a statistically isolated regression; it prevents promoting
the static-prefix result as a proven gameplay win. The temporal diagnostics
cannot substitute for these final spatial captures.

**Decision:** retain spatial reconstruction as a useful early-image experiment,
but leave it opt-in. It improves very low-sample appearance without proving
faster settling or uniformly better edits. Visible ranking likewise remains
opt-in with no demonstrated gameplay quality win. Optimize/requalify before
changing stable defaults; do not hide these tradeoffs behind sample coverage.

## Adaptive wider fresh sharing (`3131623`, study `0e3b7e0`)

Up to four coplanar receivers share fresh suffixes below 256 history, two below
1024, and mature faces return to the ordinary independent kernel. Prefer donors
three/six voxels away, then one. All proposal densities and receiver-specific
first-hit emitter MIS remain balanced; primary next-event estimation stays
local. No energy clamp, stale-path reuse or history reset is added.

Mixed groups and independent mature faces dispatch separately, preserving the
normal independent lane layout. Two-member groups still use the four-member
kernel: this is a known efficiency limitation, not evidence against all adaptive
sharing. The history cutoffs are explicit experiment parameters, not analytical
convergence thresholds.

Native Apple M4/Metal study: four disjoint windows, identical .125 importance,
16 paths/dispatch, 65536-path independent reference. Three 400-face fixtures
(open sky, black emitter room, colored skylight room) plus 2935 dense cave faces.
GPU times below are median cumulative compute timestamps; corner errors are
worst-window p95 relative RGB, **not FLIP**:

| Cave paths | Independent error / ms | Paired error / ms | Wider error / ms |
| ---: | ---: | ---: | ---: |
| 64 | .129976 / 19.28 | .109567 / 21.54 | .103150 / 25.43 |
| 256 | .059847 / 79.19 | .049032 / 95.83 | .046104 / 114.65 |
| 1024 | .027733 / 338.63 | .022476 / 387.76 | .021899 / 525.61 |
| 2048 | .019143 / 662.18 | .015004 / 770.27 | .017260 / 849.92 |

At 64 and 256 the wider proposal lowers cave error ~6% versus pairs but costs
~18–20% more. At 2048 it costs more and is less accurate than retaining pairs.
The other fixtures do not establish a universal cost-adjusted win either.
Do not promote this candidate. Exact mature-path parity, mixed groups and
partial sample/workgroup tails pass; detailed five-band drift and costs are in
`target/wide-quality.log`, with native parity in `target/wide-parity.log`.

This is **not ReSTIR PT**. [Compatibility-guided selection](https://research.nvidia.com/labs/rtr/publication/junkins2026compatibility/)
studies the error/correlation tradeoff; geometric distance alone does not solve
it. [Stochastic pairwise MIS](https://research.nvidia.com/labs/rtr/publication/hedstrom2026stochastic/)
targets inexpensive selection from many contributing neighbors and is a
promising different architecture, not implemented or qualified by this run.
NAADF remains the shared visibility/traversal accelerator, not the reconstruction
or reuse-weighting method.

## Verification boundaries

Spatial CPU/GPU oracle tests preserve raw/temporal snapshots exactly, constants,
gap topology, partial batches and disabling. Native resident ownership parity
still covers growth, retirement/recycling, caps and model restarts. The visual
ranking tests cover colored/stale direct light and positive nonvisual support.
At the integration checkpoint, 260 regular Rust tests and strict all-target
Clippy pass, including real Bevy PBR shader composition; 73 native/research
tests remain explicitly ignored rather than falsely reported as run. Four
Python analysis tests pass. No routine toggle stress was run. Failed candidates remain local branches and
do not replace the qualified baseline merely because their implementation runs.

## Target-evidence display footprint (`159e37e`)

`VOXEL_GI_SETTINGS=config/gi-display-evidence.json` selects a target of 512
weighted samples for presentation. `spatial_evidence_target: 0` retains the
fixed-neighbor experiment; `spatial_reconstruction: 0` disables both. These
controls do not change physical accumulation, history retention, direct sun,
simulation snapshots, or transport sampling. Stable defaults remain unchanged.

The worker walks unique, connected coplanar faces in breadth-first order. It
cannot jump a gap or borrow around a solid corner. Radius is bounded at 16
voxels and candidate count at 1024; the receiver's predicted retained history
limits work to at most twice `ceil(target/history)` candidates. Mature receivers
skip neighborhood lookup. The GPU reads current retained counts and the same
raw statistics/temporal RGB used by the fixed filter, not filtered neighbors.

For donor history `n` and geometry/radiance compatibility `g`, its weight is
`w = n*g`. Approximate support is `(sum w)^2 / sum(w*w/n)`, including the local
estimate. The spatial Gaussian widens with `target/local_history`. Borrowing
stops when support reaches the target; a partial final donor makes its influence
vanish continuously as local history approaches 512. Missing or incompatible
donors can prevent reaching the target. Capped histories and correlated surface
lighting make this a **footprint heuristic**, not 512 independent rays or an
accuracy certificate. At 512 local history the output is ordinary temporal RGB.

GPU results occupy existing presentation words in the 128-byte resident record.
The variable neighborhood uses a 16-byte header per selected face plus 16 bytes
per candidate, in reused worker/GPU scratch. It adds no persistent per-face
measurement state. A bounded gather can still be expensive during cold start;
runtime worker cost and transport freshness must be qualified, not inferred
from the unchanged ray count.

The native oracle covers unequal histories (1, 16, 128, 511, 2048), subsequent
edit caps, partial jobs, and disabling. Physical five-band means, moments,
counts, sequence, temporal state and CPU snapshots match the independent control
exactly. This establishes separation, not edit-image quality.

Matched image protocol (`0f59dd5`): add
`VOXEL_GI_VALIDATE_EVIDENCE_SWEEP=1` to the headless cave reference command.
It captures raw, fixed-neighbor and target-evidence presentation from the same
1–2048 path prefixes in four sequence windows. `cave-evidence-support.json`
records support/footprint separately from the FLIP gate. Use
`tools/spatial_quality.py --labels samples spatial evidence` with the desired
`--counts`; `tools/cave_sample_contact_sheet.py` accepts the same label/count
selection for matched visual rows. Real edit checks use `house-edits` and
`response`, with background loading and the production distance/history curve.
`tools/spatial_edit_quality.py CONTROL CANDIDATE` compares final captures to
both a common observed tail and each run's own tail. Neither is certified truth;
tail disagreement must accompany response measurements.

### Qualification, 2026-10-15

Native `0f59dd5` (algorithm `159e37e`), analysis `35827a4`, Apple M4/Metal,
16 GiB, Rust 1.95, development profile, 2560×1440 offscreen, PPD 67. Preferred
corners/textures/no baked AO, RGB sampling importance 1/nonvisual .125, 16 paths
per dispatch, 3 bounces, reach 768. No path reuse or visible-refinement ranking.
Only documentation/analysis files changed while the native runs were active.
These timings are **not windowed FPS**. Production retention stayed
`whole-range-power4-scale12-v1`; no distance or history thresholds changed.

Frozen cave: `cave-1792092971027649000-28601`. All 149 reference/prefix captures
completed with valid targets, nonblank images, verified assets and clean logs.
Both 131072-path replicas and eight cosine checks qualified (RGB replica p99
.008163, maximum .015107); the geometry audit has no unknown/mismatched voxels.
Four-window worst results, fixed 20-neighbor filter versus target-512:

| Local paths | Fixed FLIP mean | Target FLIP mean | Fixed worst tile | Target worst tile |
|---:|---:|---:|---:|---:|
| 2 | .265068 | .170434 | .736870 | .733848 |
| 4 | .194065 | .144360 | .553852 | .649679 |
| 8 | .150120 | .127865 | .402353 | .560265 |
| 16 | .115862 | .105786 | .397465 | .458664 |
| 64 | .074893 | .077138 | .294684 | .262844 |
| 128 | .062834 | .067199 | .231214 | .246388 |
| 256 | .048520 | .055659 | .159564 | .172156 |
| 512 | .039253 | .042425 | .107136 | .120657 |
| 2048 | .023629 | .024272 | .064411 | .067606 |

The target clearly reduces very early average error and broad residual energy:
at 16, 64-pixel OKLab residual RMS .007635→.006958. But it is not uniformly more
accurate: at 16 the worst tile is 15% worse, and from 64 onward average error
is worse than the fixed filter. One inspected worst target tile at 16/window 3
is near the green ceiling lighting (pixel 512,32); its green component is
underestimated, not merely noisy. At one path there is essentially no benefit:
zero empirical variance cannot reliably guide compatibility. Raw, fixed and
target all first pass the calibrated sustained image gate at the tested **2048**
checkpoint; this sweep did not include 1280 or 1536.

Average borrowing across the cave is approximately 132 faces at 4 local paths,
32.4 at 16, 3.94 at 128, 1.98 at 256, and zero at 512. At 16 about 95% of faces
reach the heuristic support goal; isolated/incompatible surfaces need not do
so. At one path none reach it. Full metrics: `target/evidence-image-quality.json`
and the run's `cave-evidence-support.json`. The matched overview is
`samples-spatial-evidence-overview-window-0.png` in that run directory.

#### Actual edits with background terrain

Fixed/target house pair: `house-edits-1792093458682517000-28932` /
`house-edits-1792093639640916000-29108`. Both 180-second runs applied all seven
solid removals across the three openings, captured nine valid images and logged
no render errors. Relative to the **same control 29-second observed tail**:

| Edit, checkpoint | Fixed FLIP mean | Target FLIP mean |
|---|---:|---:|
| Skylight, 400 ms | .074187 | .062518 |
| Skylight, 10 s | .023861 | .027709 |
| West door, 400 ms | .028103 | .049834 |
| East door + turn, 400 ms | .028448 | .050581 |
| East door + turn, 10 s | .023985 | .040932 |

Own-tail measurements retain the direction of the early door weakness.
Cross-run tail FLIP means are .019574/.020081/.025872, so this is not a certified
error/time-to-settle measurement or a perfectly isolated filter comparison.
Transport was 389.69M→385.91M paths (-1.0%, one pair); startup worker p95
4.974→7.054 ms, skylight 3.959→4.243 ms. Offscreen frame p99 mixed by phase
(startup 29.084→29.677 ms; skylight 28.666→27.259 ms). No FPS win is claimed.

Emitter pair: `response-1792093820455383000-29217` /
`response-1792093850887483000-28931`, 30 seconds each, 14 valid captures each,
cavity/add/remove events verified, clean logs. At one second, common-tail FLIP
means were .034884→.044356 for the cavity, .231545→.332383 for emitter addition,
and .260820→.364451 for removal. Own-tail measurements show the same weaknesses;
cross-run tail differences are .021127/.047856/.033760. Transport fell
77.93M→69.78M paths (-10.5%); emitter-added worker p95 3.894→5.371 ms.
The extra gather cost and differing scheduling confound the display bias; do
not attribute the entire response difference to stale-donor weighting alone.
Results: `target/evidence-house-quality.json`, `target/evidence-response-quality.json`.

Reverse-order repeat (target then fixed): target
`response-1792094265294212000-29450`, fixed
`response-1792094295805574000-29449`; same native binary/settings, valid captures,
events and clean logs. At one second, fixed→target common-tail FLIP was
.032147→.052019 (cavity), .221747→.179750 (emitter addition),
.322415→.267075 (removal). Own-tail comparisons agree with those directions.
Thus the emitter response ranking **reverses**, not a reproducible regression
or win. Tail disagreements remain .024958/.050520/.035189. The cost repeats:
78.94M→71.01M paths (-10.0%), added-phase worker p95 4.092→5.585 ms.
Repeat analysis: `target/evidence-response-repeat-quality.json`.

**Decision:** keep target evidence as an opt-in early-smoothing experiment,
not a new default and not a transport-sampling win. The user-proposed footprint
rule produces the intended wider early/narrower late behavior and attractive
early images, but does not improve the settled sample threshold, worsens some
local lighting detail, and costs throughput. Real edits do not establish a
consistent response win. Do not retune the validated distance/history equation
to hide these presentation limitations. Before promotion, reduce repeated
neighborhood lookup/upload cost and qualify donor freshness/real lighting
gradients explicitly. In particular, a donor's retained count is not proof
that it has already measured the latest edit. Its contribution must not be
mistaken for new evidence at the receiver. Current measurements do not isolate
that possible source of bias from scheduling effects.

All 263 regular Rust tests and strict all-target Clippy passed; 74 tests remain
explicitly ignored. Targeted native fixed/adaptive display oracles, resident
growth/retirement/parity and unselected-face cap ownership passed separately.
Four Python analysis tests passed and actual final captures were inspected.

## Interpreting weights, confidence and quality targets

These are separate quantities; do not label all of them "samples":

- **Path sequence** records actual tracing progress. It keeps advancing across
  edits and is not increased by spatial reconstruction.
- **Retained history weight** controls the influence of new measurements on the
  physical mean. It initially follows the sample count, but edit retention and
  the history ceiling make it different from lifetime rays traced.
- **Display confidence** is a smoothed reliability heuristic, not a confidence
  interval. Its target is `clamp(n/(n+8)/(1+uncertainty), .125, 1)`, where
  uncertainty is derived from RGB moments and retained weight. Ordinary corner
  reconstruction weights its connected face values by this scalar.
- **Display evidence target** sizes the optional wider spatial footprint. It
  combines donor history, distance and variance-aware radiance compatibility.
  It does not multiply by the display-confidence scalar or raise that scalar
  after borrowing. Estimated variance therefore influences sharing, but there
  is no independent statistical-confidence certificate behind the target.

In the matched prefix images, confidence is deliberately equal for all faces
to isolate the spatial filter. Real edit captures use actual runtime confidence
and uneven retained histories. Both are needed; a static prefix is not evidence
of response to edits.

The four-path target-512 image borrows from approximately 132 neighboring faces
on average, not four rays total. At 128 paths it borrows from about four. The
rapidly narrowing footprint can make the early image look smoother even while
the later image is numerically closer to the reference. Preserve both visual
review and error measurements instead of equating smoothness with accuracy.

The earlier **1280-path** full-image qualification belongs to **RGB priority
plus paired path reuse**, documented in [resident path reuse](resident-path-reuse.md#qualified-full-images).
Only RGB priority is integrated by default; paired reuse remains opt-in. The
spatial experiments use independent RGB-prioritized paths. The calibrated FLIP
limits remain fixed when an estimator improves; 2048 is their original visual
calibration checkpoint, not a universal minimum or a new limit to recalibrate
for every candidate.

## Higher evidence targets: matched sweep and real edits (`1b327e6`)

The target sweep remains on `codex/gi-evidence-target-sweep`. The tested 512
baseline is preserved on `codex/gi-display-evidence` at `0d3de57`. No defaults,
transport estimator, retention equation, temporal response or BVW code changed.
New opt-in presets are `config/gi-display-evidence-{1024,2048,4096}.json`.

Add `VOXEL_GI_VALIDATE_EVIDENCE_TARGETS=1` to the headless cave reference command
with `VOXEL_GI_REFERENCE_PATHS=131072`. This renders targets 512/1024/2048/4096,
the fixed filter, and raw display from identical independent RGB-prioritized
prefixes in four windows. The 512 filename label remains `evidence`; larger
targets use `evidence1024`, `evidence2048`, `evidence4096`. New 1280 and 1536
checkpoints close the gap in the previous independent-path measurement.

Native source `1b327e6`, Apple M4/Metal, 16 GiB, Rust 1.95, development profile,
2560×1440 offscreen, PPD 67, preferred corners/textures/no baked AO, RGB priority
1/.125, 16 paths/dispatch, 3 bounces/reach 768. Only the documentation clarification
later committed as `15fb33c` was dirty during runs. No concurrent GPU test,
compiler or image analysis during timed edit runs. Timings are offscreen app
cadence, **not interactive FPS**. Native CPU/GPU reconstruction parity passes
all four targets with heterogeneous histories, edit caps and disabling; physical
and temporal statistics remain exactly unchanged. `cargo build` passes.

Frozen run `cave-1792099117908647000-31496` completed all 245 reference/sweep
captures plus the three initial gameplay captures. All targets are valid and
nonblank; assets and logs are clean. Geometry unknown/mismatch counts are zero.
The two 131072-path replicas qualify (RGB p99 .008163, maximum .015107) and all
eight independent cosine checks pass. An earlier preflight run
`cave-1792099088641452000-31449` was stopped before baking to correct the reference
path-count environment variable; it is not evidence.

Worst-window FLIP means, with the same fixed reference and limits:

| Local paths | Target 512 | Target 1024 | Target 2048 | Target 4096 |
|---:|---:|---:|---:|---:|
| 4 | .144360 | .145792 | .158555 | .166316 |
| 16 | .105786 | .108015 | .115122 | .126041 |
| 128 | .067200 | .063669 | .062571 | .065743 |
| 512 | .042424 | .045224 | .041212 | .040597 |
| 1536 | .026919 | .026919 | .029959 | .027672 |
| 2048 | .024274 | .024274 | .024274 | .029901 |

Higher targets visibly keep walls smoother at 128; the 2048 target improves
average FLIP there by 6.9%. However its worst tile worsens .246388→.338659,
and broad 64-pixel residual RMS rises .003533→.003625. At 4096, that RMS rises
to .004481. The lost detail includes local green-ceiling illumination and warm
emitter halos; smoother broad color is not necessarily more accurate lighting.
This is a visual tradeoff, not grounds to dismiss the user's preference for
reduced spatial noise.

Unfiltered RGB-prioritized independent paths at 1280 give mean/tile
.028932/.084217 (both fail). At 1536 they give .026919/.075842 (mean passes,
local patch fails the .072964 limit). At 2048 they give .024274/.067599 and
pass both. Targets 512/1024/2048 all first pass at 2048; target 4096 has no pass
through 2048, with mean/tile .029901/.106808 there. No new sampler minimum or
lower perceptual threshold is inferred from smoothness.

At 128, targets 512/1024/2048/4096 borrow approximately 4/8/16/33 faces. At 16
they borrow 32/65/130/252. Thus target 2048 with a normal 16-path runtime batch
closely matches the footprint of target 512 at four paths. Radius/work bounds
still apply: at four paths, no face reaches the nominal 4096 target, and only
47% reach 2048. The support is heuristic, not independently traced evidence.

### Streamed edit screening

All four 30-second `response` runs completed 14 valid captures, actual cavity
removal/emitter add/removal, background terrain and clean logs. Production
distance/history retention stayed unchanged. Run identifiers, in execution order:

- 512: `response-1792099624655748000-31707`
- 4096: `response-1792099655111771000-31739`
- 1024: `response-1792099685576612000-31749`
- 2048: `response-1792099716098897000-31757`

| Target | Total paths | Emitter-added worker p95 | Cavity / add / remove FLIP at 1 s |
|---:|---:|---:|---|
| 512 | 77.55M | 5.43 ms | .042085 / .186251 / .288134 |
| 1024 | 73.10M | 7.45 ms | .083768 / .497574 / .372278 |
| 2048 | 63.04M | 13.20 ms | .104611 / .342952 / .323701 |
| 4096 | 53.78M | 22.89 ms | .120316 / .487730 / .448919 |

Errors use the same 512-run observed tail, **not certified truth**. Own-tail
errors retain all nine worse-at-one-second directions. Tail disagreements
(cavity/add/remove) are .040553/.068593/.035539 at 1024,
.055411/.080759/.042048 at 2048 and .078255/.115856/.057378 at 4096.
These are preliminary single-run screenings with varying scheduling; they do
not isolate stale-donor bias from reduced transport or prove universal edit
regressions. Total throughput falls 5.7%/18.7%/30.6%. Frame p99 is mixed by phase,
not an established frame-pacing improvement. No higher-target house or windowed
qualification was run: this screening already gives no promotion case.

**Decision:** retain all larger targets as opt-in visual comparisons. None is
an overall winner over 512 on the measured accuracy/cost/edit criteria, and
4096 is not a suitable default. Wider reconstruction remains visually valuable;
the next architectural target is cheaper neighborhood processing and better
preservation of real illumination gradients/freshness, not simply a larger
global support goal. Do not change the established retention curve or accuracy
gate to compensate. Targeted evidence is sufficient here; no routine stress or
full-suite rerun was needed for presets and this diagnostic-only extension.

Detailed results: `target/evidence-target-image-quality.json`,
`target/evidence-target-edit-quality.json`, and the frozen run's
`cave-evidence-support.json`. The combined image
`evidence-evidence1024-evidence2048-evidence4096-overview-window-0.png` in that run
shows 4/16/128 paths, with the reference in the final column.

## Preserving the 2048 look while reducing cost (`bfffadf`)

The user selected 2048 as the preferred visual target after reviewing the
matched images. Keep that target fixed while optimizing implementation cost;
do not substitute less smoothing and call it a speedup. The numerical match
with the original 2048-path visual calibration is not an equivalence: borrowed,
correlated display support is not 2048 independent measurements at a receiver.

Selected implementation branch: `codex/gi-evidence-topology`. Baseline
`963d0ca` is preserved on `codex/gi-evidence-target-sweep` (its native code is
unchanged from `1b327e6`). Two independently verified checkpoints:

- `ba3226d`: worker-owned, stable-slot coplanar adjacency replaces repeated
  world-position hash lookups. Each node stores four neighboring slot indices
  and its incarnation. Insertion/retirement/recycling maintain both endpoints;
  reseeding updates identity without rebuilding geometry. Only topology is
  cached, never radiance, donor lists or history-dependent decisions. The same
  bounded breadth-first walk preserves donor order and all weighting rules.
  Storage is 20 bytes per allocated slot plus vector spare capacity, only while
  adaptive reconstruction is enabled; disabling releases the topology.
- `bfffadf`: donor records shrink from 16 to 8 bytes, retaining full slot and
  incarnation identities. Coordinates modulo 64 fit in an existing spare
  resident metadata word. Since the gather radius is 16, the GPU recovers exact
  signed local distances across periodic boundaries. These tags are distance
  metadata, never identities. Resident records remain 128 bytes. Donor upload
  traffic is halved, not total GI traffic or total GI memory.

Transport, retention, temporal smoothing, compatibility, support target and
the partial-final-donor rule are unchanged. No BVW change. Spatial filtering
remains opt-in; use `VOXEL_GI_SETTINGS=config/gi-display-evidence-2048.json`.

### Cost qualification

Apple M4/Metal, 16 GiB, macOS 26.6.2, Rust 1.95, optimized development profile,
2560×1440 offscreen; preferred corner display, textures on, baked AO off,
independent RGB-prioritized paths, 16 paths/dispatch, 3 bounces/reach 768,
4 ms trace budget, spatial strength 1/target 2048. Both profiles include
background terrain. No concurrent compiler, GPU test or image analysis during
timed runs. All runs completed with valid nonblank captures, expected edit
events and clean render logs. These are **offscreen app-cadence measurements,
not windowed FPS or a 60-FPS qualification**.

Isolated CPU benchmark: 9216 planar faces, ten alternating measurements,
median gather time. Hash lookup → indexed adjacency is 32.409→22.711 ms at
history 16, 3.155→1.925 ms at 128, and .543→.288 ms at 512 (30–47% lower).
Building this topology took .243 ms and 184320 bytes. This is not a whole-frame
speedup and does not include the rest of transport.

Two full baseline/selected pairs were run in opposite order:

| Pair | Version | Cave paths | Startup worker p95 | Later worker p95 | Startup / later frame p99 | Response paths |
|---|---|---:|---:|---:|---:|---:|
| 1 | Baseline | 151.79M | 21.82 ms | 12.81 ms | 29.09 / 30.70 ms | 65.80M |
| 1 | Selected | 162.79M | 14.53 ms | 8.52 ms | 26.63 / 30.16 ms | 65.76M |
| 2 | Baseline | 153.84M | 22.75 ms | 12.66 ms | 29.74 / 29.76 ms | 71.13M |
| 2 | Selected | 158.55M | 13.13 ms | 8.30 ms | 27.79 / 29.55 ms | 64.07M |

Cave worker p95 falls 33–42%; throughput rises 7.2% and 3.1%. Startup near-field
age p95 improves 973→791 ms and 995→777 ms; later freshness is similar
(863→833 ms, 847→849 ms). Response throughput is unchanged in pair 1 but
**9.9% lower in pair 2**. Its per-phase frame-time and freshness results are
mixed. Do not generalize the cave gain to all workloads, dismiss the response
regression as proven measurement noise, or claim faster frames by ignoring
transport. The robust result is cheaper CPU preparation and smaller uploads.

Run IDs (suffixes include timestamp and process; all under `target/gi-validation/`):

| Version | Response | Cave |
|---|---|---|
| Baseline pair 1 | `response-1792107474745847000-34431` | `cave-1792107505189303000-34429` |
| Adjacency only | `response-1792107828685116000-34739` | `cave-1792107859098245000-34732` |
| Selected pair 1 | `response-1792108050081340000-34927` | `cave-1792108080500009000-34921` |
| Cooperative trial | `response-1792108373892550000-35165` | `cave-1792108404351211000-35159` |
| Selected pair 2 | `response-1792108593091946000-35285` | `cave-1792108623514375000-35309` |
| Baseline pair 2 | `response-1792108709871339000-35419` | `cave-1792108740278205000-35436` |

Baseline pair 1 used the already-built baseline executable while adjacency
source was being prepared; those source changes were not in that executable.
Later native runs correspond to the listed checkpoint commits. Final executable
was rebuilt from `bfffadf` after returning from the baseline repeat.

### Images, edit response and limits

Native CPU/GPU oracle checks preserve the original filter output within 1e-5
at all four targets, with heterogeneous histories, edit caps and wrap-crossing
coordinates. Physical means, moments, counts, sequences and temporal snapshots
remain unchanged. Thus the expensive frozen prefix sweep does not need repeating
for these exact-formula optimizations.

Live cave 15-second mean FLIP against the existing frozen reference improves
.131208→.121307 and .130913→.122640. Broad residual RMS improves
.010328→.008660 and .010209→.008826. At 30–60 seconds the compared images are
nearly identical in measured error (mean FLIP about .116–.117). These live
streaming worlds differ from the frozen reference's residency: this is a
comparative image check, **not a new certified convergence pass**. Candidate
15/60-second images and one-second emitter add/remove captures were inspected.

One-second edit FLIP, baseline→selected, against each pair's common observed
tail (not ground truth):

| Pair | Cavity | Emitter add | Emitter remove |
|---|---:|---:|---:|
| 1 | .073439→.087561 | .451351→.385253 | .357466→.324168 |
| 2 | .080694→.074624 | .388635→.389079 | .304026→.355465 |

Own-tail comparisons retain the cavity/removal direction reversals; addition
is effectively tied in pair 2. Cross-run tail disagreements are
.017038/.037697/.016762 in pair 1 and .020871/.023120/.023783 in pair 2.
No reproducible universal edit-settling improvement is established.

A further cooperative GPU trial (`2321005`, `codex/gi-evidence-cooperative`)
evaluates 32 donors in parallel but preserves the ordered final fold and
partial-donor rule. It passes parity/lifecycle checks; compared with selected
pair 1 it increases response paths 3.3% but reduces cave paths 4.0%. Keep it
separate, not selected: neither correctness nor one favorable profile proves
a cost win. The next isolated cost investigation should distinguish GPU
reconstruction from tracing/scheduling before altering the visual formula.

**Decision:** retain indexed adjacency and compact uploads as the selected
2048 implementation, with the response-throughput limitation above recorded.
Do not promote a universal FPS/settling claim or silently change global
defaults. Focused topology/spatial unit tests, native four-target/two-origin
oracle, native growth/retirement/recycling/disable/re-enable lifecycle checks,
strict all-target Clippy and build passed. No routine stress/full-suite rerun.

Detailed artifacts: `target/evidence-topology-bench.log`,
`target/evidence-packed-{native,lifecycle,clippy}.log`,
`target/evidence-optimized-cave-quality.json`,
`target/evidence-optimized-edit-quality.json`, and
`target/evidence-optimized-repeat-edit-quality.json`.

## Unfiltered cost baseline and a separate smoothness criterion

The earlier 3–7% cave throughput improvement compares optimized 2048 with
**unoptimized 2048**, not with spatial reconstruction disabled. Fresh short
screenings use the same `bfffadf` executable (HEAD `c61f397`, documentation-only
delta), identical settings except spatial off versus strength 1/target 2048,
and the same hardware/profile/background streaming/view as above. Cave order
was off→on; response order on→off. Analysis-only Python files were being
written during the final response run, but no compiler, image analysis or
other GPU test ran concurrently. Both pairs completed, with 3/14 valid
captures respectively, all response events and clean logs.

| Workload | Spatial off paths | Optimized 2048 paths | Throughput gap |
|---|---:|---:|---:|
| Cave, 62 s | 212.23M | 156.55M | -26.2% |
| Edit response, 30 s | 76.67M | 72.09M | -6.0% |

Cave startup/later worker p95 rises 3.00/2.25→13.31/8.32 ms; offscreen frame
p99 is 29.14/29.05→29.38/31.07 ms and near-age p95 700/780→790/853 ms.
Response worker p95 rises about 1.87–2.55→7.55–10.96 ms, with mixed frame tails
and freshness. These are preliminary single pairs, **not a stable universal
percentage, windowed FPS or visual-settling times**. The worker is asynchronous;
its time is not main-thread blocking time. Cave 15-second captures were
inspected. Runs:

- Off: `cave-1792113315861287000-37296`, `response-1792113608627091000-37499`.
- On: `cave-1792113429380722000-37397`, `response-1792113517307951000-37436`.

### Distinguish preferred early appearance from final fidelity

FLIP remains the existing settled-fidelity criterion. It measures perceived
difference when alternating against truth, not a user's preference between
two still-imperfect renderings ([FLIP authors' explanation](https://developer.nvidia.com/blog/?p=29284)).
The separate question is whether a still-converging wall looks spatially calm.
Our old full-frame one-pixel residual roughness is texture-dominated; coarse
residual RMS also includes broad bias and missing real emitter halos. Neither
alone explains the user's preference for wider smoothing.

`tools/spatial_noise_quality.py` adds a **diagnostic**, not a replacement
perceptual model. It measures reference-relative OKLab band energy in two fixed
smooth wall interiors, away from emitters, silhouettes and sunlight edges.
Four-pixel residual averaging suppresses material texels; a binomial
Laplacian-style pyramid then isolates bands at source-pixel spacings 4/8/16/32.
For each crop/window the score is `sqrt(sum(mean(||band||²)))`; take the maximum
across both crops and four sequence windows. Signed mean lightness error and
between-window dispersion are separate diagnostics. The latter is neither
temporal flicker nor a confidence interval. Multiscale perceptual models such
as [NLPD](https://arxiv.org/abs/1701.06641) motivate this separation, but this
simple RGB diagnostic is **not an implementation or validation of NLPD**.

The two regions were fixed before scoring candidates. On the existing frozen
four-window sweep, the preferred target-2048 envelope across local prefixes
4/16/64/128/256/512/1024/2048 has maximum **.00142691185** (at four paths).
`tools/quality_profiles/cave-spatial-noise-2048.json` records that as a
**provisional noise ceiling**, normalized to 1.0, including reference checksum,
view/crop coordinates and calibration provenance. This does not claim the user
reviewed every window/crop. Hold this ceiling fixed for subsequent experiments;
do not refit it to each candidate. The same images selected/calibrated the
diagnostic, so its agreement is not independent validation. Combined-image
viewing also differs from native resolution.

| Display target | Noise at 16 local paths | Noise at 128 local paths |
|---|---:|---:|
| No filter | 7.06 | 2.31 |
| 512 | 1.48 | 1.68 |
| 1024 | .83 | 1.30 |
| 2048 | .75 | .89 |
| 4096 | .66 | .56 |

This reproduces the stated distinction: 1024 and 2048 are both below the
proposed noise ceiling at 16, but only 2048 is below it at 128. It also exposes
the intermediate-history noise increase as the footprint contracts. Lower is
not automatically better: 4096 smooths further but loses real lighting detail
and fails the existing final FLIP gate through 2048 local paths. Uniformly
incorrect brightness can have zero spatial noise. Therefore:

1. Early presentation must meet the noise ceiling **and** retain full-image
   fidelity/detail and acceptable brightness/color bias versus the preferred
   preset at matched sample counts. There is no noise-only promotion.
   Once noise is acceptable, prefer better fidelity/lower cost rather than
   rewarding further smoothing for its own sake.
2. Final fidelity still uses the unchanged mean/tile FLIP limits in
   `cave-user-2048.json`. A pleasant early frame is not certified settled.
3. Validate the noise criterion on held-out prefixes, views and edits before
   treating it as an automated acceptance gate. No runtime policy changed here.

The user's proposed maximum blending extent is a promising joint quality/cost
experiment. Target 1024 borrows about 65 faces at 16 paths; target 2048 borrows
about 130 there, but only 16 at 128. A tighter maximum footprint could limit
the early case without weakening later reconstruction. Actual support depends
on topology, unequal history and compatibility; these averages do not establish
an optimal cap. Bound footprint rather than simply lowering global strength,
which would reintroduce the noisy receiver value. GPU-resident/hierarchical
neighborhood processing remains a further cost hypothesis, not a proven win.

Reproduce with `python tools/spatial_noise_quality.py FROZEN_RUN`. Results:
`target/evidence-smooth-wall-noise.json`; performance logs:
`target/evidence-cost-{off,on}-{cave,response}.log`. Five focused noise tests
and four existing spatial analysis tests pass. No Rust/runtime modification,
build or full-suite rerun for this analysis extension.

## Maximum footprint and search-cost experiments (2026-10-15)

These experiments hold the preferred 2048 support target, strength, physical
sampling, temporal history and corner presentation fixed. `spatial_max_donors`
is a hard **candidate-count** ceiling (excluding the receiver), not a radius,
blend-strength limit or certified independent-sample count. Zero preserves the
existing radius-16/work bound. A connected walk can still travel farther around
a hole than a compact patch with the same number of candidates.

### Fixed caps: useful option, not a universal replacement

Native `5b5282f` exports the existing qualified four-window raw prefixes through
the native filter oracle, without retracing transport. Replay
`cave-1792115454656276000-38373` has 65 planned frames plus its initial live frame,
an exact receiver-identity/order check, clean logs and a complete geometry audit.
The reference and all 16 uncapped control images reproduce the originals within
rare one-code-value final-render rounding. The reference wall crops are exact;
whole-reference changed pixels are 0.0204%, maximum 1/255, mean 2.75e-7.
`tools/spatial_replay_quality.py` enforces those equivalence bounds before
applying the unchanged provisional noise calibration and FLIP comparison.

Maximum noise across both walls/four sequence windows, normalized to the
previously preferred envelope (1.0); this is **not** a new settling gate:

| Maximum candidate donors | 4 paths | 16 paths | 64 paths | 128 paths |
|---|---:|---:|---:|---:|
| Original adaptive bound | 1.000 | .747 | .687 | .890 |
| 128 | 1.457 | .715 | .687 | .890 |
| 64 | 1.942 | .825 | .687 | .890 |
| 32 | 3.004 | 1.401 | .689 | .890 |

At 16 paths, worst-window whole-image FLIP is .115131 uncapped, .114656 with
128, .108261 with 64, and .105425 with 32. Lower FLIP does not make the visibly
noisier four-path wall preferable. All caps weaken that four-path appearance;
keep them opt-in, especially if reducing batch size later. At 64+ paths the
64/128 caps do not change the search at all; at 128+ none of these caps changes
it. The maximum-neighborhood idea is therefore most relevant to very early
and uneven post-edit histories, not the whole refinement period.

A preliminary streamed cave pair (same native `5b5282f`, HEAD `6aba4f6` only
adds Python analysis) compares cap128 `cave-1792115735902187000-38617` with
uncapped `cave-1792115840593454000-38703`: 168.35M vs 167.04M paths (+0.8%,
not a reliable throughput gain); startup/settling worker p95 9.37/5.90 vs
13.50/8.34 ms (about 30% lower). Offscreen frame p99 29.65/28.18 vs
34.37/29.60 ms; near-age p95 763/834 vs 789/819 ms. This does not justify
trading away the four-path look. Presets remain available for visual comparison.

### GPU walk: correct, but rejected on this hardware

Branch `codex/gi-evidence-gpu-walk`, runtime `9197051`, test checkpoint `a9bd4f2`,
moves the ordered surface BFS onto the GPU, uploading changed slot adjacency
and using disjoint work queues. It preserves the support weights, radius and
nonrecursive inputs. Native oracles pass all four targets, three caps, all six
orientations, coordinate wraps and unequal histories; lifecycle tests cover
growth, retirement, reuse and disable/re-enable. It is not rejected for accuracy.

It cuts startup CPU work but loses complete-pipeline throughput: cave 131.44M
vs 167.04M paths (-21.3%), with settling near-age p95 1360 vs 819 ms. Response
59.11M vs 74.15M (-20.3%). Cave startup worker p95 falls 13.50→4.53 ms, but
later work and freshness regress. A likely follow-up would need a different
GPU traversal/layout and sparse-update strategy, not simply tuning the cap.
Do not merge this version. The branch preserves the unsuccessful experiment.

GPU runs: `cave-1792116111708896000-38917`,
`response-1792116191248769000-38993`; response control:
`response-1792115988692092000-38795`. All these comparisons use Apple M4/Metal,
debug opt1/deps opt3, 2560×1440 headless capture, default budgets, background
terrain and preferred corner modes. They are not windowed FPS measurements.

### Exact-output compact CPU walk

`66b954d` keeps the search on the worker but precomputes the invariant bounded
grid transitions in an 8.5 KiB table. Its queue uses compact relative-cell
indices; the existing resident GPU metadata still supplies distances. No
per-face radiance/neighborhood cache is added, and no footprint is reduced.
The old coordinate walk remains an independent debug/test oracle, excluded
from release builds. The ordered six-orientation topology and native
four-target/three-cap presentation comparisons pass, including edit weights.

Nine rotating-order isolated trials over 9216 slots: indexed→compact median
search costs are 22.746→12.174 ms at history16, 1.920→1.058 ms at128, and
.289→.187 ms at512. These establish a 35–47% search-cost reduction, not an
equivalent total-frame or settling-time gain. Strict Clippy and build pass.
First streamed screening: cave `cave-1792116669726919000-39496` traces 167.77M
paths versus 167.04M control (+0.4%); response
`response-1792116748614010000-39596` 75.42M versus 74.15M (+1.7%). Neither is
a reliable total-throughput gain. Cave startup/settling worker p95 improves
13.50/8.34→11.71/7.04 ms; offscreen frame p99 34.37/29.60→33.53/28.26 ms,
near-age p95 789/819→774/840 ms. Response worker p95 drops in all four phases,
while tails/freshness remain mixed. Keep this exact-output reduction in CPU
work, but do not claim it has closed the complete-pipeline throughput gap.

### Compact GPU donor packets: not selected

`codex/gi-evidence-packed-samples` at `9122626` stores a derived 48-byte packet
per resident slot: raw RGB/count, standard error/position, temporal RGB/identity.
This avoids repeated square roots and scattered resident reads. Packets update
on accumulation and unselected history caps, survive growth, refresh on enable,
and release on disable. They never feed physical/temporal history. The native
oracle and 20-step lifecycle pass; each live packet is checked against state,
including partial-face retention. Strict Clippy/build pass.

It does not earn its memory: cave 167.46M vs compact-walk 167.77M paths (-0.2%),
response 76.54M vs 75.42M (+1.5%). Those single-pair differences are inconclusive;
there is no convincing speedup to offset the additional allocation. Preserve
the branch, do not promote. Runs `cave-1792117066633965000-40014` and
`response-1792117164239305000-40130`; same settings/hardware/view as above.

Reproduction artifacts: `target/footprint-replay/manifest.json`,
`target/footprint-quality.json`, `target/footprint-{base,cap128,gpu,fast}-*.log`.
Run the ignored `export_footprint_replay` test with `GI_FOOTPRINT_SOURCE` pointing
at the saved qualified sweep and `GI_FOOTPRINT_OUTPUT` at a new directory;
render its manifest through `VOXEL_GI_VALIDATE_NEAR_HISTORY`. Compare using
`python tools/spatial_replay_quality.py SOURCE REPLAY`. The contact-sheet tool
accepts `--images REPLAY` while keeping `SOURCE` metadata authoritative.
