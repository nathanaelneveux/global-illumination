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
worse at 64. **Both first pass the unchanged user-calibrated FLIP gate at 2048**,
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
target all first pass the calibrated sustained image gate at **2048**.

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
