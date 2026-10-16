# Repeating hallway outliers: diagnosis and guiding experiment

**Current decision:** guiding is experimental and defaults **off**. The
[full-image qualification](#whole-cave-image-gate-revokes-default-guiding) fails
the unchanged cave visual target at 2048 paths, despite its real hallway benefit.
The independent audits also found and fixed deterministic path-truncation
(`6c9d848`) and grazing emitter self-shadowing (`803f148`) bugs. Those corrections
remain on both estimators; no smoothing/history adjustment compensates for them.

## Reproduced failure (2026-10-16)

Production baseline: `c02d99a`, Apple M4 / Metal, macOS 26.6.2, development
profile, preferred corner display with uncapped 2048-evidence reconstruction.
Three saved 182-second live hallway runs repeat the same high-energy batch at
floor receiver `(-54,-12,-20)`, face 2, sequence 2240–2255. In the latest run
(`live-hallway-1792161245235760000-71464`), that batch contributes about 82% of
the receiver's accumulated red energy at the end. Its mean is about 2.9 times
the two-million-path reference, despite continuing to receive new samples.
The adjacent edit does not fix a deterministic undersampled path distribution.

`hallway_outlier_paths` reproduces the batch independently on CPU and GPU.
Path 2243 travels from the distant floor to its nearby ceiling, then 74 voxels
back toward the lamp. A surviving roulette step leads to a terminal connection
near the lamp. The contribution is large but physically valid. CPU/GPU agree;
the batch merge uses the ordinary 16/new-history weight. This is evidence of
severe variance, not evidence that all reported bad faces have the same cause.
The original isolated fixture has a different global emitter distribution, so
its exact contribution differs from the streamed world's.

## Experimental proposal, not a lighting-model change

The experimental `emitter_guiding` setting mixes cosine sampling with a
smooth source-directed spherical lobe. A task chooses a geometry/source-only
anchor; every path uses its explicit PDF in throughput and emission MIS,
including terminal connections. No radiance clipping, borrowed samples,
history resets, or emitter falloff changes are involved. All five bands use
the same PDF-compensated weights. A nonzero cosine component preserves support;
correct weights alone did not make the old path-termination policy unbiased.

The lobe uses the [Henyey–Greenstein distribution](https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions)
as a proposal, **not** as a physical medium. This follows the importance-sampling
principle described in [PBRT's path tracer](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer).
It is source-informed analytic guiding, not the learned SD-tree algorithm in
[Practical Path Guiding](https://jannovak.info/publications/PathGuide/index.html).

Anchor selection uses the existing light hierarchy but never changes its
physical light-selection probabilities. A bounded search tests up to eight
candidate voxel centers for visibility. Concentration inside a 16-voxel source
region controls proposal strength; distant, occluded sources must not dilute an
isolated lamp simply because they share a global tree. The small flat-light
distribution uses its strongest projected source directly. The fixed radius
and concentration/focus scales are measured proposal heuristics, not physical
lighting distances or analytically established universal optima.

Rejected stages remain in experimental commits: secondary-bounce-only guiding,
hard cones, and global concentration each failed important comparisons. The
loaded hallway fixture includes the disconnected cave's emitters: without it,
an apparently good guide chose an occluded cave torch instead of the hallway
lamp. An isolated-room win is not a gameplay win.

## Raw evidence so far

`conditional_guiding_prefix_study` uses four disjoint sequence windows, no
display smoothing or spatial borrowing, and two independent 1,048,576-path
control references. References are block averaged below the history cap.
The `hallway_loaded` fixture has 237 floor receivers; the cave samples its
existing research wall/ceiling receivers. These are sparse raw-field metrics,
**not FLIP**, full-image certification, or windowed FPS.

For the radius-2 candidate with distance focus denominator `256 * radius²`:

| Fixture / paths | Relative RGB norm error p95, control → candidate | p99 |
| --- | --- | --- |
| Loaded hallway / 2048 | .991 → .273 | 3.394 → .505 |
| Loaded hallway / 8192 | .729 → .166 | 1.611 → .220 |
| Cave / 2048 | .028 → .029 | .037 → .035 |
| Cave / 8192 | .014 → .014 | .018 → .019 |

Evidence: `target/conditional-guiding-focus256-study.json`, shader diff from
`c341b1c` changes only focus 64 → 256. References reused explicitly from
`target/conditional-guiding-loaded-study.json`; bad supplied reference files
fail rather than silently regenerating them. The cave is approximately neutral,
not a claimed sampler improvement. Dark hallway relative errors remain material.

CPU/GPU matched 16-path prefixes pass for hallway-with-competing-lights and cave
receivers across three windows. Proposal integration against constant incident
light and the terminal emitter's numerical area integral pass, including all
five bands. Runtime opt-in: `VOXEL_GI_SETTINGS=config/gi-guiding.json`.
Actual-dispatch throughput and live streamed captures must be qualified before
changing defaults. The first live run was concurrent with a test build: use it
for lighting diagnosis, not comparative performance claims.

## Gameplay qualification and rejected shortcut

The initial live guided hallway (`live-hallway-1792166631031324000-74640`)
reduces the 180-second p95 raw relative RGB norm error .489 → .182 against the
saved frozen reference. Its neighboring cave run (`live-cave-1792166920149121000-74833`,
`1f828bd`) has clean captures but traces only 309M paths versus the saved 483M
control. Neutral cave accuracy per path does not justify that global throughput
cost. These results did not qualify a default.

`80d8633` tried stopping the anchor search when a candidate was weak. **Rejected**:
the full-world hallway run `live-hallway-1792167337797769000-75058` stops before
finding its lamp and reproduces the bright batch. The queued cave run was
terminated once this failure was known; it is not completed evidence.
Weakness of one candidate cannot certify that remaining candidates are weak.

This exposed a fixture limitation: the original `hallway_loaded` includes only
part of the competing cave. The corrected `hallway_loaded_full` includes its
east side and ceiling/crystals. Its reference is newly computed, not relabeled
from the partial scene. `target/conditional-guiding-front-study.json` gives:

| Full competing cave / paths | p95 relative RGB norm, control → guided | p99 |
| --- | --- | --- |
| Hallway / 2048 | 1.022 → .278 | 2.934 → .600 |
| Hallway / 8192 | .743 → .172 | 1.930 → .229 |

The revised optimization rejects the entire guide query only when the light
tree's bounds lie behind the receiver's hemisphere. That cannot suppress the
hallway lamp search; the bounded visible-source search remains intact. Weak
mixtures are disabled only **after** selecting an anchor. Each face shares its
anchor across cooperative lanes, and zero-guide PDFs avoid lobe arithmetic.
This is still proposal selection, never occlusion culling of transported energy.

The exact offending 16-path batch in the full fixture changes from red .05073
to .000114; a focused regression records this, alongside native CPU/GPU parity
including the offending receiver. Constant-illumination and area-integral tests
protect the PDF/energy contract. This single batch is not an accuracy certificate;
the disjoint-window and full-world comparisons remain necessary.

## Retained implementation and default decision

Retain `2676282`'s GPU preparation pass (restored by `488d21c`). It builds one
proposal per selected face into a 16-byte scratch record following the existing
result/sky payload. The path pass reads it; CPU upload/readback sizes and
persistent per-world storage do not grow. This removes the guide-search stack
from the path kernel. Both preparations and tracing are included in measured
GPU work. Inline preparation remains a test comparison, not another default.

The subsequent split cosine/guided dispatch experiment `72d678b` was slower and
was reverted, not hidden: its normal-world path count fell to 83.18M and cave
median history to 291. Native numerical parity alone did not qualify that cost.

Final long streamed hallway run:
`live-hallway-1792168449690158000-75726`, exact `2676282`, explicit guiding on.
Compare to saved control `live-hallway-1792161245235760000-71464` at `c02d99a`.
Both are M4/Metal development builds, 2560×1440 offscreen, background terrain,
preferred corners/textures, 2048-evidence display, sixteen paths/dispatch,
4-ms adaptive batch target. No concurrent build or second GI app in the final
candidate. The reference is the control's frozen two-million-path hallway field;
this comparison does not assert that streamed scenes are identical at every
instant or that the sparse field metric is FLIP.

| Time | Control raw RGB relative error p95 / p99 | Guided p95 / p99 |
| --- | --- | --- |
| 30 s | .951 / 2.560 | .298 / .412 |
| 60 s | .583 / 1.941 | .259 / .543 |
| 120 s | .567 / 1.017 | .201 / .246 |
| 180 s | .489 / 1.320 | .170 / .223 |

At 180 seconds the tracked bad face is 13.6% below the reference, versus 192.8%
above it in the control. It has passed sequence 2243; this is not merely avoiding
the failure by running too slowly to reach that batch. Remaining relative error
is material: this does not certify every face as converged or solve every cave
artifact. The raw cave fixture is approximately neutral per path.

Normal-world `structures` cost check (same settings/hardware/viewport):

| Run / implementation | Paths in 62 s | Cave-site median history | Cave phase cadence p95 / p99 |
| --- | ---: | ---: | --- |
| `structures-1792167860822437000-75346`, control | 142.51M | 1023 | 24.47 / 30.72 ms |
| `structures-1792168385470000000-75731`, prepared guiding | 119.97M | 879 | 24.51 / 29.12 ms |

This is a **trade-off**, not a global speedup: about 16% fewer raw paths and
14% fewer cave-site samples within the unchanged compute budget. Frame tails
are comparable; offscreen cadence is not windowed FPS or proof of 60 FPS.
Captures are nonblank, assets/targets valid, and render logs clean. The earlier
unused test-only import warning was subsequently removed; it was not a rendering
failure. Do not interpret slightly better p99 as a measured speedup.

Historical promotion, **superseded by the whole-cave image gate below**:
enable guiding by default for the demonstrated large reduction in persistent
hallway error, while retaining `config/gi-guiding-control.json` and the tuning
checkbox. No smoothing strength, retained-history rule, bounce count, light
range, or BVW code changes accompany this decision. The residual raw-throughput
cost is a follow-up target, not concealed by increasing the budget. Runtime
switching preserves valid measurements; use fresh runs or matched frozen
prefixes for sampler comparisons. Frozen candidate image sweeps now explicitly
carry and record the guiding flag; their independent references stay unguided.

## Independent correctness audit

The audit separates estimator correctness from display, scheduling, variance,
and cost. Tests/harness: `0fbcc06`, `409d812`. Runtime correction: `6c9d848`.
Hardware: Apple M4, Metal, macOS 26.6.2, Rust 1.95, optimized development test
profile. Native compute only; no viewport, smoothing, reconstruction, history
invalidation or gameplay-FPS claim. References reuse traversal/material snapshots,
but use a separate SplitMix64 cosine-only integrator: no emitter selection/PDF,
MIS, guiding, roulette, or low-throughput cutoff. Scalar DDA traversal parity
remains a separate check.

The two actual hallway outliers, receivers 193 and 208, were tested at one, two
and three diffuse hits. Each independent reference has two 33,554,432-path random
replicas. GPU variants have two 1,048,576-path windows, block-averaged below the
runtime history cap. Disjoint QMC windows are not independent IID trials: their
sample variance is diagnostic, not a statistical confidence interval. The IID
shader ablation provides the statistical comparison below.

Three-bounce red means (raw energy, not display RGB or FLIP):

| Receiver | Independent cosine mean ± standard error | Guided IID with old cutoff | Guided IID after removal |
| --- | --- | --- | --- |
| 193 `(-59,-12,-20)` | 0.000160213 ± 0.000003446 | 0.000141501 | 0.000154868 |
| 208 `(-54,-12,-20)` | 0.000134841 ± 0.000003151 | 0.000119775 | 0.000130918 |

The legacy guided estimator is 11.7% / 11.2% low (5.35 / 4.66 combined estimated
standard errors). Disabling **only** its `throughput < 0.002` cutoff restores
9.45% / 9.30% relative to those old means. The corrected estimates are within
1.52 / 1.21 combined standard errors of the independent reference. Removing
roulette as a separate control agrees too. Across both sites, all depths and
nonzero bands, corrected guided IID differs by at most 2.04 estimated standard
errors. This bounds what these measurements establish; it is not a universal
unbiasedness proof or a claim that every face is visually settled.

The cutoff is invalid: a small importance weight does not bound the radiance
encountered later. Guiding creates many small-weight, important paths. A new
known-answer two-wall test demonstrates the error without Monte Carlo uncertainty:
sequence 19 returned zero instead of 9.490358 in each band. Both the CPU test and
native GPU exact-path replay pass after removal; the GPU's legacy ablation
reproduces zero. Exact-zero termination remains; roulette compensates
surviving paths and is retained. This follows the distinction between biased
deterministic truncation and compensated roulette in
[PBRT](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency).

Other focused checks:

- Light-selection frequencies versus reported PMFs pass for flat and hierarchical
  distributions with independent uniforms, pooling sparse histogram bins.
- Joint path-direction / conditional-light dimensions cover the product domain;
  testing marginal uniformity alone would miss correlations.
- An enclosed five-band emissive furnace matches its analytic finite geometric
  series in the independent oracle and native guided/unguided transport, all
  three depths and three receivers. Native tolerance is 1.5%.
- All 14 tracer tests, WGSL audit variants, and native guided CPU/GPU prefix parity
  across hallway, loaded hallway and cave pass after the correction.

Evidence under ignored `target/`: `hallway-correctness-reference.json` and its
log (pre-fix runtime with the audit harness), `hallway-cutoff-audit.json` (causal
ablation), `hallway-correctness-audit.json` / `hallway-correctness-fixed.log`
(corrected runtime), `furnace-correctness-fixed.log`, and
`cutoff-fix-native-parity.log`. These numerical accuracy runs sometimes overlapped
test compilation; they are not timing evidence.

The separately built, isolated dispatch-cost check uses 2,048 faces × 16 paths,
two warmup rounds and 24 rotating paired rounds. On `6c9d848` plus only the new
cost-test harness, median GPU milliseconds were legacy/corrected 2.470/2.447
(loaded hallway) and 4.012/3.964 (cave). Treat these as comparable, not a speedup.
The correctness fix adds no measured dispatch slowdown here; guiding still costs
more than unguided sampling (1.992/3.819 ms respectively). This is not a repeat
of the full-world throughput comparison or a windowed frame-pacing certificate.
Evidence: `cutoff-removal-dispatch-cost.json` and matching log.

### Invalidated systems and follow-up queue

1. **Removed:** deterministic nonzero-throughput truncation. It is available only
   as the explicitly named `legacy_guided_iid_cutoff` test ablation, not a runtime
   toggle/fallback. Any future pruning needs a justified error bound or correctly
   compensated stochastic termination, followed by this audit.
2. **Invalidated as proof:** CPU/GPU parity and a high-sample reference sharing the
   production estimator. They remain useful regression tools, but cannot certify
   bias. Earlier guided-reference accuracy claims need requalification; the raw
   user-calibrated FLIP threshold is not lowered by this work.
3. **Not a correctness certificate:** the runtime confidence/stability heuristic.
   In dark regions its energy floor can report high confidence alongside large
   relative errors. Retain it only for scheduling with the existing fair audit;
   revisit heavy-tail-aware uncertainty before using it as a stopping rule.
4. **Follow-up:** the expanded cave-field audit below is complete. Renewed early-prefix
   and FLIP qualification of the corrected guide and reducing guide overhead remain.
   The original unguided hallway outlier still exhibits severe variance; this
   audit does not establish that it was caused by the newly identified cutoff.
   No additional estimator is removed without measured or analytic evidence.

## Expanded cave qualification: false grazing shadows

The follow-up gate (harness `e4c318b`) includes 165 cave receivers: the existing
wall/ceiling grid plus material/normal-stratified floor, lava, spire, crystal and
torch sites. This is broader coverage, not all chamber surfaces. It also tests
16/64-block segment ranges, one-hit transport, black/white/absorbing/spectrally
imbalanced five-band furnaces, and an analytic point light. Same M4/Metal/test
profile as above, no viewport or display filter; two 262,144-path CPU replicas
and GPU windows per site. Compilations overlapped some accuracy checks, so these
are not timing measurements.

Two downward torch receivers failed: `(16,10,5)` and `(5,10,32)`, face 3.
The red means were 2.43%/2.22% low, infrared 4.66%/4.17% low, up to 17.17
combined estimated standard errors. Both guiding modes failed identically.
The discrepancy exists at one diffuse hit and survives disabling roulette;
an independent cosine-only GPU estimator agrees with the CPU oracle.

The cause was emitter shadow-segment construction: shifting the start along its
normal while preserving the unshifted direction and length. At grazing angles,
the ray reaches the selected emitter earlier than the fixed `.008` endpoint
trim and incorrectly reports an obstruction. CPU and GPU now connect **both
offset endpoints**, leaving physical PDFs/geometry factors unchanged. No extra
rays, brightness clamps, history changes, reconstruction changes or BVW edits.

A black rectangle emitting unit energy has closed-form diffuse view factor
`0.0993681879` at the test receiver. With 1,048,576 IID native paths:

| Estimator | Mean | Difference from analytic |
| --- | --- | --- |
| Legacy shadow segment | 0.0958255364 | -3.565% |
| Corrected shadow segment | 0.0993615022 | -0.0067% |
| Independent cosine-only GPU | 0.0992326729 | -0.1364% (within 0.47 estimated SE) |

All five bands are checked. A deterministic CPU test fails on the old code and
passes on the fix; inserting a real black wall still blocks the connection.
The expanded gate now passes with no rejected bands; cave IID maximum discrepancy
is 3.254 estimated SE. This does not establish universal correctness or zero
variance. Near-zero channels may still have poorly resolved relative error.

Evidence: `target/transport-field-1792178923546.json` (failure),
`target/transport-field-1792179470433.json` (corrected),
`target/grazing-plane-analytic.log`, `target/grazing-emitter-{before,after}.log`.
The intermediate `transport-field-1792179374406.json` is **not** a passed gate:
its new planar fixture accidentally exposed emitting vertical edges excluded
from its analytic answer. Embedding that rectangle in a black floor corrected
the fixture; CPU and GPU already agreed on its original nonplanar geometry.
No threshold was loosened. Fifteen tracer tests and strict all-target Clippy
pass on the fix. The endpoint construction follows
[PBRT](https://pbr-book.org/4ed/Shapes/Managing_Rounding_Error#SpawnRayTo), without
claiming PBRT's interval-error guarantees for our fixed voxel-scale epsilon.

## Whole-cave image gate revokes default guiding

Clean `c16e0ba`, Apple M4/16 GiB/Metal, macOS 26.6.2, development profile,
2560x1440 headless, preferred corner sun/indirect, textures on, AO off.
Run `target/gi-validation/cave-1792179617714840000-79804` uses background terrain
and the reference-only residency camera. No second GPU workload or compilation
overlaps the run. This is not windowed FPS or an edit-response certificate.

The geometry audit checks 125,440 chamber cells with no unknowns/mismatches.
Two 131,072-path reference windows cover all 15,163 chamber/halo faces; their
RGB discrepancy p99/max is 0.815%/1.505%, with all eight independent CPU checks
passing. The 93 reference-stage captures plus three prelude captures pass their
nonblank/target guards, with clean render logs. Held reference PNGs have identical
SHA-256 hashes. The reference, guided 2048 window 0 and unguided 2048 window 3
were visually inspected. The two reference images' mean FLIP is .007535.

The candidate and control use identical RGB importance `[1,1,1,.125,.125]`,
512-face batches and four disjoint raw prefix windows. No reuse, history policy,
temporal filtering or adaptive spatial reconstruction participates. Ordinary
connected-corner display remains. The reference uses disjoint unguided/equal-band
paths; shared-estimator agreement is supplemented by the independent audits above.

Worst across windows:

| Paths | Unguided FLIP mean / worst tile | Guided FLIP mean / worst tile |
| ---: | --- | --- |
| 16 | .16121 / .50937 | .16787 / .51688 |
| 128 | .07359 / .20823 | .07679 / .27934 |
| 512 | .04242 / .12047 | .04414 / .14632 |
| 1024 | .03122 / .08642 | .03273 / .11502 |
| 2048 | .02427 / .06789 | .02545 / .10367 |
| 4096 | .01899 / .05800 | .02023 / .08561 |
| 8192 | .01548 / .04185 | .01615 / .05301 |

The unchanged `cave-user-2048` profile includes raw face/corner error as well as
FLIP mean <=.02740086 and worst tile <=.07296444. **Unguided first passes at 2048;
guided first passes at 8192**, sustained across all tested larger counts and
windows. Guided at 2048 also exceeds the profile's raw-face maximum in one
window (.13488 vs .11969). Sparse receiver percentiles previously concealed this
spatially concentrated regression. This is not established as new estimator bias:
independent high-sample means pass, and both variants' image errors decline.
It is enough to reject a universal default. The stricter historical engineering
FLIP .01 gate still fails through 8192 for both; it has not been redefined.

Decision: restore `emitter_guiding: false`; preserve `config/gi-guiding.json` as
an explicit experiment. Retain RGB priority and the approved uncapped 2048-evidence
display. Do not hard-code a cave/hallway exception, reset history, increase
smoothing, or relax quality thresholds to rescue guiding. Its sparse-hallway
variance reduction is real, but that problem is **not solved by the default**.
A replacement/adaptive proposal must qualify both the hallway and this full-image
cave gate. Path reuse remains opt-in; its historical results need requalification
after the correctness fixes, including its separate reconnection visibility.

Reports: `target/transport-qualification-{guided,control}-analysis.json` and
`target/transport-qualification-{guided,control}-quality.json`, generated by
`cave_sample_analysis.py --perceptual` (plus `--control`) and `visual_quality.py`
with the unchanged `tools/quality_profiles/cave-user-2048.json`. Native compute
timestamps were unavailable in the image run; do not substitute elapsed bake
time for a GPU performance comparison.

The separate prebuilt endpoint cost test (`c16e0ba` plus the harness in `c6aca80`)
uses 2048 faces x16 paths, two warmups and 24 alternating paired rounds, guiding
on, no app/compiler/analysis overlap. Upper-middle median GPU milliseconds,
legacy/corrected: cave 3.975/3.992, loaded hallway 2.050/2.081. This single pair
shows small added arithmetic cost, not a speedup or a claim of exactly zero
overhead. Evidence: `target/endpoint-visibility-dispatch-cost.{json,log}`.
After the capture run, native guided CPU/GPU prefix parity and the 20-step
resident accumulation/lifecycle parity checks pass (all five bands). Their logs
are `target/visibility-fix-{native,resident}-parity.log`; these accuracy checks
overlapped CPU image analysis and are not timing evidence.

Final defaults integration: clean `92dbfe3`,
`cave-1792180515078134000-80241`, 62-second offscreen cave with background
streaming. The report confirms guiding off, RGB importance .125 for simulation,
uncapped 2048-evidence reconstruction, 75 ms display response, 16 paths/dispatch
and the unchanged 4 ms budget. All three capture guards pass, assets are verified,
render logs are clean, and the 60-second image was visually inspected. This is
startup/streaming/display integration evidence, not a new independent reference,
quantified live convergence result, or a 60-FPS certificate. The default/preset
test, strict all-target Clippy and development build also pass. No changes to
BVW, history retention, ray range or bounce count were made in this qualification.
