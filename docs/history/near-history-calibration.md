# Near-edit history calibration

Experiment branch: `codex/gi-near-history-calibration`, based on the preserved
`voxel-global-illumination` baseline `56e4589`. Runtime defaults are unchanged.
The inherited 48-voxel near boundary and 16-sample floor are not analytically
qualified constants. This study separates their effects before replacing them.

**Decision:** do not promote a new constant from this study. A 256 floor is a
better isolated-cave-edit noise/response compromise than 16 in the ordinary and
moderate controls, but it still fails repeated-cap pressure. The 2,048 floor
that passes that pressure delays genuine changes. Neither changing the near
radius nor globally retaining more history establishes a faster general
solution. Preserve the runtime baseline and use these measured response/noise
curves when qualifying early-sample improvements. No windowed FPS or measured
gameplay-latency win is claimed.

## Objective and measurement

Minimize fresh paths, then measured gameplay time, until the post-edit image is
within the **unchanged** `cave-user-2048` FLIP limits and remains there. A first
crossing is insufficient: all later recorded checkpoints must pass, with at
least two later observations. This certifies the recorded checkpoints, not the
intervals between them. Also measure the error peak and whether repeated edits
keep pushing a previously acceptable image outside the limits.

The fixed limits are mean FLIP <= 0.027400863220808837 and worst 32x32 tile mean
<= 0.07296444475650787, evaluated by official FLIP 1.7 at 67 pixels/degree on
2560x1440 LDR sRGB captures. They came from the user's accepted 2,048-path image;
1,536 paths were visibly different. We do not raise these limits to make a
candidate pass. Statistical history is neither actual sample count nor proof
of convergence.

The native diagnostic freezes the audited, composed cave after its ordinary
background-streaming prelude. It records the production estimator's moments
for 15,163 existing face receivers: a 4,096-path warm history followed by 256
independent 16-path batches, in two disjoint sequence windows. An off-camera
torch at cave-local (5,10,56) is removed/reinserted. Its receiving geometry is
unchanged in the capture. This isolates retained-cache response; it does not
qualify newly exposed faces, mesh publication or the invalidation scheduler.

For each state, two independent 131,072-path references are averaged. Reference
replica disagreement is measured against the same visual limits. References
use the unchanged native estimator, previously checked against the independent
oracle; this study does not add a new independent oracle for each edited state.

`tools/near_history_calibration.py` replays identical native batches through
weighted moments and the production display resolve, including confidence.
The final images use the real voxel material, corner reconstruction, textures,
tone mapping and direct sunlight. A Rust/Python numerical fixture checks the
display calculation. Time zero is the old display evaluated against the **new**
target, not the previously correct image.

Replay seconds are an explicit uniform per-face clock, **not measured gameplay
latency or FPS**. The initial 0.85 s per 16-path publication is near the median
in an earlier background-loaded cave trace. The useful controlled comparison
is fresh paths to sustained acceptance. A promising policy still needs an
actual runtime edit test, including main-thread/frame-time tails and lighting
freshness, before promotion.

## Separating bias, variance and distance

With retained weight H and n fresh paths, the raw mean after a single edit is

`m(n) = (H * old_mean + n * new_sample_mean(n)) / (H + n)`.

If the means were noise-free, the residual old-lighting fraction would be
`f = H / (H + n)`. Rendered blends of the two references estimate which f is
visually acceptable for this particular change. If f_max is acceptable, the
bias-only bound is `n >= H * (1 / f_max - 1)`. This is not a full settling-time
prediction: fresh-sample noise, display smoothing and scheduling also matter.

An unchanged-lighting control applies a cap every K fresh paths. Under an IID
approximation, its steady cycle-end effective sample count is `2H + K`.
For K=128, floors 16 and 256 imply only 160 and 640 effective samples, whereas
H=1024 implies 2,176. This supplies a search bracket, not a QMC or FLIP guarantee;
the native/rendered measurements decide whether it actually suffices.

Noise-free spatial controls install the new reference within radius R and
leave the old reference outside. These measure the visible residual change
outside R, independently of sampling noise. They do not establish a universal
physical radius: emitter power, visibility and bounced sunlight change the
answer. In the initial floor sweep R changes only the near/far floor boundary;
the existing distance prior still uses its inherited 48-voxel scale.

Sampler improvements should reduce measured work Q at the same visual limits,
then trigger requalification of history weights. They must not silently change
the appearance goal. A higher near floor with a lower far floor is also unsafe
to promote casually: distance-band compression relies on nondecreasing caps
with distance. Uniform higher floors across the affected domain avoid that
nonmonotonicity in the diagnostic, but are not runtime defaults.

## Initial ordinary-torch result

The full settled change from removing this one ordinary torch is already
below the visual limits: mean FLIP 0.01019, worst tile 0.02575. This is a real
small-edit control, not evidence that all emitter edits are negligible.

| Policy, near radius 48 | Removal sustained acceptance | Addition sustained acceptance | Repeated unchanged-lighting cap, K=128 |
| --- | ---: | ---: | --- |
| Retain history | 0 fresh paths | 0 fresh paths | All 17 captures pass |
| Near floor 16 | Not confirmed within initial checkpoint tail | 2,048 fresh paths | 15/17 fail |
| Near floor 256 | 2,048 fresh paths | 2,048 fresh paths | 15/17 fail |

The 16-floor removal passes at 3,072 and 4,096, but the initial plan has only
one observation after 3,072, so it does not meet the sustained-tail rule. Later
plans add a 3,584 checkpoint; old captures and results are not relabeled.
For the repeated unchanged-lighting control, peak worst-tile error is 0.22169
at floor 16, 0.16199 at 256 and 0.05726 when retaining history. Thus unnecessary
caps can take an acceptable image out of tolerance for thousands of fresh
paths. This is why a stable, unchanged-lighting control alone cannot choose a
policy: stronger real changes must also test the retained-history lag.

The higher-history pressure sweep uses a uniform floor throughout the affected
domain (R=278), not a near floor followed by the old lower far floor. Both
disjoint sequence windows start within tolerance. Across 17 captures/window:

| Floor | Failed captures, windows 0 / 1 | Peak worst-tile FLIP, windows 0 / 1 |
| --- | ---: | ---: |
| 512 | 13 / 13 | 0.13735 / 0.11001 |
| 1,024 | 10 / 3 | 0.10340 / 0.08190 |
| 2,048 | 0 / 0 | 0.07253 / 0.06410 |

2,048 is the first tested passing floor for this pressure interval, not an
exact minimum or a robust safety margin: window 0 is close to the 0.07296 tile
limit. The IID effective-count estimate correctly supplied a useful search
range but did not predict the actual FLIP pass at H=1,024. More frequent edits,
different visibility and a changed sampler need fresh qualification.

## Bright-source stress control

Scaling only the edited torch's emission by 64 makes the settled change large:
mean FLIP 0.16631, worst tile 0.38501. The two bright-state reference replicas
differ by mean 0.01516 / tile 0.05460: below the acceptance limits, but material
uncertainty for fine comparisons. The 4,096-path bright warm image itself fails
(0.04166 / 0.13918). Its removal trajectories therefore do **not** qualify
settled-start response latency. The source-absent warm image does pass, so the
addition trajectories have a valid starting point.

Near floors 16, 64 and 256 at R=48, and full retention, all fail sustained
acceptance through 4,096 fresh paths. On removal, 16 and 64 first pass at the
last checkpoint, which is not a confirmed tail. On addition, all four policies
still fail at the last checkpoint. This is a sampler/response stress result,
not permission to loosen the visual goal.

Reference-blend controls isolate stale bias from incoming sample variance
(the high-sample references themselves are not infinitely exact). For both
directions, 1/8 retained old-lighting fraction fails while 1/16 passes. For
H=2,048, these correspond to 14,336 and 30,720 fresh paths respectively. Thus
the floor that survives ordinary null-change pressure carries a substantial
response cost on this change even before fresh-sample variance is included.
These are tested fraction brackets, not an exact crossing or a universal bound.

Updating the reference perfectly only within R=48 still leaves mean 0.1164 /
tile 0.2883 error; even R=64 leaves a failing tile near 0.225. The near boundary
is not a physical influence limit. This result does not itself reject the
runtime's wider distance curve, which also updates receivers outside R=48.

The moderate-source control is deliberately separate: a 64x stress light
whose own 4,096-path warm image is noisy cannot select ordinary near-edit
defaults by itself.

## Moderate-source recovery and confirmation

An 8x edited torch gives a settled change of mean FLIP 0.04103 / tile 0.10407,
above both limits. Reference-replica disagreement is 0.00807 / 0.02078, and all
four warm controls (two states, two windows) pass. This is the valid
settled-start genuine-change comparison missing from the 64x stress control.

Lowest recorded passing checkpoint with no later failure, in fresh paths:

| Policy | Removal, windows 0 / 1 | Addition, windows 0 / 1 |
| --- | ---: | ---: |
| Near floor 16, R=48 | 3,072 / 3,072 | 3,072 / 3,584* |
| Near floor 256, R=48 | 3,072 / 3,072 | 3,072 / 3,584* |
| Uniform floor 1,024 | 3,072 / 3,072 | 3,584* / 4,096* |
| Uniform floor 2,048 | 3,584* / 4,096* | No pass / no pass |

`*` Too few later observations for the predeclared sustained-tail rule. These
are observed crossings, **not** qualified sustained responses. The common
3,072 removal result has later 3,584 and 4,096 confirmations in both windows;
the preceding 2,048 checkpoint fails. Thus the measured crossing bracket is
(2,048, 3,072], not an exact claim that 3,071 would fail. Full retention never
passes in window 0 through 4,096; it was not repeated in window 1.

The 256 floor reduces peak worst-tile error versus 16:

| Transition | Floor 16, windows 0 / 1 | Floor 256, windows 0 / 1 | Reduction |
| --- | ---: | ---: | ---: |
| Removal | 0.19076 / 0.21648 | 0.12091 / 0.12761 | 36.6% / 41.1% |
| Addition | 0.17048 / 0.18556 | 0.12965 / 0.13980 | 23.9% / 24.7% |

This is an early **image-error peak**, not a measurement of energy overshoot or
a perceptual video-flicker score. It is a useful improvement at the same
observed crossing checkpoints, not a 2x settling-speed win. Source strength,
occlusion/skylight edits, new faces and actual scheduling are not universally
qualified by these two views of one off-camera torch edit.

The first-window radius controls keep H=16 and change R to 32 or 64. Both
directions still sustain at 3,072. R=32 reduces peak error somewhat; R=64 raises
peak whole-image error (removal 0.06569 versus 0.05132 at R=48; addition 0.06868
versus 0.05360) without a speed benefit. No radius candidate earns promotion,
and these controls were not redundantly repeated in the second window.

For the moderate change, the reference-blend half-old/half-new image passes
(removal 0.02564 / 0.06525; addition 0.02542 / 0.06373). All audited receivers
are within 80.88 voxels, so the 256-floor arm has cap 256 everywhere. Its
bias-only reference model would therefore meet that tested fraction after 256
fresh paths, yet the actual displayed removal needs the observed 3,072
checkpoint. The gap implicates incoming-sample error and display response,
not just residual old-lighting bias. It is not an additive attribution or a
promise that removing the display filter would improve appearance.

The reference-only spatial control fails at R=48 (mean 0.02869, tile 0.07179)
and passes at R=64 (0.00465 / 0.05515), in both directions. In contrast, the
ordinary torch needs no update to pass and the 64x torch still fails at R=64.
**Source-independent distance alone cannot analytically determine a universal
near radius from these results.** Actual runtime receivers outside the near
floor boundary still receive updates through the wider retention curve.

These measurements give the next sampling experiment two independent gates:
reduce work to the unchanged FLIP goal after a real edit, and avoid repeatedly
taking a settled image out of tolerance under unchanged-lighting pressure.
Recalibrate weights against the new sampler only after passing both. A blanket
larger history, a larger aggressive-reset radius, or a looser threshold is not
an evidence-backed substitute.

## Provenance and reproduction

Hardware: Apple M4 integrated GPU, Metal, 16 GiB, macOS 26.6.2; Rust 1.95,
Bevy 0.19.1. Dev app opt1 / dependencies opt3. Offscreen 2560x1440,
shared-corner direct sunlight, reconstructed indirect, textures on, baked AO
off. Native transport reach 768, three hits; no BVW changes. CPU image analysis
may overlap diagnostic rendering; these are not frame-pacing measurements.

- Native recorder: `c171d1e` ordinary tape; `a68a1d8` corrected single-source
  scaling implementation for the bright tape. Source code was unchanged by
  subsequent Python/documentation commits during recording.
- Ordinary tape: `target/gi-validation/cave-1791937825604004000-73963`.
- Ordinary plan: `target/near-history-normal-plan/plan.json`, analyzer
  `0aaf691`; render `target/gi-validation/cave-1791938552928262000-75754`;
  result `target/near-history-normal-analysis.json`.
- Initial normal reference-replica difference: mean 0.007548, worst tile
  0.017470. Both 4,096-path warm images pass the appearance gates.
- Higher-history pressure plan: `target/near-history-churn-v2-plan/plan.json`;
  render `target/gi-validation/cave-1791939619275898000-76943`; result
  `target/near-history-churn-v2-analysis.json`. Native code unchanged from
  `a68a1d8`; analyzer `20a92a9`; no algorithm diff. Concurrent review commits
  `40205fd`/`55f74f1` only add/edit a source-review document.
- Bright tape: `target/gi-validation/cave-1791938695493179000-75916`;
  plan `target/near-history-bright-plan/plan.json`; render
  `target/gi-validation/cave-1791939754242635000-77041`; result
  `target/near-history-bright-analysis.json`, analyzer `20a92a9`, no algorithm
  diff. The extra high-floor bright plan was generated but not rendered:
  reference-blend results already expose the substantial stale-bias cost.
- Analyzer `dcc2e09` adds explicit settled-start qualification and a fifth
  focused passing test. It leaves earlier image measurements unchanged; the
  bright warm-start limitation above was checked directly in the older report.
- Moderate tape: `target/gi-validation/cave-1791939958286106000-77130`, recorded
  at `a638230` with unchanged native code from `a68a1d8`, scale 8. First-window
  plan `target/near-history-moderate-plan/plan.json`, render
  `target/gi-validation/cave-1791940628008696000-77385`, result
  `target/near-history-moderate-analysis.json`. Confirmation plan
  `target/near-history-moderate-confirm-plan/plan.json`, render
  `target/gi-validation/cave-1791940981405782000-78049`, result
  `target/near-history-moderate-confirm-analysis.json`. Both use analyzer
  `d5faa7d`, no dirty algorithm diff. All native logs are clean; capture status,
  target dimensions, geometry audit and nonblank images pass. The first
  window's 128-path floor-16/floor-256 removal images were visually inspected.
- `d5faa7d` adds explicit policy shortlists to avoid duplicate Cartesian arms;
  all six focused Python tests pass. No native rebuild was needed for the
  analyzer-only changes. Native applications exited normally after capture.
- Native verification: two focused near-history Rust tests, production
  Clippy with warnings denied and dev build at `a68a1d8`. Four focused Python
  tests also pass at `20a92a9`. Runtime code has not changed since those checks.

Record with `VOXEL_GI_VALIDATE_NEAR_HISTORY=record` and the existing cave,
background, headless and exit validation flags. Optional
`VOXEL_GI_NEAR_EMITTER_SCALE` changes only the edited voxel's emission in the
diagnostic snapshot, not the global material palette. Generate a plan with
the Python tool's `plan` command; set `VOXEL_GI_VALIDATE_NEAR_HISTORY` to its
absolute `plan.json` path to render; run `assess` against that capture folder.
The plan hashes its native inputs. Keep large tapes, captures and reports in
ignored `target/`, not Git.
