# Cave history-cut attribution — October 13, 2026

This is a diagnostic follow-up to [runtime retention](runtime-retention.md),
not another policy experiment. Production history thresholds, eight-event page
journals, tracing budgets, sampling, display smoothing and BVW are unchanged.

**Finding:** the overflow summary discards distance information and can reduce
history even where the current distance curve would retain it. In the detailed
recording this explained 70.94% of post-startup weight loss. Fix the summary
before retuning the curve; the cave reference target was nearly stationary.

Implementation follow-up: [distance-preserving overflow summary](distance-band-retention.md).
It changes the bounded representation rather than tuning away these measured cuts.

## What the diagnostic measures

At the same 36 cave faces / 144 reconstructed corners used for runtime
qualification, record each **accepted** transport publication, its previous
weight and its spatial cap. Count reductions before the fresh paths are merged:
a 272-to-256 cap followed by 16 paths would otherwise hide behind an unchanged
sample count. Include caps applied to unselected faces. Source/model changes and
superseded-scene results are separately flagged.

Decompose the actual cap into geometry, coalesced sunlight, folded-near and
folded-far evidence. Equal minima form one cause set, not multiple independently
counted cuts. Raw sunlight points additionally distinguish known-to-known
changes, first measurements, unresolved traces and recovery from unknown. They
are diagnostic sidecars; they never split production events or alter bounds.

The second recording keeps an opt-in, 1,024-event shadow journal on watched
receiver pages. It answers what **the same distance policy** would have capped
at the observed history weight if event details had not been folded away.
This is a one-step counterfactual, not a replay of an alternative history or a
proposal to ship a much larger journal. Analysis rejects missing shadow history
for a consumed overflow instead of interpreting it as no change.

Every 30 seconds, retain an immutable scene snapshot. After the 182-second
workload, trace each of the six snapshots with two disjoint 65,536-path windows,
the same windows at every checkpoint. Connected corner neighborhoods and source
revisions must remain compatible. Block averages stay below the production
history ceiling. These references bypass runtime history and scheduling, but
use the existing finite-depth estimator. Common sequences reduce difference
noise; window disagreement is sensitivity evidence, **not** a confidence bound.

Target motion uses RGB vector difference divided by the norm of the
componentwise brightest mean across checkpoints (floor .02). It is a sparse
world-space diagnostic, not FLIP or a whole-image invisibility certificate.
Thirty-second checkpoints also cannot prove that every intervening change was
harmless or detect every short-lived excursion.

## First recording

Native code: `6256fc0`, based on production `263f91a`. Only an untracked Python
test was added during the run; no native dirty diff. The analyzer's policy-ID
guard was corrected afterward to match the existing versioned report identity.

Apple M4 integrated GPU / 16 GiB / Metal, macOS 26.6.2; Bevy 0.19.1, registry
BVW 0.17, Rust 1.95, development application opt1 / dependencies opt3. Offscreen
2560×1440, preferred shared-corner sunlight / reconstructed indirect, textures
on, baked AO off. Normal world cave structure with the moving background loader
at 4.32 voxels/s; 16 paths/face, 4-ms trace target, reach 768, three bounces,
75-ms display response. No second native application or compiler during capture.
**Instrumented offscreen cadence is not a gameplay-FPS measurement.**

Run: `target/gi-validation/cave-1791930264791222000-67575`.
Log: `target/retention-attribution.log`.
All five captures passed camera/target/assets/nonblank guards; the 60-second
capture was also inspected. No render errors or CPU fallback. 429.04 million
paths, maximum 1,544 loaded pages.

From 30 through 180 seconds, 7,435 accepted watched-receiver jobs contained
93 actual weight reductions:

| Binding mechanism | Cuts | Weight removed | Share of removed weight |
| --- | ---: | ---: | ---: |
| Folded-far journal fallback | 78 | 82,449 | 94.48% |
| Geometry distance cap | 6 | 1,503 | 1.72% |
| Sunlight bounds | 9 | 3,312 | 3.80% |

All nine binding sunlight caps matched first-measurement evidence, rather than
known-to-known changes (weaker co-occurring causes are not excluded). Geometry-bound cuts were roughly
229–260 voxels away (median 230); sunlight bounds had median distance 233.
The inherited local 16-at-48 rule did **not** bind any post-startup cut. There
were no concurrent source/model changes or superseded-scene jobs in those cuts.

Across the evaluated checkpoint pairs, maximum corner target motion was about
**.0040%**, and paired-delta window disagreement at most **.0079%**. Absolute
corner reference-window disagreement was about .606% p99; common sequence
differences cancel most of that noise. Face-center target motion was at most
.0074%. These are tiny measured changes, not proof of exact zero change.

The first recording establishes that fallback dominates the loss. It cannot
recover the original causes hidden inside those 78 folded summaries; that is
the purpose of the second recording's shadow journal.

## Detailed overflow recording and decision

Native code: clean `7047395` (only documentation edits during capture; subsequent
analyzer output-format changes do not alter the native experiment). Same machine,
profile, settings and 182-second workload. Run
`target/gi-validation/cave-1791930786191963000-67933`; log
`target/retention-attribution-shadow.log`. All five captures passed validation,
with clean render logs and no CPU fallback. 388.22 million paths and maximum
1,561 loaded pages. The 180-second capture was also inspected. Work counts and
residency differ; this is a repeated attribution
study with additional instrumentation, **not a performance A/B**.

From 30 through 180 seconds:

| Binding mechanism | Cuts | Weight removed | Share of removed weight |
| --- | ---: | ---: | ---: |
| Folded-far journal fallback | 64 | 79,017 | 88.50% |
| Sunlight bounds | 16 | 10,271 | 11.50% |
| Geometry distance cap alone | 0 | 0 | 0% |

The shadow journal recovered every consumed overflow, with no diagnostic gaps:

- **49 of the 64 overflow cuts needed no reduction at all** at the observed
  weight under the existing distance policy.
- The other **15 still warranted a cut under that policy, but a gentler one**.
  All were first-measurement sunlight evidence, not a known-to-known change.
- Folding alone removed **63,343 excess units of statistical weight**: 80.16%
  of overflow loss and **70.94% of all post-startup loss**. The remaining 25,945
  units were all attributable to first-measurement sunlight evidence (the 15
  hidden inside overflow plus 16 explicit sunlight cuts).
- First-measurement sunlight supplied the **binding** cap, but other evidence
  co-occurred. At the same observed weights, geometry alone would reduce 29
  histories by 24,258 units (27.17% of all loss). Known-to-known sunlight points
  alone would reduce eight histories by 3,373 units (3.78%). These are overlapping
  alternatives, not additive shares of actual loss or measurements of received
  radiance. Unresolved/recovered traces would cause no cuts. Thus simply removing
  first-measurement sunlight evidence would leave most of the remaining geometry
  cuts; absence of a *binding* known-sun cap is not absence of known changes.
- There were no source/model changes. Thirteen cuts arrived with accepted
  older-scene transport jobs; they are recorded rather than misrepresented as
  current-scene sky results. Production stale-sky rejection is unchanged.

Concrete example: site 27 at 149.94 seconds retained 3,304 samples' worth of
history. The nearest intervening sunlight bounds were **289.86 voxels away**;
the unchanged distance curve required no cap. Because the page journal had
overflowed, publication instead capped it to **256**, discarding 3,048 units.
Nearby pages can receive an event whose bounds are outside a particular
receiver's influence radius. Folding loses that distinction and applies its
fixed floor. The same failure also strengthens cuts inside the radius: site 24
at 142.06 seconds had weight 3,168 and an unfolded cap of 19,459 (therefore no
reduction), but also fell to 256.

Frozen-scene corner motion again remained about **.0040% maximum**, with about
.0079% maximum paired-delta disagreement. That supports a nearly stationary
cave lighting target despite continuing loss of statistical history. It does
not classify every transient between checkpoints or certify the whole image.

**Conclusion:** most measured loss is unnecessary bookkeeping conservatism, not
evidence that the cave's correct lighting is moving substantially. Fix the
information-losing overflow summary before changing the distance/history curve.
Then remeasure remaining distance-based geometry/first-measurement evidence
against the fixed quality goal. Unknown must not simply be relabeled unchanged.
No production fix or threshold change is included in this diagnostic branch.

## Where 16 / 48, 384 and 2,048 came from

- **16 retained samples** originated in local relighting (`37a4eed`), then joined
  the shared edit/streaming policy at `b48cfe2`. **48 voxels** was already the
  edit-relight radius in the initial checkpoint `a81fe5e`. They are inherited
  tuning constants, not values derived from the current visual-error threshold.
- They have regression evidence, but that is not an optimality proof. The older
  near-history taper was rejected after a closure regression; the modest
  16-to-24 experiment `efdcb2b` was reverted by `18c0bd1` because its small,
  unpaired smoothness benefit did not justify slower response. Sampling and
  architecture have changed since those experiments.
- **384 blocks** is an empirical distance at which the tested remote changes
  were below the acceptable-error scale. The distant sun-bounce counterexample
  prevents treating it as a universal bound. See [distance history](distance-history.md).
- **2,048 paths** is the user's accepted appearance threshold, used to establish
  fixed FLIP and supporting cache-error limits in `cave-user-2048.json`. It is
  an empirical quality anchor, not an analytically guaranteed sample count.
- The new bias formula combines an assumed distance-change prior with that
  measured work scale. Its local unconstrained cap rounds to 11, but the
  inherited floor raises it to 16. Thus the formula did not independently
  discover or validate 16 or 48.

## Reproduction

Use the long-cave command in [runtime retention](runtime-retention.md), with
`VOXEL_GI_VALIDATE_HEADLESS=1`, `VOXEL_GI_RETENTION=calibrated` and
`VOXEL_GI_VALIDATE_RETENTION_AUDIT=1`. Keep `TEMPORAL`, `BACKGROUND`, `CAVE_LONG`
and `EXIT` enabled. The audit runs its six-scene oracle after saving the timed
report; wait for process completion, not just `status.json` publication.

Analyze with `python3 tools/retention_attribution.py <run-directory>` in an
environment with NumPy. Raw output is `retention-audit.json` plus
`retention-reference.json`, alongside the normal temporal sequence and captures.
Normal launches do not allocate diagnostic records or shadow entries. The
opt-in shadow journal increases metadata/copying cost, so neither diagnostic run
qualifies main-thread cost, windowed frame pacing, or a proposed optimization.

Tracked verification: 10 retention-filter Rust tests, six compact-worker tests,
eight sky-cache tests, strict production Clippy and a development build. The
shadow-journal change reran the retention tests/Clippy/build; unchanged worker
and sky paths reuse the first instrumentation checkpoint's results. Four Python
tests cover overlap accounting, incomplete evidence rejection and fixed-scene
paired differences. No full suite or routine toggle stress was needed.

Analysis artifacts (ignored, alongside the large captures):

- `target/retention-attribution-summary.json`, SHA-256
  `0055b02df4d2c891d42f7eab3ec5a21c4e2e89af4549adb72585127942f2eec2`.
- `target/retention-attribution-shadow-summary.json`, SHA-256
  `8f1acdc07afdb262c03a403cc4ac27bca15e6dcc60a8ed39cc302995b66c11f6`.
