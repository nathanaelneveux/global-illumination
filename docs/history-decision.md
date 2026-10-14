# History policy decision

Branch `codex/gi-history-decision`, baseline `423e188`. This round separates
three decisions: maximum statistical weight, the near-edit distance equation,
and whether source-type rules are justified. **Decision: select power 4 as the
near-edit policy candidate, keep the large statistical ceiling and existing
far rule, and do not add source-type heuristics.** This completes the offline
policy selection; runtime defaults remain unchanged pending live integration.

## Decision contract (before new measurements)

Keep transport, source strengths, display smoothing, native sample sequences,
corner reconstruction, textures, camera and the human-calibrated FLIP gate
fixed. A weight is not a sample count or a quality certificate.

1. Primary objective: fewer fresh paths to **sustained** visual acceptance,
   including two later passing observations. Record final failures as censored,
   not as an invented settling time. Qualify settled-start response only when
   its old-state warm control passes.
2. Compare scene families equally (hallway, house, cave), not by number of
   cameras. Within each family report both directions and native windows.
   Prefer fewer unresolved responses, then lower geometric-mean settling work.
   Report the worst regression as well as the aggregate. A material regression
   is not hidden by many easy wins.
3. Break close settling results with early mean/tile FLIP integral over the
   first 512 paths and overshoot beyond the old/new reference endpoint interval.
   A normal bright-to-dark transition is stale bias, not brightness overshoot.
4. Compare zero-history edits with cold starts using the **same incoming paths**.
   Raw estimator error, display error and retained-display initialization are
   distinct. Unchanged-lighting resets quantify avoidable re-noising; they are
   not an automatic veto merely because their noise matches fresh sampling.
5. A static cap must also remain acceptable after reaching the cap. Passing at
   exactly 2,048 fresh paths does not certify a perpetual capped filter, nor
   every scene. Retain the existing large cap if a lower cap has a failing tail.
6. Do not add emitter/wall-specific policies: a wall can occlude a strong source.
   Direct-sun changes provide spatial evidence, not a universally reliable
   high/low-impact classification. No new impact heuristic is fitted this round.

If no distance-only candidate meets these tradeoffs, make an explicit rejection
decision with its measured reason rather than indefinitely fitting thresholds.
Offline native-material replay qualifies appearance/work, not interactive FPS,
mesh publication, IR/radiation accuracy or actual scheduling latency.

## Candidate equations

Keep the qualified far-distance prior fixed. Define

```
q(d) = min(1, 48² / max(d², 48²))
F(d) = max(256, ceil(10.24 / (q(d) - 0.03)))
H_p(d) = round(F(d) * min(d / 48, 1)^p), if q(d) > 0.03
         no history reduction, otherwise
```

Screen powers 2, 4 and 6 against the current 48/16 curve. The unrounded
equation is continuous on its finite domain, equals 256 at 48 blocks, has
no near-distance bands and preserves the existing far curve. Integer rounding
is storage quantization, not a hand-authored staircase. Tiny weights can round
to zero. The existing no-cut boundary is about 277.13 blocks (278 conservative
integer reach), inside the earlier 384-block visual bound. No new physical
light cutoff is introduced.

Test the 2,048 maximum independently of edit caps; do not conflate the two.
The current GPU merge is `alpha = batch / min(weight + batch, maximum)`.
At the cap this becomes an exponential moving average, not the last 2,048
samples or a frozen arithmetic average. For independent 16-path batches its
asymptotic effective sample count would be `2 * maximum - 16`; QMC correlation
and display filtering prevent using that IID estimate as a perceptual promise.
The statistical basis is the [NIST EWMA variance formula](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm).

## Evidence to collect

- Reuse the original four-window static cave sweep for its error-by-count data.
- Native hallway cold/static capped-versus-uncapped trajectories and matched
  zero-history edits; screen the whole edit equation at multiple distances.
- Complete equation replays, with each corner donor's own edit distance, on
  hallway source/occlusion, house source/opening and cave source cases.
- Repeat shortlisted comparisons on the second native window; do not exhaust
  every constant/radius pair or rerun unrelated renderer stress.

Large captures and path tapes remain in ignored `target/`; record exact tape,
plan, executable and source identities with results. Assessments may compact
reproducible intermediate captures after hashing, retaining references and
selected review images. Never accept a blank/unfocused gameplay capture.

## Completed independent decisions

### Reject a universal 2,048 statistical ceiling

Same hallway offset-8 tape, both native windows, 16-path publications, production
display filter and material. Compare ceilings 2,048 and 65,536 separately from
edit caps. Cold controls begin at zero; static controls begin at the same
4,096-path warm mean. The lower-cap static control clamps its initial weight,
not the recorded warm mean: it is a policy-transition test, not a capped warm
prefix. The cold control has the proposed cap from its very first batch.

| Control, window 1 | Ceiling 2,048 | Ceiling 65,536 |
| --- | ---: | ---: |
| Cold start, mean FLIP at 2,048 fresh paths | .01905 | .01905 |
| Cold start, mean FLIP at 4,096 | **.02907, fails** | .02325, passes |
| Unchanged warm scene, mean FLIP after 4,096 | **.02989, fails** | .01859, passes |
| Unchanged warm scene, any failed observations | 1 | 0 |

The mean limit remains .0274008632. Both windows of the large-cap warm control
remain accepted throughout; window 0 of the lower cap also passes. This is a
measured regression in one sequence, not a claim that every 2,048-cap result is
bad. It does not support adopting that cap as a quality-preserving default.
The original cave's accepted 2,048 prefix does not guarantee an
arbitrarily long weighted stream, other cameras, or arbitrary player geometry.

Reference uncertainty matters here: the two independent 131,072-path reference
images differ by .01396 mean FLIP, although they pass the reference agreement
guard. Repeating the final comparison against each replica changes the lower
cap's pass/fail classification. Cold-start FLIP is .02526 / .03262 for the
2,048 cap versus .01892 / .02716 for the large cap; warm-static FLIP is
.02612 / .03340 versus .01505 / .02308. The larger cap is better against
**both** replicas and passes against each, but the lower cap's marginal failure
against their combined reference is not a reference-independent proof of
crossing a human visibility boundary. Keep the larger cap because no benefit
has been qualified that compensates for this measured accuracy loss, not
because this experiment proves a universal minimum. Details are in
`target/decision-cap-reference-sensitivity.json`.

A smaller global cap would speed changes with no local invalidation, but near
edits already reduce weight to far below 2,048. With retained H=256, at 2,048
fresh paths the uncapped stale fraction is 11.11%; a 2,048 ceiling only reduces
it to about 11.03%. It cannot substantially improve that early response before
the ceiling is reached. Keep the existing ceiling; do not conflate it with
slowing scheduling. The scheduler already uses per-band variance, mean drift
and display lag to enter sparse audits, not a universal 2,048-count stop.
No IR/radiation cap is qualified by these RGB images.

Evidence: `target/decision-static{2048,65536}-d8-assessment.json`. Both runs
complete their native audits. The first launcher inherited `RUST_LOG=warn`,
so its info-level output-directory line was absent; its completed native
status was resolved explicitly and assessed without rerendering. The launcher
now requests info logs. This was an orchestration failure, not a black capture
or a waived reference check.

### Separate cold-start noise from reset noise

On the native hallway tape, zero-history lamp addition and a dark-seeded cold
start have **exactly identical raw RGB and filtered display RGB at every
recorded checkpoint in both windows**. Initial reconstruction confidence
differs (retained old face versus new unknown face), so rendered images need
not be exactly identical. At offset 8/window 1, the cold image reaches 108.48%
of reference mean rendered luminance at 256 paths; the earlier zero-history
edit reaches about 109.1%. Both subsequently recover. Thus this observed
brightening overshoot is predominantly early sampling/reconstruction behavior,
not a separate energy overshoot introduced by clearing statistical history.

This does not make all zero resets harmless: resetting an unchanged bright
surface unnecessarily discards good evidence, and retaining a bright display
while its statistical confidence drops differs from starting dark. Report that
cost separately rather than demanding the edit equation fix cold-start noise.

The original static cave sweep was reanalyzed for endpoint luminance using all
76 retained full-resolution sample images; existing qualified FLIP measurements
were reused. `target/decision-original-static-cave.json` records every count and
window. Worst-window FLIP and maximum positive 32px-tile luminance error:

| Fresh paths | Mean FLIP | Worst tile FLIP | Positive tile delta Y |
| ---: | ---: | ---: | ---: |
| 16 | .18008 | .50576 | .07679 |
| 32 | .13629 | .40969 | .05279 |
| 64 | .10301 | .31540 | .04261 |
| 128 | .07985 | .24249 | .02170 |
| 256 | .06029 | .16118 | .02469 |
| 384 | .04987 | .15126 | .01157 |
| 512 | .04625 | .12367 | .01007 |
| 768 | .04116 | .10946 | .00773 |
| 1,024 | .03668 | .10453 | .00637 |
| 1,536 | .03061 | .08128 | .00559 |
| 2,048 | .02740 | .07296 | .00404 |
| 3,072 | .02318 | .06539 | .00367 |
| 4,096 | .02104 | .06865 | .00341 |
| 6,144 | .01842 | .06562 | .00332 |
| 8,192 | .01704 | .06170 | .00228 |
| 12,288 | .01470 | .04989 | .00178 |
| 16,384 | .01338 | .04048 | .00138 |
| 24,576 | .01115 | .03614 | .00125 |
| 32,768 | .01013 | .03051 | .00106 |

Whole-image brightness is a poor overshoot detector: even the 16-path cave
means are within about 1.03% of reference, while bright and dark local errors
cancel. Maximum 32px-tile **positive linear-display-luminance error** drops from
.07679 at 16 paths to .00404 at 2,048. These are absolute normalized display
luminance units, not percentages of physical radiance. The hallway's uniform
close-up makes its mean-brightness ratio more informative than the cave's.

## Complete-equation screening, native window 0

Each exposed corner donor uses its own nearest changed-voxel distance, not the
central pixel's distance or a uniform history. The geometry-event floor is
the variable under test; deterministic target-state sunlight is applied at
time zero. This is not a replay of the live event journal, sunlight notification
delivery, source-revision handling or streaming scheduler.

| Qualified edit | Current 48/16 | Power 4 | Power 6 |
| --- | ---: | ---: | ---: |
| Hallway offset 1, lamp off | >4,096 | 128 | 128 |
| Hallway offset 1, lamp on | 1,024 | 1,024 | 1,024 |
| Hallway offset 8, lamp off | 1,024 | 128 | 128 |
| Hallway offset 8, lamp on | 3,072 | 512 | 512 |
| Hallway offset 16, lamp off | 512 | 512 | 256 |
| Hallway offset 16, lamp on | 2,560 | 2,048 | 2,048 |
| House emitter off | 1,536 | 1,024 | 1,024 |
| House emitter on | 2,560 | 2,560 | 2,560 |
| Visible roof opening / closing | 256 / 256 | 256 / 256 | 256 / 256 |
| Small doorway opening / closing | 1,024 / 1,024 | 1,024 / 1,024 | 1,024 / 1,024 |
| Cave moderate source off / on | 2,560 / 3,072 | 2,560 / 3,072 | 2,560 / 3,072 |
| Hallway screen insertion | 1,536 | 1,536 | 1,536 |

Numbers are sustained **observed checkpoints**, not exact minima or elapsed
seconds. This screen uses 512-path spacing in the later range. The earlier
finer cave baseline measured removal at 2,304; its 2,560 entry here is the same
unchanged trajectory observed on this screen's grid, not a runtime regression.
All arms within this screen share the grid. Screen removal in the hallway
still has a failing pre-edit warm control, so its encouraging recovery is not
counted as qualified settled-start response.

The squared curve resolves the closest extinction failure, but needs 1,024
paths for offset-16 lamp removal versus the baseline's 512, and is slower than
the steeper curves at offset 8. Shortlist powers 4 and 6 for new window-1 paired
checks rather than exhaust more constants. A policy missing those holdouts
cannot receive a final aggregate ranking from the decision tool.

The sixth-power curve's extra speed at offset 16 comes with a measured noise
cost. On the small doorway opening, its peak mean FLIP is .05382 versus
.04051 for both baseline and power 4, with the same 1,024 settling checkpoint.
In the offset-16 null control its peak is .05559 versus .05013 for power 4 and
.03931 for baseline. Both new curves do re-noise nearby unchanged lighting;
that is reported, not hidden or used as an automatic veto.

In the cave, power 4 lowers the removal peak worst-tile FLIP from .19076 to
.14130 (about 26%) without changing this screen's settling checkpoint. Addition
peak tile changes .17048 to .16316, a smaller benefit. Power 6 is slightly
noisier there. These are appearance error peaks, not measurements of physical
energy overshoot or video flicker. Endpoint luminance excursions are recorded
separately in each assessment.

Evidence: `target/decision-curves-*-assessment.json`, corresponding native logs
and provenance, and `tools/history_decision_analysis.py`. Native Rust/rendering
remains the `423e188` baseline; all recorded executable hashes agree. Plans use
the committed common-path replay at `7f10dfb`; later changes add content-addressed
metric reuse and decision aggregation, not different lighting calculations.
Apple M4, Metal, 16 GiB, macOS 26.6.2 / 25G83, Rust 1.95.0, Bevy 0.19.1;
development optimized/debuginfo, 2560x1440 offscreen, preferred corner look,
textures on and AO off. Background terrain streams during each normal prelude,
then the measurement field is frozen. Some CPU assessments overlap another
run's prelude; no frame-pacing or gameplay-latency claims are made.

## Second-window checks and measured weaknesses

Shortlisted policies use a second disjoint native sample window, with the same
reference, material, old-state qualification and sustained-tail rule. This is
sequence replication, not a second independent scene/reference calibration.

| Qualified edit, window 1 | Current 48/16 | Power 4 | Power 6 |
| --- | ---: | ---: | ---: |
| Hallway offset 8, lamp off | 1,024 | 128 | 128 |
| Hallway offset 8, lamp on | 256 | **1,024** | **1,024** |
| House emitter off | 1,024 | 512 | 512 |
| House emitter on | 1,024 | **1,536** | **1,536** |
| Visible roof opening / closing | 256 / 256 | 256 / 256 | 256 / 256 |
| Cave moderate source off | 3,072 | 3,072 | 3,072 |
| Cave moderate source on | Tail unconfirmed | Tail unconfirmed | Tail unconfirmed |

The cave-add arms first pass at 3,584 and also pass at 4,096. All lack the
predeclared **two** subsequent passing observations, so none gets a sustained
settling claim. This is insufficient tail observation, not a failing final
image. The cave null controls have the same limitation. It does not distinguish
the policies, so more native samples solely to declare this common tail would
not resolve the choice.

The hallway's 4x brightening-work regression is material, even though its
window-0 brightening improves 6x and removal improves 8x in both windows.
House brightening also regresses 1.5x in window 1. These are the expected risk
of exposing low-sample noise instead of carrying a stabilizing old mean; the
matched cold control supports that explanation. Neither curve dominates the
baseline in every direction/sequence. Do not promise uniformly faster edits.

Power 4's cave window-1 removal peak tile is .22326 versus baseline .21648
(slightly worse), despite a lower peak mean FLIP (.05181 versus .05397).
Addition peak tile improves .18556 to .15644. Thus the window-0 removal-peak
improvement is not a universal worst-tile win either. Endpoint luminance
excursions and null-change controls remain available separately in the
assessments; FLIP includes both noise and lag.

## What the equation means analytically

Before a statistical ceiling is reached, the expected stale-state bias after
`N` fresh paths and retained weight `H` is

```
bias = H / (H + N) * (old_mean - new_mean)
H <= N * epsilon / (abs(old_mean - new_mean) - epsilon)
```

The inequality applies only when the change exceeds epsilon, and only to bias:
it excludes new-sample variance, display filtering and reconstruction. Distance
does not determine the unknown change in lighting. The near equation is an
**empirically selected prior**, not an analytically proven light bound or a
globally optimal curve. Arbitrary emission strengths and occluders preclude
promising a distance-only universal optimum. The inherited far equation is
held fixed here; its .03 normalized bias allowance is not numerically the
.02740 perceptual FLIP gate.

Likewise, keep 2,048 as the current measured work anchor for the accepted cave
quality, not a guarantee for every static scene. Recalibrating a better sampler
against the **same** image-quality gate can change that work anchor. Scaling
history with it preserves the analytical bias/work ratio; the perceptual
response and null-noise tradeoff still need a focused recheck. Do not claim
that halving sample variance automatically qualifies halving every history.

## Determination

Choose **power 4**, not power 2 or power 6, for the next runtime integration.
Inside 48 blocks the selected equation simplifies to

```
H(d) = round(256 * (d / 48)^4)
new_weight = min(old_weight, H(d))
```

Use the full equation above outside that interval. Examples are 0 at 8 blocks,
3 at 16, 16 at 24, 51 at 32, and 256 at 48. These are examples of one equation,
not distance bands. Preserve measured means, moments, path sequence and display
history; a zero statistical weight lets the next native batch replace the
estimator but does not immediately paint the face black. Never increase a
face's weight to the curve: it is a cap, not invented evidence.

The final comparison contains 23 qualified settled-start edit/window cases:
9 hallway, 10 house and 4 cave. Twenty-one settle for both baseline and the
shortlisted curves. Both new curves additionally resolve the closest hallway
extinction failure; the cave-add tail remains unconfirmed for all three.

| Policy | Qualified cases with sustained tail | Equal-family geometric work ratio | Worst measured slowdown |
| --- | ---: | ---: | ---: |
| Current 48/16 | 21 / 23 | 1.000 | 1.0x |
| Power 4 | 22 / 23 | .800 | 4.0x |
| Power 6 | 22 / 23 | .778 | 4.0x |

The work ratio is computed only on the **same 21 jointly settled cases**, first
within each scene family and then equally across families. It does not assign
a fictional duration to failures or count a cold/null control as an edit win.
It is an offline response-work score, not 20% faster gameplay. For power 4 the
family ratios are hallway .550, house .933, cave 1.000. Cave settling speed is
unchanged: reducing early sampling noise remains a separate improvement target.

Power 6's aggregate advantage over power 4 is only about 2.8%, contributed by
one additional faster hallway-removal observation. It has more doorway and
null-control error and less favorable cave early error. With the user's
established preference for a smooth response, choose power 4's less aggressive
tradeoff. This is a measured compromise, **not** an all-cases improvement or a
claim of a universal optimum. Power 2 was screened out before holdouts and is
not ranked against the complete final set.

Do not introduce emitter-versus-wall or brightening-versus-darkening rules.
The occluder fixture demonstrates why geometry categories are not reliable
impact categories. The sunlight test supplies a visible floor patch and
checks both directions, but target sunlight is installed directly in these
replays: it does not qualify changing the runtime's separate sunlight-event
floor or notification mechanism.

Keep the current maximum statistical weight of 65,536. Keep 2,048 as a
human-calibrated quality/work anchor, not a perpetual ceiling or a scheduling
switch. The original cave table, the harder occluded-hall warm-control failure,
and the old strong-source stress case do not support certifying every static
world at that count. No new GPU, memory or FPS saving is claimed by this choice.

### Exclusions and integration boundary

The extra offset-32 hallway replay completed native capture/geometry audits,
but its positive reference is nearly black at normal exposure (mean sRGB
.00849, standard deviation .00422). It fails the unchanged .01 variation guard.
Its bracketing bright witnesses agree, but that does not make the dim reference
a qualified visual result. No thresholds were relaxed; this run contributes
**no** quality or performance pass. The queued offset-48 replay consequently
did not run. The far equation is unchanged, not newly requalified by those
stations. Evidence is retained at `target/decision-far-d32*` and
`target/gi-validation/hallway-1792010324600602000-92822`.

The next integration is bounded: apply the chosen curve to geometry retention,
remove the current flat-near-zone early exit in `bounds_limit`, and qualify
exact-event and compressed-journal enforcement under background streaming.
The storage summary's distance bands are conservative compression, not the
new policy's definition; their approximation must not silently replace the
per-receiver equation tested here. Keep direct-sun event behavior and all
transport/display settings fixed for that first gameplay comparison. This
round does not qualify repeated live edits, active terrain event delivery,
perceptual IR/radiation correctness or 60 FPS. No BVW changes were made.

### Reproduction and verification

`target/decision-final-paired.json` records the final input assessment hashes,
all per-arm errors and pairwise work ratios. Each `decision-*-provenance.json`
records its launch commit, tracked dirty diff, plan hash, native executable hash
and complete GI launch environment. Recorded executables all hash to
`3d150537301f1d422b02a8ab887df344cb1c71d764a5020db512e65c7ea6f7f8`.
The baseline native implementation remains `423e188`; this round changes only
measurement tools and documentation. Two static runs and twelve completed
curve assessments are the accepted evidence, not the rejected far run.

With NumPy/Pillow/FLIP 1.7 available and `tools` on `PYTHONPATH`:

```sh
python3 tools/near_history_run.py target/decision-repeat-d8-plan/plan.json --prefix target/decision-new-repeat-d8 --compact
python3 tools/history_decision_analysis.py compare target/decision-curves-*-assessment.json target/decision-repeat-*-assessment.json --output target/decision-final-paired.json
python3 -m unittest near_history_calibration.Tests history_distance_analysis.Tests history_decision_analysis.Tests distance_history_analysis.Tests
```

The final focused Python verification passes 24 tests across those modules.
Successful replay launches use `cargo run --locked --offline`, native geometry
and capture-count checks, valid offscreen targets and clean rendering logs;
this is not a rerun of the unrelated full Rust suite. Reproducible intermediate
PNGs were compacted only after assessment and hashing. Original static-sweep
images, native tapes, references, metrics and selected trajectory images remain.
