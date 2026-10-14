# History policy decision

Branch `codex/gi-history-decision`, baseline `423e188`. This round separates
three decisions: maximum statistical weight, the near-edit distance equation,
and whether source-type rules are justified. Runtime defaults are unchanged
until the evidence below supports a specific promotion.

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
  zero-history edits; include a near and a mid-distance receiver.
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
bad. It is sufficient to reject that cap as a universal quality-preserving
default. The original cave's accepted 2,048 prefix does not guarantee an
arbitrarily long weighted stream, other cameras, or arbitrary player geometry.

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
.14127 (about 26%) without changing this screen's settling checkpoint. Addition
peak tile changes .17048 to .16323, a smaller benefit. Power 6 is slightly
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
