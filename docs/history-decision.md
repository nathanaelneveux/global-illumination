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
