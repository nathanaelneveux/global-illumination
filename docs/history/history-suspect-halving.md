# Finite suspect-history experiment

Branch: `codex/history-suspect-halving`, based on `1aeb931`. This implements the
user's exact proposed order in an isolated policy study. **It is not an alternate
`cargo run` lighting mode.** Runtime, display smoothing and defaults are unchanged.

## Rule

An affected receiver becomes suspect for the next **16 individual path samples**.
For each one, compare the raw sample with the **pre-merge** historical mean, merge
it at the current history weight, and only then halve the resulting weight if
the sample disagrees substantially. Agreement also consumes one suspect sample.
After the sixteenth sample, normal accumulation resumes. Another event rearms
the window but does not itself discard history. A cut changes future weighting,
not the just-merged mean or raw second moment. There is no replay, variance gate,
batch-mean substitution, immediate reset, radiance clamp or permanent decay.

The unspecified "large margin" starts at 50% relative disagreement in any of
the five bands, using `abs(old-new) > margin * max(abs(old), abs(new), 1e-6)`.
A single shared effective weight covers RGB, IR and ionizing energy. Sensitivity
arms use 25%, 75% and 90%, plus RGB-vector-only comparisons at 50% and 90%.
These vary only the disagreement definition, not the requested sequence/window.
RGB-only arms deliberately cannot detect simulation-only changes.

Effective weights are fractional: 4,000 plus one sample, then halved, is 2,000.5.
With an initial mean of one and consistently disagreeing zero-valued samples,
after `n` samples the weight is `1 + 3999 / 2^n` and the mean is
`4000 / (3999 + 2^n)`. The first sample leaves brightness at 99.975%; after 16,
5.75% remains. The seventeenth sample still merges, but no longer halves weight.
Tests cover these exact values and order, agreement, a lone outlier, rearming,
multi-band behavior and rejection of missing individual-path data.

## Matched evidence

- Harness commit `928edd2`: schema-2 tapes expose each of the 16 paths in the
  first post-event batch. All other batches remain aggregated. Every individual
  bank reconstructs the original native mean and second moment in every band,
  within floating-point reduction tolerance. All 84 original control evaluations
  reproduced exactly; this is not a comparison against a different sample stream.
- Policy and study driver: `eb268a3`. The final evidence uses that commit plus
  the one-emitter-removal fixture committed with this document; policy code is
  unchanged. All 240 earlier evaluations reproduced exactly after adding it.
- Apple M4 integrated GPU / Metal, dev opt1/dependencies opt3. Eight native cases,
  three disjoint live prefixes, 4,096 warm paths, 8,192 fresh paths per receiver.
  Two independent 131,072-path reference replicas per scene. No viewport or FPS.
- The ninth workload derives a **single** null event from the unchanged cave
  tape. It is a synthetic suspect marker, not a simulated terrain edit. This
  separates a one-window false alarm from repeated invalidation.
- The original cave-null-streaming workload marks every receiver suspect every
  128 paths. This is an adverse repeated-event control, **not** an implementation
  of the proposed "retain history on streaming, suspect only on edits" policy.
  If streaming does not mark a face suspect, this rule reduces to ordinary
  retained accumulation; the retain control measures that case. The separate
  single-null and hidden-emitter cases still test false alarms on edit markers.
- RGB p95 reference-replica disagreement: cave .55%, emitter scenes <=.081%,
  one-emitter post-edit scene .017%, skylight 1.73%. Skylight all-band disagreement
  is 13.2%; it cannot certify a fine 10% all-band opening threshold. Closed targets
  are exactly dark, so the closure/removal evidence does not have that limitation.

## Results

Fresh paths to sustained RGB p95 error below 10%, median over the three prefixes.
Error is measured against independent reference transport, not sample coverage.
Results are sampled every 16 paths, not every frame. No policy needs replay rays.

| Change | Retain all history | Local cap16 control | Suspect16, 50% five-band | Suspect16, 90% RGB-only |
| --- | ---: | ---: | ---: | ---: |
| Remove both lamps | >8,192 | 160 | 16 | 16 |
| Occlude both lamps | >8,192 | 160 | 16 | 16 |
| Add lamps | >8,192 | 160 | 32 | 32 |
| Remove one lamp, keep the other | >8,192 | 144 | 32 | 128 |
| Close one-block skylight | 5,360 | 32 | 16 | 16 |
| Open one-block skylight | 5,296 | 4,608 | 5,296 | 5,296 |

The strongest result is real: the bounded post-merge window removes stale energy
quickly, including the skylight closure missed by the preceding matched-replay
study. It also helps partial source removal. Rare openings remain hard: almost
none of the first 16 paths see the new opening, so the window expires without
enough change evidence. No later history cuts are allowed by this rule.

The tradeoff is false disagreement in steady illumination. For a **single**
suspect marker in the unchanged cave, median RGB p95 error after the 16-sample
window is:

| Policy | Immediate error | Fresh paths to sustained RGB 10% |
| --- | ---: | ---: |
| Retain | 2.27% | Already below; first observation 16 |
| Cap16 | 24.63% | 304 |
| Cap256 | 3.58% | Already below; first observation 16 |
| Suspect16, 50% five-band | 146.25% | 400 |
| Suspect16, 50% RGB-only | 53.45% | 368 |
| Suspect16, 90% RGB-only | 2.35% | Already below; first observation 16 |

The five-band rule halves 654–655 of the 656 possible times in this first window;
RGB-only at 50% cuts 329–344 times, and at 90% cuts 53–58. This is not evidence
that the lighting changed: individual paths are noisy even when their average
is well estimated. The finite lifetime does prevent permanent decay after a
single event, and all arms eventually recover; it does not prevent a large
transient excursion while suspect.

A high RGB threshold is not a general fix. In the sealed-room hidden-emitter
case, lighting is physically unchanged but the emitter proposal changes. Its
first-window RGB p95 error reaches 266% for the primary arm and 308% for 90%
RGB-only, versus .30% for retention. With repeated events its late compressed
face variation is 7.20 / 6.54, respectively, versus .0072 retained and 1.22 cap16.
These are **measurement** variation/error metrics, not observed screen flicker.
Similarly, repeatedly marking the unchanged cave gives final RGB p95 errors of
17.46% / 10.25%, versus 1.19% retained and 6.33% cap256.

Decision: preserve the exact experiment on its branch, **do not promote it to a
general runtime history policy**. The response-speed benefit is established, but
raw-sample disagreement is not reliable enough to keep stable evidence stable.
This decision is not based solely on treating streaming as repeated edits, nor
solely on a five-band threshold. Do not silently replace this experiment with a
batch/variance detector and call it the user's rule. Such a detector would be
a separate future comparison.

## Verification and reproduction

Eleven policy tests and two evaluator tests pass. Native recording and `cargo fmt
--check` pass for the final fixture. Strict all-target Clippy passed on the
schema-2 harness before adding the one-lamp fixture; it was not rerun for that
small test-only addition. The final study validates all individual-path banks
and evaluates 270 case/seed/policy combinations. No gameplay, visual, frame-time
or production GPU-overhead qualification is claimed. Runtime integration would
need to process individual samples without destroying normal GPU batching, and
preserve suspect state across scheduling/publication; neither cost is measured
by offline tape replay.

```sh
VOXEL_GI_HISTORY_TAPE=target/history-suspect-partial-tape.json \
  cargo test --locked record_history_qualification_tape -- --ignored --nocapture
python3 tools/history_candidate.py
python3 tools/history_qualification.py --test
python3 tools/history_suspect_study.py target/history-suspect-partial-tape.json \
  --output target/history-suspect-study-final.json
```

Large tapes/reports/logs stay in ignored `target/`. This is an opt-in study, not
an addition to routine gameplay performance testing. The default branch remains
unchanged, and no commits are published.
