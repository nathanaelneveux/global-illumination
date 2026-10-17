# Edit-directed paired history experiment

Branch: `codex/history-edit-guided`, from `5bf8385`. This is an isolated native
transport/policy study, **not an alternate `cargo run` mode**. The earlier
suspect-halving branch is preserved. Runtime shaders, accumulation, scheduling,
display smoothing, and defaults are unchanged.

## Decision

**Useful change detector; not a general replacement for the current policy.**
Targeted comparisons avoid unnecessary strong history cuts behind an occluder.
Changed sunlight surfaces also supply the missing targets for indirectly lit
ceiling faces: the four-target arm detects all 24 receivers in the skylight
fixture. However, successfully detecting a rare opening and capping history at
16 does not reduce the variance of the subsequent path samples. Diffuse-only
multibounce changes can still be missed, delaying darkening with the 256 fallback.
Do not promote this into gameplay on these results.

The distinction is **change detection versus estimation of the new lighting**.
This experiment improves the former in specific cases; it does not establish a
better estimator, convergence speedup, or frame-time improvement.

## Mechanism and controls

1. Identify actual changed optical voxels, including the former positions of
   removed blocks. Extract actual production geometry-history caps for each
   evaluated receiver; do not fabricate a reset where the scene did not change.
2. Populate and refresh the existing shared-corner sunlight cache on stable
   fixture surfaces. Known old/new corner-value differences produce additional
   target positions, just inside the associated unit faces. Unknown values do
   not count as measured darkness. Neither reference results nor future path
   samples enter target selection.
3. Rank targets in the receiver's hemisphere by cosine/distance-squared, within
   the normal transport reach. Compare the **same aimed direction in the old and
   new scene**. This deterministic diagnostic sees environment on escape, or
   emission and analytic direct light at its first hit. It does not randomly
   choose emitters, so a changed light-proposal distribution cannot itself look
   like a temporal lighting change.
4. If any known pair differs by more than 25% in any of RGB/IR/ionizing bands,
   cap effective history at 16; otherwise cap it at 256. Apply once per actual
   event, never increase existing history, and then accumulate normally. Missing
   evidence uses 256, **not** unlimited retention or a claim of unchanged light.
   There is no halving window, debounce, repeated decay, or radiance clamp.

The three candidate arms use one geometry-only target (`geometry1`), one target
from the combined edit/sun pool (`combined1`), or up to four combined targets
(`combined4`). Controls use the actual baseline cap, unconditional cap16/cap256
on the same events, and retained history. The unchanged-cave synthetic marker
has **no actual edit**, so all arms preserve history there. The nearby hidden
emitter is a real edit inside the baseline's near-change radius, not a synthetic
invalidation or a special case in the classifier.

Diagnostic values never enter accumulated lighting; aimed samples would require
a correctly weighted proposal to be used as transport samples. The normal native
GPU kernel supplies all fresh/reference transport, unchanged for every arm.

This adapts the paired-comparison principle discussed in
[A-SVGF](https://momentsingraphics.de/HPG2018.html), not its complete screen-space
algorithm. Event-target selection and the 16/256 mapping are our experiment,
not constants or mechanisms attributed to the paper. Unlike the earlier
[replay study](adaptive-history-qualification.md), directions come from edit
metadata rather than the last random path bank.

## Evidence and reproducibility

- `47046c1`: test-only detector and schema-3 native tape, plus a nearby occluded
  emitter and diffuse-only skylight open/close fixtures.
- `4f66a44`: preserve distinct target positions at shadow discontinuities.
  Initially an angular deduplication collapsed four neighboring corner targets,
  hiding useful evidence. One ray still detects 14/24 ceiling receivers; four
  distinct targets detect 24/24. A focused regression test covers this boundary.
- `337b4bd`, `7154a01`: signed cache excursions and a separate unfloored relative
  brightness metric. Existing transport and floor-based metrics are unchanged.
- `73d7e83`: diagnostic shadow connections use the production finite visibility
  horizon; only diffuse segments use the transport-horizon uncertainty guard.
  A focused point-light endpoint test prevents geometry beyond the light from
  incorrectly classifying its clear connection as unknown.
- Qualification tape: `target/history-edit-guided-tape-v2.json`, recorded at `47046c1`
  plus exactly the detector diff committed as `4f66a44`. Concurrent Python-only
  metric changes do not affect the recorder. Initial tape/logs remain in
  `target/history-edit-guided-*`; final evaluation is
  `target/history-edit-guided-study-final.json` using the policy accompanying
  this document and shared metrics through `7154a01`.
- Final confirmation: `target/history-edit-guided-tape-final.json`, recorded at
  `7154a01` plus the Rust diff committed as `73d7e83`. All sample, reference and
  diagnostic fields exactly match v2; only timing fields differ. The final
  policy report is therefore reused without claiming another evaluation run.
- Apple M4 integrated GPU, Metal, dev opt1/dependencies opt3. No viewport or
  frame timing. Eleven cases, three disjoint live prefixes, 4,096 warm paths,
  512 batches of 16 fresh paths per receiver. Two 131,072-path reference replicas
  per scene. Nearby/remote hidden-emitter events repeat every eight batches.
- All eight original cases' sample/reference data match the previous study
  exactly, excluding dispatch timings. All eleven cases match between the two
  detector recordings. All 96 original fixed-policy evaluations reproduced
  exactly before adding the unfloored metric (excluding new excursion fields).
- Final evaluation: 231 case/seed/arm combinations. Every individual-path bank
  reconstructs its native batch moments. Cap choices are identical at 10%, 25%,
  and 50% diagnostic disagreement thresholds on these fixtures; this is not a
  broadly calibrated threshold sweep.

## Results

Median across three prefixes. The following response times use the **existing
floor-based RGB p95 error below 10%, sustained through the end**, in fresh paths
per receiver. They are not frames, seconds, or sample-coverage milestones.

| Change | Baseline | Geometry target | Four combined targets |
| --- | ---: | ---: | ---: |
| Remove both emitters | 160 | 160 | 160 |
| Add both emitters | 160 | 160 | 160 |
| Remove one emitter, retain the other | 144 | 144 | 144 |
| Occlude emitters | 160 | 160 | 160 |
| Open one-block skylight | 4,608 | 4,240 | 4,608 |
| Close one-block skylight | 32 | 336 | 32 |

The apparent opening difference is not an established convergence win. With the
floor removed, none of the opening arms reaches sustained 10% relative brightness
accuracy within 8,192 paths. The sunlit opening's median final relative brightness
error is 50.7% for both baseline and four-target arm. Its reference replicas
disagree by 12.7% at p95, so this reference cannot certify a fine 10% threshold.
The all-band opening check also fails in every prefix. The detector succeeds;
the rare-path estimator remains noisy.

For **nearby hidden emitter edits**, late compressed face-cache total variation
falls from **1.219 to .230 (5.3× lower)**. The candidate stays below the floored
10% RGB error threshold at every observed batch in all three prefixes; baseline
repeated cuts do not achieve sustained stability within the test. Final snapshot
RGB p95 error is 2.39% versus 2.07%; the important win is stability across time,
not the final snapshot. The distant hidden-emitter case is identical to baseline
because that already selects 256. Untouched cave accumulation is also identical.

### Signed overshoot

Peaks are measured at the 16-path batch boundaries in linear face-cache
luminance, outside the interval containing the initial cached value and both
reference endpoints, normalized by the brighter endpoint. Bright/dark excursions
are separate; ordinary fading lag and pre-existing warm error are not overshoot.
These are not pixel/display-smoothing measurements.

| Case | Baseline worst bright / dark excursion | Four-target worst bright / dark excursion |
| --- | ---: | ---: |
| Nearby hidden emitter, repeated edits | +45.2% / −47.1% | +8.76% / −7.79% |
| Visible emitter add/remove/partial removal/occlusion | 0 / 0 | 0 / 0 |
| Skylight opening | +5,078% / 0 | +5,078% / 0 |
| Skylight closure | 0 / 0 | 0 / 0 |

The large skylight percentages are relative to very dim physical references,
not screen brightness. They expose rare-path spikes; they should not be read as
equivalent perceptual percentages. Preserving 256 everywhere reduces the worst
opening excursion to +509%, but slows closure. Successfully detecting every
changed receiver and selecting 16 restores baseline's opening noise as well as
its closure speed. This is the unresolved tradeoff, not a detector false alarm.

### Diffuse-only skylight

With the directional sun removed, the same ceiling receivers see the opening
only through a diffuse bounce. The edited roof location lies outside their
downward-facing hemisphere. All three directed arms miss those receivers and
fall back to 256. Knowing the edited location supplies a useful source signal,
but does not itself construct the required two-segment path.

Both old floor-based error metrics misleadingly look settled after the first
batch in this extremely dim fixture. The new relative-brightness metric exposes
the error, but the two diffuse-only reference replicas disagree by **57.4% p95**;
do not claim precise opening accuracy or exact relative-reference response from
this reference budget. A stronger reference would be needed for that claim.

The closure tail is unambiguous without such a reference: the post-edit fresh
samples are exactly zero in the sealed scene. Reducing the **initial cached
brightness** by 90% takes 144 fresh paths at cap16 versus 2,304 at cap256. This
follows the measured policy's `cap / (cap + fresh_paths)` decay and is independent
of whether the pre-edit cache had converged. Thus the missed closure is genuinely
slower, not a noisy-reference artifact. It is lag, not an overshoot.

## Work and integration limits

| Diagnostic case | One-target extra traversal segments/face/event | Four-target extra traversal segments/face/event |
| --- | ---: | ---: |
| Both-emitter add/remove | 2 | 4 |
| One-emitter removal | 2 | 2 |
| Emitter occlusion | 2 | 8 |
| Nearby/remote hidden emitter | 2 | 2 |
| Sunlit skylight | 4 | 16 |
| Diffuse-only ceiling / no actual edit | 0 | 0 |

Counts include both old/new queries and analytic shadow segments. A traversal
segment is **not a full Monte Carlo path**; do not combine them into a fabricated
equal-cost convergence number. Repeated hidden edits cost 128 diagnostic segments
per receiver over the complete 8,192-fresh-path workload. No runtime GPU overhead
or FPS is established by these counts.

The sunlit fixture separately uses 2,662 cold and 2,662 post-edit sunlight-cache
segments over its stable surface set. Cold population is fixture setup, not an
incremental event cost. The update is shared work with the existing direct-sun
system, but not free. This test reconstructs whole fixture surface lists/cache
and scans changed regions; **that is not a scalable gameplay event pipeline**.
`record_cpu_ms` includes setup and every arm and is not a candidate frame time.

A production version would need bounded event target indexing, old/new geometry
residency (the new target direction was not necessarily sampled before the edit),
incremental sunlight-change publication, and GPU-side diagnostic scheduling.
Their memory/upload/scheduling costs are unmeasured. Newly exposed receivers,
unknown residency transitions, analytic light/material-table changes, arbitrary
multi-edit sequences, and further diffuse bounces are not qualified here.
Sun targets come from the current visible directional-light cache, not a general
IR/ionizing/point-light change cache; the diagnostic geometry path checks all
five bands, but the unchanged multi-band emitter fixture is not a dedicated
IR-only change qualification.

The next investigation should separate variance reduction from history selection
instead of adding another reset exception. Edit/sun-directed transport proposals
with correct weighting, or a separately qualified multibounce change signal,
are possible follow-ups, **not implementations or accepted winners here**.

## Run only when relevant

```sh
cargo test --locked edit_guided
VOXEL_GI_HISTORY_TAPE=target/history-edit-guided-tape-final.json \
  cargo test --locked record_history_qualification_tape -- --ignored --nocapture
python3 tools/history_qualification.py --test
python3 tools/history_edit_guided.py --test
python3 tools/history_edit_guided.py target/history-edit-guided-tape-final.json \
  --output target/history-edit-guided-study-final.json
```

Verification: six detector tests, four shared evaluator tests, four policy tests,
native recorder, individual-path moment checks, margin classification check,
`cargo fmt --check`, strict test Clippy, and diff review. No gameplay capture or
frame pacing claim; first-gate shortcomings do not justify an expensive streamed
qualification yet. Use `history_edit_guided.py`, not `history_candidate.py`
(the latter remains the preserved suspect study from this branch's ancestry).
Large reports/tapes/logs stay ignored under `target/`; all commits are local.
