# Adaptive history qualification

Baseline: `a3adafb`, including the accepted terminal-emission mixture. These
experiments qualify **world-space temporal-history mechanisms**, not complete
DDGI or A-SVGF renderers. The preferred corner presentation, transport model and
display smoothing remain fixed. No candidate is a production default yet.

The subsequent [finite suspect-window study](history-suspect-halving.md) tests
the user's distinct compare/merge/**then**-halve rule on actual individual paths,
not the DDGI branch's variance-gated, pre-merge batch adaptation.

## Research selection (2026-10-10)

- [Production DDGI, section 4.3](https://jcgt.org/published/0010/02/01/paper-lowres.pdf):
  reduce irradiance hysteresis when new illumination disagrees with history;
  use the published two-threshold rule as a control, then measure a
  variance-aware adaptation rather than assuming single-batch disagreement
  implies a changed distribution.
- [A-SVGF](https://momentsingraphics.de/HPG2018.html): matched-seed reevaluation
  isolates temporal change from random-number changes. Our receivers have stable
  world coordinates, so screen reprojection is unnecessary. Sparse gradient
  reconstruction remains relevant; a missed path is not evidence of no change.
- Current [NRD history-confidence guidance](https://github.com/NVIDIA-RTX/NRD#history-confidence)
  still recommends matched-seed gradients and geometry-aware spatial filtering
  of the change evidence. This supports testing A-SVGF's mechanism rather than
  replacing it merely because the original paper is older. NRD also uses short
  and long histories for anti-lag; it warns that these require bias/noise tuning.
- [AMD GI-1.2](https://gpuopen.com/learn/gi-1-2-multibounce-indirect-rendering/)
  improves multibounce cache transport, while [FSR Radiance Caching](https://gpuopen.com/manuals/fsr_sdk/techniques/radiance-cache/)
  uses an online neural representation. Neither is a drop-in solution to our
  statistical history problem; a renderer/cache replacement would confound this
  comparison. This is a focused research check, not a claim of exhaustive SOTA.

## Qualification gates

First use deterministic, native-GPU sample tapes so all history candidates see
the same fresh paths. Include the many-emitter cave, exact-dark emitter edits,
one-block skylight opening/closing, unchanged high-variance lighting and a
physically hidden emitter that changes the sampling distribution. Keep RGB, IR
and ionizing measurements; never substitute display-filtered values as evidence.
References must be analytic/exact where available, otherwise two disjoint
high-sample replicas with reported disagreement.

Measure sustained error, false history reductions, residual old energy, temporal
variation, actual fresh paths and additional replay work. Spatial reconstruction
of change evidence must not cross incompatible faces. Replaying a path is work,
not a new independent sample. Compare equal fresh work **and** equal total ray
opportunities; virtual batches are not frames or gameplay latency.

Candidates that fail the fixed-work correctness/noise gate remain isolated on
their experiment branches. Only a passing candidate warrants integration and the
182-second streamed headless cave plus an edit/response workload. That second
gate must include runtime overhead and history/publication correctness. A good
native result alone is not a gameplay qualification or a reason to change defaults.

## DDGI branch: first-gate result

`codex/history-ddgi` evaluates the published absolute thresholds, scale-relative
thresholds, and variance-aware repeated halving (64/128 fresh-path windows,
two/three heuristic standard errors). These are linear five-band history
adaptations, not DDGI's perceptually encoded irradiance atlas or spatial filter.

Native tape: `eb02319` harness, generated at `a3adafb` plus that exact harness
diff; Apple M4 / Metal, dev opt1/deps opt3, no viewport or frame timings. Three
live prefixes per case; each starts with 4,096 paths and receives 8,192 fresh
paths in 16-path batches. Repeated null events occur every eight batches.
Controls: `target/history-controls.json`; candidates: `target/history-ddgi.json`.
Reference replica RGB p95 disagreement: cave .55%, emissive room <=.081%,
skylight 1.73%. Skylight all-band p95 disagreement is 13.2%, so it cannot certify
a fine 10% all-band opening threshold; the closed target is exactly dark.

Median over the three prefixes:

| Policy | Unchanged cave final RGB p95 error | Late face variation | Removal: sustained RGB 10%, fresh paths |
| --- | ---: | ---: | ---: |
| Retain | 1.19% | .089 | Not within 8,192 |
| Repeated cap256 / local cap16 on edits | 6.33% | 2.86 | 160 |
| DDGI absolute thresholds | 61.71% | 50.20 | Not within 8,192 |
| DDGI relative, low floor | 53.31% | 72.68 | 16 |
| Variance-aware halving, 128-path window / 3-error threshold | 1.27% | .352 | 1,040 |

Variation is compressed **face-measurement** total variation over the final
4,096 fresh paths, not the prior streamed-cave corner metric or a pixel metric.
The conservative halving arm still makes 75–86 false history reductions across
41 unchanged cave faces. It avoids the hidden-emitter false resets, but does not
detect the dim skylight closure; it inherits the long-retention tail there.
Lowering its absolute floor does not fix that uncertainty problem. Relative
thresholding responds rapidly to darkening but repeatedly destroys good history.

Decision: **do not promote these DDGI adaptations**. The variance-aware arm is
a useful tradeoff control, not a general winner: more noise than retention,
~6.5× the local-cap removal work, and a missed real geometry change. No expensive
streamed run is warranted for a first-gate failure. This does not disqualify DDGI
in its intended probe/filtering pipeline. Verified two evaluator tests, two
policy tests, full three-prefix tape evaluation and diff review. Runtime code
is unchanged; no gameplay build/run is claimed for these Python policy studies.

DDGI candidate code and branch-local evidence: `e371890`.

## A-SVGF/NRD branch: first-gate result

`codex/history-asvgf` tests matched-seed reevaluation of the last 16 or 64
pre-event paths, with local, coplanar-spatial, and sparse-spatial variants.
Continuous confidence uses `(1-relative_gradient)^4`; a second mapping retains
history below a 25% change threshold and rejects it above. These mappings are
our adaptations, not constants attributed to either paper. Spatial evidence is
averaged within 14 voxels on the same plane/orientation. This fixture-level
reconstruction is not a production connectivity filter. It filters signed
lighting differences, not cached radiance or simulation samples.

Same native tape and controls as `eb02319`: Apple M4 / Metal, dev opt1/deps opt3,
seven frozen-scene cases and three live prefixes. Warm history 4,096 paths,
8,192 fresh paths, 16 per batch. Artifacts: `target/history-asvgf.json` and
`target/history-controls.json`. Each target has two 131,072-path references;
cave RGB p95 replica disagreement .55%, emitter room <=.081%, skylight 1.73%.
Skylight all-band disagreement is 13.2%, limiting fine opening-error claims.
The closed skylight target is exactly dark, so stale closure tails do not depend
on a noisy reference.

All paired variants make **zero false reductions** in the unchanged cave, with
the same 1.19% median final RGB p95 error and .089 late face variation as retained
history at equal fresh work. Dense 16-path replay detects emitter addition,
removal and occlusion in all three prefixes: sustained RGB 10% after 16 fresh +
16 replay paths, versus 160 fresh paths for cap16. Correlated replay is charged
as work but never added as independent transport evidence.

| Detector | Hidden-emitter null: late variation | Skylight closure: sustained RGB 10%, total paths/site, three prefixes |
| --- | ---: | --- |
| Retain | .0072 | 5,360 / 5,872 / 4,784 |
| Cap16 | 1.219 | 32 / 32 / 32 |
| Local matched 16 | 1.132 | 5,376 / 5,888 / 4,800 |
| Local matched 64 | .229 | 5,424 / 4,160 / 4,848 |
| Spatial matched 64 | .063 | 4,656 / 80 / 4,848 |
| Sparse spatial matched 64 | .131 | 5,376 / 5,888 / 4,800 |
| Spatial matched 64, threshold | .0072 | 4,656 / 80 / 4,848 |

Variation is compressed face-measurement total variation over the final half of
the fresh-path tape, not a gameplay corner/pixel metric. The hidden emitter is
physically invisible to the sealed receiver room, but it changes the light
sampling proposal. Equal seeds then select different paths. Spatial averaging
plus a threshold handles this null case; raw local differences do not.

The skylight exposes the decisive limitation: the 16-path bank misses it on all
24 faces at edit time in all three prefixes. The 64-path bank finds evidence on
only 1 / 3 / 0 faces. Spatial reconstruction extends that to 8 / 19 / 0 faces;
it cannot reconstruct a change when no detector sampled it. The last prefix
retains the stale lighting essentially unchanged by the detector. This is not
fixed by a higher-confidence mapping or by ordinary display smoothing.

Cost: one replay event costs 16 or 64 additional paths per dense receiver.
For repeated null events every eight 16-path batches, those variants add 12.5%
or 50% ray opportunities; sparse 64-path replay averages 12.5%. Actual runtime
dispatch, memory and CPU overhead are unmeasured because the first correctness
gate failed. These are not frame-time speedups. No ray-bank storage or replay
work has been added to the ordinary game.

Decision: **promising change detector, not qualified as a replacement history
policy**. It preserves stationary evidence and rapidly detects well-sampled
changes, but absence of a replay difference cannot certify unchanged transport.
No candidate is promoted. A later experiment needs a principled bounded-lag
policy for unresolved changes, or better change-conditioned sampling, before a
full streamed implementation is warranted. Verified evaluator tests, two
paired-policy/boundary tests, three-prefix evaluations and diff review. Runtime
code is unchanged; this is not a full A-SVGF renderer or a gameplay run.

A-SVGF candidate code and branch-local evidence: `e544d37`.

## Equal-work closeout and reproduction

The comparisons above deliberately expose both equal-fresh-work quality and the
extra cost of replay. At an equal budget of 8,192 **total** path opportunities
per face (median across three prefixes):

| Policy | Unchanged cave RGB p95 error | Hidden-emitter null RGB p95 error |
| --- | ---: | ---: |
| Retain | 1.19% | .408% |
| Repeated cap256 | 6.33% | 2.068% |
| DDGI conservative variance halving | 1.27% | .408% |
| A-SVGF local 16 | 1.23% | 9.856% |
| A-SVGF sparse spatial 64 | 1.22% | 1.920% |
| A-SVGF spatial 64, threshold | 1.42% | .599% |

This reinforces the same decision, not a frame-time claim: sparse/filtered paired
evidence is promising, while naive same-seed differences are unreliable when
the emitter proposal changes. Neither branch solves the rare-path geometry case.
The default game therefore stays on the accepted sampler and existing history
policy; only the shared harness and research findings are merged. No default,
presentation mode, or runtime storage changed. Strict all-target Clippy passes
on the common harness; candidate evaluations/tests are recorded above.

The two branches are **qualification studies, not alternate `cargo run` modes**.
Their `tools/history_candidate.py` modules are intentionally absent from the
default branch. Stop the game before switching branches. Generate the common
tape once, then evaluate either candidate on exactly that tape:

```sh
VOXEL_GI_HISTORY_TAPE=target/history-qualification-tape.json \
  cargo test --locked record_history_qualification_tape -- --ignored --nocapture
git switch codex/history-ddgi
python3 tools/history_candidate.py
python3 tools/history_qualification.py target/history-qualification-tape.json \
  --output target/history-ddgi.json
git switch codex/history-asvgf
python3 tools/history_candidate.py
python3 tools/history_qualification.py target/history-qualification-tape.json \
  --output target/history-asvgf.json
git switch voxel-global-illumination
```

These are opt-in studies, not additions to the routine performance checklist.
Future work should use paired differences as positive change evidence while
treating missing evidence as unresolved, with a separately qualified bounded-lag
response. Tests must still include rare openings and sampling-proposal changes;
a fixed history cap or a brightness-specific exception is not silently counted
as successful change detection.
