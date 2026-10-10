# Adaptive history qualification

Baseline: `a3adafb`, including the accepted terminal-emission mixture. These
experiments qualify **world-space temporal-history mechanisms**, not complete
DDGI or A-SVGF renderers. The preferred corner presentation, transport model and
display smoothing remain fixed. No candidate is a production default yet.

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
