# Research and design decisions

The engine combines established techniques; it is not a full implementation of
any one paper. Separate a reference's reported results from results measured in
this repository. In particular, voxel traversal speed, estimator variance and
display stability are different problems.

This review used the primary sources below. The NAADF publication page and vxpg
author abstract were accessible; their large/full PDFs were not retrieved in
this review. Do not treat those entries as a fresh equation-by-equation audit of
the papers. Solari's entry is explicitly versioned, not a claim about its latest
implementation or platform compatibility.

## What the references contribute

### NAADF, 2026

The [NAADF publication](https://www.cg.tuwien.ac.at/research/publications/2026/ulschmid-2026-naadf/)
describes nested axis-aligned distance fields for voxel traversal and incremental
updates. Its complete renderer also uses compressed lighting samples and
temporal/spatial reuse, including a 32-frame bounce-sample history and ReSTIR
spatial resampling. Acceleration alone is not the whole lighting method.

Here: nested empty-space bounds, incremental geometry, empty/uniform/mixed brick
packing and content deduplication are implemented. We do **not** implement that
complete renderer, its bounce-sample history or ReSTIR reuse. Our per-face means
and moments are not equivalent to retained, re-evaluable path samples. Published
speedups cannot be transferred to this code or machine without a paired test.

### vxray

The [repository](https://github.com/Nelarius/vxray) and
[development log](https://nelari.us/post/voxel-ray-tracing/) are especially useful
because they document reversals as well as wins. Relevant ideas include stable
low-discrepancy sequences, NAADF traversal, world-space radiance caches, wavefront
path tracing and explicit sampling of important directions. The log also records
costs from atomics and filtering on Apple hardware.

Here: persistent world-space receivers, stable sequences and sparse traversal are
implemented. We do not have vxray's screen-space reconstruction, wavefront queue
architecture or recursive world-space cache estimator. Their value depends on
our bottleneck; a recorded win in that renderer is a candidate, not a benchmark
for ours. Reusing direct-light code also requires preserving our existing
analytic-light and emissive MIS accounting rather than double-counting energy.

### DDGI and production irradiance probes

[Dynamic Diffuse Global Illumination (2019)](https://research.nvidia.com/publication/2019-05_dynamic-diffuse-global-illumination-ray-traced-irradiance-fields)
and [Scaling Probe-Based Real-Time Dynamic Global Illumination (2021)](https://jcgt.org/published/0010/02/01/paper-lowres.pdf)
address interpolation, visibility and history explicitly. The production paper
separates irradiance and visibility filtering, discusses perceptual encoding,
and adapts irradiance hysteresis after substantial changes. Long history trades
noise for response time; it does not create missing lighting information.

Our inference: keep simulation measurements separate from display, preserve
visibility/normal boundaries during interpolation, and use change evidence to
manage history. Blindly resetting history after a large four-path brightness
jump could mistake sampling noise for a real event. Our connected-coplanar
reconstruction and uncertainty-limited display filter are independent adaptations,
not DDGI or its directional depth-moment visibility model.

### SVGF

[SVGF (2017)](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced)
combines temporal accumulation, variance estimation and image-space wavelet
filtering. Its relevance is the separation of noisy measurements and their
reconstruction, not a mandate to put our authoritative cache in screen space.

Here: moments and temporal presentation serve related purposes. There is no
SVGF implementation, motion-vector reprojection or screen-space TAA in this GI
path. A future screen-space finishing filter must remain optional presentation;
offscreen simulation queries cannot depend on it.

### Solari, versioned Bevy 0.17 design

The [Bevy 0.17 release](https://bevy.org/news/bevy-0-17/) and its author's
[implementation discussion](https://jms55.github.io/posts/2025-09-20-solari-bevy-0-17/)
describe separate direct/indirect estimators and a position/normal/LOD-keyed
world irradiance cache. Cache update, reading and blending have explicit phases.
The discussion also identifies energy loss, stale illumination and outliers as
quality issues in that version.

Our inference: world-space reuse is well-established, but it introduces feedback
and invalidation obligations. Our paths currently evaluate explicit bounces;
replacing those bounces with cache lookup changes the estimator and needs its own
reference tests. No Bevy upgrade or Solari backend migration is required merely
to apply these architectural lessons.

### Path guiding

[Practical Path Guiding (2017)](https://jannovak.info/publications/PathGuide/index.html)
learns a spatial/directional importance distribution for incoming radiance.
[Real-time Voxel-based P-Guiding (2024)](https://weschang.com/publications/vxpg/)
targets useful guiding under limited real-time samples. The latter's voxel
organization does not mean it is specifically a block-world GI renderer.

These suggest a relevant next experiment: help paths find small sky openings or
emitters more often. Neither method is implemented here. A proposal distribution
must have a known PDF, retain support for newly appearing sources, and preserve
correct weighting. Reference implementation licenses must be reviewed separately;
this audit copies no external implementation code.

## Why indirect corners are still an experiment

Direct sun asks a known-direction visibility question. A shared position can
reuse that ray across face normals. Indirect transport estimates a hemisphere
integral: its normal, surface connectivity and origin policy matter. Collapsing
all faces at a position into one scalar value loses information at hard edges.

The CPU topology census in `reconstruction.rs` counts adapting the **current
oriented estimator** to `(integer corner, face normal, coplanar component)`:

| Fixture | Face-center states | Position-only corners | Oriented connected corner states |
| --- | ---: | ---: | ---: |
| 32 × 32 flat patch | 1,024 | 1,089 | 1,089 |
| Single cube | 6 | 8 | 24 |
| Two diagonally touching top faces | 2 | 7 | 8 |
| Prototype house selection | 1,019 | 1,044 | 1,324 |

This is a state-count test, **not a speed benchmark or a universal lower bound**.
A directional/full-sphere corner representation could share more, but is a
different estimator with different storage, visibility and sampling costs.
Primary corners may improve spatial appearance; they do not automatically reduce
the number of paths or the time to discover newly arriving indirect light.

Retain face transport plus virtual corners as the tested default. If implemented,
compare a corner estimator at equal work and equal image/field error, not merely
equal samples per receiver. Preserve disconnected-surface, hard-edge, emitter and
multi-band tests before considering promotion.

## Next speed experiments, in order

1. **Measure useful work after this cleanup.** Attribute time to scene assembly,
   scheduling, GPU tracing, readback/merge and display publication. Pair frame
   tails with fresh-field latency; an idle renderer with stale light is not faster
   GI. The audit removes redundant work, not a proven 2–3× speedup.
2. **Improve directions at the same ray budget.** Prototype a small world-space
   directional guide with a cosine-mixture fallback. Compare to the current
   estimator in opening/closing rooms, emitter addition/removal and steady scenes.
   A guide must relearn after changes, recheck visibility and consider infrared/
   ionizing sources even when invisible. Verify all bands against the reference.
3. **Reduce CPU/GPU transfers if profiling warrants it.** GPU-resident moments
   could remove repeated state uploads/readbacks. Preserve explicit versions,
   topology ownership, bounded readback and demand-driven simulation queries.
   Do not combine this storage experiment with a new estimator in one commit.
4. **Then evaluate primary corners or reusable path samples.** Both are broader
   changes than tuning presentation. Sample reuse needs current visibility,
   probability accounting and finite-memory/lifetime policies. A temporal blend
   of old RGB cannot substitute for those obligations.

All four are proposals, not promised gains. Hold the preferred corner look and
current temporal filter fixed while investigating speed. Change one mechanism
per checkpoint and retain a failed experiment's evidence if it is rejected.
