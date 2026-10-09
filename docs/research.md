# Research and design decisions

The engine combines established techniques; it is not a full implementation of
any one paper. Separate a reference's reported results from results measured in
this repository. In particular, voxel traversal speed, estimator variance and
display stability are different problems.

NAADF's implementation findings below are pinned to a source revision, separate
from its paper's reported results. The vxpg entry uses the author's abstract,
not an equation-by-equation paper audit. Solari's entry is explicitly versioned,
not a claim about its latest implementation or platform compatibility.

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

#### Source implementation findings

Source revision [3ea3bb6](https://github.com/cg-tuwien/NAADF/tree/3ea3bb6a836aa4a30277c055f812d65a9d7d6766),
MIT licensed. The construction experiments below are independent changes to our
builder/cache, not copied shader implementations.

- **Bound construction has two schedules.**
  [boundsCommon.fxh](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/world/data/boundsCommon.fxh)
  performs three XYZ rounds for each 4³ block in GPU group-shared memory.
  [boundsCalc.fx](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/world/data/boundsCalc.fx)
  schedules top-level 4³ groups by bound size and axis using queues and indirect
  dispatch. Its `anyBoundsIncrease` flag is written but does not gate requeueing
  at this revision; this is not an early-fixed-point implementation. Our full-cycle
  early-exit experiment is separate and was rejected for mixed-workload overhead.
- **Edits repair bounds separately from lighting history.**
  [ChangeHandler.cs](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/World/Data/ChangeHandler.cs)
  and [worldChange.fx](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/world/data/worldChange.fx)
  identify affected groups, shrink unsafe bounds and requeue refinement. Small
  voxel/block fields are rebuilt locally. We retain transactional CPU-built
  snapshots; publishing conservative geometry before full accelerator refinement
  remains a possible larger change, not an implemented feature.
- **Traversal uses the packed directional bounds directly.**
  [rayTracing.fxh](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/render/rayTracing.fxh)
  preselects bound shifts from the ray direction, uses reciprocal direction and
  steps all tied axes with a vector mask. It also separates integer/fractional
  positions. Our vector box-step experiment is now retained, with the original
  XYZ boundary policy; the earlier reciprocal-only candidate did not show a win.
  Their step limits are 60–120 by
  ray type, versus our conservative 2,048; shortening ours is not a free equivalent
  optimization because incomplete traversal has explicit unknown semantics.
- **GI retains individual path samples, including their geometry.**
  [renderGlobalIllum.fx](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/render/versions/base/renderGlobalIllum.fx)
  stores 32-byte lit samples and 16-byte unlit samples. Lit records retain the
  original receiver, direction and bounce-hit information; only one eighth of
  unlit records is retained. It also applies radiance roulette and samples
  atmosphere on escaped paths at probability 1/16 with compensation. These are
  storage/work tradeoffs, not something our per-face means and moments already do.
- **Reuse includes rejection, weighting and a visibility ray.**
  [renderSampleRefine.fx](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/render/versions/base/renderSampleRefine.fx)
  reprojects history into 8×8 screen buckets, admits up to 32 lit candidates and
  writes at most seven into an eight-slot refined allocation. Unlit counts are
  weighted by eight. Its optional leveling and distance-outlier suppression
  deliberately modify sample radiance. In
  [renderSpatialResampling.fx](https://github.com/cg-tuwien/NAADF/blob/3ea3bb6a836aa4a30277c055f812d65a9d7d6766/NAADF/Content/shaders/render/versions/base/renderSpatialResampling.fx),
  twelve neighborhood candidates undergo normal/depth, PDF-ratio and Jacobian
  checks; weighted selection is followed by visibility testing. Direct sun still
  gets a separate visibility ray. This is much more than temporal blending.

**Implications for this engine:** exact uniform-field construction and topology-keyed
top-level reuse are now measured wins in [the experiment record](experiments.md#naadf-construction-experiments).
The largest unimplemented GI opportunity from this source is world-space reuse
of individual samples. A receiver-owned design must retain path geometry, proposal
density and all five energy bands; reconnect with current visibility; invalidate
source/geometry dependencies; and preserve unbiased zero-sample accounting before
adding any deliberate display-only bias. Screen buckets, RGB compression and
darkening heuristics are not suitable as the authoritative simulation field.

The isolated fresh-emitter resampling experiment (`fcfca22`) tested four new
candidates followed by one selected visibility ray, not retained-path reuse.
It used [weighted reservoir selection](https://www.pbr-book.org/4ed/Sampling_Algorithms/Reservoir_Sampling)
over all five bands, with the original MIS contribution inside the corrected
estimate. It regressed against our stratified emitter sampler and was rejected.
This result does not test or reject NAADF's spatial reuse or
[ReSTIR GI](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing).
Future reuse experiments must compare error per measured execution time and
preserve source diversity; adding candidate selection alone is not a win here.

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

The [October 8 experiment results](experiments.md) record completed branches.
Emitter-only scrambled Sobol sampling and explicit sky demand are now defaults.
Uniform AADF construction and topology-keyed top-level reuse also remove measured
scene-building work without changing transport samples.
Vector box stepping, sparse GPU metadata uploads and unchanged packed allocation
reuse now remove execution/preparation work as well; see the measured scope in
the experiment record. Re-evaluable path samples remain unimplemented.
The four-sector local guide, dense wavefront implementation and broader Sobol
variants were not promoted. The experiments below remain research directions,
not claims that those tested variants should be retried unchanged.

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
