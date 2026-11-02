# GI performance experiment results

The [October 15 resident/path-reuse study](resident-path-reuse.md) records the
current runtime-ownership and early-sample experiments, including spectral
sampling priorities, qualified RGB images and rejected variants.

Retained changes improve emitter sampling, avoid repeated sky work, accelerate
geometry construction and traversal, and reduce geometry preparation/uploads.
Other candidates remain on separate local branches. The preferred shared
corner appearance, four paths per dispatch, three-bounce estimator, history
policy and temporal display filter are unchanged. There is no new debounce,
per-corner indirect transport, reservoir history or screen-space dependency.

The [October 9 follow-up](research-2026-10-09.md) records decisions on resident
ownership, edit differences, ReSTCV, many-light trees and cache correction. It
also includes the new playable cave and matched offscreen integration results.

## Baseline and decisions

Baseline `8b863cb` is preserved as `experiment/gi-baseline-20261008`. Paired shader
comparison infrastructure is `9186a55`. Winners were merged into
`voxel-global-illumination` at `f51e5df`; no commits were pushed.

| Experiment branch suffix after `experiment/gi-` | Commit | Decision |
| --- | --- | --- |
| `directional-guide` | `f8fba72` | Reject four azimuth sectors: higher room error and disturbed open-sky sampling; added probe memory was not justified. |
| `terminal-work` | `ab4cfdd` | Keep isolated: removing unused terminal-bounce preparation showed only a small preliminary gain. |
| `sampler-cost` | `ac9af28` | Keep isolated: constant Halton radices gave no compelling overall gain. |
| `sampler-dimension-recheck` (this branch) | see section | Reject all-dimension Sobol again: Halton's small-count correlation does not show in field error; hallway error is estimator-dominated. |
| `ray-reciprocals` | `0850c91` | Reject: no demonstrated throughput gain from explicit reciprocal reuse. |
| `wavefront` | `f0044d6` | Reject dense per-bounce passes: extra scratch and dispatches, slower open terrain, no room gain. |
| `scrambled-sobol` | `2034e48` | Do not promote all dimensions: strong sky/emitter improvement, but worse early room error. |
| `hybrid-sampler` | `70cb3f8` | Reject primary-Sobol/secondary-Halton mixture: did not resolve the room tradeoff. |
| `emitter-sobol` | `5f3cee2` | Retain: lower emitter error, original path directions and nonemissive results preserved. |
| `explicit-sky-work` | `a250cbb` | Retain: remove repeated warm sky lookups/copies from indirect-only work. |
| `aadf-convergence` | `5fecb63` | Reject full-cycle early exit: loaded islands improve, mixed terrain and bricks regress. |
| `uniform-aadf` | `21d2409` | Retain: solve empty/full acceleration volumes exactly, without iterative sweeps. |
| `topology-cache` | `b57d853` | Retain: share top-level acceleration across edits that preserve region residency and occupancy. |
| `vector-box-step` | `f7e57aa` | Retain: vectorize empty-space stepping while preserving the scalar XYZ boundary policy. |
| `directional-planes` | `ffbabde` | Reject: integer plane selection adds no gain over vector stepping. |
| `geometry-upload-deltas` | `9bac5c8` | Retain: rebuild/upload the directory only for residency changes and patch changed headers. |
| `packed-region-reuse` | `be1386a` | Retain: preserve unchanged region allocations while still updating their skip bounds. |
| `emitter-resampling` | `fcfca22` | Reject fresh four-candidate RIS: increased colored-emitter noise and GPU cost. Not temporal/spatial path reuse. |
| `task-encoding` | `0020e7d` | Retain: encode complete task records directly into reusable upload storage. |
| `fresh-spatial-reuse` | `be32d40`, harness through `bd60c3f` | Keep isolated: two-receiver reuse lowers ceiling error at matched GPU budgets, but real task ordering pairs only 28% of faces. No end-to-end win established. |
| `fresh-spatial-four` | `337bb87` | Reject this four-receiver implementation: substantially slower and invalid at high sample count. |
| `spatial-neighborhoods` | `8ed313f` | Defer promotion: 94% pairing, but higher worker costs without an established end-to-end win. |
| `recursive-cache-contract` | `9a6c078` | Defer recursive termination: feedback changes bounce depth and source-removal semantics; reject unversioned mean reuse. |
| `shared-angular-guide` | `a8ceaef`, lint fix `2de5d91` | Reject fixed shared 64-bin proposal: more error and CPU work in all three fixtures. |
| `resident-tick-prototype` | `34c3476` | Keep architecture direction: matching fixed-slot transport with fewer readbacks; not production world residency. |
| `incremental-brick-repair` | `4196b88`, merge `bdac2dc` | Retain exact local repair. Defer coarse-to-fine/background publication. |

Rejected variants are not dormant production toggles or extra runtime resources.
A rejected implementation does not disprove its broader research family: the
guide lacked spatial sharing and a rich angular distribution; the wavefront test
did not compact live paths or separate shadow queues.

## Emitter sampling

The implementation follows the shuffled, padded, hash-scrambled Sobol approach
described by [Sobol-Burley](https://github.com/cessen/sobol_burley), based on
[Burley 2020](https://jcgt.org/published/0009/04/01/). Its reverse-bit scramble is
adapted under the included `crates/voxel_gi/licenses/sobol_burley-MIT.txt`. No external direction
tables or dependency were added. The first two Sobol dimensions are computed
arithmetically; separate dimension pairs have separate seeds. Existing light
PDFs, power weighting, visibility and five-band MIS remain unchanged.

Apple M4 / Metal, Rust 1.95.0, development profile from `validation.md`. Native
isolated compute, no viewport. Baseline shader `8b863cb` versus candidate
`5f3cee2` code; 100 receivers, four paths per batch, eight disjoint aligned
1,024-path runs with alternating execution order. Open sky and two colored
ceiling emitters have independent analytic references; the skylight room uses
one shared 65,536-path baseline reference from a disjoint sequence block.

| Five-band error metric | Baseline | Emitter Sobol |
| --- | ---: | ---: |
| Emitter normalized absolute error p95, 64 paths | 0.063570 | 0.006746 |
| Emitter normalized absolute error p95, 1,024 paths | 0.004275 | 0.002146 |
| Open-sky error, every checkpoint | Original | Identical |
| Nonemissive skylight-room error, every checkpoint | Original | Identical |

Normalization is `abs(error) / (0.02 + abs(reference))`, pooled across receivers,
replicates and five bands. The 64-path emitter p95 is **9.4 times lower**, not a
9.4-times whole-engine speedup or a universal convergence guarantee. Emitter
GPU batch median was 0.08292 → 0.08054 ms; reduced noise is the principal win.
No sample-count or scheduling budget was reduced to obtain it.

Evidence: `target/emitter-sobol-paired-study.log`,
`target/emitter-sobol-native-parity.log`. Distribution tests cover dyadic 2D
stratification through high sequence blocks, normalized cosine sampling and exact
preservation of all twelve old path dimensions. Native CPU/GPU standard and
emissive-room references report RMS 0.000000, including edits and extra bands.

## Path-dimension sampler recheck (October 17)

Rotated Halton correlates its high prime bases at small counts. Roulette (dimension
9, base 29) and the terminal light/cosine choice (dimension 10, base 31) have a mean
absolute correlation of 0.58 over 16 samples. Their 4x4 coverage is 3.5x worse than
independent samples at 16 paths and 6.5x worse at 256. The terminal-direction pair
(17/19) is about 2x worse at 16 paths. The terminal mixture introduced dimensions
6, 7 and 10 on October 10, after the October 8 all-dimension Sobol check.
`sampling.rs` checks joint coverage only at 65,536 samples.

The hypothesis was that this explained persistent per-face early noise. The
ignored native test `sampler_dimension_study` tests it. It compares production
Halton, padded Owen-scrambled Sobol for every dimension, and an IID counter control
on the same estimator and fixtures. Fixtures: 237 loaded-hallway sites and 165 cave
material sites; six disjoint starts; prefixes 16 to 2048. References are two
disjoint 262,144-path IID blocks. Apple M4 / Metal, optimized dependency test
profile, no app running. Analysis: `tools/sampler_dimension_analysis.py`.

Median per-face relative RGB error (max channel / reference max channel):

| Fixture | Paths | Halton | Sobol | IID |
| --- | ---: | ---: | ---: | ---: |
| Cave | 16 | .1943 | .1733 | .1994 |
| Cave | 256 | .0416 | .0437 | .0537 |
| Cave | 2048 | .0133 | .0142 | .0196 |
| Hallway | 16 | .5407 | .5192 | .5581 |
| Hallway | 256 | .4202 | .4067 | .4098 |
| Hallway | 2048 | .2627 | .2543 | .2636 |

**Decision: reject.** The sequence correlation is real, but it does not appear in
field error. Halton stays slightly ahead of all-Sobol in the cave from 64 paths.
In the hallway all three samplers are equal, and the mean squared error is
dominated by heavy tails for every sampler. The hallway's 26% median error at
2048 paths does not depend on the sampler: it comes from the estimator. Its
reference blocks disagree by up to 15% at p95, so hallway values are diagnostic
only. Evidence: ignored `target/sampler-dimension-study.json`.

## Explicit sunlight demand

An indirect refinement batch no longer implies a sunlight refresh. Missing
corners still request work immediately; measured black is valid. Geometry and
sun revisions retain their urgent/global refresh queues. A combined job can do
both, but unrequested zero-initialized sky output never replaces valid display.
This changes CPU work ownership, not the sunlight estimator or ray budget.

At `a250cbb`, Apple M4 / Metal, same development profile: 8,192 warm floor
receivers × four paths, 40 alternating pairs, no viewport. Compare the old
implicit sky-demand behavior to explicit demand in the same binary, with
identical five-band means/moments, direct energy, sky values and sequences.

| Time in milliseconds | Implicit sky demand | Explicit demand |
| --- | ---: | ---: |
| CPU preparation and decode p50 | 0.705 | 0.285 |
| CPU preparation and decode p95 | 0.790 | 0.345 |
| Whole worker p50 | 2.316 | 1.874 |
| Whole worker p95 | 2.388 | 1.947 |
| GPU p50 | 0.377 | 0.375 |

A second 40-pair run measured CPU 0.681 → 0.297 ms and worker 2.268 → 1.880 ms.
That is **56–60% less preparation/decode time and 17–19% less worker time** for
this workload, not those percentages off total frame time. Both modes traced
the same indirect paths and zero new sunlight rays; the saving is lookup/copy
work that remained even when visibility was cached.

Evidence: `target/explicit-sky-demand-paired.log`,
`target/explicit-sky-demand-confirm.log`, `target/explicit-sky-parity.log`.
Focused tests cover queued overlap, unknown retries, measured-black validity,
untouched sunlight and mixed sky/partial-face simulation parity.

## Integration checkpoint

At `f51e5df`: 132 regular tests pass, 16 opt-in tests ignored, strict all-target
Clippy and development build pass. Native emitter removal still reaches exact
zero transport in its first four-path batch; the unchanged display falls below
10% in eight batches and 1% in twelve at the fixture's virtual 50 ms cadence.
That is a controlled response test, not a gameplay latency measurement.

### House edits

Baseline binary built at `8b863cb`; candidate binary built at clean `f51e5df`
(only documentation edited during the run). Apple M4 / Metal, development
profile, seed 7, terrain period 256, textures on, baked AO off, preferred corner
modes. Bevy offscreen 2560 × 1440, 60 Hz schedule runner; no simultaneous compiler
or GPU benchmark. Both ran 180 seconds with nine valid, nonblank captures, assets
verified and no render warnings/errors. Both removed one roof voxel and three
voxels in each doorway at the scheduled times, confirmed by `house.json`.
Early/late skylight and late turned-corner images retain textured dark interiors
and smooth corner shading. Sparse captures cannot establish every-frame stability.

| Offscreen observation | Baseline | Combined candidate |
| --- | ---: | ---: |
| Maximum loaded pages | 1,087 | 960 |
| Traced paths | 117,404,264 | 100,178,948 |
| Skylight cadence p99 | 31.82 ms | 29.88 ms |
| West-door cadence p99 | 31.63 ms | 30.40 ms |
| Turn cadence p99 | 34.87 ms | 32.11 ms |
| Skylight near-field age p95 | 1.98 s | 1.92 s |
| Turn near-field age p95 | 1.93 s | 1.95 s |
| Startup maximum frame | 621.90 ms | 617.32 ms |

Other phases are mixed: first-corner median cadence was 19.24 → 22.09 ms and
returned-corner p99 was 28.10 → 30.19 ms. Different streamed world sizes and
asynchronous scheduling prevent attribution of a whole-game speedup or slowdown
from this pair. These are offscreen cadence observations, not windowed FPS.
Large startup/streaming spikes remain; the controlled CPU benchmark, not these
cadence percentiles, establishes the removed-work win.

Artifacts under `target/gi-validation/`:
`house-edits-1791489290305296000-58498/` and
`house-edits-1791489612635699000-63528/`.

### Source changes

Combined `f51e5df`, documentation-only dirty files, same hardware/profile/viewport
and preferred settings: the 72-second `sources` workload completed with five
valid captures, verified assets and clean render logs. Active-emitter and
recovered-terrain images were inspected. It traced 90,937,568 paths, reached
100% coverage and loaded at most 410 pages. Active-source near-field age p95 was
140.73 ms; streaming p95 was 141.87 ms. Offscreen cadence p99 was 34.89 ms and
41.91 ms respectively; this single integration run is not a speed comparison.
Artifact: `target/gi-validation/sources-1791489826571058000-66834/`.

## Research still open

The [vxray wavefront design](https://nelari.us/post/voxel-ray-tracing/) motivated
the execution experiment, but its benefit did not transfer to this dense queue
on M4. More expressive guiding, edit-difference transport, retained re-evaluable
samples and NAADF-style resampling remain separate experiments. None is implied
by changing the sample generator or avoiding redundant cache access.

A 32-frame history can be a reasonable latency budget at 60 Hz, but half a second
of delay is not evidence of correct visibility or probability accounting. Reuse
experiments still need current-geometry/source checks, bounded state and both
brightening and darkening references before promotion.

## NAADF construction experiments

The next series starts at `af4a724`, preserved as
`experiment/gi-baseline-naadf-source-20261008`. Source findings are pinned in
[the research notes](research-design-notes-2026-10.md#source-implementation-findings). These are
CPU acceleration-construction changes; GI paths, source evaluation, history,
presentation and the GPU shader are unchanged. No external implementation was
copied. Winners were merged at `ee136fc` and `01d7e43`; rejected early termination
remains isolated, with no dormant runtime switch.

All timings below: Apple M4, Rust 1.95.0, development `opt-level=1`, shared-cache
settings from `validation.md`, isolated CPU execution with no viewport. Baseline
and candidate alternate in one test binary after warmup. Frozen test-only
references keep the original algorithms available without production resources.

### Exact uniform fields

`dac6dcf` freezes the `af4a724` builder, independently retaining its neighbor rule.
Its baseline control timings agree with the unchanged production implementation.
Candidate `21d2409` was measured as `dac6dcf` plus its `naadf.rs` diff: 20 measured
pairs after four warmups, 256 builds per small-volume observation.

| Construction p50 | Full sweeps | Uniform fast path |
| --- | ---: | ---: |
| Empty 4³ brick | 3.635 µs | 0.089 µs |
| Solid 4³ brick | 0.529 µs | 0.037 µs |
| Mixed planar 4³ brick | 2.015 µs | 2.007 µs |
| Mixed scattered 4³ brick | 2.574 µs | 2.588 µs |
| 48×16×48 terrain field | 25.952 ms | 25.707 ms |

The win is avoiding relaxation and a second allocation for uniform volumes,
including uniform bricks inside mixed regions. Mixed timings are effectively
unchanged, not a separate claimed improvement. Every field equals the full-sweep
reference. Tests exhaust all 2³ occupancy patterns and cover seeded volumes,
uniform fields, zero/single-cell dimensions and each 255-cell axis.

The rejected `5fecb63` candidate stopped only after an unchanged full XYZ cycle.
It reduced the loaded-island fixture from 7.321 to 1.953 ms, but top terrain rose
25.938 → 26.827 ms and scattered occupancy 27.952 → 29.230 ms. Its bookkeeping
also slowed empty/mixed bricks. This result does not reject NAADF's queued GPU
refinement, which uses a different schedule.

Evidence: `target/aadf-baseline-control.log`, `aadf-uniform-paired.log`,
`aadf-uniform-tests.log`, `aadf-uniform-world-tests.log`,
`aadf-convergence-paired.log` and `aadf-convergence-tests.log`.

### Topology keyed reuse

Top-level skip bounds depend on region membership and empty/nonempty occupancy,
not every voxel/material change. `b57d853` shares an immutable `Arc<[Aadf]>` when
those inputs match. Edits still replace changed region objects and generate
normal transport/source changes. Last-solid removal, first-solid addition and
residency changes rebuild bounds; older snapshots remain valid.

Harness `19dc564` preserves the old rebuild-on-any-region-change decision.
Its control pair measured 3.445 versus 3.450 ms. Candidate measurements use that
commit plus the two-file diff committed as `b57d853`, including the uniform-field
winner. Each pair deletes one surface voxel among 8,192 resident regions
(1,024 pages), building a fresh successor of the same immutable baseline.

| Complete scene construction | Old rebuild policy | Topology reuse |
| --- | ---: | ---: |
| First run p50 | 3.447 ms | 0.145 ms |
| First run p95 | 3.664 ms | 0.162 ms |
| Repeat p50 | 3.483 ms | 0.143 ms |
| Repeat p95 | 3.551 ms | 0.151 ms |

Forty measured alternating pairs per run follow four warmups. Approximately 24×
faster **scene construction for this edit**, not 24× FPS or faster ray integration.
References match materials, material masks, every voxel/brick/top-level bound and
residency. Ordinary tests also cover material-only edits, occupancy transitions,
negative coordinates, same-count island replacement and source invalidation.

Evidence: `target/topology-baseline-control.log`, `topology-cache-paired.log`,
`topology-cache-confirm.log`, `topology-cache-tests.log`.

### Integrated verification

Code committed as `b57d853` and merged unchanged at `01d7e43`: 135 regular tests
pass, 18 opt-in tests ignored; strict all-target Clippy and development build pass.
Native Metal CPU/GPU reference RMS is 0.000000 for ordinary and emissive fixtures,
with edit upload/sky invalidation and extra-band checks passing. No toggle stress
run was needed. Logs: `target/naadf-source-integration-tests.log`,
`naadf-source-clippy.log`, `naadf-source-build.log`, `naadf-source-native-parity.log`.

At `01d7e43` with only documentation edits during the run, the 35-second
`streaming` workload completed on M4/Metal, development build, 2560×1440 offscreen,
preferred corner look, textures on and AO off. All three captures had valid
targets and nonblank content; streaming/recovery images were inspected and render
logs were clean. It loaded at most 635 pages and ended at full coverage.
Streaming cadence p99 was 43.38 ms with a 248.00 ms maximum; near-field age p95
was 1.95 seconds. Large stalls remain. This is an integration check, not a paired
gameplay speed claim, and these timings are not windowed FPS.

Artifact: `target/gi-validation/streaming-1791494602912343000-36808/`;
log: `target/naadf-source-streaming.log`.

## Traversal and GPU geometry updates

Baseline `946888c` is preserved as `experiment/gi-baseline-traversal-20261008`.
Three compatible winners were merged at `de2336f`, `b15aa25` and `27df3bd`.
These changes retain the original paths, three bounces, five energy bands, history
and presentation. They do not implement path-sample reuse or reduce sampling work.
All isolated measurements below use Apple M4/Metal, Rust 1.95.0, the same
development/shared-cache profile and no viewport.

### Vector box stepping

`f7e57aa` replaces the dynamic XYZ loop in `advance_box` with vector arithmetic
and ordered selects. It preserves the scalar rule even when a crossing is slightly
negative: clamp the accepted distance before comparing the next axis. It does not
adopt NAADF's simultaneous tied-axis normal, shorten rays or add a cache.

Paired shader baseline `946888c` versus its WGSL-only diff committed as `f7e57aa`:
30 alternating warmed pairs, identical advancing sequences, four paths per face.
The follow-up `ffbabde` integer-plane selection candidate showed no added benefit
(room GPU median 1.406 → 1.430 ms) and remains isolated.

| GPU batch p50 | Baseline | Vector step | Repeat baseline | Repeat vector |
| --- | ---: | ---: | ---: | ---: |
| 1,024 open-floor faces | 0.063 ms | 0.051 ms | 0.060 ms | 0.049 ms |
| 8,100 faces in nine emissive rooms | 1.817 ms | 1.416 ms | 1.798 ms | 1.419 ms |

Room GPU time falls 21–22%. Whole-worker medians are essentially unchanged
(4.714 → 4.724 ms; repeat 4.606 → 4.619 ms), and p95 timings are mixed. This frees
GPU execution capacity but does not establish equivalent whole-engine throughput
or frame pacing. Five-band moments, direct lighting, sample counts and sky output
pass the paired comparison; inactive lanes and 1/7/64-path cases are included.

The native boundary test covers 210,001 cases across brick/region scales, negative
coordinates, zero and near-zero components, equal crossings and small negative
crossings. Normals equal the independent scalar oracle; distances agree within
floating-point tolerance. Native CPU/GPU transport RMS is 0.000000 in ordinary
and emissive fixtures, including edits and extra bands.

Evidence under `target/`: `vector-box-step-paired.log`,
`vector-box-step-confirm.log`, `vector-box-step-native-parity.log`,
`vector-box-step-boundaries.log`, `directional-planes-paired.log`.

### Sparse metadata uploads

`9bac5c8`, measured as `de2336f` plus its committed three-file diff, changes
`PackedScene::update` to return directory changes and header/cell element ranges.
Directory membership is unchanged by ordinary voxel edits. Existing buffers get
coalesced patches; growth triggers a complete array upload, so sparse updates
cannot omit old contents when allocations change. Buffer-identity binding checks
are unchanged.

The native benchmark alternates deleting/restoring one surface block among 8,192
regions (1,024 pages), tracing 64 receivers × four paths. Forty measured pairs
follow four warmups. The test-only reference shares dirty bookkeeping but forces
the former directory rebuild and full metadata uploads.

| First comparison | Full metadata | Sparse metadata |
| --- | ---: | ---: |
| CPU preparation p50 | 1.257 ms | 1.079 ms |
| CPU preparation p95 | 1.543 ms | 1.320 ms |
| Whole worker p50 | 2.850 ms | 2.673 ms |
| Uploaded geometry per edit | 393,216–393,600 bytes | 16–400 bytes |

Repeat preparation p50 was 1.263 → 1.093 ms and worker 2.830 → 2.693 ms.
Lighting outputs are equal. The transfer reduction is over 99.8% in this
occupancy-preserving edit fixture, not for loading new terrain or changed
top-level bounds. Evidence: `geometry-deltas-native.log`,
`geometry-deltas-confirm.log`, `geometry-deltas-parity.log` under `target/`.

### Unchanged packed allocations

`be1386a`, measured as `b15aa25` plus its two-file diff, avoids map
removal/reinsertion and no-op allocation handling for unchanged region identities.
Their top-level bounds still get checked and patched: unchanged voxel data does
not imply unchanged skip distances after neighbouring residency/occupancy changes.

The same fixture rotates three variants in 40 measured trios after four warmups.
The reference bypasses the new identity lookup; exploratory runs that charged
that lookup to every variant are not used for promotion. All variants retain
common dirty bookkeeping, and their complete transport outputs match.

| First corrected comparison | Full metadata | Sparse metadata | Sparse plus allocation reuse |
| --- | ---: | ---: | ---: |
| CPU preparation p50 | 1.469 ms | 1.253 ms | 0.871 ms |
| CPU preparation p95 | 1.812 ms | 1.564 ms | 1.078 ms |
| Whole worker p50 | 3.084 ms | 2.853 ms | 2.475 ms |
| Whole worker p95 | 3.426 ms | 3.195 ms | 2.690 ms |

Repeat preparation p50 was 1.459 / 1.247 / 0.874 ms and worker p50 was
3.056 / 2.843 / 2.474 ms. Allocation reuse saves about 30% of preparation versus
sparse uploads alone. Together they save about 40% of preparation and 19–20% of
whole-worker time versus the full-metadata reference. These are edit-dispatch
costs, not overall FPS or convergence claims.

Evidence: `target/packed-region-reuse-fair-control.log` and
`target/packed-region-reuse-fair-confirm.log`. Native comparisons cover buffer
growth, unload, recycled headers, negative coordinates and empty-region changes.
A CPU upload mirror also checks every header's bounds against the current scene.

### Integrated verification

At code `be1386a`, merged unchanged at `27df3bd`: 136 regular tests pass, 21
opt-in tests ignored, strict all-target Clippy and development build pass. Native
CPU/GPU RMS remains 0.000000 for ordinary and emissive fixtures. Targeted native
growth/recycling checks cover the upload changes; toggle stress was not needed.
Logs under `target/`: `traversal-upload-integration-tests.log`,
`traversal-upload-clippy.log`, `traversal-upload-build.log`,
`traversal-upload-native-parity.log`.

The integrated `edits` profile at `27df3bd` (documentation-only dirty changes)
completed offscreen on Apple M4/Metal using the development/shared-cache build,
2560 × 1440 target and preferred corner presentation. All 18 captures had valid
targets and nonblank content; render logs had no warnings/errors. The response
log records all five synthetic edits. The profile's generic edit counter does
not count these scripted events and must not be used as their verification.
Inspected early, 400 ms and recovered captures show textured room/surface geometry
and the resulting cavities. Some 50 ms captures precede mesh publication; this
check does not establish instant edit response or a flicker improvement.

Offscreen cadence p50/p95 was 22.60/30.61 ms during underground edits and
22.67/30.77 ms during the surface edit. Nearby measurement-age p95 was 1.26 s and
1.77 s respectively; final coverage was 100%, not proof of convergence. Startup
still included a 685 ms frame. There is no paired application baseline here,
and these timings are not windowed FPS or a gameplay speedup claim.

Artifact: `target/gi-validation/edits-1791505230889909000-94935/`;
log: `target/traversal-upload-edits.log`.

## Fresh resampling and task serialization

Baseline `d893bc4` is preserved at `experiment/gi-baseline-resampling-20261008`.
Both experiments use Apple M4/Metal, Rust 1.95.0, development/shared-cache profile
and no viewport. They change separate mechanisms and were tested independently.

### Fresh emitter resampling rejected

`fcfca22` is a GPU-only experiment: four fresh emitter candidates are scored
before tracing one selected shadow ray. The weight is the maximum of the original
MIS-weighted contribution's five bands. Selected estimates are corrected by
`weight_sum / (4 * selected_weight)`; zero candidates remain in that denominator.
The complementary cosine-hit estimator is unchanged. There is no persistent
reservoir, reconnection or temporal reuse, and the CPU tracer remains the baseline.

The paired sampler study used eight sequence blocks, 100 receivers and four
paths per dispatch, comparing both kernels to the same analytic/high-sample
reference. At 64 paths, colored-emitter normalized error p95 rose from 0.006746
to 0.074010; at 1,024 paths it rose from 0.002146 to 0.018359. GPU batch median
rose from 0.069334 to 0.080083 ms. The no-emitter open sky and skylight room outputs
were unchanged. Selecting one colored source adds variance that defeats this
fixture's existing low-discrepancy sampling. Do not promote this candidate or
extrapolate its failure to retained-path reuse.

WGSL validation and the reporting benchmark passed; those passes mean the study
ran, not that quality passed. Logs: `target/emitter-resampling-wgsl.log` and
`target/emitter-resampling-paired.log`. A high-sample bias/occlusion study and CPU
implementation were unnecessary after this clear early rejection.

### Direct task encoding retained

`0020e7d`, measured as `d893bc4` plus its three-file diff, removes the intermediate
typed GPU task vector. A reusable byte vector receives one complete 32-byte record
at a time. Word-at-a-time appends were slower in an initial trial and are not the
retained implementation. The test-only reference uses the generic ShaderType
encoder; the comparison shares the remaining dispatch code rather than running
two archived executables.

Forty alternating measured pairs follow four warmups at 17, 1,024 and 8,192 faces,
four paths each. Cold mixed sky/indirect batches also check correctness; timed
batches are indirect-only. Full five-band means/moments, direct energy, sequences,
sample counts and corner sky outputs match exactly.

| 8,192 faces | Generic | Direct | Repeat generic | Repeat direct |
| --- | ---: | ---: | ---: | ---: |
| Preparation p50 | 0.2326 ms | 0.1595 ms | 0.1793 ms | 0.1233 ms |
| Preparation p95 | 0.3112 ms | 0.2048 ms | 0.2157 ms | 0.1554 ms |
| Worker p50 | 1.9473 ms | 1.8677 ms | 1.8751 ms | 1.8292 ms |
| Worker p95 | 2.1185 ms | 1.9966 ms | 2.0247 ms | 1.8910 ms |

Preparation falls about 31%, while worker medians fall 2–4%. At 1,024 faces the
preparation saving is 0.011–0.012 ms; 17-face differences are noise-sized. This is
reduced CPU dispatch overhead, not improved GPU tracing, convergence or FPS.
Scratch retains its largest dispatched capacity (256 KiB of task contents for
8,192 indirect-only faces), replacing recurring allocations; it is not a cache
allocated per loaded world face.

Verification: byte-layout oracle, three focused GPU unit tests, native CPU/GPU
ordinary/emissive RMS 0.000000, sky-only/partial-face parity, buffer growth/rebinding
and strict all-target Clippy. No presentation, scheduler, estimator or history
changes required another image profile. Logs under `target/`:
`task-encoding-record-paired.log`, `task-encoding-record-confirm.log`,
`task-encoding-gpu-tests.log`, `task-encoding-native-parity.log`,
`task-encoding-sky-parity.log`, `task-encoding-growth.log`, `task-encoding-clippy.log`.

## Fresh spatial path reuse

Baseline `8836ddf` is preserved at
`experiment/gi-baseline-spatial-reuse-20261008`. The default remains unchanged;
these GPU estimator experiments are isolated on their branches. CPU fallback
still uses the original explicit-path estimator. No persistent path reservoir,
temporal reuse, display filter or scheduling policy was added.

### Two receiver prototype

`be32d40` records a fresh path's first hit, direction and five-band suffix, then
shares it with one compatible receiver. Adjacent task pairs must have the same
normal, lie in the same plane and be within three voxels. A current-geometry
connection ray checks visibility. Balance MIS weights compare both receivers'
proposal densities at that hit; the first-hit emitter MIS is reevaluated for the
destination. Unsupported connections leave the original self contribution intact.
Primary emitter next-event estimation stays local. This is pairwise fresh-path
MIS, not weighted reservoir selection or NAADF's complete reuse pipeline.

Harness `6ead4ae` adds ceiling receivers, high-sample checks, visibility support
tests and a throughput comparison that permits different indirect estimates.
`bd60c3f` adds matched compute budgets. All isolated results below use Apple
M4/Metal, Rust 1.95.0, development/shared-cache profile and no viewport. Baseline
shader is exactly `8836ddf`; candidate shader is `be32d40`. Runs used the harness
diffs later committed as the cited harness commits.

The ceiling fixture has 96 underside receivers around a skylight, eight disjoint
sequence blocks and four generated paths per batch. Both kernels use the same
65,536-path reference from a separate sequence block. At 1,024 generated paths,
normalized five-band error p95 falls 0.133091 → 0.094336, about 29%. GPU batch
median rises 0.1141 → 0.1365 ms; sharing adds connection rays, so this is not an
equal-ray comparison. Whole-worker medians are effectively unchanged in this
small fixture (1.752 → 1.753 ms).

The matched-budget study retains every four-path estimate and selects the last
batch within each median cumulative GPU budget across eight sequence blocks.
Normalization remains `abs(error) / (0.02 + abs(reference))`, pooled across
receivers, blocks and bands.

| GPU budget | Baseline generated paths | Reuse generated paths | Baseline error p95 | Reuse error p95 |
| --- | ---: | ---: | ---: | ---: |
| 4 ms | 132 | 112 | 0.52848 | 0.40525 |
| 8 ms | 264 | 224 | 0.33487 | 0.25331 |
| 16 ms | 532 | 448 | 0.21317 | 0.16004 |
| 24 ms | 800 | 668 | 0.15758 | 0.12967 |

This is encouraging quality per GPU time, not a universal speedup or interactive
response measurement. The analytic colored-emitter fixture regresses slightly at
64 paths (p95 0.006746 → 0.007961), though it improves at 1,024 paths
(0.002146 → 0.001537). In the 8,100-face room throughput test, GPU p50/p95 rises
1.425/2.125 → 1.702/3.377 ms and worker p95 rises 6.061 → 6.464 ms.

Focused checks pass: shader validation; original-estimator agreement with sharing
disabled at 1/7/64 paths and 17 receivers; and a sealed two-room test with
interleaved receivers across an opaque wall. The latter checks 1/7/64/65,536 paths,
inactive lanes and exact zero for invisible bands on each side (RGB versus
infrared/ionizing emitters). High-sample analytic error is small; the ceiling's
65,536-path discrepancy p95 is 0.01546 against the finite-sample reference, not
proof of zero bias. Ordinary exact CPU/GPU estimator parity is not claimed for
the changed estimator. Strict all-target Clippy passes through `bd60c3f`.

Evidence under `target/`: `fresh-spatial-paired.log`,
`fresh-spatial-ceiling-support.log`, `fresh-spatial-throughput.log`,
`fresh-spatial-high-confirm.log`, `fresh-pair-equal-compute.log`,
`fresh-pair-final-clippy.log`.

### Real workload pairing

Optional diagnostic `88d5930` counts compatible pairs without changing work
ordering. One preliminary offscreen `response` pair used `6ead4ae` plus that
`gpu.rs` diff; the baseline substituted the exact `8836ddf` shader into the same
Rust code. Both were built before running, on M4/Metal with the development
profile, 2560 × 1440 target, preferred corner modes, textures on and AO off.
Both completed all three scripted events, produced 14 valid nonblank captures,
and had clean render logs. Emitter-settled and recovered-room captures were
inspected in both; sparse images cannot establish every-frame stability.

Only **27.98%** of candidate faces were paired over 1,280 batches. Cavity
offscreen cadence p95 was 30.325 → 30.456 ms; emitter-added cadence p95 was
30.723 → 31.476 ms, with nearby measurement-age p95 1.641 → 1.858 seconds.
Other phases were mixed. These are offscreen cadence observations, not windowed
FPS, and establish no end-to-end improvement. Before promotion, build deliberate
neighborhoods without starving urgent or all-world work, then repeat paired
quality/freshness measurements including their CPU and memory costs.

Artifacts under `target/gi-validation/`: candidate
`response-1791509890472653000-65987/`, baseline
`response-1791509968724481000-67446/`. Logs: `target/fresh-pair-response.log` and
`target/fresh-pair-baseline-response.log`.

### Four receiver implementation rejected

`337bb87` extends sharing to four nearby coplanar receivers, with three connection
rays per generated path and more workgroup scratch. Against `8836ddf`, ceiling
GPU median rises 0.115 → 0.579 ms; the 8,100-face room median rises 1.428 →
7.109 ms and worker median 5.008 → 11.208 ms. Although 1,024-path ceiling p95
improves to 0.07773, the 65,536-path check produces invalid brightness (normalized
error p95 about 6,690). A completed reporting benchmark is not a correctness pass.
This candidate is rejected, not merged, and the large-count failure remains
unresolved on its branch. It does not disprove more selective reuse designs.

Same hardware/profile as above; code measured as `6ead4ae` plus the shader diff
committed at `337bb87`. Evidence: `target/fresh-four-wgsl.log`,
`fresh-four-ceiling.log`, `fresh-four-throughput.log`.

## Five direction decision pass

Baseline `28585a0` is preserved at `experiment/gi-baseline-five-decisions-20261008`.
Only exact brick repair is merged into normal gameplay. The resident prototype
is the strongest next architecture direction; a decision to pursue it is not a
claim that production residency or simulation ticks are implemented.
All compute/CPU studies below use Apple M4, Metal where applicable, Rust 1.95.0
and the development/shared-cache profile. They have no viewport unless stated.
Measurements used the cited commit's code as a dirty diff on its branch parent.

### Spatial neighborhoods deferred

`8ed313f`, based on `bd60c3f`, sorts a task-index permutation into 2×2 coplanar
tiles and scatters results back to their original receivers. Scheduler membership
and publication ranges do not change. The native scatter oracle passes at 17/100
receivers with independently advancing sequences. Strict Clippy passes.
The ceiling study retains 23–26% lower error p95 at 4–16 ms GPU budgets.

The offscreen `response` run at 2560×1440, preferred corner modes, textures on and
AO off raised compatible pairing from the previous 27.98% to **94.02%**. All
14 captures were valid/nonblank, assets resolved and render logs were clean;
emitter-settled and recovered-room images were inspected. Cavity/add/remove
worker p95 was 5.699/5.986/6.742 ms, versus 5.039/5.121/5.138 ms in the earlier
original-shader control. Near-field age p95 was 1.941/1.780/1.783 seconds versus
1.850/1.641/1.684. Offscreen cadence p95 was lower in this run, but this preliminary
comparison does not establish causation or windowed FPS. Additional worker cost
and mixed freshness do not justify default promotion on controlled error alone.

Decision: **defer**, retaining the branch for selective reuse in difficult
receivers rather than enabling reconnection rays everywhere. Temporal reservoirs
remain separate work, not a failed experiment. Evidence under `target/`:
`neighborhood-{scatter,quality,clippy,response}.log`; captures in
`target/gi-validation/response-1791510698889403000-79059/`.

### Recursive cache termination deferred

`9a6c078` adds a geometry-backed 216-receiver white-furnace contract test with
separate cache read/write generations. The uniform interior makes each iteration
exact, eliminating interpolation and Monte Carlo noise. The existing explicit
32,768-path CPU reference agrees with the three-bounce analytic value.

At reflectance 0.8 and emission 0.1, the red-band three-bounce value is 0.244;
recursive feedback approaches 0.5. In a deliberately high-reflectance 0.98 band,
52.39% of old energy remains after 32 complete updates with emission removed,
even without an additional history filter. A complete generation reset restores
exact zero. These are solver iterations, not frames or a calibrated radiation
material. More bounces are not inherently wrong, but they change the model.

Decision: **defer recursive termination**, and reject unversioned reuse of the
existing means as a drop-in optimization. Choose finite-depth layers or an
explicit recursive model with dependency/reset rules before integrating it.
The [versioned Solari design](https://jms55.github.io/posts/2025-09-20-solari-bevy-0-17/)
likewise separates cache reads from blending and spreads recursive paths across
updates. Resident storage alone will not resolve feedback semantics. Focused test
and strict Clippy pass; this was a correctness/response study, not a speed trial.
Logs: `target/recursive-cache-{contract,clippy}.log`.

### Fixed shared angular guide rejected

`a8ceaef` tests 16 azimuth × 4 radial bins in cosine-hemisphere parameter space,
shared across oriented 8-voxel cells. Sixty-four pilot paths train the proposal;
those paths also contribute to the estimate and measured time. A known-density
50% cosine fallback preserves support. Eight disjoint 1,024-path blocks compare
against a common analytic/native 65,536-path reference.

| Fixture | Baseline error p95 at 1,024 paths | Guided error p95 | CPU total p50 baseline → guided |
| --- | ---: | ---: | ---: |
| Open sky | 0.000394 | 0.003497 | 29.94 → 35.04 ms |
| Skylight room floor | 0.18638 | 0.22530 | 110.13 → 116.07 ms |
| Skylight ceiling | 0.13309 | 0.16302 | 100.65 → 111.76 ms |

Decision: **reject this fixed proposal** before GPU integration. This does not
reject adaptive spatio-directional trees from
[Practical Path Guiding](https://jannovak.info/publications/PathGuide/index.html).
Nonemissive fixtures suffice for the rejection; emitter MIS and live retraining
were not implemented. Density/support tests pass. Commit `2de5d91` corrects a
fixture lint and the preceding commit's mistaken Clippy-pass statement; strict
Clippy then passes. Evidence: `target/shared-guide-{quality,support}.log` and
`target/shared-guide-clippy-fixed.log`. The reporting study completing is not a
quality pass.

### Resident accumulation and tick snapshots retained as a direction

`34c3476` keeps five-band means/moments, count and advancing sequence in GPU
storage for stable fixture slots. Batches can submit without CPU readback;
snapshots return accumulated measurements after the chosen number of batches.
CPU counts/timestamps remain unchanged until delivery. Generation changes discard
old accumulation, including two queued but unread batches before source removal.
The reset snapshot is exactly zero. The experiment does not retain/reuse paths.

Twenty alternating measured pairs follow four warmups. Each batch traces the
same four paths; the candidate does not replace several batches with fewer paths.

| 1,024-face workload | Ordinary worker total p50 | Resident total p50 | Confirmation ordinary → resident |
| --- | ---: | ---: | ---: |
| One batch and one snapshot | 1.824 ms | 1.824 ms | 1.888 → 1.891 ms |
| Three batches and one resident snapshot | 5.456 ms | 2.275 ms | 5.359 → 2.298 ms |
| Eight batches and one resident snapshot | 14.293 ms | 3.189 ms | 14.692 → 3.446 ms |

Confirmation uses nonzero environment energy in all five bands. Means, moments
and direct values agree within `1e-5 * (1 + magnitude)` tolerance; integer
counts/sequences match exactly. Seventeen-face tail cases pass. This is
back-to-back compute with amortized submission/wait/readback, not a simulated
20 Hz tick, gameplay frame-pacing result or production memory measurement.

Decision: **keep as the next architecture direction**. Do not merge the fixed-slot
harness as a world implementation. Stable generation-owned slots, bounded
in-flight snapshots, real sample ages, scheduling feedback and GPU display
publication are integration requirements. Task headers still upload in this
prototype; it does not prove their elimination. Strict Clippy passes.
Logs: `target/resident-tick-{study,confirm,clippy}.log`.

### Exact brick repair retained instead of background refinement

`4196b88`, merged unchanged at `bdac2dc`, reuses complete voxel bounds for each
occupancy-identical brick of a changed dense region. Material-only changes still
update material data/masks and region identity. Region-level brick bounds are
reused only if the brick occupancy map matches. Changed bounds are fully rebuilt;
the resulting fields exactly equal the original constructor, so traversal quality
does not temporarily degrade and there is no second publication phase.

Forty alternating warmed trios, each with 64 constructions, compare a frozen
`28585a0` constructor, new cold construction, and incremental repair. The internal
constructor specializes away reuse comparisons for cold builds.

| Region workload | Original p50 | New cold p50 | Repair p50 | Isolated repeat original → repair |
| --- | ---: | ---: | ---: | ---: |
| Terrain, one removal | 38.360 µs | 38.298 µs | 5.174 µs | 37.654 → 5.053 µs |
| Scattered, one removal | 176.143 µs | 176.307 µs | 6.771 µs | 177.568 → 6.781 µs |
| Terrain, material change | 37.500 µs | 37.328 µs | 4.436 µs | 36.482 → 4.357 µs |
| Scattered, material change | 178.674 µs | 179.055 µs | 4.021 µs | 175.415 → 3.947 µs |

The single-edit changed-region construction is about 7.4×/26× faster in these
fixtures, not whole-scene construction or FPS. Cold timings are effectively
unchanged. The first attempted confirmation overlapped Clippy and is excluded;
the table uses the subsequent isolated repeat.

Decision: **retain exact repair; defer coarse-to-fine background publication**.
Queued top-level repair remains a possible response to measured topology-changing
streaming costs, not a necessary part of this local win. Exact-oracle checks cover
200 edits, IDs 0/255, empty/uniform transitions and old snapshot preservation.
Seven world tests pass; the focused oracle reran after cold specialization. Native
ordinary/emissive CPU/GPU RMS is 0.000000, including edit reuse. Strict all-target
Clippy and diff review pass. No estimator, display or buffer-lifetime change
required another toggle-stress or image run.

Evidence under `target/`: `incremental-bricks-specialized.log`,
`incremental-bricks-confirm-isolated.log`, `incremental-bricks-world-tests.log`,
`incremental-bricks-native.log`, `incremental-bricks-clippy.log`.
