# GI performance experiment results

Four compatible changes are retained: better emitter sampling, explicit sky
work, exact uniform-field construction and topology-keyed acceleration reuse.
Other candidates remain on separate local branches. The preferred shared
corner appearance, four paths per dispatch, three-bounce estimator, history
policy and temporal display filter are unchanged. There is no new debounce,
per-corner indirect transport, reservoir history or screen-space dependency.

## Baseline and decisions

Baseline `8b863cb` is preserved as `experiment/gi-baseline-20261008`. Paired shader
comparison infrastructure is `9186a55`. Winners were merged into
`voxel-global-illumination` at `f51e5df`; no commits were pushed.

| Experiment branch suffix after `experiment/gi-` | Commit | Decision |
| --- | --- | --- |
| `directional-guide` | `f8fba72` | Reject four azimuth sectors: higher room error and disturbed open-sky sampling; added probe memory was not justified. |
| `terminal-work` | `ab4cfdd` | Keep isolated: removing unused terminal-bounce preparation showed only a small preliminary gain. |
| `sampler-cost` | `ac9af28` | Keep isolated: constant Halton radices gave no compelling overall gain. |
| `ray-reciprocals` | `0850c91` | Reject: no demonstrated throughput gain from explicit reciprocal reuse. |
| `wavefront` | `f0044d6` | Reject dense per-bounce passes: extra scratch and dispatches, slower open terrain, no room gain. |
| `scrambled-sobol` | `2034e48` | Do not promote all dimensions: strong sky/emitter improvement, but worse early room error. |
| `hybrid-sampler` | `70cb3f8` | Reject primary-Sobol/secondary-Halton mixture: did not resolve the room tradeoff. |
| `emitter-sobol` | `5f3cee2` | Retain: lower emitter error, original path directions and nonemissive results preserved. |
| `explicit-sky-work` | `a250cbb` | Retain: remove repeated warm sky lookups/copies from indirect-only work. |
| `aadf-convergence` | `5fecb63` | Reject full-cycle early exit: loaded islands improve, mixed terrain and bricks regress. |
| `uniform-aadf` | `21d2409` | Retain: solve empty/full acceleration volumes exactly, without iterative sweeps. |
| `topology-cache` | `b57d853` | Retain: share top-level acceleration across edits that preserve region residency and occupancy. |

Rejected variants are not dormant production toggles or extra runtime resources.
A rejected implementation does not disprove its broader research family: the
guide lacked spatial sharing and a rich angular distribution; the wavefront test
did not compact live paths or separate shadow queues.

## Emitter sampling

The implementation follows the shuffled, padded, hash-scrambled Sobol approach
described by [Sobol-Burley](https://github.com/cessen/sobol_burley), based on
[Burley 2020](https://jcgt.org/published/0009/04/01/). Its reverse-bit scramble is
adapted under the included `licenses/sobol_burley-MIT.txt`. No external direction
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
[the research notes](research.md#source-implementation-findings). These are
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
