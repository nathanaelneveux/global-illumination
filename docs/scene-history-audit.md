# Scene-history audit — October 10, 2026

## Decision

Remove the global measured-history restart caused by adding/removing/loading an
emitting voxel. Use the existing spatial scene-change policy instead. Do not
replace it with a localized emitter-specific reset, another display filter, or
an emitter-removal timer. Geometry already describes a source being removed,
replaced, or hidden behind an opaque surface.

This is a stability/response tradeoff, not a proof that history has no cost.
Local live removal takes roughly another half to one second in the tested room.
The previous [house-turn investigation](temporal-regression-investigation.md)
found that removing the reset returned variation to the older stable range with
fresh transport still running. The extinction test below shows that the general
history path reaches the exact dark answer, but not in one batch.

Accepted by the user's visual test and fast-forwarded into the regular
`voxel-global-illumination` branch on October 10, 2026. The experiment branch
`codex/unified-scene-history` remains as a checkpoint; use the regular branch
for current play/testing. Nothing has been pushed.
The [policy closeout](#policy-closeout) finishes the remaining analysis, removes
the redundant material-ID reset, and records which other policies stay unchanged.

## Changes and checkpoints

| Commit | Scope |
| --- | --- |
| `00a8032` | Observe the response fixture every Update, including all five energy bands, moments, sequence and revisions. No solver change; reset-enabled control. |
| `d6eb72b` | Ablate only the scene-emitter global reset; preserve all existing geometry, sampling and display policies. |
| `e1d7935` | Add the opt-in native extinction reference for removal, replacement and occlusion. |
| `d2abf3e` | Delete now-unused changed-material classification; add actual scene-publication/history regression; clarify simulation freshness; rename a misleading explicit-restart test. |
| `9d656f7` | Centralize global model replacement in `invalidate_model`, making the remaining restart watermark explicit and inseparable from revision advancement. |
| `6fd5918` | Test-only matched-work policy controls and isolated emission-cache rebuild cost; production remains the `f5b4043` baseline. |
| `0ce0f12` | Existing receiver material swaps use local retained history; preserve revision rejection/new-face semantics and initialize new display clocks explicitly. |

Scene and material `Arc` identities still rebuild the emission distribution when
needed. Distribution updates never depended on the removed source-revision bump.
The ordinary geometry revision makes every receiver eligible, including distant
ones. Sixteen local / 256 distant retained samples are weights, not ray counts
that are preserved individually. Sequences continue advancing. Display values
are retained independently; there is no new debounce or slower filter.

The publication regression compares deleting material 2 and emitting material 4
through `apply_scene_task` and actual dispatch selection. Both preserve current
means/display/sequence, mark transport stale, and retain 16/256 local/remote
weight. Neither advances the analytic-source/model-reset epoch. Add/remove/unload
and unchanged-scene delta tests remain, without emission-specific classification.

## Removal evidence

### Exact-reference native experiment

At `e1d7935`, clean tree, Apple M4 / Metal:

```sh
cargo test --locked scene_history_source_extinction_response -- --ignored --nocapture
```

A closed room with two RGB/IR/ionizing ceiling sources is warmed with 2,048 paths
per receiver. All other surfaces absorb, and environment/analytic light are zero.
Remove the sources, replace them with opaque nonemitters, or enclose them in
opaque blocks. Each has an exact post-change reference of zero in all five bands.
For 100 floor receivers, compare forced zero-history restart with the actual
geometry plan's 16-sample cap, then trace 256 four-path batches. CPU/GPU first-batch
parity is checked; every subsequent mean is checked against the extinction law.

All three operations give the same results:

| Policy | Raw five-band residual below 10% | Worst RGB display residual below 10% |
| --- | ---: | ---: |
| Forced reset | 4 fresh paths | 28 fresh paths |
| Ordinary spatial history | 148 fresh paths | 156 fresh paths |

Retained mean follows `old_mean * retained / (retained + fresh_paths)` within
tolerance. Every RGB display trace decreases monotonically. The display test uses
a virtual 50 ms between four-path batches, **not a measured gameplay latency**.
Native runtime was 3.24 seconds. Log: `target/history-extinction-native.log`.
The fixture exercises local history, not the 256-sample remote tail, repeated
changes, all source intensities, or a calibrated physical radiation model.

### Live room, background terrain loading

One matched pair, Apple M4 (10-core GPU) / Metal, Bevy 0.19.1, Rust 1.95.0,
development profile with shared Cargo intermediates. Bevy offscreen 2560×1440,
60 Hz ScheduleRunner. Textures on, baked AO off, corner sun and reconstructed
indirect. Four paths, three diffuse hits, reach 768, 4 ms target, 8,192 receiver
cap, 75 ms response / 50 ms upload, adaptive defaults. No simultaneous compiler
or GPU benchmark during capture. Build-root guards passed.

```sh
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_PROFILE=response VOXEL_GI_VALIDATE_BACKGROUND=1 \
VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD" VOXEL_GI_VALIDATE_EXIT=1 \
RUST_LOG=warn cargo run --locked
```

The emitter is added at 20 seconds and removed at 25. For each of two floor
receivers, estimate the pre-add red level from the median at 18.5–20 seconds;
estimate the lit level at 24–25. Measure the last excursion above 10% of their
difference in the five-second removal window. This independent pre-add state is
an approximate noisy reference, **not high-sample ground truth**.

| Receiver, +Y face | Reset control | Ordinary scene history |
| --- | ---: | ---: |
| `(9,38,-9)` | 1.200 s | 1.709 s |
| `(10,38,-9)` | 1.302 s | 2.257 s |

Final residual fractions: control −0.023 / 0.003; candidate 0.018 / 0.042.
The lamp here has RGB and IR; the native test supplies the nonzero ionizing case.
Both runs completed with 14 valid nonblank captures, verified dimensions/assets,
and clean render logs. Candidate emitter-lit, one-second removal and recovered
images were inspected; sparse pictures do not certify every intervening frame.
The candidate's temporary unused-field compile warning was eliminated by cleanup.

- Control `00a8032`, clean: `response-1791633622239167000-50676`,
  `target/history-reset-response-control.log`; 34,820,688 paths.
- Candidate `d6eb72b`, clean: `response-1791633695897349000-50976`,
  `target/history-unified-response-candidate.log`; 32,265,996 paths.

Run directories are below ignored `target/gi-validation/`. These measurements
justify the response tradeoff, not a speedup or windowed frame-pacing claim.

### House with continuous background loading

Same machine/profile/settings as above; `house`, temporal recorder enabled,
background loader enabled. Cave present; seven valid captures each, verified
assets and clean render logs. Both turned-ten-second images were inspected.
One preliminary pair, with documentation-only dirty files. Streaming timing is
asynchronous, not a deterministic replay of identical loaded chunks per frame.

| Policy | Median corner excess variation, 30–50 s | p95 | Minimum accepted fresh paths / receiver | Strict 0.100 gate |
| --- | ---: | ---: | ---: | --- |
| Reset control `00a8032` | 0.09674 | 0.17155 | 552 | Fail |
| General scene history `9d656f7` | 0.05641 | 0.10584 | 608 | Fail |

The candidate is 38% lower at p95 in this pair, but **does not pass** the strict
camera-only threshold under this heavier workload. Do not raise the threshold
to conceal that result or claim all streaming noise is solved. Fresh work remains
active; repeated distant geometry caps, not a frozen cache, remain a candidate
cause. The original camera-only controls and reset ablation are recorded in the
prior investigation and must not be conflated with this workload.

Offscreen second-corner cadence p99 was 45.79 → 44.47 ms, maximum 252.48 →
181.84 ms; sampled near-field age p95 1.97 → 1.85 s. Startup maxima were about
613 ms in both. These one-pair observations are **not a measured speedup or
interactive FPS**. Total traced paths: 106,362,476 → 100,984,624; maximum loaded
pages 1,428 → 1,348. Differing asynchronous residency is retained as a limitation.

- Control: `house-1791634394903159000-53518`,
  `target/history-reset-house-background-control.log`.
- Candidate: `house-1791634259209506000-53028`,
  `target/history-unified-house-final.log`.

### Original camera-only gate, final candidate

At `9d656f7` with documentation-only changes, the same `house` workload without
the additional loader passes: median 0.04199, p95 0.06708, all 100 corner series
complete, minimum 716 accepted fresh paths at each roof receiver during 30–50 s.
Seven valid nonblank captures; assets/dimensions verified, no render warnings or
errors. The turned-ten-second image was inspected. This independently reproduces
the earlier successful reset ablation after cleanup, with the cave present.

Artifacts: `house-1791634558563083000-54094`,
`target/history-unified-house-camera-only.log`. There is no independent house
accuracy reference: passing this gate proves neither convergence nor correctness
at every receiver. Combined with the removal reference, it supports keeping the
general policy while tracking continuous-streaming stability separately.

## Code-path audit and disposition

This review covers the runtime GI boundaries, CPU/GPU transport and both shaders,
host integration, and diagnostic/regression paths. It is not a formal proof for
arbitrary procedural worlds or a new benchmark of every previously tested path.

| Area | Finding / disposition |
| --- | --- |
| Scene changes, BVW ingestion, publication | **Remove** emitter classification/reset. Scene bounds/journal already cover optical edits and residency. Preserve page-atomic publication, exact-edit urgency and revision rejection. The eager edit path rejects in-flight local work; the lazy journal also covers streaming/distant work, so they are not redundant. Existing 40 ms edit / 350 ms stream coalescing and 1 s maximum wait are unchanged. |
| Remaining restarts | Global table/model replacement has one explicit operation. **Remove** the per-receiver material-ID reset after the null control below; local scene retention suffices. Newly exposed faces still have no compatible measurements. |
| Analytic sources | Keep canonical snapshots, old/new influence journals and 4/64 limits. The moving-light control does not establish 16 as a better default: less raw error but worse displayed tracking/steps. Shared mechanisms need not force identical tuning. These limits are heuristics, not physical necessities. |
| Storage, NAADF, traversal | Immutable region reuse, occupied-frontier semantics and CPU reference remain coherent. Unknown transport space and visible-only sunlight have intentionally different query semantics. No house/cave/material-ID rule found in traversal. Shortened reach, epsilon and step/horizon bounds remain explicit approximations. Update: the former throughput cutoff was rejected and removed by the [independent estimator audit](hallway-path-outliers.md#independent-correctness-audit). |
| Emissive sampling | Keep next-event sampling, five-band power, CDF/tree crossover and matching MIS densities. These distinguish integration strategies for light sources, not gameplay types. Full rebuild on unrelated scene changes is **confirmed avoidable work** in the isolated cave control below; incremental reuse is next performance work, not implemented by this audit. |
| GPU worker/storage | Persistent upload buffers, identity-keyed bindings and finite failure fallback are necessary ownership rules, not historical lighting patches. One outstanding batch; CPU fallback pool only on use. Serial/test kernels and native references remain deliberately opt-in. No storage/readback rewrite in this audit. |
| Scheduler | Every loaded receiver remains eligible; camera priority does not restrict transport. Keep sweep, demand and audits. Retain the RGB-dark minimum: extra discovery paths find more rare portal hits, but neither 64 nor 256 certifies darkness. Five-band variance/drift protects observed uncertainty; it cannot bound undiscovered paths. Bright IR alone is not evidence that dark RGB has been sampled sufficiently. |
| Direct sunlight | Deterministic shared-corner visibility remains separate from stochastic diffuse transport but shares scene, worker and submission. Shadow-swept invalidation is not superseded by distance-based *statistical* retention. No per-pixel fallback when a corner is temporarily missing. |
| Display/history/reconstruction | Preserve measured-state vs presentation separation. CPU new-face seeding has new-scene connectivity; shader fallback bridges the earlier mesh/cache gap using old cache evidence. They cover different availability phases. The shader prior is an approximation, not certified current-geometry visibility. No debounce, brightness clipping or smoothing change here. |
| Display buffer lifetime | Generation-owned buffers/mirrors, transactional topology and capacity checks remain required. Do not remove them because toggle stress no longer fails. Existing targeted coverage reused; no routine stress run. |
| Simulation API | Correct documentation: check both source and transport freshness, age and uncertainty. Refreshed estimates may still contain bounded pre-change energy; display priors/filters never enter the field. Physical heating/dose/material transitions remain host responsibilities. |
| Host world/render/physics/input | Structure/palette choices stay outside GI; generation and fallback share stamps. AO is optional and off by default. Texture preparation and unused native sun-shadow suppression remain consumer-driven. Collider workers preserve old collision until replacement and validate jobs; unrelated to lighting history. |
| Diagnostics and tests | Runtime UI is optional, expensive tracing opt-in. Every-Update recorder distinguishes means, display and CPU render mirror; it is not GPU readback. Renamed the old native "emitter removal" test because it explicitly restarted history and could not prove the production removal path needed a reset. No dependency removed merely for having one current consumer. |

## Policy closeout

The remaining questions now have dispositions. Keep the smooth-history baseline
and return to throughput work; there is no evidence here for another blanket
reset or a blanket increase in history length. This is **readiness to iterate**,
not a declaration that continuous streaming, rare-path discovery or physical
simulation accuracy is solved.

### Reproducible controls

Tests are in `src/voxel_gi/history_audit.rs`, registered under
`voxel_gi::gpu::refinement`. All five numerical bands use identical retention
weights. These are opt-in controls, not additions to routine gameplay testing:

```sh
cargo test --locked policy_ -- --ignored --nocapture --test-threads=1
cargo test --locked policy_retention_extinction_bounds -- --nocapture
```

Evidence: `f5b4043` plus precisely the test-only diff committed as `6fd5918`.
Apple M4 / Metal, Rust 1.95, development test profile (application opt-level 1,
dependencies 3), shared Cargo intermediates. No window or viewport. Default
three-hit/768-reach transport; four fresh paths per batch. Virtual 50 ms between
display resolves, 75 ms response. The candidate policies receive identical fresh
moments; changes in old weight do not change their ray sequences or budgets.
No compiler or gameplay capture ran concurrently with the numerical controls.

Four native controls passed in 8.64 seconds; the CPU rebuild-cost control and
exact-weight unit also passed. Strict all-target Clippy passed. The four native
controls were run before adding the independent CPU cost test and tightening the
f64 threshold comparison's roundoff tolerance; their unchanged code was not
rerun for the checkpoint. Logs: `target/history-policy-native.log`,
`target/history-policy-emission-cost.log`, `target/history-policy-clippy.log`.
Reported relative errors use a per-band 0.002 floor and the worst band for each
receiver/batch, then a percentile across those observations. They are not
perceptual scores or certified confidence intervals.

### 1. Receiver material identity — remove the reset

A sealed room with two colored RGB/IR/ionizing emitters, absorbing surfaces and
100 floor receivers is warmed with 2,048 paths. Replace the floor receivers'
material IDs with an optically identical alias. The pre/post-change native
four-path batches match in all bands (relative tolerance `1e-6`), with continuing
sequences. Independent ceiling-area quadrature supplies the reference. Across
the first 64 fresh paths, worst-band relative-error p95 is **0.05726 with reset**
versus **0.04459 with 16 retained samples**. This is a null-control estimator
result, not a measured visual or timing improvement.

Decision implemented in `0ce0f12`: material replacement invokes ordinary local
relighting, not `invalidate`. Position and orientation still define the same
incident-energy receiver. Changed reflection/emission is a scene change; the
retained estimate is bounded and explicitly stale. The earlier exact extinction
test already covers non-null source replacement/occlusion through this scene
policy. No material-property equality check or source-specific branch is added.

Actual page-publication and late-worker tests cover nonemitting/emitting swaps
and simultaneous new face exposure: means/moments/sequence/display stay intact,
weight is capped at 16, freshness/stability clear, self-emission updates but is
not counted as incident energy, and old jobs cannot overwrite the receiver.
New faces have zero measurements. A separate material-0/2 publication test
protects the new display clock, which is now initialized explicitly rather than
incidentally by the removed reset. The full-probe restart helper is test-only;
production global-model restarts remain in compact work publication.

### 2. Analytic light history — keep 4/64 for now

A reflective sealed room, 25 floor receivers and one five-band point light moving
through 16 poses. Each pose has two disjoint 16,384-path references; their
worst-band relative disagreement p95 is **0.00772**. Four cycles move one pose per
four-path service; the first cycle is excluded from scoring. This is deliberately
fast motion in service units, not a measured character/light speed in gameplay.

| Local cap | Raw relative-error p95 | RGB display relative-error p95 | RGB display step p95 |
| --- | ---: | ---: | ---: |
| 4, current | 0.46293 | 0.35689 | 0.04837 |
| 16, proposed common local cap | 0.36235 | 0.40216 | 0.06356 |

Longer history reduces raw error here but worsens displayed tracking and steps;
the display filter responds differently to the smoother, more confident yet
lagging estimate. This is not a universal win for either constant. **Do not
promote identical analytic/geometry limits from this evidence.** Preserve the
shared journal/work/publication architecture without changing tuning merely for
symmetry. Remote analytic 64 and missing-transition behavior are not new empirical
optima; existing safeguards remain unchanged. Exact extinction below also bounds
their response cost independently of spectrum, including hidden-only light.

### 3. Dark discovery — retain the safeguard, do not call it convergence

In an absorbing room with a one-voxel roof portal and five-band environment,
800 receiver/sequence combinations still observed no nonzero indirect path in
**774 cases at 64 paths**, versus **693 at 256**. This intentionally difficult
fixture proves that additional discovery work helps, but also exposes how weak
either finite minimum is as a darkness/accuracy guarantee. The test counts native
path discovery, not time spent in the adaptive scheduler or a full minimum-policy
A/B run. Audit rays remain necessary.

Keep the current 256-sample RGB-dark minimum rather than reducing it blindly to
64. A max-over-all-bands shortcut is not a valid generalization: bright, stable
IR does not prove a rare RGB path was discovered, and the reverse is also true.
The existing five-band variance/drift tests cover **observed** hidden-band
uncertainty; zero observations can still produce zero estimated variance.
Simulation clients must not interpret settled coverage or zero variance as a
bound on undiscovered energy. Source-aware discovery/sampling is a future
accuracy/throughput experiment, not an emitter-removal exception. This control
does not establish a universal 256-sample optimum or quantify cave scheduling cost.

### 4. Repeated distant updates — mechanism confirmed, no blanket cap increase

The sealed two-emitter room is warmed with 8,192 paths. Add/remove nonemitting
geometry 128 voxels away every eight batches, for 256 four-path batches. The
actual geometry journal gives a remote limit of 256. Matched native batches with
and without the distant geometry agree within `1e-6`: the closed room cannot see
the update. Compare the same fresh moments under different repeated caps:

| Cap | Raw worst-band error p95 | RGB compressed-display total variation p95 |
| --- | ---: | ---: |
| 64 | 0.02450 | 0.02320 |
| 256, current remote scene limit | 0.00959 | 0.00677 |
| 1,024 | 0.00330 | 0.00175 |
| 65,536, effectively retain warm evidence here | 0.00134 | 0.00021 |

The reference is independent ceiling-area quadrature. Display variation sums
the maximum channel step after `RGB/(0.25+RGB)` compression per receiver; it is
**not the house recorder's corner excess-variation metric**. This confirms the
mechanism by which repeated caps disturb an unchanged solution, not that they
alone caused the earlier house gate failure.

For a real extinction, retained energy is `old * H/(H+N)` after one cap. Exact
f64 controls give the following fresh-path counts strictly below 10%; repeated
caps are a separate every-batch example, not the eight-batch workload above:

| Retained weight | One change | Cap reapplied every four-path batch |
| --- | ---: | ---: |
| 4 | 40 | 16 |
| 16 | 148 | 44 |
| 64 | 580 | 152 |
| 256 | 2,308 | 596 |
| 1,024 | 9,220 | 2,364 |

All bands obey the same weight law; this is a sample-domain bound, not seconds.
Thus increasing remote history fourfold buys smoothness by making a genuine
remote extinction require roughly four times as much replacement evidence.
**Keep 256 for this baseline.** Pursue better change/dependency knowledge or more
fresh work before a blanket increase. Do not skip distant updates solely because
they are distant: bright sources and occluders can matter at any allowed reach.
The previous continuous-background house result remains a failure of the strict
0.100 p95 gate; it has not been relabeled a pass or rerun for these controls.

### 5. Emission-distribution maintenance — confirmed next speed target

The cave plus 64 nonemitting background pages contains 576 regions and 4,585
emissive faces. Alternate one remote nonemitting material ID for 32 repetitions;
scene construction is outside the measured span. `EmissionCache::prepare`
rebuilds a new distribution each time despite identical face identities and
probabilities. Median **1.480 ms**, p95 **1.519 ms**. Reusing the identical scene
and material Arcs returns the same distribution; its median elapsed time is below
the timer's resolution (printed as zero), not literally zero CPU work.

This cost is on the worker, not a demonstrated 1.5 ms main-thread stall or frame
gain. It is a concrete avoidable preparation cost and a clean next experiment:
reuse distributions when emitting regions **and their exposure dependencies**
are unchanged, then consider incremental affected-region maintenance. A changed
nonemitter in an adjacent region can expose an emitter, so checking only emitter
material IDs is incorrect; the existing cross-region occlusion test protects
that boundary. Preserve proposal probabilities/MIS and five-band sampling.
No distribution/storage algorithm was changed in this audit.

### Next phase and remaining limits

1. Measure safe emission-distribution reuse under cave plus background loading.
2. Improve fresh-path throughput/service latency while keeping this history and
   presentation baseline fixed; compare freshness and tails, not just FPS.
3. Revisit dependency-aware invalidation and rare-path discovery with dedicated
   correctness references. They are separate experiments, not reasons to reopen
   emitter-specific resets.

Global material-table replacement still conservatively restarts the field even
for unused entries; scoped optical dependencies would be a separate optimization.
The controls do not exhaust arbitrary light motion, scene scales, intensities,
hidden-band-only gameplay cases, or simultaneous rendering/streaming schedules.
No new knobs, debounce, alternate corner transport, or physical radiation model
were added. The accepted smoothness/response tradeoff is now explicit and tested;
speed improvements can target work instead of weakening that baseline.

Retain the existing integration limits: axis-aligned unit voxels, Y-up environment,
f32 positions, opaque diffuse normalized bands, bounded rays, finite buffers and
CPU-owned statistical state. This is a more coherent prototype, not a claim that
arbitrary world scales or physical IR/radiation have been solved.

## Verification checkpoints

### Original scene-reset removal

At `9d656f7`, with documentation-only changes:

- `cargo test --locked --all-targets`: **157 passed, 29 opt-in ignored**.
- `cargo clippy --locked --all-targets -- -D warnings`: passed.
- `RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps --document-private-items`:
  passed.
- Native five-band extinction evidence at `e1d7935` is reused: subsequent changes
  remove dead metadata and centralize the unchanged global-model operation;
  neither changes its kernel, fixture or numerical resolve. Not claimed as rerun.
- Development binary rebuilt explicitly after branch changes; final build-root
  guard passed. No baseline/candidate compiled concurrently with a capture run.
- Documentation diff/links reviewed; original checkout remains clean at
  `ad6ba8e`. All commits local. No toggle-stress or broad ignored-native suite.

### Policy closeout, `0ce0f12`

- `cargo test --locked --all-targets`: **160 passed, 34 opt-in ignored**.
  Strict all-target Clippy, development build and warning-denied private rustdoc
  passed. Logs: `target/history-policy-integration-tests.log`,
  `target/history-policy-integration-clippy.log`, `target/history-policy-build.log`,
  `target/history-policy-doc.log`. No broad ignored-native or toggle-stress run.
- Numerical controls from `6fd5918` above are reused; the cleanup changes
  publication behavior, not their kernels, fixtures or statistical integration.
  Actual publication/async rejection is covered by the two new regular tests.
- Final `response` headless run, background loader on, runtime `0ce0f12`, clean
  at launch with documentation-only edits during the run. Same M4/Metal, dev
  profile and preferred 2560×1440 settings as the earlier live pair: four paths,
  three hits, reach 768, 4 ms / 8,192 cap, 75 ms display / 50 ms upload, adaptive
  defaults, textures on/AO off. Build-root guard passed; no compiler/GPU test
  ran concurrently. **14 valid nonblank captures**, assets/dimensions verified,
  clean render logs. Dig-300 ms, emitter-settled and room-recovered inspected.
  This is an integration check, not a newly paired performance experiment or
  a visual material-alias A/B (that change has a native null control).
- Using the earlier approximate pre-add/lit reference method, the two floor
  receivers' final above-10% display excursions occur at **1.801 / 2.224 s**
  after removal; final normalized residuals **0.0226 / 0.0440**. The smooth
  baseline's response remains in the previous observed range. This noisy
  five-second window is not independent high-sample accuracy certification.
  Run `response-1791636085121577000-58934`,
  `target/history-policy-response-final.log`; 31,089,636 traced paths, maximum
  593 loaded pages. Offscreen removal-phase p99 40.94 ms / maximum 50.96 ms;
  startup maximum 599.38 ms. Not windowed FPS or a speedup claim.
- The earlier heavy-streaming house gate still fails; it was not rerun or
  relaxed. Camera-only evidence remains separate. Documentation is updated to
  distinguish completed policy decisions from future performance work.
