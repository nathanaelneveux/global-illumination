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

Work is local on `codex/unified-scene-history` in the `temporal-regression` worktree.
The user's original detached `ad6ba8e` checkout is preserved for comparison.

## Changes and checkpoints

| Commit | Scope |
| --- | --- |
| `00a8032` | Observe the response fixture every Update, including all five energy bands, moments, sequence and revisions. No solver change; reset-enabled control. |
| `d6eb72b` | Ablate only the scene-emitter global reset; preserve all existing geometry, sampling and display policies. |
| `e1d7935` | Add the opt-in native extinction reference for removal, replacement and occlusion. |
| `d2abf3e` | Delete now-unused changed-material classification; add actual scene-publication/history regression; clarify simulation freshness; rename a misleading explicit-restart test. |
| `9d656f7` | Centralize global model replacement in `invalidate_model`, making the remaining restart watermark explicit and inseparable from revision advancement. |

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
| Remaining restarts | Global table/model replacement now has one explicit operation. The conservative per-receiver material-ID reset remains; it is not emitter-specific and was not tested by this ablation. Newly exposed faces have no compatible measurements. Do not confuse these cases with globally discarding unchanged receivers. |
| Analytic sources | Canonical snapshots and complete old/new influence journals remain. Their 4/64-sample limits differ from scene 16/256 limits; these are heuristics, not a physical necessity. Consolidating their *policy* requires a moving/removing-light comparison, not merely a shared helper. |
| Storage, NAADF, traversal | Immutable region reuse, occupied-frontier semantics and CPU reference remain coherent. Unknown transport space and visible-only sunlight have intentionally different query semantics. No house/cave/material-ID rule found in traversal. Shortened reach, epsilon, step/horizon bounds and throughput cutoff remain explicit approximations. |
| Emissive sampling | Keep next-event sampling, five-band power, CDF/tree crossover and matching MIS densities. These distinguish integration strategies for light sources, not gameplay types. Rebuilds remain keyed by scene/material snapshots. Full emitter-tree rebuild on unrelated scene changes is potential remaining work, not measured away by this reset removal. |
| GPU worker/storage | Persistent upload buffers, identity-keyed bindings and finite failure fallback are necessary ownership rules, not historical lighting patches. One outstanding batch; CPU fallback pool only on use. Serial/test kernels and native references remain deliberately opt-in. No storage/readback rewrite in this audit. |
| Scheduler | Every loaded receiver remains eligible; camera priority does not restrict transport. Preserve all-world sweep, forced simulation demand and adaptive audits. The RGB-dark 256-sample minimum and fixed error floors are heuristics. Five-band variance/drift already participates; RGB-only minimum deserves its own hidden-band/rare-light ablation. |
| Direct sunlight | Deterministic shared-corner visibility remains separate from stochastic diffuse transport but shares scene, worker and submission. Shadow-swept invalidation is not superseded by distance-based *statistical* retention. No per-pixel fallback when a corner is temporarily missing. |
| Display/history/reconstruction | Preserve measured-state vs presentation separation. CPU new-face seeding has new-scene connectivity; shader fallback bridges the earlier mesh/cache gap using old cache evidence. They cover different availability phases. The shader prior is an approximation, not certified current-geometry visibility. No debounce, brightness clipping or smoothing change here. |
| Display buffer lifetime | Generation-owned buffers/mirrors, transactional topology and capacity checks remain required. Do not remove them because toggle stress no longer fails. Existing targeted coverage reused; no routine stress run. |
| Simulation API | Correct documentation: check both source and transport freshness, age and uncertainty. Refreshed estimates may still contain bounded pre-change energy; display priors/filters never enter the field. Physical heating/dose/material transitions remain host responsibilities. |
| Host world/render/physics/input | Structure/palette choices stay outside GI; generation and fallback share stamps. AO is optional and off by default. Texture preparation and unused native sun-shadow suppression remain consumer-driven. Collider workers preserve old collision until replacement and validate jobs; unrelated to lighting history. |
| Diagnostics and tests | Runtime UI is optional, expensive tracing opt-in. Every-Update recorder distinguishes means, display and CPU render mirror; it is not GPU readback. Renamed the old native "emitter removal" test because it explicitly restarted history and could not prove the production removal path needed a reset. No dependency removed merely for having one current consumer. |

## Remaining systematic experiments

1. Continuous streaming can repeatedly cap unchanged remote history at 256.
   Study sustained-reference error and volatility versus the amount of old
   influence, without masking it with a source-class switch or starving refresh.
   A source farther than 48 voxels is not necessarily weak.
2. Test whether material-ID replacement at an otherwise identical receiver needs
   a zero-history restart: incident energy has the same spatial definition, but
   replacement changes the scene. Compare bounded retention, stale-result
   rejection and source on/off response before removing that convention.
3. Evaluate one configurable history policy across analytic/scene changes,
   including continuous motion, repeated edits and hidden-band-only sources.
   Sharing a journal is already done; using identical constants is not yet
   validated. Global material-table changes are conservatively global even if a
   changed entry is unused; scoped material dependencies would be a separate
   optimization.
4. Profile incremental emission-distribution maintenance and dark-receiver
   scheduling with the cave and background loader. Neither should be justified
   by an emitter-removal exception.

Retain the existing integration limits: axis-aligned unit voxels, Y-up environment,
f32 positions, opaque diffuse normalized bands, bounded rays, finite buffers and
CPU-owned statistical state. This is a more coherent prototype, not a claim that
arbitrary world scales or physical IR/radiation have been solved.

## Final verification

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
