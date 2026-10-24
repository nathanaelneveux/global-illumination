# Validation and regression tracking

Use the smallest check that answers the current question. The purpose is faster,
stable gameplay, not maximizing test minutes. GI owns visual RGB; heat/radiation
are a separate future simulation, not a GPU lighting delivery contract.
The full suite is an integration checkpoint, not the default for every tweak.

For resident transport, distinguish `snapshot_latency_ms` (submission to CPU
delivery) from worker occupancy and GPU timestamps. It is not a blocking wait.
Validation reports' receiver records describe world-cache receivers, not the
displayed view-owned faces; use captures for presentation.

Normal runs use nonvisual importance 0 (display-only GI);
`simulation_sampling_importance: 1` reproduces the historical equal-band control.

The paired image study uses `VOXEL_GI_VALIDATE_REUSE_SWEEP=1` with the existing
headless cave/reference/sample-sweep flags. It records a resident independent
control in the same frozen scene. Analyze it with `cave_sample_analysis.py
RUN --perceptual --control`, and analyze the candidate without `--control`.
Both require the independent reference's qualification and completed capture
status; neither relaxes the user-calibrated RGB gates. The
[resident/path-reuse results](history/resident-path-reuse.md) document exact commits,
settings and evidence. These opt-in studies are not routine toggle stress.

Current appearance acceptance is the user's [2,048-path cave target](history/cave-sampling.md#user-calibrated-visual-target),
with 1,536 and 1,024 as visible-failure controls. Use the fixed, versioned profile
for sampler comparisons; do not turn 2,048 into a history cap, use a raw RGB
outlier as a universal perceptual threshold, or waive temporal/FPS qualification.
Keep the original stricter engineering metrics alongside this human-calibrated
target, and keep IR/ionizing tolerances independent.

The [whole-range integration](history/whole-range-history.md#runtime-integration) records
the current policy. The [earlier runtime qualification](history/runtime-retention.md)
records the preceding calibration, independent references and windowed controls.
The legacy 16/256 control, the floor summary and the retention audit were
removed on the surface-cache branch (last present at `8cab848`), as were the
study harnesses the [historical records](history/README.md) use: near-history
calibration, model comparison, footprint replay, the retain-all ablation and
their Python analyses. Check out that commit to reproduce them. Check
`spatial_retention` and `measured_retention` in the report;
`config/gi-retention-curve.ron` selects the whole-range curve without measured
flux. Coverage and retained weight are still not convergence.
The blocked windowed capture is not a frame-pacing qualification.

For adaptive-history research, the [DDGI/A-SVGF qualification study](history/adaptive-history-qualification.md)
records the opt-in matched native sample tapes, isolated branches, first-gate
failures and required runtime gate. Neither detector is a gameplay default;
do not add those studies to ordinary throughput testing.

## Reproducible setup

Use the checked-in lockfile. On a fresh machine, fetch dependencies normally;
`--offline` below assumes they are already cached. Record the exact commit, dirty
diff, compiler/profile, hardware/backend, viewport, settings and workload. The
validation report does not record every one of those for you.

```sh
git rev-parse HEAD
git status --short
rustc --version
cargo build --locked --offline
```

Build before timing. Do not run another application instance, compiler or GPU
benchmark during a paired measurement. A preliminary baseline/candidate pair is
enough to guide a substantial iteration; repeat pairs when differences are small
or noisy. Retain logs and captures under ignored `target/`, not Git.

## Choose the relevant check

### Separate sampling accuracy from presentation

Correctness precedes variance/quality tuning. The
[independent estimator audit](history/hallway-path-outliers.md#independent-correctness-audit)
found a bias that CPU/GPU parity and shared-estimator references missed. For changes
to sampling, MIS or termination, check distribution/PDF agreement and a known-answer
integral first, then independent cosine-only means by depth at the actual suspect
receivers. Separate IID statistical uncertainty from QMC prefix variation; do not
interpret disjoint QMC windows as independent random trials. Only then compare
time to FLIP acceptance. No display filter participates in an estimator audit.

Focused checks: `low_throughput_paths_can_still_reach_bright_radiance`,
`light_selection_frequencies_match_reported_probabilities`,
`path_and_conditional_emitter_dimensions_have_joint_support`,
`independent_oracle_matches_furnace_geometric_series`, and ignored native
`native_furnace_matches_geometric_series` and
`native_low_throughput_path_keeps_bright_suffix`. The ignored
`hallway_estimator_correctness_audit` collects evidence, **not** an automatic
accuracy pass: inspect independent reference precision and replica agreement.
Set `VOXEL_GI_AUDIT_PATHS=1048576 VOXEL_GI_AUDIT_ORACLE_PATHS=33554432` to reproduce
the measured audit; setting oracle paths to zero explicitly omits it for GPU
ablations and requires a separately retained unchanged-scene reference.

`transport_field_correctness_gate` extends this to the cave's wall/ceiling grid
and material/normal-stratified floor/source sites (165 receivers, not every cave
face), shorter ranges, spectral furnace extremes, and a point source. It uses
two 262,144-path independent CPU replicas and GPU windows per site. IID variants
have a six-estimated-standard-error diagnostic gate; QMC variation is recorded,
not labeled IID uncertainty. Analytic QMC cases require <=1.5% error. Outputs
are timestamped `target/transport-field-*.json`; inspect precision, not only the
pass flag, especially for near-zero channels. `cave_witness_estimator_isolation`
compares the two formerly failing receivers at each depth with roulette off and
with a separate cosine-only GPU estimator. `grazing_emitter_plane_matches_view_factor`
checks the closed-form integral and reproduces the faulty visibility as a
test-only legacy ablation; the ordinary CPU `grazing_emitter_connection...` test
also checks real blockers. All three native checks are opt-in/ignored.

The preferred resident display now uses uncapped 1024-evidence reconstruction
after the frozen surface-cache qualification (2,048 remains the historical
independent-path visual target, not a changed sampler gate).
Do not let that hide estimator regressions or inflate apparent sample efficiency:

- **Sampler qualification:** compare raw frozen-scene prefixes at matched path
  counts and native compute budgets against the same qualified reference, over
  the same disjoint sequence windows. `sample_sweep` reads physical means directly
  and renders `samples-*` without temporal/adaptive spatial filtering, regardless
  of gameplay defaults. Ordinary corner interpolation and materials stay fixed.
  Reuse variants need their own raw-prefix comparison, including correlation.
- **Presentation qualification:** apply reconstruction to those identical saved
  prefixes, then compare noise, FLIP, bias/detail and edit response separately.
  Borrowed evidence does not increase the physical sample count. A smoother image
  alone cannot lower the calibrated raw-sampler threshold.
- **Gameplay cost:** compare full presentation with matched settings and background
  loading. `config/gi-display-unfiltered.ron` explicitly disables adaptive spatial
  reconstruction; unlike frozen prefixes it retains normal temporal response.
  Omitting `VOXEL_GI_SETTINGS` now enables reconstruction and is no longer an off
  control. Report cadence/freshness separately from fixed-prefix accuracy.

### Near/far lifecycle checks

Before evaluating far-tier image quality, use the focused native tests
`geometry_tests` and `maintenance_tests` (`--ignored --test-threads=1`). They
cover held render generations through growth/bounds changes, sparse edit reuse,
GPU retirement during poll-only/budget-exhausted dispatch, and zero-near-receiver
startup/update/idle. `gpu_headless_renderer_prepares_cache_and_material_generations_together`
checks the material integration. These are lifetime checks, not sampler or
gameplay performance qualification.

The far consumer's raw gate is `cargo test --locked --offline
voxel_gi::gpu::view_tests -- --ignored --test-threads=1 --nocapture`.
It covers matched prefixes against near transport, frozen-cache termination,
black/unknown/inactive accounting, and analytic multi-bounce tails. The `wave`
tests compare one/three/eight-hit partitions with unsplit paths, including
zero-budget pauses, idempotent completion and generation/receiver replacement.
The uncapped-tail furnace test aggregates only after each cohort finishes;
partial or early-completed samples must not establish a false convergence claim.
These test readbacks are intentionally synchronous; gameplay must keep the queue
and records on the GPU. The record layout is uncompressed qualification data,
not a compressed reservoir layout. Live integration now has valid offscreen
cave captures; see [far-view runtime evidence](history/far-view-runtime.md).
`native_view_runtime_pipelines` and `view_runtime_shader_is_valid` cover the
runtime entry points and both single/multisample prepasses. Use the same
`gameplay` profile in a baseline worktree and a positive-radius candidate for
movement/edit qualification. CPU field diagnostics cover near receivers only;
they cannot establish far-image stability. Far timing logs report raw owners,
known completions, history, pending tails and display coverage separately.

`native_view_discovery_keeps_thin_surfaces_without_growing_storage` is the focused
native coverage gate (`--ignored --nocapture`). Synthetic depth/normal prepasses
contain six depth planes plus one-pixel side planes in one crowded tile, backed
by real transport geometry. Empty tiles elsewhere provide spare slots. It checks
borrowing beyond four local slots, even/odd grid edges, stable stationary face
identities, cross-tile motion and removal/reappearance in the prepass. A nine-slot
case exercises bounded global exhaustion. The actual path-reprojection stage
checks that tagged raw means and QMC indices follow retained faces, while newly
allocated faces do not inherit a recycled slot's unrelated history. Pool/output
layout assertions and single/MSAA pipeline tests cover the binding contract.
This is a coverage/identity test, not rendered quality evidence. Pair it with
cave/streaming captures and total preparation-plus-discovery cost. The older
fixed-tile baseline injection fixture remains in its historical branch, not as
a second runtime implementation.

For the opt-in `view_cache_exchange`, run `native_two_way_cache` with
`--ignored --test-threads=1 --nocapture`. It checks raw transfers (not summed
histories), alternating worker/view sequences, unfinished tails, stale-generation
write rejection, edit retention, and an analytic furnace without reconstruction.
The furnace also asserts that unused reservations do not skip QMC indices based
on path length. `native_view_batches_preserve_raw_prefixes_and_pending_tails`
keeps the exchange-disabled frontend's batching behavior covered.

Rendered exchange checks use background loading and the same +cache settings on
both commits: `house` for reveal/return, `response` for offscreen emitter edits,
and `cave` for sustained many-emitter lighting. Set
`VOXEL_GI_VALIDATE_CACHE_HANDOFF=1` with `hallway` to move an already-visible
source wall into the resident radius and back while preserving its view identity.
`tools/view_cache_quality.py PROFILE BASELINE_RUN CANDIDATE_RUN OUTPUT_DIRECTORY`
compares actual captures, validates offscreen target dimensions/nonblankness, and
produces paired images. Its early-to-own-later FLIP measures remaining visible
change, not reference accuracy; final cross-run differences are separate. It
does not establish interactive FPS. See [two-way results](history/two-way-cache.md).

For shared-kernel refactors, `shader_candidate_benchmark` accepts a fully
composed baseline WGSL via `VOXEL_GI_BASELINE_SHADER`. It checks five-band moment
parity and alternating compute timings against that exact shader. Use the
targeted `shader` filter for layout/composition checks; avoid the broad `view_`
filter, which also selects unrelated view-factor and upload studies.

Production probes keep no CPU measurements (only test fixtures do).
`slim_progress_receives_gpu_stability_without_changing_sample_counts` and
`mirror_record_never_replaces_a_newer_cpu_result` cover HUD progress and
publication ordering natively.

2026-10-21 lifecycle smoke: `a5edd91` (only documentation changed during the
run), dev profile, Apple M4/Metal, 2560×1440 offscreen. Command: `RUST_LOG=warn
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 VOXEL_GI_VALIDATE_BACKGROUND=1
VOXEL_GI_VALIDATE_FRAME_TIMINGS=1 VOXEL_GI_VALIDATE_PROFILE=gameplay
VOXEL_GI_VALIDATE_EXIT=1 VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD"
VOXEL_GI_SETTINGS=target/near-far-lifecycle-64.json cargo run --locked --offline`.
The settings file contained `{"receiver_radius":64,"cpu_statistics":false}` (since removed);
other settings are defaults (corner sun/indirect, 1024 evidence, textures on,
AO off). Evidence:
`target/gi-validation/gameplay-1792609609777779000-76370/` and
`target/near-far-lifecycle-64.log`.

All five captures reported valid targets and nonblank images; preferred and
recovered captures were also visually inspected. Clean render log, 80 scripted
edits and five moves completed, slim coverage/bootstrap reported 75.21%/75.20%
under continued background loading. This establishes runtime/capture integrity,
not convergence or a frame-pacing improvement. The far indirect contribution
was still absent at that historical checkpoint, with visibly darker distant
shaded faces at this small radius. It is present in the subsequent live runtime.
The full-statistics-only edit-freshness metric is unavailable in this slim run;
do not interpret its zero observations as instantaneous response. No matched
baseline or windowed performance claim is attached to this smoke check.

Report average FLIP across windows, worst-window mean FLIP, and worst regional
tile separately. These answer different questions; an average must not conceal a
localized regression. The provisional smooth-wall noise envelope complements,
not replaces, the existing user-calibrated FLIP guard. Change a quality threshold
only with new matched evidence and visual agreement, not to pass a candidate.

The frozen cache sweep, whole-cave reference image and their FLIP/noise tools
rendered through the world display and were removed with it on 2026-10-23
(last present at `3cd98b2`); see [the cave protocol](history/cave-reference.md).

| Change | First useful check |
| --- | --- |
| Documentation/workflow | Diff and link review; no renderer run |
| History, revisions, scheduling | Focused CPU unit tests and response fixtures |
| Traversal, materials, emitter PDF/MIS | CPU reference tests and native CPU/GPU parity |
| Geometry packing/uploads | Native incremental scene parity and bounds checks |
| Display buffer ownership/material extraction | Targeted native generation-lifetime regression |
| Temporal or spatial presentation | `house-edits`, `response` or `compare`, with valid images |
| Streaming / main-thread stalls | `streaming` or `gameplay`, plus field freshness |
| Analytic sources / multi-band transport | Source tests and `sources` |
| Meaningful integrated checkpoint | Regular suite, strict Clippy and selected native checks |

Discover test names rather than maintaining an exhaustive command list here:

```sh
cargo test --locked --offline -- --list
cargo test --locked --offline missed_source_updates
cargo test --locked --offline gpu_matches_cpu -- --ignored --nocapture
cargo test --locked --offline --all-targets
cargo clippy --locked --offline --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --no-deps --document-private-items
```

Ignored native tests create real GPU devices or intentionally expensive
benchmarks. Select by name; do not routinely run all ignored tests. The optional
`stress` profile remains for relevant buffer/material/cache-generation changes or
a returning toggle symptom, not ordinary sampling or performance work.

For execution-only shader experiments, `shader_candidate_benchmark` reads an
archived WGSL file from `VOXEL_GI_BASELINE_SHADER`, alternates warmed baseline and
candidate runs, and checks five-band result equality at identical paths. Sampler
changes deliberately alter those paths: use `sampler_candidate_benchmark` instead.
It compares eight disjoint sequence blocks to the same analytic/high-sample
reference at equal path counts. Neither benchmark establishes gameplay FPS.
`explicit_sky_demand_benchmark` isolates warm sky-cache CPU work and verifies
unchanged transport and displayed sunlight. These remain opt-in, not a new
mandatory suite for ordinary edits.

For scene/source history changes, the focused native
`scene_history_source_extinction_response` test uses an exact dark reference in
RGB, infrared and ionizing bands after removing, replacing or covering emitters.
It checks retained-weight decay and monotonic display at fixed fresh path counts;
its virtual display cadence is not a gameplay latency measurement. Pair it with
`response` and `VOXEL_GI_VALIDATE_BACKGROUND=1` when changing live removal policy.
The response recorder observes every Update, including means/moments, display,
sequence and revisions. Use the [temporal recorder](history/temporal-regressions.md) for
the house-turn stability gate; keep background-loader and ordinary camera-only
streaming results separate. A failed gate stays failed, even when a candidate
improves relative to its control. See the [audit evidence](history/scene-history-audit.md).

The [completed policy controls](history/scene-history-audit.md#policy-closeout) cover
material-ID null changes, moving analytic lights, rare portal discovery,
repeated distant updates, and emission-cache rebuild cost. Run the relevant
`policy_` test with `--ignored --nocapture` only when touching that policy;
`policy_retention_extinction_bounds` is a regular exact-weight unit test.
These fixed-work/virtual-cadence studies do not establish gameplay FPS. The
material-swap cleanup's ordinary publication/late-job tests run without a GPU.
Keep the accepted history/filter settings fixed for the next throughput A/Bs.

`many_light_tree_study` uses the old shader supplied by
`VOXEL_GI_BASELINE_SHADER` and compares mixed/torch/lava cases at matched GPU
budgets. Warm short batches after a long reference pass before measuring.
`native_light_tree_selection_pdf_matches_cpu` checks both selection and reverse
MIS density. Sampler changes cannot pass an identical-path assertion by design;
use distribution/PDF checks and independent high-sample field references.
`terminal_mixture_native_study` compares terminal-emission estimators using an
archived `VOXEL_GI_BASELINE_SHADER`, checks CPU/GPU parity and measures interleaved
fixed-work GPU time. Pair sampler changes with the ordinary independent area-
quadrature test and the relevant all-band source response test; do not routinely
repeat the full study. See the [cave accuracy evidence](history/cave-history-retention.md).
The edit-difference, resident-ownership and control-variate studies live only on
their [recorded experiment branches](history/research-2026-10-09.md), not in the default
test suite. Check out a branch only after stopping the app: shader hot reload
would otherwise contaminate a running comparison.

For acceleration construction, `aadf_construction_benchmark` compares exact
bounds and paired CPU timings against the frozen full-sweep builder.
`scene_topology_reuse_benchmark` compares a one-block edit among 8,192 resident
regions against rebuilding top-level bounds after any region change. Both are
opt-in, have no viewport and measure construction, not frame time. Ordinary
oracle tests cover topology, material changes, residency and immutable snapshots.

`native_box_steps_match_scalar_boundary_policy` checks 210,001 native box steps
against the scalar XYZ rule, including zero/tiny directions, ties and offsets.
`sparse_geometry_upload_benchmark` rotates full metadata, sparse metadata and
sparse metadata plus allocation reuse at identical paths. Its companion
`sparse_geometry_uploads_survive_growth_and_residency_changes` exercises growing
buffers, removed/recycled headers and edits. These target the changed paths; they
do not require the solved toggle stress workload.

`task_encoding_matches_shader_layout_and_clears_reused_tails` compares dispatch
bytes with the generic shader-layout encoder, including signed coordinates,
large sequences and shrinking/mixed task arrays. The opt-in
`task_encoding_benchmark` compares preparation and worker time at 17, 1,024 and
8,192 faces with identical transport outputs. Native sky-only/partial dispatch
and resource-growth tests cover the integration without a toggle-stress run.

### Quality-lever comparisons

Use **Esc → Lighting controls…** for exploratory comparisons. Save defaults to A
and a candidate to B; change one lever at a time. Model Apply/Load resets measured
history progressively but preserves display history, so a screenshot immediately
after loading is not an independent candidate result. Compare settled appearance
and edit response as well as cost; a dark field or starved transport is not a win.

For repeatable starts, export applied RON, save it outside tracked source (for
example `target/reach16.ron`), and set `VOXEL_GI_SETTINGS` for either interactive
or scripted runs. Partial files are supported:

```ron
(transport: (reach: 16.0, bounces: 3))
```

The report records reach, depth, batch samples and existing work settings. Keep
the profile, background-terrain option, build and viewport fixed. A shortened
reach removes some light paths rather than reproducing Minecraft attenuation.
Budget, batch samples, stability thresholds and temporal response answer separate
questions; do not change them together in the first reach comparison.

Focused checks are `tuning_` (CPU configuration/widget/preset contracts),
`voxel_gi::gpu::tuning_tests -- --ignored --nocapture` (native variable-model parity,
sealed-roof and empty-streaming-frontier checks), and optional
`tuning_panel_offscreen_capture -- --ignored --nocapture`. The latter saves
`target/gi-tuning-panel.png` using production widgets and Bevy's image capture,
without a window. It is UI layout evidence, not gameplay lighting or FPS evidence.
No routine toggle stress is needed for tuning.

## End-to-end workloads

Defaults stay shared-corner sunlight, corner-reconstructed indirect, textures on,
baked AO off. Only `compare` and `stress` deliberately switch display modes.
Durations start after required assets are ready and exclude process compilation.

| Profile | Seconds | Intended question |
| --- | ---: | --- |
| `gameplay` | 72 | Preferred-look settling, edits and streaming |
| `streaming` | 35 | Turn 180° and move into newly visible terrain |
| `motion` | 32 | Fixed spawn eye; repeat exact views 0.15° apart to expose selection flicker |
| `house` | 90 | Revisit dark ceiling corners without edits |
| `house-edits` | 180 | House corners, roof skylight, wall doors and turn |
| `response` | 30 | Underground cavity and emitter on/off response |
| `edits` | 30 | Early frames after underground and surface edits |
| `tunnel` | 60 | Dig a one-wide, two-high hillside tunnel for 20 blocks; paired early/later views |
| `sources` | 72 | Analytic/emissive source changes, multi-band field and streaming |
| `cave` | 62 | Fixed view of many-emitter chamber; automatically selects cave geometry/palette |
| `structures` | 62 | Normal-world house/path, cave entrance, and underground interior |
| `day-cycle` | 183 | Canyon under a rotating sun: history, sky retrace and flicker |
| `compare` | 29 | Controlled display-mode visual comparison |
| `stress` | 72 | Optional toggle/buffer lifetime stress |

Every run records the process physical footprint (`footprint -p`, which
includes Metal/GPU allocations on Apple silicon) every 2 s on its own thread:
`report.json` `memory` has the peak, last value and series, and `footprint.txt`
the final category breakdown (IOAccelerator/graphics vs heap). It is the whole
process, not GI's share. `tools/base_spec_suite.sh <label>` runs the base-spec
suite on a target machine (windowed gameplay and day-cycle for frame pacing;
headless response, house-edits and day-cycle with fixed-sun references for
lighting and memory) and archives the results to copy back; `BASE_SPEC_ONLY`
selects runs by name.

`VOXEL_GI_VALIDATE_TURN_BURST=1` makes `streaming` turn 180 degrees at 15 s
and stand still, capturing 24 consecutive frames (`turn-*`) plus `turn-1s` and
`turn-3s`; light-component captures (`VOXEL_GI_VALIDATE_LIGHT_COMPONENT`) also
apply to `streaming`. Count pure-black pixels per frame to see how long newly
viewed terrain waits for GI geometry.

`VOXEL_GI_VALIDATE_PLACE_EMITTER=1` makes `edits` place an emitter instead of
rock at 19 s and capture 16 consecutive frames from the edit
(`emitter-place-*`), to count frames a newly drawn block lags the GI geometry.

The `tunnel` profile removes the upper block at each depth, then the lower block
250 ms later, while advancing one depth every two seconds. Keep background
streaming enabled. `tools/tunnel_quality.py BASE CANDIDATE OUT` compares the
400 ms and 1700 ms views at six depths and generates a contact sheet. These are
temporal-change measurements, not reference accuracy; inspect geometry before
interpreting them. The 50 ms views still contain the lower block. The same
`face-validity` diagnostic is available for separating absent lighting support
from mesh/transport publication skew. Its blue 4x4 execution witness is excluded
from mask counts and does not relax ordinary black-frame capture checks.

The `motion` profile keeps the generator's spawn eye fixed (independent of
startup physics), uses the ordinary terrain camera's 45° vertical FOV, and
alternates two exact orientations 0.15° apart. `VOXEL_GI_VALIDATE_VIEW_COVERAGE=1`
replaces terrain shading with an identity-coverage mask: green is exact-face
support, yellow is coplanar-only support, red has no indirect support. This
measures visible holes that the selected-owner `far_uncovered` counter cannot.
Do not use mask runs to compare normal material cost or lighting quality.
`tools/motion_quality.py RUN...` compares normal captures: exact-return FLIP,
plus an analytically aligned small-turn luminance residual with a two-pixel
low-pass to suppress texture resampling. The latter is a motion-stability
diagnostic, not the calibrated raw-reference FLIP threshold. Run analysis outside
timed gameplay. Interval `far_discovery_mean_ms` and `far_discovery_receipts`
aggregate every existing GPU receipt; prefer them over sparsely logged last-frame
costs for moving/staggered workloads.

For motion diagnosis, `VOXEL_GI_VALIDATE_LIGHT_COMPONENT=sun|indirect|combined`
captures lighting without terrain textures, using the same transport and cached
corner shadows. It is mutually exclusive with `VOXEL_GI_VALIDATE_VIEW_COVERAGE`.
Reports record `view_diagnostic` (0 appearance, 1 combined irradiance, 2 coverage,
3 indirect, 4 sunlight, 5 exact corner sunlight). The `sun-reference` component
traces each displayed face's own four canonical corners, bypassing borrowed
view-cache shadows. Compare it against `sun` at the same camera to detect
persistent wrong shadows even when exact-view return tests are stable. It is an
expensive diagnostic, not a proposed gameplay fallback or a performance run.
`VOXEL_GI_VALIDATE_MOTION_STATIC=1` keeps the camera
stationary through the same capture schedule and background streaming, to
separate ongoing refinement from motion-sensitive reconstruction. `VOXEL_GI_VALIDATE_MOTION_WALK=<voxels/s>`
instead walks forward from the same eye from 15 s and captures two bursts of
16 consecutive frames (`walk-a-*`, `walk-b-*`). Exact returns cannot see
flicker that discovery reuse hides on return; consecutive frames can. Use it
with `VOXEL_GI_VALIDATE_LIGHT_COMPONENT=indirect` and `tools/walk_dropouts.py`
for single-frame black flashes (4.32 voxels/s matches `streaming`). Reports record
the actual `motion_turn_degrees`; the analysis uses that angle for reprojection.
Component residuals cover the valid image overlap, including unchanged sky:
the green-terrain mask is inappropriate for white/blue irradiance. Compare like
components, not their means against textured-terrain means. Exact-view return
and repeated-turned-view FLIP need no alignment; coverage colors are rejected as
lighting evidence. Component runs are diagnostic, not normal-material timings.

For tiny camera-dependent dark specks, use `VOXEL_GI_VALIDATE_LIGHT_COMPONENT=face-validity`
(mode 6) with a view-first configuration. Red independently marks missing indirect
support; green marks a material lookup that is not an exposed transport face.
`tools/view_validity.py RUN...` counts both nonzero pixels and their resolved
channel sums. The old green/yellow/red coverage palette can hide partially covered
MSAA failures: red and green samples resolve to yellow, indistinguishable from
valid borrowed support. Do not use only fully red pixels or whole-image FLIP to
clear sparse edge flicker. This diagnostic excludes sky-mixed silhouette pixels;
two flags in one resolved pixel need not describe the same MSAA sample. It uses
additional geometry lookups and is not a normal-material performance run.

The `streaming` profile now detaches the camera from the character controller
and replays the generator's spawn eye, a 180-degree turn at 15s and 64.8 voxels
of absolute travel over the next 15s. It still streams normal/background terrain;
it is not a controller/collision test. Verify the recorded camera positions in
paired runs. Older Transform-only body motion could be overwritten by physics,
so those runs are not a reliable matched-path performance baseline.

For unattended lighting/capture checks, use Bevy's offscreen target:

```sh
RUST_LOG=info \
VOXEL_GI_VALIDATE=1 \
VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_BACKGROUND=1 \
VOXEL_GI_VALIDATE_PROFILE=house-edits \
VOXEL_GI_VALIDATE_EXIT=1 \
cargo run --locked --offline > target/house-edits.log 2>&1
```

Validation does not automatically enable terminal statistics. Use the JSON frame
and work timelines for performance evidence; synchronous per-page console output
can contaminate main-thread timings. Set `VOXEL_GI_LOG_STATS=1` explicitly when
investigating verbose statistics, and label those runs separately. A quiet run
must be compared with a quiet control, not credited as a gameplay optimization
against an older verbose validation run. Normal gameplay already defaults to
quiet statistics.

The capture target is 2560 × 1440. This needs neither desktop focus nor an unlocked
screen, but system sleep can still suspend execution. Do not change security or
screensaver settings. Omit `VOXEL_GI_VALIDATE_HEADLESS` for a windowed run, then
verify its real window size, focus and unobscured captures. Never compare offscreen
app cadence to windowed FPS or present it as interactive frame pacing.

The `cave` profile uses the preferred corner look and a fixed 65° vertical FOV,
capturing at 15/30/60 seconds. Fixed warm-up is not certified convergence. Use
`VOXEL_GI_SCENE=cave cargo run --locked` for interactive exploration. The finite
chamber uses a four-chunk spawning distance; do not compare its cadence against
the normal ten-chunk terrain workload. RGB, IR and ionizing transport use the
same fixture palette as the native compute studies, with no cavity-specific GI.

The `structures` profile stays in the normal world, moving its detached view at
12/24 seconds and capturing at 10/22/45/60 seconds. It checks integration of the
nearby cave and house through ordinary streaming, not character locomotion.
Normal sunlight/environment and the ten-chunk spawning distance stay active.
Do not compare this workload with isolated `cave` timings. See
[structure ownership and verification](structures.md).

The `day-cycle` profile uses the canyon view and rotates the sun about world Z
from the canyon sun at `VOXEL_GI_VALIDATE_SUN_RATE` degrees per second (default
0.3, a 20-minute day), starting at `VOXEL_GI_VALIDATE_SUN_OFFSET` degrees.
`VOXEL_GI_VALIDATE_SUN_STEP` quantizes the angle to whole steps. A fixed-sun
reference is rate 0 with the offset the moving run reaches at a capture time;
`tools/day_cycle_analysis.py cont=<run> ref18=<run> ref36=<run> ref54=<run>`
compares them.
Captures come in pairs one second apart (30/31 ... 180/181 s): the sun moves a
fraction of a degree between them, so their FLIP is mostly flicker. `day.json`
records traced paths, sky-corner retraces and revisions each second.

First results (gi-day-night on 4915387, Apple M4, headless, full mirror). A
fixed sun settles to median history ~1,000 samples by 180 s at ~4.3M paths/s,
sky retrace stops after ~40 s, and paired-frame FLIP is ~.001. A sun moving
continuously at 0.3°/s publishes a global source change every frame
(source revision +43/s): median history stays 0, traced paths drop to
0.04-0.13M/s, sky corners are retraced at ~18k/s without end, and the frame
never receives sunlight or bounce (FLIP to the fixed-sun reference .49-.56,
tile max .98; paired frames identical). Stepping the sun 1° at a time keeps
median history 0 (p90 16-20), paths 1.2-1.7M/s, FLIP to reference .12-.16
and paired-frame tile spikes up to .96 at steps.

### Background-terrain variants

Add `VOXEL_GI_VALIDATE_BACKGROUND=1` to any profile, including the small `house`,
`house-edits`, `response`, `edits`, `sources`, and `cave` fixtures. An inactive BVW
loading camera walks at 4.32 voxels/s from (-96,40,0) toward -Z throughout the run.
It generates and meshes real terrain through the ordinary residency/NAADF/GI
publication paths; it does not render another view or take over GI camera priority.
The measured view and its scripted edits remain unchanged. This is a continuous
background-load variant, not merely a room surrounded by already-loaded chunks.
Use this variant for end-to-end refinement/response promotion decisions; omit the
flag when diagnosing an isolated fixture. Do not require every profile each time:
one affected large world plus the relevant small-room countercheck is sufficient.

`cave` with this flag uses the nearby normal-world structure and its entrance
landing view, the normal palette/environment and ten-chunk draw distance. Without
it, `cave` retains the isolated chamber control. The report identifies the variant,
loader updates and scene-revision timeline; compare matching variants only. The
small fixed-work native compute studies remain isolated estimator controls, not
evidence of throughput under streaming. Use these composed end-to-end profiles
before promoting their winners.

Each run creates a unique `target/gi-validation/<profile>-<stamp>-<pid>/` directory
with status, report and images. Schema 3 identifies offscreen/window capture;
older schema-2 records in the archive retain their original limitations. The
harness validates assets, target dimensions, completed/nonblank captures and
workload actions. Independently review render logs and images. A completed
workload or nonblank image alone is not a quality/performance pass.

## Interpret the measurements

`live-cave` and `live-hallway` hold the actual normal-world structure views for
182 seconds, with captures at 15/30/60/120/180 seconds. The hallway camera stands
beside the emitter looking toward the dark end, unlike the historical `hallway`
overview looking toward the lamp. Use `VOXEL_GI_VALIDATE_BACKGROUND=1` for a
moving terrain loader. Neither profile freezes sampling, forces uniform
histories, or replaces scheduling with an offline prefix bake. These views diagnose live publication and sampling; they do not supersede the
raw matched-prefix sampler controls or establish windowed frame pacing.
The explicit `gi-display-unfiltered.ron` control also disables the final
render-frame transition. Physical prefix exports are unchanged by that
transition; it must not be mistaken for improved sampler accuracy.

- `VOXEL_GI_VALIDATE_FRAME_TIMINGS=1` adds per-frame intervals and the previous
  Update's six GI system-body times to `measurements.frame_timeline`, with
  names in `gi_main_stage_names`. Capture exclusions are explicit. Pipeline
  overlap means the remainder is not a BVW-only measurement. Use this when
  investigating stalls instead of inferring them from 250-ms averaged timings.
  These are opt-in CPU body timings, not GPU timestamps.
- `VOXEL_GI_VALIDATE_NO_VSYNC=1` (windowed validation only) presents with
  `AutoNoVsync` and three frames of latency. Bevy's default Fifo locks every
  option to the display rate, hiding headroom; on macOS `AutoNoVsync` alone
  still blocks most frames on the two-drawable default. Reports record the
  actual `present_mode`. Compare uncapped intervals only with other uncapped
  runs. Add `RUST_LOG=warn,prototype_voxel_template::voxel_gi=info` for the
  view tier's GPU receipts (`GI view transport`: trace, presentation,
  discovery, pool), logged every 120 frames.
- Coverage is at least one path sample per exposed face. Bootstrap targets eight.
  Record published/loaded geometry and pending ingestion alongside percentages;
  an early tiny field can be 100% covered while most terrain is not published.
- Settled is a revisitable heuristic over samples, estimated error, drift and
  display residual. It is neither unbiased error against ground truth nor proof
  that all paths found a small opening. Faces still receive periodic audits.
- Compare frame-time tails with transport throughput, near/far age, source/
  geometry freshness and edit latency. CPU and GPU stages overlap; do not sum
  their times. Capture-associated frames are separately excluded from cadence.
- Inspect temporal change as well as the final frame. Fixed house receivers
  help distinguish display jumps from camera movement and topology turnover.
  Relative RGB steps need an absolute signal floor and cannot establish accuracy.
- For speed-to-correct-light claims, use fixed seeds and equal-work references or
  a higher-sample reference. Measure both brightening and darkening. Three seconds
  of smooth adjustment can be preferable to two seconds of objectionable noise;
  report latency and noise separately rather than naming one “convergence.”
- Never count display priors, filtered colors or interpolated neighbors as new
  simulation samples. Include infrared and ionizing channels in estimator tests.

## Profiling with Tracy

Offscreen frame interval is pinned near 20 ms by the 60 Hz runner (it did not
move with resolution, GI display or transport load on 2026-10-19), so it does
not measure CPU or GPU cost. Use Tracy spans for CPU work:

- Bevy 0.19 links Tracy 0.13.1 (protocol 76). Build `capture` and `csvexport`
  from the Tracy 0.13.x source (`cmake -B build -S capture`, then
  `cmake --build build`; the same for `csvexport`). Homebrew's 0.14 cannot connect.
- Start `tracy-capture -o target/prof/run.tracy -f -s 80`, then run the profile
  with `cargo run --features tracy` and `RUST_LOG=info` (`warn` hides the spans).
- `tracy-csvexport -e -s '|' run.tracy` gives self time per span. Divide by
  the frame count for ms per frame. Span names contain commas, hence `|`.
- GI worker stages have their own spans (`gi_trace_worker`, `gi_prepare_inputs`,
  `gi_evidence_gather`, `gi_face_index_upload`, `gi_submit`, `gi_readback`,
  `gi_apply_retention`, `gi_scene_build`). Bevy systems get spans automatically.
- On Metal, Bevy records no GPU pass timestamps. GI transport has its own
  (`transport_execution_ms`); use Xcode's Metal debugger for render passes.

## Shared Cargo cache

The tracked `.cargo/config.toml` isolates workspace artifacts with the checkout-local
`tools/rustc-workspace.sh`. It optionally includes ignored `.cargo/local.toml` for
machine-specific settings. This machine's local file shares compatible dependency
intermediates with related Bevy prototypes:

```toml
[build]
build-dir = "/Users/nathanaelneveux/Developer/.cargo-build/bevy-0.19"

[profile.dev]
opt-level = 1
debug = "line-tables-only"
incremental = true

[profile.dev.package."*"]
opt-level = 3
debug = "line-tables-only"

[profile.dev.package.wgpu-types]
debug-assertions = false
```

Configure a suitable absolute path in `.cargo/local.toml` on another machine; do
not check a user's path into the project. Keep the tracked wrapper configuration
in place. Cargo hashes the resolved `rustc-workspace-wrapper` path into workspace
artifact names, so different checkouts do not alias the application while registry
dependencies remain shared. The POSIX wrapper only executes the original compiler
command; it changes no compiler flags. Do not move it to one global path. See
[Cargo's workspace-wrapper contract](https://doc.rust-lang.org/cargo/reference/config.html#buildrustc-workspace-wrapper).

Final binaries, rustdoc and validation artifacts remain local under `target/`.
Distinct feature/profile fingerprints can still require separate cached builds.
Do not clean the shared cache as a routine test step or delete other prototypes'
outputs to make a benchmark look clean. Historical worktrees without the tracked
configuration need their own copy of the wrapper/config before sharing intermediates.

### Release launch regression, October 16

On `906e50e`, `cargo run --release --locked --offline` reused an older executable
from the shared cache, then loaded the current terrain shader. Pipeline creation
failed because the old material layout lacked resident-buffer binding 105.
Cached compiler warnings named `validation_mirror`, absent from the current code;
Cargo's fingerprint log nevertheless classified the application as fresh. This
matches the documented [same-name workspace collision](https://github.com/rust-lang/cargo/issues/7740).

A two-checkout, dependency-free release reproduction ran A's code when asked to
run B. Adding `env!("CARGO_MANIFEST_DIR")` alone **did not prevent** reuse on Cargo
1.95.0. The checkout-local workspace wrapper passed A/B/A with correct identities.
The permanent focused regression uses the real config/wrapper:

```sh
python3 tools/test_shared_build_cache.py
```

The application also logs its compiled checkout at info level in every profile;
that is diagnostic provenance, not the isolation mechanism. This fix changes
neither transport nor presentation. Baseline logs, fingerprint trace and reduced
reproduction are under `target/startup-crash-*` and
`target/shared-cache-check.oG1D6B/`. Verification uses this commit's build/config
diff on the Apple M4 / Metal, macOS 26.6.2, Rust/Cargo 1.95.0. The development
headless `edits` workload passed on unchanged runtime code before this build-only
fix; it is not evidence of release frame pacing. With the final wrapper/config,
`cargo run --release --locked --offline` rebuilt only the application and stayed
running past startup with the correct build-root log and no render errors. A
subsequent release build reused that application successfully. The existing egui
Metal bindless warning remains; no dependency, shader or lighting setting changed.

Keep focused, buildable local commits. Record verification and limitations in
their messages; do not combine unrelated estimator, storage and presentation
changes merely because they share one session.
