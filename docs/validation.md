# Validation and regression tracking

Use the smallest check that answers the current question. The purpose is faster,
stable gameplay and useful simulation measurements, not maximizing test minutes.
The full suite is an integration checkpoint, not the default for every tweak.

For adaptive-history research, the [DDGI/A-SVGF qualification study](adaptive-history-qualification.md)
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
sequence and revisions. Use the [temporal recorder](temporal-regressions.md) for
the house-turn stability gate; keep background-loader and ordinary camera-only
streaming results separate. A failed gate stays failed, even when a candidate
improves relative to its control. See the [audit evidence](scene-history-audit.md).

The [completed policy controls](scene-history-audit.md#policy-closeout) cover
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
repeat the full study. See the [cave accuracy evidence](cave-history-retention.md).
The edit-difference, resident-ownership and control-variate studies live only on
their [recorded experiment branches](research-2026-10-09.md), not in the default
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

For repeatable starts, export applied JSON, save it outside tracked source (for
example `target/reach16.json`), and set `VOXEL_GI_SETTINGS` for either interactive
or scripted runs. Partial files are supported:

```json
{"transport":{"reach":16,"bounces":3}}
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
| `house` | 90 | Revisit dark ceiling corners without edits |
| `house-edits` | 180 | House corners, roof skylight, wall doors and turn |
| `response` | 30 | Underground cavity and emitter on/off response |
| `edits` | 30 | Early frames after underground and surface edits |
| `sources` | 72 | Analytic/emissive source changes, multi-band field and streaming |
| `cave` | 62 | Fixed view of many-emitter chamber; automatically selects cave geometry/palette |
| `structures` | 62 | Normal-world house/path, cave entrance, and underground interior |
| `compare` | 29 | Controlled display-mode visual comparison |
| `stress` | 72 | Optional toggle/buffer lifetime stress |

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

`cave` and `structures` also save `cave.json`: fixed receivers on all four walls,
including five-band measurements, moments, history, sequence, RGB display and
measurement age. Missing receivers are recorded, not silently excluded. Use it
to distinguish lack of work from sampling error; neither history nor coverage is
ground-truth convergence.

Each run creates a unique `target/gi-validation/<profile>-<stamp>-<pid>/` directory
with status, report and images. Schema 3 identifies offscreen/window capture;
older schema-2 records in the archive retain their original limitations. The
harness validates assets, target dimensions, completed/nonblank captures and
workload actions. Independently review render logs and images. A completed
workload or nonblank image alone is not a quality/performance pass.

## Interpret the measurements

- `VOXEL_GI_VALIDATE_FRAME_TIMINGS=1` adds per-frame intervals and the previous
  Update's seven GI system-body times to `measurements.frame_timeline`, with
  names in `gi_main_stage_names`. Capture exclusions are explicit. Pipeline
  overlap means the remainder is not a BVW-only measurement. Use this when
  investigating stalls instead of inferring them from 250-ms averaged timings.
  `previous_display_stage_ms` further splits display publication into poll,
  install, topology, snapshot start/advance, values and ledger, with names in
  `display_stage_names`. These are opt-in CPU body timings, not GPU timestamps.
- `work_timeline` includes `display_preparing`, `display_catchup_keys` and
  `display_topology_pending`. Check them alongside geometry, first-sample debt,
  throughput and receiver age: small frame costs obtained by indefinitely
  delaying the replacement cache are not a pass. Private catch-up must drain;
  ordinary live lighting must continue while it does. Record replacement latency
  as well as its per-frame cost.
- `VOXEL_GI_VALIDATE_TEMPORAL=1` also supports `cave` and `structures`: 36 fixed
  exposed wall faces and their 144 reconstructed corners, with `site_faces` in
  schema 2. Both the isolated chamber and composed normal-world structures are
  checked for exposure. Missing sites are failures, not stable black samples.
  See the [streaming investigation](streaming-performance.md) for exact runs.
- `VOXEL_GI_VALIDATE_REFERENCE=1` adds two offline 262,144-path references per
  fixed face on the frozen final scene, **after** timings/captures. It requires
  headless validation with exit enabled. Use `tools/reference_analysis.py` to
  compare actual RGB error, replica disagreement and sustained accuracy, not
  just history count. `VOXEL_GI_VALIDATE_CAVE_LONG=1` extends the cave workload
  to 182 seconds; optional `VOXEL_GI_VALIDATE_PATH_DIAGNOSTIC=<receiver-index>`
  replays individual CPU/GPU paths to diagnose outliers. These are opt-in
  investigations, not a new routine validation requirement. Full protocol and
  limitations: [history-retention study](cave-history-retention.md).
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

## Shared Cargo cache

This machine uses an ignored `.cargo/config.toml` to share compatible intermediates
with related Bevy prototypes:

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

Configure a suitable absolute path on another machine; do not check a user's path
into the project. Final binaries, rustdoc and validation artifacts remain local
under `target/`. Distinct feature/profile fingerprints can still require separate
cached builds. Do not clean the shared cache as a routine test step or delete
other prototypes' outputs to make a benchmark look clean.

Keep focused, buildable local commits. Record verification and limitations in
their messages; do not combine unrelated estimator, storage and presentation
changes merely because they share one session.
