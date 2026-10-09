# Validation and regression tracking

Use the smallest check that answers the current question. The purpose is faster,
stable gameplay and useful simulation measurements, not maximizing test minutes.
The full suite is an integration checkpoint, not the default for every tweak.

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

`many_light_tree_study` uses the old shader supplied by
`VOXEL_GI_BASELINE_SHADER` and compares mixed/torch/lava cases at matched GPU
budgets. Warm short batches after a long reference pass before measuring.
`native_light_tree_selection_pdf_matches_cpu` checks both selection and reverse
MIS density. Sampler changes cannot pass an identical-path assertion by design;
use distribution/PDF checks and independent high-sample field references.
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
