# Validation and regression tracking

Use the smallest check that answers the question. GI is visual RGB lighting;
heat and radiation have no simulation query or qualification contract here.
Sampling accuracy, presentation quality and gameplay cost are separate results.

## Reproducible setup

Record the commit and dirty diff, compiler/profile, hardware/backend, viewport,
settings and workload. Build before timing. Do not run another game instance,
compiler or GPU benchmark during a paired measurement. Keep raw logs and images
under ignored `target/`, not Git. `--offline` commands assume cached dependencies.

A baseline/candidate pair can guide a substantial change. Repeat small or noisy
differences before claiming a gain. Keep shared-corner sunlight, reconstructed
indirect, textures on and baked AO off unless testing those controls explicitly.

## Choose the relevant check

| Change | First useful check |
| --- | --- |
| Docs/comments | Diff, links, formatting; doctests for changed examples |
| Host input / scene construction | Focused page, region and connector tests |
| Traversal / emitter PDFs / materials | Native path tests and analytic fixtures |
| Surface-cache transport | Native furnace and view/cache exchange tests |
| Geometry uploads / buffer lifetime | Native growth and generation tests |
| History / edit response | Focused retention tests, then `response` or `house-edits` captures |
| Spatial / temporal presentation | Paired `house`, `cave`, `motion` or `tunnel` captures |
| Streaming / stalls | `streaming` or windowed `gameplay`, with progress and freshness |
| Release integration | Crate tests/examples, doctests, Clippy, rustdoc, package verification |

The public crates and lab are separate Cargo packages. The lab is not published.
For the release crates:

```sh
cargo test --locked -p bevy_voxel_gi -p bevy_voxel_gi_bvw -p bevy_voxel_gi_debug --lib --examples
cargo test --locked -p bevy_voxel_gi -p bevy_voxel_gi_bvw -p bevy_voxel_gi_debug --doc
cargo clippy --locked -p bevy_voxel_gi -p bevy_voxel_gi_bvw -p bevy_voxel_gi_debug --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --no-deps -p bevy_voxel_gi -p bevy_voxel_gi_bvw -p bevy_voxel_gi_debug
cargo fmt --all --check
```

Discover native test names with `cargo test -p bevy_voxel_gi --lib -- --list`.
Select the affected path rather than running every ignored benchmark:

```sh
cargo test -p bevy_voxel_gi --lib native_view_runtime_pipelines -- --ignored
cargo test -p bevy_voxel_gi --lib native_two_way_cache -- --ignored --test-threads=1
cargo test -p bevy_voxel_gi --lib cached_furnace_converges -- --ignored
```

Native adapters honour `WGPU_BACKEND`, `WGPU_DX12_COMPILER`,
`WGPU_ADAPTER_NAME` and `WGPU_POWER_PREF`. Check the printed adapter backend.
A requested unavailable backend must fail instead of silently testing another.
DX12 needs DXC; see [platform setup](testing-other-gpus.md).

### Raw sampling and presentation

For sampler changes, compare matched raw prefixes without temporal or adaptive
spatial filtering. Borrowed support and initialization priors are not independent
samples. For recursive-cache transport, use analytic furnace and native cache
checks; a finite three-bounce reference has a different fixed point.

The [cave sampling study](history/cave-sampling.md#user-calibrated-visual-target)
records a visually accepted 2,048-path prefix and visibly worse 1,536/1,024
controls for its estimator and scene. This is historical calibration, not a
universal sample threshold or a reason to cap today's histories at 2,048.
The original frozen world-display sweep and its analysis tools are no longer
in the current tree. Reproduce them at their recorded revision, or build a
new matched-prefix harness when qualifying a new sampler.

For presentation, compare actual images at matched scene/settings: settling
and reveal/return with `tools/view_cache_quality.py` on `house`/`cave`, edits
with that tool on `response`, tunnels with `tools/tunnel_quality.py`, and
motion with `tools/motion_quality.py`. `tools/gates.sh` bundles these checks;
it is an integration gate rather than a routine test for every small change.

Use `config/gi-display-unfiltered.ron` to disable adaptive reconstruction and
`config/gi-far-tier-off.ron` for receiver-only lighting. Keep raw-estimator and
filtered gameplay conclusions separate. World-cache receiver records do not
describe the final displayed view.

Require valid assets, camera/target dimensions, nonblank captures and clean
render logs. Windowed evidence must also be focused and visible. Prefer
`VOXEL_GI_VALIDATE_HEADLESS=1` for unattended image checks, and do not change
screensaver/security settings. Headless app cadence does not establish
interactive FPS. Windowed headroom checks use `VOXEL_GI_VALIDATE_NO_VSYNC=1`.
Compare uncapped runs only against uncapped runs.

The solved toggle/buffer-generation regression has focused coverage. Run the
optional `stress` profile only for related buffer, material or generation
changes, renderer upgrades or a returning symptom.

### Quality-lever comparisons

Use **Esc → Lighting controls** for exploratory comparisons; change one lever at
a time. Model Apply resets measured history progressively but preserves display
history, so a screenshot immediately after applying is not an independent
candidate result. `VOXEL_GI_VALIDATE_DEBUG_UI=1` renders the overlay (panel
open) into validation captures to check the UI itself, not lighting. Compare settled appearance
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
`bevy_voxel_gi::gpu::tuning_tests -- --ignored --nocapture` (native variable-model parity,
sealed-roof and empty-streaming-frontier checks), and optional
`tuning_panel_offscreen_capture -- --ignored --nocapture`. The latter saves
`target/gi-tuning-panel.png` using production widgets and Bevy's image capture,
without a window. It is UI layout evidence, not gameplay lighting or FPS evidence.
No routine toggle stress is needed for tuning.

## End-to-end workloads

Defaults stay shared-corner sunlight, corner-reconstructed indirect, textures on,
baked AO off. Only `compare` and `stress` deliberately switch display modes.
Scripts, phase boundaries and captures count from GI's first view receipt.
`gi_ready_s` records the wait after terrain assets are ready; frame and work
timelines include that wait. Compare baselines using this same clock contract.
A run fails if GI publishes no view receipt within 120 seconds.

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
| `sources` | 72 | Analytic/emissive RGB source changes and streaming |
| `cave` | 62 | Fixed view of many-emitter chamber; automatically selects cave geometry/palette |
| `structures` | 62 | Normal-world house/path, cave entrance, and underground interior |
| `day-cycle` | 183 | Canyon under a rotating sun: lag behind fixed-sun references and flicker |
| `compare` | 29 | Controlled display-mode visual comparison |
| `stress` | 72 | Optional toggle/buffer lifetime stress |
| `canyon` | 62 | Long daylight paths and high terrain walls |
| `hallway` | 62 | Fixed single-emitter overview |
| `live-cave` / `live-hallway` | 182 | Normal-world structures under live scheduling |
| `lod` | 40 | Coarse geometry and receiver pressure during flight |
| `overlook` | 70 | Travel, rise above terrain and hold on far-band faces |

On macOS, each run records the process physical footprint (`footprint -p`, which
includes Metal/GPU allocations on Apple silicon) every 2 s on its own thread:
`report.json` `memory` has the peak, last value and series, and `footprint.txt`
the final category breakdown (IOAccelerator/graphics vs heap). Other platforms
currently report no memory samples. It is the whole
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
(mode 6) with the default view-owned configuration. Red independently marks missing indirect
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
the normal ten-chunk terrain workload. It shares the native fixtures' palette,
with no cavity-specific GI.

`VOXEL_GI_VALIDATE_CAVE_SKYLIGHT=open` turns `cave` into a high-skylight
response test (`src/validation/skylight.rs`): a 7×7 ceiling hole opens at 15 s
and its sun lands 38 voxels below, on stone laid over the lava so the bounce
onto the east wall changes. `=pre` opens it at startup and is the settled
same-time reference. Captures run 0.1 s to 30 s after the opening;
`tools/skylight_quality.py PRE label=OPEN ...` prints FLIP and the worst 32 px
luminance tile against the reference. `VOXEL_GI_VALIDATE_LIGHT_COMPONENT`
applies (`indirect` isolates bounce). The distance rule alone leaves the floor's
history; the edit's shadow footprint is what relights it.
With `=pre`, `VOXEL_GI_VALIDATE_SKYLIGHT_SUN_RATE=-1` turns the sun from 15 s
so the patch slides across a light-stone floor, and
`VOXEL_GI_VALIDATE_SKYLIGHT_AWAY=1` faces the far corner: the patch reaches the
screen only as bounce through off-screen cache faces. Fixed references are rate
0 at `VOXEL_GI_VALIDATE_SUN_OFFSET=-10` and `-25` (their `sun-40s` captures
against the moving run's `sun-25s` and `sun-40s`). Receipts at 9b1d92e:
.0087/.0121, reference noise .0042. `VOXEL_GI_VALIDATE_SKYLIGHT_DARK=1` turns
every emitter off, so sunlight through the openings is the only light: there
main reads .046/.069 (reference noise .0047; 25 deg is ~9% too bright, a
gradual dimming no visibility flip reports), and world-cache flips
(`experiment/cache-sun-flips`) read .019/.078.

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
records traced paths and batches each second.

Current receipts (2026-10-24, Apple M4, headless): FLIP of the moving run
against ref18/ref36/ref54 is .018/.023/.034 (tile max .04/.05/.02), paired
frames .005-.011; reference self-noise is .002. Removing all sunlight-flip
retention events raises the lag to .024/.042/.059, so shadow borders matter.

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
  runs. Add `RUST_LOG=warn,bevy_voxel_gi=info` for the
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
  independent path samples.

## Profiling with Tracy

Headless validation uses a 60 Hz schedule runner and limits GPU work to one
frame in flight. This provides backpressure when GPU work exceeds the runner
period; offscreen app cadence is still not windowed frame pacing.
`VOXEL_GI_VALIDATE_GPU_PACING=0` disables it. Use Tracy spans for CPU work:

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

The tracked `.cargo/config.toml` only includes the ignored `.cargo/local.toml`
for machine-specific settings, so a fresh clone builds anywhere (Windows has no
POSIX shell for a wrapper script). This machine's local file shares compatible
dependency intermediates with related Bevy prototypes and isolates workspace
artifacts with the checkout-local `tools/rustc-workspace.sh`:

```toml
[build]
build-dir = "/absolute/path/to/shared/bevy-0.19"
rustc-workspace-wrapper = "tools/rustc-workspace.sh"

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
not check a user's path into the project. Set the wrapper with any shared
`build-dir`. Cargo hashes the resolved `rustc-workspace-wrapper` path into workspace
artifact names, so different checkouts do not alias the application while registry
dependencies remain shared. The POSIX wrapper only executes the original compiler
command; it changes no compiler flags. Do not move it to one global path. See
[Cargo's workspace-wrapper contract](https://doc.rust-lang.org/cargo/reference/config.html#buildrustc-workspace-wrapper).

Final binaries, rustdoc and validation artifacts remain local under `target/`.
Distinct feature/profile fingerprints can still require separate cached builds.
Do not clean the shared cache as a routine test step or delete other prototypes'
outputs to make a benchmark look clean. Historical worktrees without the tracked
configuration need their own copy of the wrapper/config before sharing intermediates.


The [historical validation notes](history/validation-design-notes-2026-10.md)
retain the removed study commands and older measurement protocols. Use their
recorded revisions when reproducing those studies; they are not current
release instructions.
