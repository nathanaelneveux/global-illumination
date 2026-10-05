# Voxel GI Lab

This is a fork of `prototype_voxel_template` with persistent, world-space, per-voxel global illumination for the complete loaded voxel world. It is not a camera-following light volume and it does not trace one GI path per screen pixel.

The implementation draws on [vxray](https://github.com/Nelarius/vxray), its [development log](https://nelari.us/post/voxel-ray-tracing/), and [NAADF: Globally Illuminated Voxel Worlds Accelerated with Nested Axis-Aligned Distance Fields](https://www.cg.tuwien.ac.at/research/publications/2026/ulschmid-2026-naadf/). This is an independent compute implementation, not a port of the paper's complete renderer. BVW remains responsible for streaming, meshing, and editing; transport uses its own sparse acceleration structure. It runs on Bevy 0.19 without hardware ray-query support.

## What is implemented

- Each loaded 32³ page is split into eight 16³ regions, with nested 4³ brick and voxel distance fields. Empty and uniform regions are compressed. On the GPU, bricks also have empty/uniform/mixed tags: empty bricks carry AADFs, uniform bricks carry a material, and mixed bricks point to 64 packed 16-bit voxel cells. Identical mixed bricks share reference-counted storage with copy-on-write edits and full-content equality checks. Unchanged regions are shared between snapshots; GPU geometry is persistent and receives changed-region patches.
- Missing regions inside the streamed domain are unknown, not open sky, for transport/simulation queries. Those rays terminate conservatively. Display-only direct-sun rays instead continue across gaps and test all loaded geometry beyond them, matching the geometry that Bevy renders rather than inventing opaque streaming slabs. Geometry outside the loaded set cannot cast display shadows.
- Every exposed solid voxel owns a six-direction ambient-cube probe. Surface sampling is independent of camera visibility and screen resolution.
- GPU compute traces four new paths per **selected exposed face** per transport dispatch, with cosine-weighted directions, three diffuse bounces, visibility-tested analytic sources, Russian roulette, and power-weighted emissive-face sampling with multiple importance sampling. A CPU reference/fallback uses the same transport model.
- Independently rotated low-discrepancy sequences continue advancing after the 256-sample history limit. Eight samples is only a bootstrap milestone, not convergence. Settled faces reduce their update frequency rather than stopping forever; see change-driven scheduling below.
- Indirect display reconstruction averages connected, coplanar neighboring face samples at **virtual voxel corners**, then bilinearly blends across each face. Direct sun instead traces visibility at the actual four unit-face corners, preserving contact shadows that miss face centers. Greedy quads stay intact: no additional mesh vertices are needed. Missing faces and disconnected diagonals are excluded from indirect reconstruction to limit light leaking across edges.
- Direct-sun corners are interned by integer world position across faces, chunks, and dispatches. Each missing corner traces once per visible directional source, then projects the result onto all six receiving normals. A 100×100 coplanar patch has 10,201 unique corners (versus 10,000 centers or 40,000 duplicated corner references); subsequent unchanged batches trace **zero additional sky-corner rays**. Fragmented surfaces have a higher boundary-to-area ratio, so corner counts are not universally equal to face counts.
- A persistent CPU-owned sky cache feeds both compute and CPU fallback. Sun direction/color changes invalidate it; point lights, indirect environment changes, camera movement, and display toggles do not. Geometry changes invalidate spatial buckets intersecting the changed regions' shadow sweeps, up to the same 768-voxel range as tracing, including remote receivers. Air-only streaming preserves visibility; retired receiver entries are pruned. This is independent of the continuously sampled indirect/simulation field and adds no debounce.
- New faces start with conservative dark indirect light or a low-confidence estimate from fresh, same-facing pre-edit neighbors connected through loaded air. They never initialize to unoccluded sky. A separate presentation history filters measured indirect RGB; confidence-weighted corner reconstruction favors measured faces. Deterministic direct-sun visibility responds without temporal filtering, and seed reuse is restricted to identical geometric corners.
- Bevy PBR retains ordinary visible direct/specular lighting. The extension adds cached indirect RGB, standalone/custom-emitter RGB, and voxel self-emission without counting ordinary direct light twice. Simulation queries receive direct plus indirect energy before display smoothing.
- Edits invalidate local histories, retain existing display values, and reject incompatible in-flight geometry results. A lazy global transport revision also wakes distant indirect receivers without clearing their display or scanning the world during invalidation. Nearby work is prioritized; at least one quarter of transport candidate slots are reserved for an uninterrupted all-loaded-world sweep. Moving sources shorten histories instead of repeatedly clearing the world.
- Adaptive scheduling ranks bounded rotating windows by age, source revision, and per-band relative uncertainty. RGB cannot mask uncertain infrared/ionizing transport. Edit/bootstrap/simulation requests take priority, while the reserved world sweep still prevents starvation. Selection has a per-face work mask distinct from the voxel's exposed-face topology, so an uncertain face does not force its settled neighbors to retrace.
- Scene building, cache construction, dispatch, and readback run asynchronously. The GPU tracer adapts its batch size toward a configurable time budget. A CPU mirror provides gameplay estimates; this is not a wholly GPU-resident authoritative field.
- Sparse display uploads preserve stable slots during edits. Topology patches preflight all allocations and hash insertions before publishing anything. Growing the cache keeps updating the previous display, and the replacement merges newer measurements in its installation frame rather than briefly showing the old worker snapshot. Invalidation preserves display blending weights as well as values; replacement weights change with the filtered estimate. No additional edit debounce is used.
- Display-cache rebuilds publish fresh buffer identities and generation-owned patch queues. In-flight render frames retain their old generation; paired hash/probe uploads are bounds-checked and deferred updates are retained.

The core lives in [`src/voxel_gi/`](src/voxel_gi):

```text
BVW streamed chunks + edits
          |
          v
32³ pages -> 16³ regions -> 4³ bricks -> 4³ voxel NAADFs
          |                         |
          |                         +-> hierarchical world-space rays
          v
exposed-voxel ambient cubes <-> continuous GPU multi-band path transport
          |                                      |
          v                                      +-> unfiltered simulation estimates
world-voxel GPU hash -> virtual corners -> custom BVW PBR material
```

## Loaded-world coverage

Coverage is defined by BVW's streamed chunk set, not a camera-following light volume. Newly loaded pages enter the same scene and despawned pages leave it. All loaded exposed surfaces receive background updates, including offscreen terrain; nearby surfaces receive more frequent updates. A bounded work budget necessarily means distant estimates can be older, particularly during streaming.

Streaming publication is staged by spatial page, nearest-first, rather than merging a whole turn-around's newly visible terrain in one frame. The probe index holds stable heap allocations so index growth does not relocate the large ambient cubes. Geometry/NAADF work and GPU encoding/mirror construction run on workers; snapshot packing is incremental (at most 8,192 probes or approximately 1.5 ms per tick). GPU allocations reserve geometric headroom, and live updates use a separate catch-up ledger while a replacement builds. ECS publication and GPU allocation/upload still have nonzero costs; this is not a hard real-time guarantee. Edit builds can overtake a streaming publication without dropping unrelated pending pages. Simulation estimates return `None` for receiver pages awaiting publication instead of reporting old topology as current.

Chunk collision scans/construction use two bounded workers, prioritized near the player. Results are checked against current entity, LOD and voxel hash before installation; old collision remains until its replacement is ready. Physics component publication is capped at two chunks per frame.

`VoxelGiStats` separates first-sample `interactive_coverage`, eight-sample `bootstrap_coverage`, mean history, oldest measured-sample age, and refresh progress. None is a certified convergence/error bound. Check per-surface age and source revision when freshness matters to gameplay. The temporary display fallback for an unmeasured face is never returned as a simulation measurement.

## Change-driven scheduling and NAADF additions

Direct sunlight and GI use the **same voxel ray-traversal geometry**, but have different sampling/update policies. Cached direct sun is deterministic visibility at shared corners; diffuse environment light and bounces are stochastic multi-band path transport. Bevy's per-pixel sunlight comparison mode still uses its own raster shadowing. F7 changes presentation, not the transport model.

Sky repairs have an independent urgent queue and loaded-world refresh cursor, but **share the worker submission with due indirect work**. Duplicate receivers are merged; additional sky-only receivers have a zero transport mask, so they generate no GI paths and cannot modify simulation sample counts, moments, or timestamps. Sky-only submissions remain possible when no indirect work is due, and fully cached sky-only batches submit no GPU commands or rebuild the emissive distribution. Changed sky values bypass the normal display upload interval. This avoids spending two out of three worker turns on sky while indirect lighting waits. Geometry/sun generations prevent obsolete sky values from being published in either kind of batch.

Indirect faces become eligible for reduced-frequency updates only after at least 64 samples by default, four stable batches, low relative error and drift in **every energy band**, and a sufficiently settled display value. They remain eligible for background verification every two seconds; eligibility is not a service-time guarantee. Explicit simulation refresh requests override resting state. Geometry, streaming, material, and source changes wake faces through a revision check. The mechanism follows NAADF Section 4.2's selective-sample-generation principle, adapted to world-space faces rather than TAA/pixel history; it is not the paper's resampling estimator.

Indirect invalidation is intentionally conservative and **currently global**, unlike the spatially targeted direct-sky cache. There is no claim of complete per-path dependency tracking. This avoids permanently stale distant light after an edit, including insertion into previously empty space, but frequent edits/moving sources can keep indirect work active. The rotating audit remains necessary: a stable low-sample estimate is not proof that a rare light contribution does not exist. Simulation freshness includes a transport revision and each requested face's own age; a fresh top face cannot make an old side-face measurement appear fresh.

Global wake-up is separate from evidence destruction: remote histories retain up to 64 samples of temporal weight, rather than falling back to four on every streamed air page or local light movement. Known local geometry changes still invalidate measurements, and global source changes/nearby moving lights still use the short four-sample response. Distant changes therefore have a bounded temporal response, not instant exact convergence. Freshness metadata remains conservative. Candidate scanning is also independent of emitted work: settled probes do not consume ray slots, and underfilled cheap batches cannot collapse the scheduling budget. Searches remain bounded to at most 8,192 candidates per selection, plus the bounded priority windows.

From NAADF Sections 3.1-3.2, the GPU scene now uses tagged uniform bricks, 16-bit voxel cells, and deduplicated mixed bricks. No extra storage-buffer binding is required. The CPU traversal reference remains independent. Freed tables and brick payloads are reused after edits/unloading; allocations retain high-water capacity. In a 32×32 one-voxel-thick floor fixture, cell storage falls from 66,560 to 1,152 bytes (directory/region headers excluded). This is a repetitive fixture, not a universal terrain compression ratio.

The paper's 32-frame screen-space TAA, lit/unlit sample lists, and biased spatial reservoir resampling are **not** implemented by these changes. They would require a separate estimator design to suit world-space persistence and the unfiltered simulation API. GPU/background AADF refinement and dynamic non-aligned voxel entities also remain separate work.

An Apple M4 Metal development-build stress run at 2560×1440 completed 79 edit/restore operations, a continuously moving point source, and five streaming moves, reaching 432 loaded pages. Median frame time was about 16.7 ms; edit-phase p99 was 25.3 ms and streaming-phase p99 was 30.3 ms. The run ended at 100% first-sample coverage. This measurement predates the additional emissive-voxel workload and the final redundant-rebuild/fairness fixes; repeat the validation below for the current revision. It is evidence for this machine and scene, not a guarantee for every GPU, terrain, or source count.

## Energy API

`VoxelEnergyField::estimate(voxel, normal)` returns an `EnergyEstimate` containing:

- `incident`: direct plus indirect RGB, infrared, and ionizing bands;
- `variance`: a running second-moment estimate, not a statistical confidence interval;
- `effective_samples`, per-query-face `age`, `sources_current`, and `transport_current`: metadata for deciding whether to use or defer a simulation update. `transport_current` additionally checks geometry/material revisions and whether all faces used by this query have been refreshed.

Unmeasured, geometry-invalidated, and buried faces return `None`. `sample` is a convenience method that returns only incident energy. These are unfiltered surface estimates, not volume samples or the smoothed display value.

`request_refresh(voxel)` queues a deduplicated, asynchronous refresh for a loaded probe when a simulation needs fresher data. It returns false for an unloaded voxel and never blocks or guarantees immediate completion. Borrowed/filtered display values do not enter this API's measurements, sample counts, or moments. Uncertainty used by the scheduler is a heuristic, not a calibrated confidence interval.

`EnergyEmitter` augments a `PointLight` or `DirectionalLight`, or works with a transform alone as a point source, for any combination of bands. `VoxelMaterialTransportTable::set` changes diffuse response and emission for a material index and triggers progressive resampling. Reflectance is constrained to [0, 1]; non-finite/negative energy inputs are rejected or sanitized. Emissive voxels both glow and illuminate other surfaces.

All bands currently use normalized transport units, **not watts, temperature, or dose**. A thermal simulation can integrate suitable fresh estimates with heat capacity, absorption, conduction, cooling, and phase-change rules, then turn a hot block into an emissive material. That simulation and physical unit calibration are not implemented here. Ionizing transport uses the same opaque diffuse model: it does not model penetration, energy-dependent attenuation, scattering physics, or biological dose.

## Supported model and limits

The implemented renderer is diffuse GI for opaque axis-aligned voxels, with Bevy point/directional sources and emissive voxel materials. Paths have three diffuse bounces and a 768-voxel segment limit. Transmission, specular GI, spot/area-light components other than voxel emitters, arbitrary non-voxel occluders, and participating media are not implemented. Neither all-world coverage nor a good frame rate guarantees instantaneous global lighting changes. Unbounded worlds and thousands of analytic lights require additional scale testing; the current analytic-source loop evaluates every source.

## Run and verify

### Preferred defaults

Normal `cargo run` now starts with **shared-corner direct sunlight and corner-blended indirect GI**, terrain textures enabled, irradiance debug disabled, and mesh ambient occlusion disabled. Corners mean naive unit-voxel face corners, not greedy-mesh vertices. Indirect transport itself still samples faces; only its display is reconstructed at corners. F7 remains the per-pixel sunlight comparison. GPU/adaptive transport, the 4 ms trace target, 75 ms indirect display response, and all-loaded-world coverage are unchanged; the rejected CPU throttle is not enabled.

A top-right HUD shows smoothed **FPS**, **Coverage** (exposed faces with at least one sample), and **Bootstrap** (progress toward eight samples per exposed face), using the existing all-loaded-world counters. Numbers refresh four times per second without extra rays or world scans. Neither percentage is convergence. Empty worlds show dashes until probes exist. Recurring terminal FPS/GI/rebuild stats are quiet during normal play; use `RUST_LOG=info VOXEL_GI_LOG_STATS=1 cargo run` to restore them. Warnings/errors remain enabled. Scripted validation retains terminal diagnostics and hides the HUD to preserve clean captures and blank-frame detection.

The HUD's **Add emitting block (E)** button places a warm emissive voxel against the face at the center of the camera view. Press **E** directly while playing, or aim, press **Esc** to release the cursor, then click the button; player movement/look pauses while using the unlocked cursor. Placement uses the same empty-adjacent-cell and GI invalidation path as right-click. These are real emissive voxels (material 4, warm RGB plus infrared, no ionizing emission), not spawned point lights. Left-click removes them normally; right-click still places grass. The four-layer texture atlas is unchanged.

### Correctness checks

Choose checks for the code being changed. The following is a **catalog, not a checklist to rerun after every edit**. Use focused unit tests during iteration; use the native test for the affected GPU path when needed. Documentation-only work needs diff review, not a build or live run. Broader suites belong at meaningful integration checkpoints.

```bash
cargo run
cargo test --all-targets
cargo test gpu_matches_cpu -- --ignored --nocapture
cargo test gpu_display_upload -- --ignored --nocapture
cargo test gpu_headless_renderer -- --ignored --nocapture
cargo test shared_sky_corner_benchmark -- --ignored --nocapture
cargo test sky_only_and_partial -- --ignored --nocapture
cargo test adaptive_idle_and_edit -- --ignored --nocapture
cargo test fused_sky_transport_benchmark -- --ignored --nocapture
cargo clippy --all-targets -- -D warnings
```

The explicit ignored tests require a native compute adapter. They compare GPU and CPU results for sky, analytic sources, all transport bands, and an enclosed emissive room, check incremental geometry uploads, validate the reported out-of-bounds offset, and exercise actual Bevy buffer/material preparation across cache generations without a window. Regular tests also cover dark tunnel edits, neighbor compatibility, display/measurement separation, temporal response, confidence-weighted corner continuity, adaptive scheduling, and global fairness.

Corner-sky regressions include a sunlit face center with two wall-adjacent shadowed corners, shared-corner agreement, loaded occluders beyond streaming gaps, conservative simulation behavior, and CPU/GPU packed-sample agreement. Failed topology patches are checked for zero partial publication. A system-level test verifies same-frame lighting catch-up after a rebuild even with a one-hour normal upload interval.

Cache tests additionally cover cross-batch deduplication, distinct normals sharing one visibility sample, multiple suns, negative coordinates, sun versus point-light invalidation, remote blocker insertion/removal, conservative shadow-bucket intersection, retrying unknown rays, and eviction during streaming. The native corner benchmark rotates duplicated/cold-shared/warm-shared modes with identical 1,024-face, four-path GI work, checks identical corner colors, and reports scheduled sky rays plus median GPU/worker times over 30 measured batches. Its duplicated mode uses the **new split kernel** with repeated corner tasks; it is not an old-revision or whole-frame FPS benchmark. Run it independently of other GPU tests for timing comparisons.

`adaptive_idle_and_edit` uses real GPU transport with a **virtual** 60 Hz scheduling clock over a simple static 256-face open-sky fixture. On Apple M4 it traced 23,552 paths versus 614,400 for continuous updates, leaving 577/600 scheduling ticks idle, with analytic-reference RMS RGB error 0.000513 in transport units. It then inserts a roof, verifies sky repair without new simulation samples, and verifies that indirect faces wake. This demonstrates work reduction and correctness in that fixture, **not** whole-game FPS, live edit latency, or an error guarantee for arbitrary scenes. Compression regressions exercise shared-block edits, material IDs 0/255, uniform bricks, recycling, and CPU/GPU traversal parity.

`fused_sky_transport_benchmark` rotates separate versus combined submissions for identical 512-face/four-path GI work plus 512 sky-only receivers, verifying identical transport values, moments, sample sequences and corner colors. An isolated Apple M4 Metal run on October 5 measured worker medians of 3.446 ms separate versus 1.806 ms combined over 40 rounds (GPU execution 0.147/0.126 ms). Both trace exactly 1,089 unique sky rays. This measures submission/readback overhead, **not** a whole-frame speedup. Regression tests also cover sparse settled windows and preserving remote histories during local source movement.

The early October 5 `regression-before.json` and `regression-continuous.json` reports are **invalid gameplay baselines**: their direct binary launches did not resolve the asset directory, so terrain was not rendered. The interrupted patched run had the same setup problem. Headless compute benchmarks are unaffected. Corrected runs explicitly set `BEVY_ASSET_ROOT` to the project directory and verify rendered-terrain screenshots; their artifacts are saved separately under `target/gi-validation/verified-before` and `verified-after`.

Corrected October 5 Apple M4/Metal development runs at 2560×1440, with both sky and toggle validation enabled, completed 79 edit/restores, five streaming moves, and 320 display-mode changes without render validation errors:

| Measurement | Before scheduler/history fixes | After fixes |
| --- | ---: | ---: |
| Static frame p99 | 30.8 ms | 25.0 ms |
| Editing frame p99 | 24.8 ms | 31.3 ms |
| Streaming frame p99 | 33.5 ms | 32.2 ms |
| Maximum sampled nearby measurement age | 2.43 s | 1.78 s |

Both had about 16.7 ms median frame time and ended at 100% first-sample coverage. During editing, mean retained history near 40 seconds rose from roughly 8 to 34 samples. This is a convergence/refresh improvement, **not an across-the-board frame-time win**: editing tails still need work. These are individual live runs, not repeated statistical trials; asynchronous streaming also produced slightly different loaded-page counts (437/433).

A separate 2 ms main-thread selection/merge-budget experiment reduced editing p99 to 21.8 ms but worsened maximum sampled nearby age to 14.79 s. It was rejected and is **not enabled or retained as a setting**. Its artifacts are in `target/gi-validation/rejected-cpu-budget`. Selection/merge timing telemetry is retained to guide optimization without starving transport. Shared-corner indirect transport and elimination of the face-center direct-simulation samples remain future experiments, not changes validated by these results.

### Current gameplay and visual procedure (report schema 2)

The normal performance workload must render the look we actually play with. It no longer cycles debug views, disables GI, or animates an unrelated point source while measuring digging. Build first, then run **one profile at a time** with the game foreground and no other benchmarks/compilation running:

```bash
cargo build
RUST_LOG=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_EXIT=1 cargo run
# Only when relevant to the change:
RUST_LOG=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_EXIT=1 VOXEL_GI_VALIDATE_PROFILE=sources cargo run
RUST_LOG=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_EXIT=1 VOXEL_GI_VALIDATE_PROFILE=compare cargo run
```

| Profile | Purpose | Workload |
| --- | --- | --- |
| `gameplay` (default, 72 s) | Frame pacing and lighting response in our preferred look | Startup 0–10 s, settled 10–25 s, edit/restore 25–45 s, five streaming moves 45–65 s, recovery 65–72 s; no mode cycling or moving source |
| `streaming` (35 s) | Turn-around/walk streaming regression | Settle 15 s at spawn, turn 180°, translate at 4.32 blocks/s for 15 s, recover 5 s; fixed preferred look. Scripted translation, not a character-controller movement test |
| `sources` (72 s) | Point-light and emissive-material updates, including IR/ionizing transport | Same timing and streaming path, but moving RGB/IR/ionizing point source plus emissive voxel replace repeated digging; remove sources for recovery |
| `compare` (29 s) | Appearance inspection, **not** performance evidence | Explicit corner sun, per-pixel sun, flat voxel, corner/flat irradiance, indirect-disabled, and restored-default captures |
| `stress` (72 s, targeted only) | Regression check for the fixed buffer-generation/mode-switch failure, **not** routine performance work | Editing, sources, streaming, and F4–F7 combinations eight times per second, followed by recovery |

The known toggle/buffer overrun is fixed and covered by native regression tests. Keep `VOXEL_GI_VALIDATE_PROFILE=stress` available for changes to buffer lifetime, cache generations, material extraction, display-mode handling, renderer dependency upgrades, or a returning symptom. Prefer a targeted headless regression test when sufficient. Do **not** routinely spend a live stress run on sampling, scheduling, lighting tuning, or documentation changes.

Each development-only run creates a **unique** `target/gi-validation/<profile>-<timestamp>-<pid>/` with screenshots, `status.json`, and (on completion) `report.json`. `running` means incomplete, `failed` means invalid setup/capture, and `completed` means the workload finished—not that lighting quality or performance passed. Old top-level reports and the legacy before/after numbers above are historical; schema-2 measurements must get a new baseline. Keep the launch log (including adapter/backend and any render errors) and record the commit/dirty diff with each candidate comparison.

Keep the game foreground throughout; screensavers can invalidate captures even without locking. Missing assets, loading timeouts, unfocused/hidden windows, uniform/black captures, and missing capture callbacks reject the run with a nonzero exit. Inspect the images and reject **any** render validation error in the log too: the automatic checks are setup guards, not scene-correctness proofs. Direct launches require `BEVY_ASSET_ROOT` set to the project directory; `cargo run` supplies it. Do not change screen/security settings as part of the harness.

Player input and manual F4–F7 switches are ignored only during validation. Scripted temporary blocks/materials/sources are restored; no terrain files are modified. All profiles return to the preferred corner settings. The player stays at the final scripted position when automatic exit is omitted. Old `VOXEL_GI_VALIDATE_SKY`/`TOGGLES` recipes and combinations with `SMOKE_EDIT`/`SMOKE_MOVE` now fail explicitly instead of silently changing the workload.

### What we judge and how to promote a change

- **Frame pacing:** compare per-phase p50/p95/p99/max and counts above 16.7/33.3/50 ms, not average FPS alone. Keep startup and recovery separate. Our working target is ~60 FPS median on the test Mac without large edit/streaming spikes; it is not a cross-hardware guarantee. Capture/readback frames and a 250 ms cooldown are counted as excluded, so capture overhead does not dominate gameplay tails.
- **Lighting responsiveness:** inspect nearby and distant face-age distributions, stale/unmeasured observations, unknown-sky observations, and first-fresh-measurement latency after edits. Superseded, untargeted, and still-pending edits are reported alongside successful latencies. Do not accept better FPS obtained by starving the field. First measurement, first-sample coverage, and retained history are **not convergence**.
- **Stability and look:** inspect textured corner sunlight, hill/wall contact shadows, chunk seams, and irradiance-only views (F6 includes cached direct sun when enabled). Field reports include nearby indirect display-step distributions and positive jumps from dark values; these are diagnostic indicators, not pixel-space noise, an error metric, or proof that flicker is absent. Check a fixed-camera sequence while digging a dark tunnel, placing/removing its roof, and stopping edits: no bright startup flash, false sky-exposed shadows, leaking across disconnected corners, or failure to settle. Still images alone cannot pass temporal stability.
- **Sources and simulations:** run `sources` and the CPU/GPU parity tests. Verify source removal recovers without stale glow and offscreen receivers continue updating. Display interpolation/filtering must not enter RGB/IR/ionizing simulation measurements; missing/freshness metadata must remain conservative. This does not validate physical heat/dose calibration or planet/asteroid coordinate warps.
- **Robustness:** when the affected path warrants it, use the native cache-generation/buffer-bounds test; add `stress` for relevant integration risk or reproduction. Require no GPU validation errors, lost updates, or crashed runs. This is retained regression coverage, not an ongoing toggle-stability project.

For performance iteration, start with a relevant isolated benchmark or one baseline/candidate **gameplay** pair on the same machine, power mode, viewport/present mode, terrain, settings, and otherwise idle system. Compare phase tails **together with** age, edit response, and temporal appearance; preserve failures as well as successes. Repeat alternating pairs when differences are within run-to-run variation or before claiming a small gain—not as a mandatory tax on every change. Promote only when the intended benefit is credible without a meaningful stability/response regression; an intentional tradeoff needs explicit agreement. Run source/visual/robustness checks only when their affected behavior warrants them. Microbenchmarks explain where time went; they do not establish a whole-game speedup.

Field statistics use a bounded systematic sample of at most 1,024 voxels every 250 ms. They count **observations**, not unique faces or independent trials; empty distributions omit percentiles rather than pretending unknown is zero. Display-step statistics only compare matching nearby faces in consecutive observations. Edit response tracks newly exposed adjacent faces (or restored voxel faces), requires a new geometry revision and fresh face timestamps, and is censored when another edit supersedes the watch. It is not a whole-world service guarantee. Screenshot/extraction cooldown exclusion and these diagnostics change instrumentation overhead, another reason not to compare against legacy reports.

Use `VOXEL_GI_BACKEND=cpu cargo run` for the bounded CPU reference fallback. It shares the model, not a throughput guarantee. Logs include coverage/history, sky-ray and batch counters, main-thread selection/merge time, and worker preparation/wait/decode timings. GPU execution and worker wait overlap; never add them. Those latest-batch values are not frame percentiles. The report records backend, viewport/present mode, terrain, and scheduling settings so mismatched runs can be rejected.

The smaller development-only `VOXEL_GI_SMOKE_EDIT=1` check removes one block after bootstrap settles. `VOXEL_GI_SMOKE_MOVE=1` moves the player 48 voxels to exercise streaming. Do not combine these with the full validation workload when comparing performance.

### Commit discipline

`aca688b` checkpoints the integrated shared-corner GI, transport fixes, preferred defaults, and schema-2 harness, verified by 85 regular tests, seven native tests, and the four live profiles. It is a regression reference, not a claim that all quality/performance work is finished.

Its October 5 schema-2 `gameplay` run on Apple M4/Metal, 2560×1440, FIFO, development build and default settings completed 80 edits/restores and five streaming moves. Editing frame p50/p99 were 16.79/28.92 ms; first-fresh-receiver latency p50/p95 was 154.61/183.44 ms (maximum 420.05 ms), with zero pending or superseded watches at completion. This is a single harness-verification run, not evidence of a comparative speedup or visual convergence. Full local artifacts: `target/gi-validation/gameplay-1791217759852541000-95006/`.

From here, commit each coherent, verified change separately and checkpoint before a new performance/architecture experiment. Keep commits buildable, include the verification actually performed and known limitations, and record benchmark results against the exact revision plus any dirty diff. Preserve a compact measurement summary in commit history or tracked documentation; screenshots/logs can remain under `target/`. Do not accumulate multiple independent performance experiments into one uncommitted state. The repository's [agent workflow](AGENTS.md) makes this the default for future work.

## Controls

- `WASD`: move
- `Space`: jump
- `Left Ctrl`: crouch
- `Shift`: sprint
- Mouse: look
- Left click: remove a voxel and invalidate its transport page
- Right click: place a voxel and invalidate its transport page
- `E` / HUD **Add emitting block**: place an emissive voxel against the aimed-at face
- `F4`: toggle voxel GI for A/B comparison
- `F5`: toggle flat per-voxel faces / blended virtual voxel corners
- `F6`: isolate cached irradiance from terrain textures and ordinary PBR lighting
- `F7`: switch direct sunlight between the voxel cache (default) and Bevy per-pixel lighting; F5 selects flat faces or blended voxel corners for the cached mode
- `F3`: toggle mesh debug overlay
- `Esc`: toggle inspector and release/capture cursor

## Configuration

Terrain settings remain hot-reloadable in [`assets/config/default.prototype.ron`](assets/config/default.prototype.ron). Mesh ambient occlusion is disabled by default so it does not bake a second approximation into the ray-traced indirect result.

For a sky-light A/B comparison, leave F5 on its default corner blending and press F7 to switch from cached corner sunlight to Bevy per-pixel sunlight; press it again to restore the default. Diffuse sky/environment lighting remains in the voxel cache in either mode. Point lights and other materials retain their normal Bevy rendering. The cached sun uses voxel-ray visibility and a diffuse Lambertian response, with Bevy's camera exposure; it intentionally omits the sun's view-dependent specular lobe. Thus this compares the full diffuse voxel-lit look, not only two shadow filters. Cached direct sun can lag edits/source changes just like the GI field. F4 can disable indirect GI without disabling the selected direct-sun mode. Switching modes changes neither transport sources nor simulation estimates and does not restart sampling.

Cached sun samples the naive unit-face corners, not greedy-quad vertices or averaged face centers. F5 bilinearly interpolates these samples, or displays their constant face average. Four compact RGBE colors replace the previous single float4 per face, keeping the GPU display-probe stride at 208 bytes. Only display color is quantized; gameplay measurements remain f32. Missing probes or unknown corner rays use Bevy's paired direct-light/shadow fallback until measurements are available, not black cache entries. A confirmed occlusion stays black. This display fallback does not change the conservative treatment of unknown regions in simulation transport.

All faces sharing a corner use the same ray origin, displaced toward the sun rather than along an individual face normal. Exact grid-plane ties consistently select the positive grid cell on CPU and GPU. Shared sky tasks reuse the existing GPU scratch buffers in a separate compute entry point; no permanent GPU binding or display-buffer expansion is needed. Warm batches also omit sky-result readback (96 bytes per face versus the previous 112); cold batches add 32 bytes per pending shared corner. The tradeoff is a sparse CPU corner cache and bucket index, whose entry count follows retained loaded receivers; allocations may retain their high-water capacity. Invalidation is conservative at 16³-region/bucket granularity, and dirty corners refresh when their receivers are scheduled, not in an instantaneous whole-world solve.

The material uses scoped naga-oil function overrides for Bevy's directional-light and shadow evaluation; no Bevy shader source is copied or modified. Ordinary materials are unaffected. Directional lights should represent global sky sources here; per-camera light-layer filtering and directional light textures are not represented in the shared world-space cache.

`VoxelGiSettings` can be inserted before `VoxelGiPlugin` or adjusted as a resource at runtime. Defaults target a 4 ms trace dispatch, up to 8,192 probes per dispatch, and a 50 ms display-upload interval. These are scheduling controls, not world-coverage limits or hard real-time deadlines. Smaller budgets reduce throughput; measure estimate age as well as FPS.

Resting behavior is controlled by `stable_min_samples` (default 64), `stable_relative_error` (default 0.12), and `stable_audit_interval` (default 2 seconds). Relative error is a scheduling heuristic, not a statistical confidence bound. Noisy faces continue refining. Deterministic sky repairs bypass the normal upload interval; they do not use the indirect display filter or an added debounce.

For a same-revision scheduling A/B baseline, use `VOXEL_GI_CONTINUOUS=1 cargo run` (also works with validation flags), or set `adaptive_updates = false`. This keeps the same ray kernels, compressed geometry, and sky cache, but continuously schedules transport faces. It is not an old-engine benchmark.

`display_response` defaults to 75 ms and controls only the presentation filter. An initial four-ray estimate has bounded influence; subsequent measurements approach persistent changes without permanent brightness clamping. This time constant is not a settling-time guarantee: cadence depends on available tracing budget. Reservoir/path reuse, GPU-resident accumulation, and wavefront tracing remain future work; neither history smoothing nor adaptive scheduling implements ReSTIR.

## License

Licensed under `CC0-1.0 OR MIT OR Apache-2.0`. See [LICENSE-CC0-1.0.txt](LICENSE-CC0-1.0.txt), [LICENSE-MIT.txt](LICENSE-MIT.txt), and [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt).
