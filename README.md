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
- New faces start with conservative dark indirect light or a low-confidence estimate from measured pre-edit neighbors connected through loaded air. Placement can also inherit the same-facing surface it just covered. They never initialize to unoccluded sky. A separate presentation history filters measured indirect RGB; confidence-weighted corner reconstruction favors measured faces. Deterministic direct-sun visibility responds without temporal filtering, and sun seed reuse is restricted to identical geometric corners.
- Bevy PBR retains ordinary visible direct/specular lighting. The extension adds cached indirect RGB, standalone/custom-emitter RGB, and voxel self-emission without counting ordinary direct light twice. Simulation queries receive direct plus indirect energy before display smoothing.
- Edits invalidate local histories, retain existing display values, and reject incompatible in-flight geometry results. A lazy global transport revision also wakes distant indirect receivers without clearing their display or scanning the world during invalidation. Nearby work is prioritized; at least one quarter of transport candidate slots are reserved for an uninterrupted all-loaded-world sweep. Moving sources shorten histories instead of repeatedly clearing the world.
- Adaptive scheduling ranks bounded rotating windows by age, source revision, and per-band relative uncertainty. RGB cannot mask uncertain infrared/ionizing transport. Edit/bootstrap/simulation requests take priority, while the reserved world sweep still prevents starvation. Selection has a per-face work mask distinct from the voxel's exposed-face topology, so an uncertain face does not force its settled neighbors to retrace.
- Scene building, cache construction, dispatch, and readback run asynchronously. The GPU tracer adapts its batch size toward a configurable time budget. A CPU mirror provides gameplay estimates; this is not a wholly GPU-resident authoritative field.
- Sparse display uploads preserve stable slots during edits. Topology patches preflight all allocations and hash insertions before publishing anything. Growing the cache keeps updating the previous display, and the replacement merges newer measurements in its installation frame rather than briefly showing the old worker snapshot. Invalidation preserves display blending weights as well as values; replacement weights change with the filtered estimate. No additional edit debounce is used.
- Display-cache rebuilds publish fresh buffer identities and generation-owned patch queues. In-flight render frames retain their old generation; paired hash/probe uploads are bounds-checked and deferred updates are retained. Sparse geometry edits publish removed/added addresses and changed face values atomically, bypassing the ordinary value-upload timer. Shader bounds use the generation's allocated capacity so old extracted materials already accept new slots.

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

Newly exposed faces prefer measured, same-facing neighbors for their indirect display prior. If none exist, they borrow from measured surfaces bordering the same air cell or a face-adjacent known-empty cell. This lets the sides as well as the bottom of a freshly dug cavity inherit the old surface's indirect lighting. Placement may inherit the equally oriented face it just covered: that old exposed face establishes that the new block occupied air. Other connections cannot cross solid/unknown cells. These approximate, low-confidence presentation priors never propagate synthetic samples or rotate direct-sun values between face orientations. Donor age reduces weight continuously instead of abruptly rejecting settled faces after two seconds.

Mesh publication can precede that receiver update. During this handoff, an absent indirect face borrows nearby equally oriented display history, including a just-covered surface; otherwise it uses surfaces bordering its visible exterior air cell or the removed voxel's old light. This temporary prior participates in the usual corner reconstruction. It does not copy emission or cached sun, write anything back, or become measurement evidence. Once the new probe is published, the seeded display and existing temporal resolve take over. Established faces take the unchanged reconstruction path; no extra rays or debounce are introduced. This addresses initial missing-data flashes, not all sampling noise or arbitrary multi-block edits before useful history exists.

## Change-driven scheduling and NAADF additions

Direct sunlight and GI use the **same voxel ray-traversal geometry**, but have different sampling/update policies. Cached direct sun is deterministic visibility at shared corners; diffuse environment light and bounces are stochastic multi-band path transport. Bevy's per-pixel sunlight comparison mode still uses its own raster shadowing. F7 changes presentation, not the transport model.

Sky repairs have an independent urgent queue and loaded-world refresh cursor, but **share the worker submission with due indirect work**. Duplicate receivers are merged; additional sky-only receivers have a zero transport mask, so they generate no GI paths and cannot modify simulation sample counts, moments, or timestamps. Sky-only submissions remain possible when no indirect work is due, and fully cached sky-only batches submit no GPU commands or rebuild the emissive distribution. Changed sky values bypass the normal display upload interval. This avoids spending two out of three worker turns on sky while indirect lighting waits. Geometry/sun generations prevent obsolete sky values from being published in either kind of batch.

Indirect faces become eligible for reduced-frequency updates only after at least 64 samples by default, four stable batches, low relative error and drift in **every energy band**, and a sufficiently settled display value. They remain eligible for background verification every two seconds; eligibility is not a service-time guarantee. Explicit simulation refresh requests override resting state. Geometry, streaming, material, and source changes wake faces through a revision check. The mechanism follows NAADF Section 4.2's selective-sample-generation principle, adapted to world-space faces rather than TAA/pixel history; it is not the paper's resampling estimator.

Indirect invalidation is intentionally conservative and **currently global**, unlike the spatially targeted direct-sky cache. There is no claim of complete per-path dependency tracking. This avoids permanently stale distant light after an edit, including insertion into previously empty space, but frequent edits/moving sources can keep indirect work active. The rotating audit remains necessary: a stable low-sample estimate is not proof that a rare light contribution does not exist. Simulation freshness includes a transport revision and each requested face's own age; a fresh top face cannot make an old side-face measurement appear fresh.

Global wake-up is separate from evidence destruction: remote histories retain up to 64 samples of temporal weight, rather than falling back to four on every streamed air page or local light movement. Local geometry changes retain at most 16 measured samples on unchanged faces, mark them stale, reject pre-edit worker results, and prioritize fresh paths. Newly exposed faces and changed materials still start with zero measurements; opening one face no longer clears every other face of that voxel. Global source changes/nearby moving lights still use the short four-sample response. Retained measurements can therefore keep coverage/bootstrap high after edits; those percentages do not certify current lighting. Inspect `transport_current`, `sources_current`, age and variance for simulations. Candidate scanning is independent of emitted work: settled probes do not consume ray slots, and underfilled cheap batches cannot collapse the scheduling budget. Searches remain bounded to at most 8,192 candidates per selection, plus the bounded priority windows.

Discrete voxel-emitter additions/removals (including loaded/unloaded emitters) and transport material-table changes are an exception: each receiver discards obsolete transport statistics lazily on its next dispatch, while preserving display history and sample-sequence progression. Old in-flight source results cannot republish obsolete energy. This prevents an extinguished emitter surviving as 16 or 64 old lit samples diluted over many batches. The existing display resolve still smooths replacement measurements; ordinary digging and moving analytic lights retain their bounded-history policies. Source reset is conservative across loaded receivers because path dependencies are not tracked. The focused `emitter_removal_replaces_energy_in_one_batch_and_fades_display` native test checks RGB/IR/ionizing removal in a sealed room and CPU/GPU agreement. Its virtual 50 ms batch cadence is not a gameplay latency measurement.

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
| `response` (30 s) | Focused cavity/emitter temporal inspection | Fixed-camera synthetic enclosed room; delete one floor block at 15 s, add a warm emitter at 20 s, remove it at 25 s; 14 captures and sampled receiver history in `response.json`; no streaming moves or mode switches |
| `edits` (30 s) | Initial geometry/lighting handoff | Same enclosed room: dig/restore its floor, place/remove an ordinary block, then inspect a roof excavation from above; 18 captures at 50/150/400 ms offsets and receiver history; no source or mode changes |
| `streaming` (35 s) | Turn-around/walk streaming regression | Settle 15 s at spawn, turn 180°, translate at 4.32 blocks/s for 15 s, recover 5 s; fixed preferred look. Scripted translation, not a character-controller movement test |
| `sources` (72 s) | Point-light and emissive-material updates, including IR/ionizing transport | Same timing and streaming path, but moving RGB/IR/ionizing point source plus emissive voxel replace repeated digging; remove sources for recovery |
| `compare` (29 s) | Appearance inspection, **not** performance evidence | Explicit corner sun, per-pixel sun, flat voxel, corner/flat irradiance, indirect-disabled, and restored-default captures |
| `stress` (72 s, targeted only) | Regression check for the fixed buffer-generation/mode-switch failure, **not** routine performance work | Editing, sources, streaming, and F4–F7 combinations eight times per second, followed by recovery |

The known toggle/buffer overrun is fixed and covered by native regression tests. Keep `VOXEL_GI_VALIDATE_PROFILE=stress` available for changes to buffer lifetime, cache generations, material extraction, display-mode handling, renderer dependency upgrades, or a returning symptom. Prefer a targeted headless regression test when sufficient. Do **not** routinely spend a live stress run on sampling, scheduling, lighting tuning, or documentation changes.

Each development-only run creates a **unique** `target/gi-validation/<profile>-<timestamp>-<pid>/` with screenshots, `status.json`, and (on completion) `report.json`. `running` means incomplete, `failed` means invalid setup/capture, and `completed` means the workload finished—not that lighting quality or performance passed. Schema 3 explicitly identifies capture mode and timing domain. Keep the launch log (including adapter/backend and any render errors) and record the commit/dirty diff with each candidate comparison.

For unattended visual/lighting work, use Bevy's offscreen screenshot API:

```sh
RUST_LOG=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 VOXEL_GI_VALIDATE_PROFILE=streaming VOXEL_GI_VALIDATE_EXIT=1 cargo run
```

This runs the real terrain camera into a 2560×1440 GPU image, captured using `Screenshot::image`, with no OS window, Winit runner, inspector, cursor capture, or swapchain. It therefore does not depend on an unlocked/foreground desktop or its drawable surface. All existing profiles can use this mode. The app uses a 60 Hz schedule runner; reports label its app-cadence timings as **offscreen**, not actual windowed FPS. Compare like-for-like offscreen runs for diagnostics, and retain a separate windowed baseline for interactive frame pacing. This does not prevent system sleep or guarantee GPU availability during every OS power/security transition.

For windowed FPS checks, omit `VOXEL_GI_VALIDATE_HEADLESS` and keep the game foreground; screensavers can invalidate window captures even without locking. Missing assets, loading timeouts, uniform/black captures, and missing capture callbacks reject either mode with a nonzero exit. Offscreen captures additionally require the expected dimensions and an active camera targeting the captured image; window captures still require a focused, visible, nonzero-size window. Inspect the images and reject **any** render validation error in the log too: the automatic checks are setup guards, not scene-correctness proofs. Direct launches require `BEVY_ASSET_ROOT` set to the project directory; `cargo run` supplies it. Do not change screen/security settings as part of the harness.

The initial October 5 offscreen harness check (Apple M4/Metal, development build, preferred corner settings) completed `streaming` with three inspected 2560×1440 terrain captures, no render validation errors, and 100% first-sample coverage. Artifacts: `target/gi-validation/streaming-1791225567702474000-13792/`. This verifies windowless capture/workload integrity, not a windowed performance improvement or temporal-quality pass.

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

The old `VOXEL_GI_SMOKE_EDIT` and `VOXEL_GI_SMOKE_MOVE` flags have been removed.
Use `edits` for focused edit presentation and `streaming` for the turn/walk
workload; keep `gameplay` for coupled performance comparisons.

### Commit discipline

`aca688b` checkpoints the integrated shared-corner GI, transport fixes, preferred defaults, and schema-2 harness, verified by 85 regular tests, seven native tests, and the four live profiles. It is a regression reference, not a claim that all quality/performance work is finished.

Its October 5 schema-2 `gameplay` run on Apple M4/Metal, 2560×1440, FIFO, development build and default settings completed 80 edits/restores and five streaming moves. Editing frame p50/p99 were 16.79/28.92 ms; first-fresh-receiver latency p50/p95 was 154.61/183.44 ms (maximum 420.05 ms), with zero pending or superseded watches at completion. This is a single harness-verification run, not evidence of a comparative speedup or visual convergence. Full local artifacts: `target/gi-validation/gameplay-1791217759852541000-95006/`.

From here, commit each coherent, verified change separately and checkpoint before a new performance/architecture experiment. Keep commits buildable, include the verification actually performed and known limitations, and record benchmark results against the exact revision plus any dirty diff. Preserve a compact measurement summary in commit history or tracked documentation; screenshots/logs can remain under `target/`. Do not accumulate multiple independent performance experiments into one uncommitted state. The repository's [agent workflow](AGENTS.md) makes this the default for future work.

## Architecture and cost audit (October 5, 2026)

The separation between **transport measurements**, **scheduling/publication**, and
**display history** is sound. Keep it: simulations need unsmoothed measurements
and explicit freshness, while rendering needs stable priors during edits. This
is still a prototype integration, not a published storage-independent library.

| Boundary | Responsibility / constraint |
| --- | --- |
| `voxel_gi/bvw.rs` | BVW events and generator snapshots become immutable 32³ pages. This is the host-specific storage adapter. |
| `world`, `naadf`, `gpu_scene`, `emission` | Immutable acceleration, persistent GPU geometry, and exact emissive-source discovery. No player or screen-space dependencies. |
| `tracer`, `gpu`, `sampling`, `sky_cache` | Multi-band face transport and shared-corner direct sun; bounded asynchronous work, with an independent CPU reference. |
| `mod`, `field`, `scheduling` | Revision-checked publication, fair all-loaded-world scheduling, and simulation queries. Readers can order themselves after `VoxelGiSystems`. |
| `history`, `reconstruction`, `render` | Display-only priors/filtering, corner reconstruction, and generation-owned GPU presentation. Never write borrowed values into simulation statistics. |
| `diagnostics`, `validation` | Explicit demo plugin: HUD, keyboard comparisons, logging, and scripted workloads. Not installed by the transport plugin itself. |

Implemented cleanup, in independently verified local commits:

- Removed the unused face-center display-sun RGB arrays (72 bytes per probe,
  approximately 22.7 MiB at the baseline's 330,838 probes). **Direct simulation
  energy and its rays remain.** Sparse display encoding no longer allocates a
  temporary vector for every patched probe.
- Retained probe allocations through page publication and stopped uploading
  identical display records. Geometry invalidation still wakes receivers; it
  does not need to upload their unchanged retained display values.
- Added material-presence masks to immutable regions. A registered but absent
  emitting material no longer causes dense scans of every mixed region. Source
  ordering, live material changes, and cross-region occlusion remain exact.
- Cached static material/light uploads and compute bind groups by buffer identity.
  Growth invalidates bindings, including idle passes, so old allocations are not
  retained indefinitely. Removed redundant NAADF sweep copies and per-brick
  occupancy scratch allocations.
- The **demo sun** skips cascaded shadow rendering in corner mode when there are
  no non-voxel mesh consumers. F7, missing material state, or other meshes restore
  raster shadows before light culling. Other lights are not modified.
- Normal play no longer scans every probe for terminal-only statistics or runs
  dormant validation systems. Removed the superseded smoke-edit/move systems;
  use the named profiles instead. CPU fallback threads are created only if used.
  Prototype albedos now live in terrain setup rather than generic table defaults.

Retained deliberately: all five transport bands, the CPU reference/fallback,
F7 comparison, simulation freshness metadata, and the bounded global audit.
These have consumers or correctness roles; deleting them would change scope or
quality. All direct dependencies have code or feature consumers (`naga_oil`'s
override feature is required). Bevy's broad default feature set remains; trimming
compiled features is not evidence of a runtime frame-rate improvement.

### Measured checkpoint

One preliminary **offscreen** gameplay pair on Apple M4/Metal, development build,
2560×1440: clean `5acbe3a` baseline versus clean `f5bdd00` candidate. Both used
textured corner sun/indirect reconstruction, baked AO off, adaptive 4 ms trace
target, 8,192-probe cap, four paths per selected face, 75 ms display response,
and the same validation instrumentation. No concurrent compilation/benchmark was
run during either workload. These are app-cadence timings, **not windowed FPS**.

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Settled frame p50 / p99 | 20.57 / 35.44 ms | 21.46 / 30.01 ms |
| Editing frame p50 / p99 | 18.53 / 39.67 ms | 18.88 / 38.23 ms |
| Streaming frame p50 / p99 | 19.98 / 39.10 ms | 19.95 / 36.31 ms |
| Recovery frame p50 / p99 | 21.40 / 30.33 ms | 20.92 / 32.94 ms |
| First fresh edit measurement p50 / p95 | 162.41 / 189.33 ms | 151.60 / 189.66 ms |
| Editing nearby measurement age p95 | 139.46 ms | 131.20 ms |
| Logged page publication median / p95 | 1.0 / 5.1 ms | 0.7 / 4.6 ms |

Both completed 80 edits/restores and five moves, with five valid nonblank captures,
clean render logs, 100% first-sample coverage and no pending/superseded edit watches.
Loaded pages differed (410 / 403); probe counts were 330,838 / 330,682. Editing
indirect display-step p95 was 0.01996 / 0.01960 in normalized units; neither this
nor still captures proves temporal convergence. The settled images retain the
preferred look. **Results are mixed, not a demonstrated across-the-board speedup.**
Repeat paired measurements before quoting small percentage gains. Diagnostics
remain enabled in validation, so this pair does not measure their removal from
ordinary gameplay.

The large hitches are **not solved**: maximum streaming frames were 188 / 196 ms,
and startup maxima were 857 / 705 ms. Whole-app profiling must distinguish GI
publication from mesh/collider installation, renderer preparation and pipeline
startup; a faster tracer alone cannot establish a stable gameplay frame budget.

Local artifacts: `target/gi-validation/gameplay-1791245232756073000-7623/` and
`target/gi-validation/gameplay-1791246484602972000-26663/`; corresponding logs are
`target/audit-baseline-gameplay.log` and `target/audit-candidate-gameplay.log`.
The later idle-binding-retirement and redundant-NAADF-copy commits were verified
separately and are **not included** in those timings. The isolated static-dispatch
pair and its limitations are recorded in commit `8b639a1`.

Strict all-target Clippy, 103 regular tests, native CPU/GPU source/geometry parity,
static-upload/binding mutation/growth, and the targeted headless renderer-generation
regression passed at the integration checkpoint. The optional broad toggle-stress
profile was not rerun. These checks do not certify all hardware or arbitrarily
large worlds.

Final `b1b7c07` code also completed the focused offscreen `edits` profile with
18 valid captures and clean render logs (only this documentation was dirty).
Artifacts: `target/gi-validation/edits-1791246801335940000-32069/` and
`target/audit-final-edits.log`. Inspected early underground digging/placement and
surface-edit frames; no new black-face or raster-shadow-mode flash was apparent
in those captures. The first scheduled 50 ms dig capture still preceded mesh
publication; the 150 ms capture contained the lit cavity. This is a sampled
regression check, not proof of flicker-free intervening frames or a latency bound.

### Waste and throughput follow-up (October 5, 2026)

The next checkpoints preserve four paths per selected face, all five transport
bands, path depth, eight-sample bootstrap, resting thresholds, preferred corner
presentation, and the existing temporal filter. No new debounce was added.

- `d4b9cca`: opt-in 250 ms coverage/bootstrap timeline, including published probe
  counts and pending scene/page work, plus selection/merge/worker distributions.
  Empty or partially published worlds must not count as completed initial terrain.
- `2c91093` / `63275d0`: reuse dispatch/candidate/dirty-set allocations; avoid
  duplicate sky receiver copies and the second lookup when merging results;
  use the engine's fast integer-coordinate probe index. Full probe snapshots
  still cross the worker boundary; compact selected-face jobs remain worthwhile.
- `ec93575`: publish cheap receiver pages within a **2 ms budget**, rather than
  unconditionally stopping after one page. At most four pages per update bound
  downstream display patching. Each page is atomic and can exceed the budget;
  `scene_publish_budget = Duration::ZERO` retains the old one-page comparison.
- `85bc560` / `83dae62`: benchmark and enable four cooperative GPU lanes per face.
  They trace the same independent paths, not reused or borrowed samples. The
  serial entry point remains a benchmark reference; normal play creates only
  the chosen transport pipeline and the separate direct-sky pipeline.
- `8473f80`: display dirty ledgers hold only slot IDs. Removed a redundant
  208-byte lighting value per dirty probe (and redundant hash-entry values);
  generation-owned byte mirrors already contain everything uploaded. Latest
  writes, atomic publication and failed-upload retry semantics are unchanged.

#### Evidence and limits

All following gameplay runs are development builds on Apple M4/Metal,
2560×1440 **offscreen**, preferred defaults, with validation diagnostics and
`RUST_LOG=warn`. No builds or other GPU workloads ran during measurement.
These are app-cadence observations, not interactive/windowed FPS. Each revision
had one gameplay run; they are preliminary comparisons, not repeated statistical
estimates. The trace budget stayed at 4 ms and the dispatch limit at 8,192.

The first quiescent initial terrain cohort with 100% coverage **and** bootstrap,
no dirty/publishing pages, and no scene task was observed at **6.14 s** on
`d4b9cca` (219,632 probes), **5.13 s** after allocation cleanup on `2c91093`
(227,675), **3.92 s** after bounded publication on `ec93575` (219,538), and
**4.02 s** with cooperative paths on `83dae62` (225,555), and **3.69 s** with compact
display ledgers on `8473f80` (219,758). Sampling is every 250 ms; late-arriving
terrain can create another cohort. The final comparison is about a **1.7× initial
completion improvement**, not a demonstrated 2–3× gain or visual convergence.

In the first allocation comparison, streaming selection p50/p99 fell from
3.52/8.92 to 1.64/2.50 ms; merge p50/p99 from 2.61/7.58 to 1.48/3.09 ms.
Whole-frame medians did not improve: streaming p50/p99 changed from 17.71/66.74
to 20.98/38.02 ms. Do not turn the stage-level win into an overall FPS claim.

The cooperative kernel was compared at fixed work on one native device/queue,
alternating 30 warm pairs and repeating the experiment. Open-floor GPU medians
were 0.116–0.117 ms serial versus 0.061–0.062 ms cooperative (1,024 faces × four
paths). Nine enclosed emissive rooms were 2.034–2.046 versus 1.838–1.879 ms
(8,100 faces × four paths). Worker medians were essentially unchanged/mixed:
submission/readback and CPU overhead limit this small-kernel win. This is an
execution-layout experiment motivated by [vxray's occupancy/wavefront
discussion](https://nelari.us/post/voxel-ray-tracing/), **not a wavefront renderer**,
ReSTIR, or a new NAADF algorithm. It uses a workgroup reduction, not extra samples.

At `83dae62`, editing first-fresh measurement p50/p95 was 145.86/169.53 ms;
editing nearby-age p95 was 140.05 ms and far-age p95 2,049.10 ms. The startup
maximum was still **636 ms**, and recovery frame p99 was 41.55 ms. Faster kernels
do not establish hitch-free gameplay. The sampled settled/streaming images
retain the preferred look; still captures do not prove flicker-free motion.

Final `8473f80` versus the instrumented `d4b9cca` baseline:

| Offscreen metric | Baseline | Final |
| --- | ---: | ---: |
| Streaming selection p50 / p99 | 3.52 / 8.92 ms | 1.65 / 2.73 ms |
| Streaming merge p50 / p99 | 2.61 / 7.58 ms | 1.53 / 3.38 ms |
| Settled frame p50 / p99 | 17.62 / 37.64 ms | 21.35 / 34.48 ms |
| Editing frame p50 / p99 | 17.24 / 43.83 ms | 18.93 / 36.42 ms |
| Streaming frame p50 / p99 | 17.71 / 66.74 ms | 20.87 / 38.65 ms |
| First-fresh edit measurement p50 / p95 | 155.78 / 181.93 ms | 148.64 / 168.74 ms |

The final run completed 80 edits and five moves, with five valid nonblank captures,
clean render logs, and no pending/superseded edit watches. Editing nearby-age p95
was 138.39 ms, far-age p95 1,989.24 ms, and indirect display-step p95 0.02358.
Streaming maximum was still **146 ms**; startup maximum **635 ms**. Frame tails
improved in this pair but medians worsened: no across-the-board FPS claim. The
smaller dirty ledger is a structural memory win, not evidence that it alone
caused the final run's timing differences.
The baseline recorded 79 edit latencies and one superseded watch, versus all 80
latencies in the final run; neither set measures visual settling time.

Local runs under `target/gi-validation/`, in revision order:

- `d4b9cca`: `gameplay-1791248100607155000-53907/`
- `2c91093`: `gameplay-1791248294774532000-57609/`
- `ec93575`: `gameplay-1791248442489438000-59890/`
- `83dae62`: `gameplay-1791248981380354000-68305/`
- `8473f80`: `gameplay-1791249153187694000-70880/`

These binaries were built from their clean code checkpoints; later source edits
during the runs did not affect the executing binary or rendering assets. Logs:
`target/perf-progress-{baseline,reuse,publication,cooperative,final}.log`.
Native fixed-work results: `target/perf-cooperative-{shared-device,repeat}.log`.
Kernel parity also covers 1/7/64 samples, incomplete workgroups, CPU reference,
geometry edits, partial faces, sky-only work and removal of emitting sources.
The integrated ledger change passed 107 regular tests, strict all-target Clippy,
and targeted native upload-retry/renderer-generation checks. Broad toggle stress
was not rerun.

### Next performance work

The remaining cost is not just ray traversal. Prioritize **page-local indexing and
compact selected-face jobs**: the main thread still selects, clones, merges and
hashes large six-face probe records, while display reconstruction performs repeated
spatial lookups per fragment. Profile those CPU stages and the full rendered GPU
frame separately before changing the estimator. Preserve offscreen fairness and
edit-response measurements in every comparison.

More selective distant-edit invalidation, GPU-resident scheduling/display resolve,
and allocator compaction need architectural experiments, not unmeasured cleanup.
Do not reduce coverage, disable bands, extend update intervals, or add debounce to
make a timing graph look better. Before extraction into Starwrought, make the
world adapter and texture/shader setup injectable, define coordinate/transform
contracts, and separate simulation readback demand from rendering demand. The
current BVW vertex layout, single axis-aligned world, and prototype texture path
are still explicit portability limits. The display hash/probe buffers also remain
monolithic and retain spare/high-water capacity. A library supporting much larger
loaded worlds needs device-limit-aware paging or explicit capacity handling;
the existing patch-bounds regression is not a proof of unlimited cache capacity.

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

Cached sun samples the naive unit-face corners, not greedy-quad vertices or averaged face centers. F5 bilinearly interpolates these samples, or displays their constant face average. Four compact RGBE colors replace the previous single float4 per face, keeping the GPU display-probe stride at 208 bytes. Only display color is quantized; gameplay measurements remain f32. Corner mode never switches an incomplete face to Bevy's per-pixel shadows. Available corners contribute immediately; unknown corners temporarily contribute no direct sun (but remain unknown in storage), so a new sunlit face can briefly be dimmer while measurements arrive. A confirmed occlusion stays black. F7 still selects the complete Bevy comparison explicitly. Simulation transport is unchanged.

All faces sharing a corner use the same ray origin, displaced toward the sun rather than along an individual face normal. Exact grid-plane ties consistently select the positive grid cell on CPU and GPU. Shared sky tasks reuse the existing GPU scratch buffers in a separate compute entry point; no permanent GPU binding or display-buffer expansion is needed. Warm batches also omit sky-result readback (96 bytes per face versus the previous 112); cold batches add 32 bytes per pending shared corner. The tradeoff is a sparse CPU corner cache and bucket index, whose entry count follows retained loaded receivers; allocations may retain their high-water capacity. Invalidation is conservative at 16³-region/bucket granularity, and dirty corners refresh when their receivers are scheduled, not in an instantaneous whole-world solve.

The material uses scoped naga-oil function overrides for Bevy's directional-light and shadow evaluation; no Bevy shader source is copied or modified. Ordinary materials are unaffected. Directional lights should represent global sky sources here; per-camera light-layer filtering and directional light textures are not represented in the shared world-space cache.

`VoxelGiSettings` can be inserted before `VoxelGiPlugin` or adjusted as a resource at runtime. Defaults target a 4 ms trace dispatch, up to 8,192 probes per dispatch, a 2 ms receiver-publication budget (at most four atomic pages per update), and a 50 ms display-upload interval. These are scheduling controls, not world-coverage limits or hard real-time deadlines. Smaller budgets reduce throughput; measure estimate age as well as FPS. `scene_publish_budget` is capped at 8 ms; zero retains one page per update, not zero progress.

Resting behavior is controlled by `stable_min_samples` (default 64), `stable_relative_error` (default 0.12), and `stable_audit_interval` (default 2 seconds). Relative error is a scheduling heuristic, not a statistical confidence bound. Noisy faces continue refining. Deterministic sky repairs bypass the normal upload interval; they do not use the indirect display filter or an added debounce.

For a same-revision scheduling A/B baseline, use `VOXEL_GI_CONTINUOUS=1 cargo run` (also works with validation flags), or set `adaptive_updates = false`. This keeps the same ray kernels, compressed geometry, and sky cache, but continuously schedules transport faces. It is not an old-engine benchmark.

`display_response` defaults to 75 ms and controls only the world-space presentation filter. Its response is bounded by sample evidence and variance even after long scheduling gaps. Brightness-compressed temporal weights suppress isolated HDR estimates in dark areas; falling changes cannot exceed the ordinary EMA response. This is a deliberately biased display filter with no permanent brightness ceiling, not a transport estimator. IR, ionizing energy and simulation RGB never use these display values. Persistent brightening/darkening still settles, but the time constant is not a settling-time guarantee: cadence depends on available tracing budget. No screen-space history, extra tracing or new debounce is added. Reservoir/path reuse, GPU-resident accumulation, and wavefront tracing remain future work; neither history smoothing nor adaptive scheduling implements ReSTIR.

## License

Licensed under `CC0-1.0 OR MIT OR Apache-2.0`. See [LICENSE-CC0-1.0.txt](LICENSE-CC0-1.0.txt), [LICENSE-MIT.txt](LICENSE-MIT.txt), and [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt).
