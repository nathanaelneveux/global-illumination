# Voxel GI Lab

This is a fork of `prototype_voxel_template` with persistent, world-space, per-voxel global illumination for the complete loaded voxel world. It is not a camera-following light volume and it does not trace one GI path per screen pixel.

The implementation draws on [vxray](https://github.com/Nelarius/vxray), its [development log](https://nelari.us/post/voxel-ray-tracing/), and [NAADF: Globally Illuminated Voxel Worlds Accelerated with Nested Axis-Aligned Distance Fields](https://www.cg.tuwien.ac.at/research/publications/2026/ulschmid-2026-naadf/). This is an independent compute implementation, not a port of the paper's complete renderer. BVW remains responsible for streaming, meshing, and editing; transport uses its own sparse acceleration structure. It runs on Bevy 0.19 without hardware ray-query support.

## What is implemented

- Each loaded 32³ page is split into eight 16³ regions, with nested 4³ brick and voxel distance fields. Empty and uniform regions are compressed. On the GPU, bricks also have empty/uniform/mixed tags: empty bricks carry AADFs, uniform bricks carry a material, and mixed bricks point to 64 packed 16-bit voxel cells. Identical mixed bricks share reference-counted storage with copy-on-write edits and full-content equality checks. Unchanged regions are shared between snapshots; GPU geometry is persistent and receives changed-region patches.
- Missing regions before occupied loaded terrain within a ray's remaining range are unknown, not open sky, for transport/simulation queries. Those rays terminate conservatively. Empty loaded regions do not create an occluding frontier. Display-only direct-sun rays instead continue across gaps and test all loaded geometry beyond them, matching the geometry that Bevy renders rather than inventing opaque streaming slabs. Geometry outside the loaded set cannot cast display shadows.
- Every exposed solid voxel owns a six-direction ambient-cube probe. Surface sampling is independent of camera visibility and screen resolution.
- GPU compute traces four new paths per **selected exposed face** per transport dispatch, with cosine-weighted directions, three diffuse bounces, visibility-tested analytic sources, Russian roulette, and power-weighted emissive-face sampling with multiple importance sampling. A CPU reference/fallback uses the same transport model.
- Low-discrepancy sequences continue advancing independently of history weights. Unchanged surfaces retain up to 65,536 samples of evidence; scene/source changes shorten or reset histories separately. Eight samples is only a bootstrap milestone, not convergence. Settled faces reduce their update frequency rather than stopping forever; see change-driven scheduling below.
- Indirect display reconstruction averages connected, coplanar neighboring face samples at **virtual voxel corners**, then bilinearly blends across each face. Direct sun instead traces visibility at the actual four unit-face corners, preserving contact shadows that miss face centers. Greedy quads stay intact: no additional mesh vertices are needed. Missing faces and disconnected diagonals are excluded from indirect reconstruction to limit light leaking across edges.
- Direct-sun corners are interned by integer world position across faces, chunks, and dispatches. Each missing corner traces once per visible directional source, then projects the result onto all six receiving normals. A 100×100 coplanar patch has 10,201 unique corners (versus 10,000 centers or 40,000 duplicated corner references); subsequent unchanged batches trace **zero additional sky-corner rays**. Fragmented surfaces have a higher boundary-to-area ratio, so corner counts are not universally equal to face counts.
- A persistent CPU-owned sky cache feeds both compute and CPU fallback. Sun direction/color changes invalidate it; point lights, indirect environment changes, camera movement, and display toggles do not. Geometry changes invalidate spatial buckets intersecting the changed regions' shadow sweeps, up to the same 768-voxel range as tracing, including remote receivers. Air-only streaming preserves visibility; retired receiver entries are pruned. This is independent of the continuously sampled indirect/simulation field and adds no debounce.
- New faces start with conservative dark indirect light or a low-confidence estimate from measured pre-edit neighbors connected through loaded air. Placement can also inherit the same-facing surface it just covered. They never initialize to unoccluded sky. A separate presentation history filters measured indirect RGB; confidence-weighted corner reconstruction favors measured faces. Deterministic direct-sun visibility responds without temporal filtering, and sun seed reuse is restricted to identical geometric corners.
- Bevy PBR retains ordinary visible direct/specular lighting. The extension adds cached indirect RGB, standalone/custom-emitter RGB, and voxel self-emission without counting ordinary direct light twice. Simulation queries receive direct plus indirect energy before display smoothing.
- Edits invalidate local histories, retain existing display values, and reject incompatible in-flight geometry results. A lazy global transport revision also wakes distant indirect receivers without clearing their display or scanning the world during invalidation. Nearby work is prioritized; at least one quarter of transport candidate slots are reserved for an uninterrupted all-loaded-world sweep. Moving sources shorten histories instead of repeatedly clearing the world.
- Adaptive scheduling ranks bounded rotating windows by age, source revision, and per-band relative uncertainty. RGB cannot mask uncertain infrared/ionizing transport. Edit/bootstrap/simulation requests take priority, while the reserved world sweep still prevents starvation. Selection has a per-face work mask distinct from the voxel's exposed-face topology, so an uncertain face does not force its settled neighbors to retrace.
- Scene building, cache construction, dispatch, and readback run asynchronously. The GPU tracer adapts its batch size toward a configurable time budget. A CPU mirror provides gameplay estimates; this is not a wholly GPU-resident authoritative field.
- Workers receive compact receiver headers and only selected faces' measurement state. Display history, topology indices, and adaptive state remain authoritative on the main world; validated results update selected faces in place. Returned job vectors and scheduler ranking buffers are reused across batches.
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

Empty-page streaming outside the occupied region bounds is bookkeeping, not a
transport invalidation: it preserves history and settled state. Filling/removing
unknown gaps **inside** those bounds still invalidates conservatively, as does
any geometry change. Dark indirect receivers (peak RGB below 0.02) require at
least 256 samples before sparse audits. Refinement uses a 0.002 per-band energy
floor, separate from the presentation filter's 0.02 floor, so dark noisy faces
are not prematurely labeled settled. The HUD can therefore report less settling
initially even while those faces obtain better evidence sooner.

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

### Local shared build cache

This checkout's machine-local `.cargo/config.toml` uses
`/Users/nathanaelneveux/Developer/.cargo-build/bevy-0.19` as Cargo's **build directory**,
matching the other voxel prototypes. Compatible intermediate dependency builds
are shared; final executables and GI validation captures remain under this
project's own `target/`. The local configuration also matches the sibling
development profiles: application optimization 1, dependency optimization 3,
line-table debug information, incremental builds, and disabled `wgpu-types`
debug assertions. It overrides the manifest's debug-info settings; record this
configuration when comparing future builds with older performance evidence.

The absolute machine-specific configuration is excluded via `.git/info/exclude`,
not committed or applied globally. A fresh clone needs its own local setup.
`cargo metadata --format-version 1 --no-deps --offline` reports the effective
`build_directory` and `target_directory`. **Do not run an unrestricted `cargo clean`**
to tidy this project: it also removes the shared intermediate cache used by the
other projects. Keep cleanup scoped and preserve validation evidence.

### Preferred defaults

Normal `cargo run` now starts with **shared-corner direct sunlight and corner-blended indirect GI**, terrain textures enabled, irradiance debug disabled, and mesh ambient occlusion disabled. Corners mean naive unit-voxel face corners, not greedy-mesh vertices. Indirect transport itself still samples faces; only its display is reconstructed at corners. F7 remains the per-pixel sunlight comparison. GPU/adaptive transport, the 4 ms trace target, 75 ms indirect display response, and all-loaded-world coverage are unchanged; the rejected CPU throttle is not enabled.

A top-right HUD shows smoothed **FPS**, **Coverage** (exposed faces with at least one sample), **Bootstrap** (progress toward eight samples per exposed face), and **Settled** (faces that passed the adaptive stability checks in the current transport revision). Settled requires at least 64 samples by default, low error/drift across RGB/infrared/ionizing bands and settled presentation for four consecutive batches. It is a heuristic, not certified convergence or an idle indicator: settled faces still receive periodic audits. Geometry/source revisions reset the settled count without clearing all lighting history, so coverage/bootstrap can remain high while settled drops. These all-loaded-world counters are maintained incrementally without HUD world scans or extra rays. Percentages show dashes until probes exist.

**GI main** and **GI GPU** show average milliseconds **per app frame**, not per dispatch, over 250 ms windows; idle windows do not repeat the last busy batch's cost. Main measures the GI main-world system bodies (change collection, source synchronization, scene publication, scheduling, result merging and display-cache preparation). It excludes asynchronous scene/trace workers and render-world uploads/drawing. GPU uses only completed transport/sky compute-pass timestamps, excluding transfers, queue waits and graphics; it shows a dash before measurement, on the CPU backend, or when valid GPU timestamps are unavailable. Completion-based accounting can lag an in-flight batch. These overlapping costs must **not** be added or mistaken for total GI subsystem cost. Hover the HUD with the cursor released for definitions. Numbers refresh four times per second.

Ordinary edits retain existing nearby faces' measured averages/moments with their history weight capped at 16 samples within 48 voxels; farther faces retain up to 64 when refreshed. These are statistical weights, not stored individual ray paths. Histories are marked stale until refreshed. Newly exposed faces start without measurements and can receive a presentation-only neighbor seed; material changes and discrete emitter additions/removals reset obsolete measurements while preserving presentation history for the temporal resolve.

Recurring terminal FPS/GI/rebuild stats are quiet during normal play; use `RUST_LOG=info VOXEL_GI_LOG_STATS=1 cargo run` to restore them. Warnings/errors remain enabled. Scripted validation retains terminal diagnostics and hides the HUD to preserve clean captures and blank-frame detection.

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
| `house` (90 s) | Ceiling refinement and camera-turn invalidation | Fixed eye inside the unmodified prototype house; turn between opposite ceiling corners at 30/50/70 s. Normal streaming remains enabled. Seven captures and 25 fixed underside receivers recorded in `house.json`; no edits, player translation, sources, or mode switches |
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

### Compact worker handoff and coordinate indices (October 6)

`12b8417` replaces full six-face worker snapshots with reusable receiver/face
vectors shared by the GPU path and CPU fallback. On this arm64 build, a one-face
job carries **264 bytes instead of 880** (70% less); sky-only work carries
144 bytes and no transport history (84% less). All six selected faces cost
864 bytes. GPU decoding no longer needs a separate probe/face mapping vector.
Publication validates revisions and updates measurements, display filtering and
adaptive state in place, preserving live topology indices and unsampled faces.
This reduces transient copying and scratch storage, **not persistent world-cache
memory**. The estimator, sample/history limits, simulation bands, direct-sky
cache and presentation filter are unchanged.

`1132d43` retains bounded ranking/deduplication buffers and uses Bevy's fast
integer-key collections for hot coordinate ledgers and display indices.
Ordered ranking, tie-breaking, sweep reservations and GPU hash/buffer formats
are unchanged. Its renderer transaction test also now supplies the timing
resource introduced in `71680e0`; the missing fixture resource was exposed by
the focused renderer tests, not a runtime failure.

All measurements below use Apple M4/Metal, 2560×1440 **offscreen**, the 72-second
gameplay profile, preferred defaults, `RUST_LOG=warn`, and the machine-local
shared-cache development profile documented above. Every executable was built
from the indicated clean revision; no builds ran during measurement. Ordinary
desktop background apps remained running. These are app-cadence observations,
not interactive/windowed FPS or fixed-work microbenchmarks. Asynchronous terrain
loading and scheduling produce different receiver counts between runs.

| Offscreen metric | Starting `71680e0` | Compact `12b8417`, first / repeat | Final `1132d43`, first / repeat |
| --- | ---: | ---: | ---: |
| Editing selection p50 | 1.70 ms | 1.36 / 1.41 ms | 1.21 / 1.28 ms |
| Editing merge p50 | 2.57 ms | 1.74 / 1.83 ms | 1.59 / 1.75 ms |
| Streaming merge p50 | 1.68 ms | 1.09 / 1.15 ms | 1.12 / 1.14 ms |
| Editing frame p50 / p99 | 19.85 / 38.00 ms | 17.84 / 36.09; 20.44 / 37.81 ms | 21.23 / 34.37; 20.52 / 36.21 ms |
| Settled frame p50 / p99 | 21.81 / 31.80 ms | 19.69 / 32.32; 22.26 / 32.86 ms | 22.32 / 32.47; 21.77 / 35.55 ms |
| Streaming frame p50 / p99 | 21.11 / 44.67 ms | 19.09 / 39.38; 21.75 / 41.83 ms | 21.73 / 38.57; 21.87 / 37.65 ms |
| First-fresh edit measurement p95 | 173.20 ms | 159.19 / 170.30 ms | 168.47 / 167.11 ms |
| Total traced paths | 76.30 M | 81.09 / 74.50 M | 74.54 / 76.15 M |

The initial compact run's whole-frame median improvement did not repeat.
Retain these changes for the smaller worker protocol and reduced CPU stage
cost, **not an across-the-board FPS claim**. The index/ranking change's edit
selection/merge improvement repeated; other phases' stage costs were mixed.
Final-repeat streaming far-age p95 was 2,751 ms versus 2,633 ms for the paired
compact repeat; edit far-age p95 was 2,057 versus 2,069 ms. Nearby indirect
display-step p95 during edits was 0.02261 versus 0.02034. These are sparse field
observations, not image-space error or proof of identical temporal noise.
Streaming maximum was still 130 ms and startup maximum 617 ms in the final
repeat; the compact repeat's streaming maximum was 50 ms. Tail improvements
are not universal and these changes do not establish hitch-free play.

The first quiescent initial cohort at 100% coverage **and** bootstrap was
observed at 3.84 s / 217,773 probes in the starting run; compact runs at
3.47 s / 213,741 and 3.69 s / 225,618; final runs at 3.59 s / 227,736 and
3.33 s / 221,711. These 250 ms observations with varying denominators are not
a demonstrated 2–3× completion gain or visual convergence measurement.
All five runs completed 80 edits, five moves and five valid nonblank captures,
with clean render logs and no pending/superseded edit watches. Inspected settled
and streaming captures preserve the preferred corner look.

Local artifacts under `target/gi-validation/`, in execution order:

- `71680e0`: `gameplay-1791294885663371000-31711/`
- `12b8417`: `gameplay-1791295676121845000-44977/`
- `1132d43`: `gameplay-1791296303114088000-54180/`
- `12b8417` repeat: `gameplay-1791296482052773000-56557/`
- `1132d43` repeat: `gameplay-1791296580887441000-58021/`

Logs: `target/perf-compact-{baseline,candidate}.log`,
`target/perf-index-candidate.log`, and
`target/perf-index-{baseline,candidate}-repeat.log`.
Compact publication passed all 63 nonempty face masks across three history/reset
modes, focused CPU orchestration/history tests, and native GPU CPU-parity,
geometry-edit, sky-only/partial-face and emitter-removal checks. The index change
passed 39 focused scheduler/renderer/orchestration tests and both native
upload-retry/renderer-generation checks. Strict all-target Clippy, formatting,
diff checks and development builds passed. No broad toggle stress rerun.

### Fixed-scene refinement diagnosis (October 6)

Coverage/bootstrap did not explain the observed brightness/color swings. The
opt-in `fixed_scene_refinement_diagnostic` (`637811b`) removes streaming, edits,
adaptive scheduling and display uploads from the experiment. It traces 100 floor
receivers, four paths per batch, through 2,048 paths each, recording unfiltered
five-band measurements and the existing display filter separately. It includes:

- An open floor with an analytic sky integral.
- A sealed, nonreflecting room with two colored ceiling emitters. Independent
  128×128 midpoint area quadrature supplies the reference, without MIS or our
  path sampler.
- A diffuse room with a colored wall and narrow skylight. A disjoint 16,384-path
  estimate supplies this reference; unlike the first two, it is **not** an
  independent proof of the integrator's correctness.

The baseline reproduced late brightness swings even with a completely fixed
scene. High-sample analytic/quadrature checks passed; native CPU/GPU comparison
also agreed. These checks rule out several simple estimator/backend failures,
not every possible transport bug. Two retained changes address measured causes:

- `cff4c39`: unchanged surfaces retain up to **65,536** samples of evidence,
  instead of permanently forgetting old evidence after 256. No extra path
  storage is allocated: existing means, moments and counts suffice. This lowers
  the static refinement noise floor. Local/remote edits still cap retained
  weight at 16/64, moving lights at 4/64, and discrete source/material changes
  still reset measurements. The 75 ms presentation response is unchanged.
- `e1a5ac0`: digit permutations break up long same-color prefixes in the high-base
  emitter dimensions of the Halton sampler. Nonzero prime-base multipliers are
  bijective and keep zero fixed; the existing uniform rotations remain. CPU and
  GPU use identical arithmetic. This is limited digit scrambling, not full Owen
  scrambling or a Sobol implementation. The motivation is the high-dimensional
  structure described in [PBRT's Halton discussion](https://pbr-book.org/4ed/Sampling_and_Reconstruction/Halton_Sampler).

A broader permutation experiment also changed the lower path dimensions. It
worsened skylight error at 64 paths (0.579 → 0.825) and was **rejected**. The final
sampler preserves all lower-dimensional samples exactly; non-emissive fixtures
therefore match the history-only change numerically.

M4/Metal, shared-cache development/test profile; same scenes, receivers and
paths, with virtual 50 ms display updates, not gameplay or wall-clock settling:

| Fixed-work metric | Baseline | History only | Final |
| --- | ---: | ---: | ---: |
| Colored emitter raw error p95, 64 paths | 0.1572 | 0.1572 | 0.06118 |
| Colored emitter raw error p95, 2,048 paths | 0.02926 | 0.004277 | 0.002327 |
| Skylight raw error p95, 2,048 paths | 0.2690 | 0.1476 | 0.1476 |
| Colored emitter late display-step p95 | 0.01705 | 0.006452 | 0.002448 |
| Skylight late display-step p95 | 0.04085 | 0.01152 | 0.01152 |
| Skylight late display-step maximum | 0.2065 | 0.1141 | 0.1141 |

Error is absolute per-band error divided by `0.02 + reference_band`, over five
bands and 100 receivers. Display step is the largest RGB channel change divided
by `0.02 + max_reference_RGB`, observed between four-path batches after 256 paths.
These are dimensionless fixture metrics, **not literal percentages of image
brightness**, image-space flicker measurements, or universal error guarantees.
The difficult skylight still has appreciable early variance; neither change
makes eight-sample bootstrap equivalent to convergence. History-only results up
through 256 paths are unchanged. No additional debounce, clamping, disabled
bands, fewer paths per batch, or slower display filtering was introduced.

The emitter kernel's median batch cost was approximately 0.052 ms versus
0.048 ms in the earlier sampler, so its benefit is fewer paths to comparable
accuracy, not cheaper individual rays. Skylight batches remained approximately
0.089–0.090 ms. These tiny fixed-work measurements are not whole-game speedups.
Artifacts: `target/refinement-{baseline,history,permutation,emitter-sampling,final}.{json,log}`.
Baseline is `484da2a` plus the harness; history is `637811b` plus the cap change;
final is `cff4c39` plus the emitter sampler and diagnostic regression bounds,
committed as `e1a5ac0`. No other application/shader changes were present.

Reproduce the opt-in diagnostic (large timelines remain ignored under `target/`):

```sh
VOXEL_GI_REFINEMENT_REPORT=target/refinement.json \
  cargo test --locked --offline fixed_scene_refinement_diagnostic -- --ignored --nocapture
```

Focused tests cover long-history accumulation, invalidation caps, unchanged
lower-dimensional sample streams, emitter distribution, shader validation and
scheduler behavior. Native checks cover analytic references, CPU/GPU parity,
geometry edits, partial-face/sky-only work, and emitter removal; strict Clippy and
development build passed. The final `response` profile at clean `e1a5ac0` saved
14 valid 2560×1440 offscreen captures with clean logs; inspected digging and
warm-emitter removal retain the preferred look. Artifacts:
`target/gi-validation/response-1791304348504440000-86802/` and
`target/refinement-response.log`. This is an integration/response check, not a
paired visual or FPS benchmark.

One final offscreen gameplay sanity run at the same code revision and preferred
settings completed 80 edits, five moves, five valid captures and zero pending or
superseded edit watches. First-fresh edit latency p50/p95 was 149.20/167.93 ms;
editing frame p50/p99 was 20.87/35.38 ms, streaming 22.11/38.93 ms. It traced
75.41 million paths across up to 408 pages. Startup/streaming maxima remained
831/121 ms: this refinement change does not fix those hitches. The earlier
`1132d43` final-repeat edit latency p95 was 167.11 ms; this is a sanity comparison,
not a fresh paired performance experiment or a measured speedup. No code or
asset edits/builds ran during measurement (only these documentation edits).
Artifacts: `target/gi-validation/gameplay-1791304485423591000-89508/` and
`target/refinement-gameplay.log`.

### House ceiling and camera-turn regression (October 6)

The fixed-scene floor diagnostic above did **not** cover the reported house
ceiling/camera-turn failure. `4b1bb1d` adds the `house` profile: a fixed eye inside
the real prototype house, opposite ceiling-corner views at 30/50/70 seconds,
normal terrain streaming, and 25 fixed roof-underside receivers sampled at about
100 ms intervals even while offscreen. Reproduce it with:

```sh
RUST_LOG=warn VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
  VOXEL_GI_VALIDATE_PROFILE=house VOXEL_GI_VALIDATE_EXIT=1 cargo run --locked --offline
```

Two causes were reproduced and addressed separately:

- `594569f`: distant **empty** chunks could turn a sparse gap into an unknown
  occluder, changing the lighting itself. Every scene revision also shortened
  world-wide histories to 64 samples. CPU/GPU frontier queries now ignore empty
  regions and respect the segment limit. Air-only changes outside occupied
  region bounds update the scene without invalidating transport. Interior gaps
  and real geometry/source changes still invalidate conservatively.
- `703ca85`: the scheduler reused the presentation filter's generous dark-energy
  floor. Ceiling faces were called settled near 100 samples, then received only
  about two new paths per second until a rare bright path woke them. A tighter
  independent refinement criterion and 256-sample dark minimum keep them working.
  The 75 ms display filter, four paths per batch, 4 ms dispatch target and fair
  world sweep are unchanged. This is still a heuristic, not certified convergence.

Apple M4/Metal, shared-cache development profile, preferred look, 2560×1440
offscreen. One exploratory sequence, not repeated statistical trials:

| House observation | `4b1bb1d` baseline | Frontier fix | Both fixes |
| --- | ---: | ---: | ---: |
| Transport revisions after 10 s | 7 | 1 | 1 |
| History-count drops across 25 faces after 10 s | 150 | 25 | 25 |
| Median ceiling history at 29 s | 128 | 400 | 1,120 |
| Median ceiling history at 89 s | 116 | 492 | 2,116 |
| Nonzero display-step p95 during 50–90 s | 0.10642 | 0.03986 | 0.01583 |
| Maximum display step during 50–90 s | 0.64703 | 0.24648 | 0.11100 |

Step means max absolute RGB change divided by `0.002 + max(old_RGB,new_RGB)`
between recorded observations. Unchanged observations are excluded from the
percentile; there were 1,621/495/6,102 changed observations respectively. This is
observed cache variation, **not image-space flicker or error against ground truth**.
The single remaining reset followed actual new terrain loading on the first
turn. More selective geometry invalidation is still future work. The stricter
test can lower the HUD's early settled percentage while producing better evidence.

All three runs completed seven valid captures with clean render logs; inspected
returned-corner captures preserve textures and show a more uniform ceiling.
Streaming discovery/order is stochastic. The frontier-only run also overlapped
focused compilation/native checks early on, so this sequence makes no FPS claim.
Local artifacts under `target/gi-validation/`:

- Baseline `4b1bb1d`: `house-1791332068681180000-88752/`.
- `4b1bb1d` plus the runtime diff committed as `594569f`:
  `house-1791332268553231000-91840/`.
- `594569f` plus the runtime diff committed as `703ca85`:
  `house-1791332452113600000-94741/`.

Logs: `target/house-{baseline-clean,frontier,dark-scheduling}.log`. The earlier
preliminary `house-1791331816480324000-83963/` predates removal of unused
player-body translations from this profile and is excluded from the table.

Final gameplay integration (`703ca85` runtime, no concurrent builds): 80 edits,
five moves, five valid captures, no pending/superseded edit watches, clean logs.
First-fresh edit p50/p95 was 150.47/171.20 ms (previous `e1a5ac0` sanity run:
149.20/167.93 ms); 74.88 million paths, 100% coverage. Editing offscreen frame
p50/p99 was 20.65/33.36 ms, streaming 21.94/39.27 ms. Startup/streaming maxima
remain 613.5/104.7 ms. This is not a fresh paired FPS experiment or a hitch fix.
Artifacts: `gameplay-1791332653004925000-97991/`, `target/house-fix-gameplay.log`.
Focused scheduler/history/worker/orchestration checks, native CPU/GPU frontier
and geometry/emitter parity, Naga validation, formatting and strict all-target
Clippy passed. No broad toggle stress run.

### Next performance work

The remaining cost is not just ray traversal. Compact selected-face worker jobs
are implemented; **persistent face storage and page-local indexing** remain
worthwhile experiments. Authoritative probes still reserve six faces and require
world-coordinate lookups. Display reconstruction also performs repeated spatial
lookups per fragment. Profile those CPU stages and the full rendered GPU frame
separately before changing the estimator. Preserve offscreen fairness and
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
