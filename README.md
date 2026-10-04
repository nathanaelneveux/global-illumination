# Voxel GI Lab

This is a fork of `prototype_voxel_template` with persistent, world-space, per-voxel global illumination for the complete loaded voxel world. It is not a camera-following light volume and it does not trace one GI path per screen pixel.

The implementation draws on [vxray](https://github.com/Nelarius/vxray), its [development log](https://nelari.us/post/voxel-ray-tracing/), and [NAADF: Globally Illuminated Voxel Worlds Accelerated with Nested Axis-Aligned Distance Fields](https://www.cg.tuwien.ac.at/research/publications/2026/ulschmid-2026-naadf/). This is an independent compute implementation, not a port of the paper's complete renderer. BVW remains responsible for streaming, meshing, and editing; transport uses its own sparse acceleration structure. It runs on Bevy 0.19 without hardware ray-query support.

## What is implemented

- Each loaded 32³ page is split into eight 16³ regions, with nested 4³ brick and voxel distance fields. Empty and uniform regions are compressed. Unchanged regions are shared between snapshots; GPU geometry is persistent and receives changed-region patches.
- Missing regions inside the streamed domain are unknown, not open sky. Rays reaching such holes terminate conservatively. The environment contributes on escape from the domain; geometry outside the loaded set cannot cast shadows.
- Every exposed solid voxel owns a six-direction ambient-cube probe. Surface sampling is independent of camera visibility and screen resolution.
- GPU compute traces four new paths per exposed face per dispatch, with cosine-weighted directions, three diffuse bounces, visibility-tested analytic sources, Russian roulette, and power-weighted emissive-face sampling with multiple importance sampling. A CPU reference/fallback uses the same transport model.
- Independently rotated low-discrepancy sequences continue advancing after the 256-sample history limit. Eight samples is only a bootstrap milestone, not convergence; sampling never stops there.
- Display reconstruction averages connected, coplanar neighboring face samples at **virtual voxel corners**, then bilinearly blends across each face. Greedy quads stay intact: no additional mesh vertices are needed. Missing faces and disconnected diagonals are excluded to limit light leaking across edges.
- Bevy PBR retains ordinary visible direct/specular lighting. The extension adds cached indirect RGB, standalone/custom-emitter RGB, and voxel self-emission without counting ordinary direct light twice. Simulation queries receive direct plus indirect energy before display smoothing.
- Edits invalidate local histories, retain existing display values, and reject obsolete in-flight geometry results. Nearby work is prioritized; at least one quarter of dispatch slots are reserved for an uninterrupted all-loaded-world sweep. Moving sources shorten affected histories instead of repeatedly clearing the world.
- Scene building, cache construction, dispatch, and readback run asynchronously. The GPU tracer adapts its batch size toward a configurable time budget. A CPU mirror provides gameplay estimates; this is not a wholly GPU-resident authoritative field.
- Sparse display uploads preserve stable slots during edits. Growing the hash cache keeps the previous display available until its replacement is ready.
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

`VoxelGiStats` separates first-sample `interactive_coverage`, eight-sample `bootstrap_coverage`, mean history, oldest measured-sample age, and refresh progress. None is a certified convergence/error bound. Check per-surface age and source revision when freshness matters to gameplay. The temporary display fallback for an unmeasured face is never returned as a simulation measurement.

An Apple M4 Metal development-build stress run at 2560×1440 completed 79 edit/restore operations, a continuously moving point source, and five streaming moves, reaching 432 loaded pages. Median frame time was about 16.7 ms; edit-phase p99 was 25.3 ms and streaming-phase p99 was 30.3 ms. The run ended at 100% first-sample coverage. This measurement predates the additional emissive-voxel workload and the final redundant-rebuild/fairness fixes; repeat the validation below for the current revision. It is evidence for this machine and scene, not a guarantee for every GPU, terrain, or source count.

## Energy API

`VoxelEnergyField::estimate(voxel, normal)` returns an `EnergyEstimate` containing:

- `incident`: direct plus indirect RGB, infrared, and ionizing bands;
- `variance`: a running second-moment estimate, not a statistical confidence interval;
- `effective_samples`, `age`, and `sources_current`: metadata for deciding whether to use or defer a simulation update.

Unmeasured, geometry-invalidated, and buried faces return `None`. `sample` is a convenience method that returns only incident energy. These are unfiltered surface estimates, not volume samples or the smoothed display value.

`EnergyEmitter` augments a `PointLight` or `DirectionalLight`, or works with a transform alone as a point source, for any combination of bands. `VoxelMaterialTransportTable::set` changes diffuse response and emission for a material index and triggers progressive resampling. Reflectance is constrained to [0, 1]; non-finite/negative energy inputs are rejected or sanitized. Emissive voxels both glow and illuminate other surfaces.

All bands currently use normalized transport units, **not watts, temperature, or dose**. A thermal simulation can integrate suitable fresh estimates with heat capacity, absorption, conduction, cooling, and phase-change rules, then turn a hot block into an emissive material. That simulation and physical unit calibration are not implemented here. Ionizing transport uses the same opaque diffuse model: it does not model penetration, energy-dependent attenuation, scattering physics, or biological dose.

## Supported model and limits

The implemented renderer is diffuse GI for opaque axis-aligned voxels, with Bevy point/directional sources and emissive voxel materials. Paths have three diffuse bounces and a 768-voxel segment limit. Transmission, specular GI, spot/area-light components other than voxel emitters, arbitrary non-voxel occluders, and participating media are not implemented. Neither all-world coverage nor a good frame rate guarantees instantaneous global lighting changes. Unbounded worlds and thousands of analytic lights require additional scale testing; the current analytic-source loop evaluates every source.

## Run and verify

```bash
cargo run
cargo test --all-targets
cargo test gpu_matches_cpu -- --ignored --nocapture
cargo test gpu_display_upload -- --ignored --nocapture
cargo clippy --all-targets -- -D warnings
```

The explicit ignored test requires a native compute adapter. It compares GPU and CPU results for sky, analytic sources, all transport bands, and an enclosed emissive room, and checks incremental geometry uploads. Regular tests cover traversal against a DDA reference, no sky leakage in a sealed room, corner continuity, sparse allocation/uploads, continuous sequences, scheduler fairness, and streaming/in-flight consistency.

For the repeatable 72-second end-to-end workload:

```bash
cargo build
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_EXIT=1 cargo run
```

Keep the game window visible and the machine otherwise idle. In development builds this captures flat/corner/GI-disabled/irradiance-only images, performs repeated edits, animates a point light, places an emissive voxel, streams terrain, restores temporary block/material/source changes, and saves phase frame-time percentiles to `target/gi-validation/report.json`. Screenshots are saved alongside it. The workload runs only when explicitly enabled; it does not edit terrain files. The player remains at the final test position if automatic exit is omitted.

Use `VOXEL_GI_BACKEND=cpu cargo run` to exercise the bounded CPU reference fallback. GPU initialization/dispatch failures also fall back with a warning. CPU fallback has the same model but not the same throughput guarantee. Startup and five-second status logs report the selected backend, coverage, history, age, and build/trace timings.

The smaller development-only `VOXEL_GI_SMOKE_EDIT=1` check removes one block after bootstrap settles. `VOXEL_GI_SMOKE_MOVE=1` moves the player 48 voxels to exercise streaming. Do not combine these with the full validation workload when comparing performance.

## Controls

- `WASD`: move
- `Space`: jump
- `Left Ctrl`: crouch
- `Shift`: sprint
- Mouse: look
- Left click: remove a voxel and invalidate its transport page
- Right click: place a voxel and invalidate its transport page
- `F4`: toggle voxel GI for A/B comparison
- `F5`: toggle flat per-voxel faces / blended virtual voxel corners
- `F6`: isolate cached irradiance from terrain textures and ordinary PBR lighting
- `F7`: switch direct sunlight between Bevy per-pixel lighting (default) and the voxel cache; F5 selects flat faces or blended voxel corners for the cached mode
- `F3`: toggle mesh debug overlay
- `Esc`: toggle inspector and release/capture cursor

## Configuration

Terrain settings remain hot-reloadable in [`assets/config/default.prototype.ron`](assets/config/default.prototype.ron). Mesh ambient occlusion is disabled by default so it does not bake a second approximation into the ray-traced indirect result.

For a sky-light A/B comparison, leave F5 on its default corner blending and press F7. Diffuse sky/environment lighting was already in the voxel cache; F7 additionally routes ordinary directional sunlight through it. Point lights and other materials retain their normal Bevy rendering. The cached sun uses voxel-ray visibility and a diffuse Lambertian response, with Bevy's camera exposure; it intentionally omits the sun's view-dependent specular lobe. Thus this compares the full diffuse voxel-lit look, not only two shadow filters. Cached direct sun can lag edits/source changes just like the GI field. F4 can disable indirect GI without disabling the selected direct-sun mode. Switching modes changes neither transport sources nor simulation estimates and does not restart sampling.

The material uses scoped naga-oil function overrides for Bevy's directional-light and shadow evaluation; no Bevy shader source is copied or modified. Ordinary materials are unaffected. Directional lights should represent global sky sources here; per-camera light-layer filtering and directional light textures are not represented in the shared world-space cache.

`VoxelGiSettings` can be inserted before `VoxelGiPlugin` or adjusted as a resource at runtime. Defaults target a 4 ms trace dispatch, up to 8,192 probes per dispatch, and a 50 ms display-upload interval. These are scheduling controls, not world-coverage limits or hard real-time deadlines. Smaller budgets reduce throughput; measure estimate age as well as FPS.

## License

Licensed under `CC0-1.0 OR MIT OR Apache-2.0`. See [LICENSE-CC0-1.0.txt](LICENSE-CC0-1.0.txt), [LICENSE-MIT.txt](LICENSE-MIT.txt), and [LICENSE-Apache-2.0.txt](LICENSE-Apache-2.0.txt).
