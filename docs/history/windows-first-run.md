# Windows first run (baseline, 2026-10-30)

The first run of voxel GI off the Apple M4, following
[testing-other-gpus](../testing-other-gpus.md) from start to finish. Later
Windows fix and performance PRs compare against this record. It makes no
code changes.

Commit `33ac7ed` (`.gitattributes`, LF checkout). Every step except the
first `minimal` launch ran there. That launch, and the initial builds, ran on
`0b06e67`, which differs only by `.gitattributes`.

## Machine

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XT (vendor 0x1002, device 0x744C), discrete |
| Driver | 32.0.31019.2002, dated 2026-05-28; wgpu reports "AMD proprietary driver 26.6.1 (LLPC)" |
| Subgroups | min 32, max 64 (Vulkan) |
| CPU, RAM | Intel Core i5-13600KF (14 cores), 31.8 GiB |
| OS | Windows 11 Pro 10.0.26300 |
| Display | 2560x1440, 180 Hz |
| Toolchain | rustc 1.99.1, `dev` profile (`build: development` in reports) |

wgpu picks **Vulkan** by default on this machine. The guide says DX12 is the
default. The DX12 runs below set `WGPU_BACKEND=dx12` explicitly.

## Results

| Step | Vulkan | DX12 |
|---|---|---|
| Build: example, lab, both test crates | pass | (same build) |
| 2. `minimal` window | pass (tester: looked correct) | not run |
| 2. `minimal.png`, 600 frames offscreen | pass, no WARN/ERROR | not run |
| 3. `voxel_gi` lib tests (`--include-ignored`) | **181 pass, 3 fail** | not run |
| 3. `voxel_gi_bvw` lib tests | 8 pass | not run |
| 4. Lab window | pass (tester: "looked and ran good", about 180 FPS) | not run |
| 5. Headless streaming, cave, house | all completed | **all fail at startup** (FXC) |

The guide runs steps 2 to 4 with the default backend, so they ran on Vulkan
here. The GPU tests always ran on Vulkan (the test adapter line says so).

### 1. CRLF checkout (fixed upstream in 33ac7ed)

On `0b06e67`, with Git for Windows' default `core.autocrlf=true`, `minimal`
panicked before opening a window:

    thread 'main' panicked at crates\voxel_gi\src\gpu.rs:45:5:
    shader lacks the traversal include marker

The call chain was `VoxelGiPlugin::build` → `install_render_assets` →
`presentation_shader_source` → `include_traversal`. `33ac7ed` forces LF
checkouts. Everything after that crash ran on an LF checkout.

### 2. Three `voxel_gi` GPU test failures (Vulkan)

- `gpu::view::runtime::coverage_tests::native_view_discovery_keeps_thin_surfaces_without_growing_storage`
  fails at `coverage_tests.rs:229`, "all six crowded planes need support". Grid 4
  passes for every thin/delayed case, and grid 3 without thin surfaces passes.
  The first failure is grid 3, `thin=true`, frame 0: plane `(4, 7)` is missing,
  with counters `[9, 10, 1, 0]`.
- `gpu::maintenance_tests::scene_edits_reach_view_history_without_world_cache_pages`
  fails at `maintenance_tests.rs:325`, "scene publication timed out".
- `gpu::view::runtime::pixel_tests::native_small_faces_are_lit_per_pixel_without_receivers`
  fails at `pixel_tests.rs:368`. At frame 3, the 64 values are a mix of `4.0` and
  `3.9980469` (4 − 2⁻⁹).

The whole suite took 46 s. The full output is in `gpu-tests.txt` in the results
bundle.

### 3. DX12: FXC rejects the GI compute shaders

All three profiles exit with code 1 about one second after starting. Each logs 87
ERROR lines, and `status.json` stays at `"running"`. FXC (wgpu's default
DX12 compiler when no `dxcompiler.dll` is next to the executable) reports:

- `error X3500: array reference cannot be used as an l-value; not natively addressable`
  in "World-space voxel transport", line 795
- `error X3511: forced to unroll loop, but unrolling failed` at lines 251 and 540

These pipelines fail to create: World-space voxel face tracing, GI change
footprints, classify_pixels, discover_receivers, load_shared_faces,
prepare_receiver_pool, publish_view, reconstruct_view, reproject_paths,
trace_pixel_queue and trace_visible_paths. The only WARN line is Bevy's
"CommandQueue has un-applied commands being dropped" during shutdown.

**Diagnostic, not part of the guide:** one streaming run with
`WGPU_DX12_COMPILER=dxc`, using the Windows SDK's DXC 1.8.2502.11 from
`PATH`. No shader errors appear, but the run fails its capture check:
"Offscreen capture is uniform/black, has unexpected dimensions, or has no active
camera targeting its image." `settled.png` is entirely black. Nothing logs at
WARN with `RUST_LOG=warn,voxel_gi=info,wgpu_hal=info`.

### 4. Memory is not sampled on Windows

Every report has `memory.samples: []` and `peak_mb: null`. The sampler uses
macOS `footprint -p`.

## Measurements (Vulkan, headless)

These are offscreen app-cadence timings (60 Hz `ScheduleRunner`, no
swapchain), in a `dev` build. Frame p50 sits at the 16.7 ms runner period, so
frame time shows hitches, not GPU headroom. Don't compare these with windowed
FPS. Times are in ms. "worker" is the GI worker; "transport" is the
transport execution.

**streaming**: completed, coverage 0.998, 659 max loaded pages, 126.5 M traced paths, 44 s

| phase | frames | frame p50 | p95 | p99 | max | >33.3 | >50 | worker p50 | worker p95 | transport p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| startup | 582 | 16.93 | 18.25 | 23.07 | 119.0 | 5 | 5 | 0.64 | 1.05 | 1.46 |
| streaming | 861 | 16.95 | 18.30 | 19.45 | 64.3 | 2 | 1 | 0.95 | 1.30 | 2.18 |
| settled | 276 | 16.96 | 17.49 | 17.99 | 18.5 | 0 | 0 | 0.58 | 0.81 | 0.50 |
| recovery | 276 | 16.94 | 17.68 | 18.30 | 21.4 | 0 | 0 | 0.63 | 0.97 | 2.04 |

**cave**: completed, coverage 1.0, 64 max loaded pages, 16.2 M traced paths, 63 s

| phase | frames | frame p50 | p95 | p99 | max | >33.3 | >50 | worker p50 | worker p95 | transport p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| startup | 875 | 16.96 | 17.50 | 18.04 | 103.3 | 3 | 2 | 0.52 | 0.73 | 0.46 |
| settling | 2710 | 16.97 | 17.46 | 17.76 | 20.4 | 0 | 0 | 0.50 | 0.63 | 0.37 |

**house**: completed, coverage 1.0, 663 max loaded pages, 59.2 M traced paths, 91 s

| phase | frames | frame p50 | p95 | p99 | max | >33.3 | >50 | worker p50 | worker p95 | transport p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| first-corner | 1737 | 16.95 | 17.63 | 18.54 | 122.5 | 4 | 3 | 0.58 | 0.92 | 0.99 |
| first-corner-returned | 1141 | 16.94 | 17.55 | 18.08 | 24.8 | 0 | 0 | 0.64 | 0.87 | 1.51 |
| second-corner | 1113 | 16.96 | 17.59 | 18.08 | 68.8 | 2 | 2 | 0.64 | 0.89 | 1.68 |
| second-corner-returned | 1163 | 16.95 | 17.48 | 17.77 | 22.3 | 0 | 0 | 0.64 | 0.83 | 1.55 |

No Vulkan run logged WARN or ERROR. Every capture is nonblank, and the house
and cave captures look plausible: bounce light indoors and emitter glow in the
cave. They have not been compared with M4 captures.

DX12 has no measurements: every profile failed before measuring.

## Measurements (Vulkan, windowed, uncapped)

Added after the first run, at `acddc43` (same GI code as `33ac7ed` apart from
the test and documentation changes since). `gameplay` profile,
`VOXEL_GI_VALIDATE_NO_VSYNC=1`, `WGPU_BACKEND=vulkan`, `dev` build. The
window was the lab's default **1280x720**, not the 2560x1440 display, so
these numbers do not describe full-resolution headroom. All 5 captures were
focused, visible, varied and presented with `AutoNoVsync`. Windowed frame
pacing: don't compare these with the headless tables above.

completed, coverage 1.0, 414 max loaded pages, 470.3 M traced paths

| phase | frames | frame p50 | p95 | p99 | max | >33.3 | >50 | worker p50 | worker p95 | transport p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| startup | 2924 | 3.19 | 3.87 | 7.12 | 124.5 | 3 | 3 | 0.59 | 0.91 | 0.22 |
| streaming | 5844 | 3.26 | 4.29 | 5.62 | 11.5 | 0 | 0 | 0.59 | 1.10 | 0.68 |
| editing | 5811 | 3.33 | 4.28 | 4.89 | 9.9 | 0 | 0 | 0.49 | 0.96 | 0.81 |
| settled | 4534 | 3.17 | 3.58 | 3.83 | 9.8 | 0 | 0 | 0.59 | 0.88 | 0.16 |
| recovery | 2108 | 3.13 | 3.83 | 4.37 | 10.4 | 0 | 0 | 0.56 | 1.06 | 0.22 |

Median frame time is 3.1 to 3.3 ms (about 300 to 320 FPS) in every phase,
and p99 stays under 7.2 ms. The only frames over 33 ms are three at startup.
No WARN or ERROR lines.

## What the tester saw

- **`minimal`** (window, Vulkan): it looked correct, with the room, pillar, lamp,
  sunlight and bounce. Nothing was reported black, flickering or missing. The
  offscreen `minimal.png` matches.
- **Lab** (window, Vulkan, vsync on): it "looked and ran good", at about 180
  FPS. That equals the 180 Hz refresh rate, so it is probably vsync-capped.
  The uncapped run above, at 1280x720, ran at about 300 FPS.

## Results bundle

These were sent separately, not committed: `minimal.png`, `gpu-tests.txt`, the
`run-{streaming,cave,house}-{dx12,vulkan}.log` files, the zipped
`target/gi-validation` (DX12 folders with only `status.json`, Vulkan folders
with reports and captures, and the DXC diagnostic), the minimal crash
backtrace from `0b06e67`, and the build log.
