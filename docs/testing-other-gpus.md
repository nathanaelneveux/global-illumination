# Testing on another GPU

Voxel GI has been validated on an Apple M4 (Metal) and an AMD RX 7900 XT
on Windows (Vulkan and DX12 with DXC). This guide qualifies another GPU or
driver. Allow about an hour, mostly for the first build; step 6 lists the
evidence to retain.

## 1. Set up

- [Rust](https://rustup.rs) 1.95 or newer (`rustup update stable`).
- Windows: Visual Studio Build Tools with the "Desktop development with C++"
  workload, which rustup offers to install. Linux: the
  [Bevy dependencies](https://github.com/bevyengine/bevy/blob/main/docs/linux_dependencies.md).
- Git, plus about 20 GB of free disk for the first build.
- Windows, for the DX12 runs: the DXC shader compiler. Without it wgpu falls
  back to FXC, which rejects GI's compute shaders (and, with them rewritten
  to compile, took over five minutes on every launch; see the
  [FXC issue](https://github.com/nathanaelneveux/global-illumination/issues/4)).
  Vulkan needs nothing extra. Bevy uses DXC by any one of these routes:

  | Route | Cost |
  |---|---|
  | `dxcompiler.dll` (13.7 MB) and `dxil.dll` (1.4 MB) in the working directory | Two Microsoft-redistributable DLLs to ship with each Windows build; no code change |
  | Bevy's `statically-linked-dxc` feature (wgpu `static-dxc`) | The host's choice: a prebuilt static library fetched at build time, Windows x86-64 MSVC only, a larger executable |
  | `$env:WGPU_DX12_COMPILER = "dxc"` with DXC on `PATH` | Per-user setup; fine for testing, not for shipping |

  For this guide the third is simplest: the Windows SDK installed with the
  Build Tools includes DXC, so put its `bin\<version>\x64` folder on `PATH`.

  DXC compiles slowly the first time (over 20 s for GI's pipelines on an RX
  7900 XT) and caches the result, so a first DX12 run can show no terrain
  for a while; later runs reached GI's first frame in about 8 s
  ([cold-cache issue](https://github.com/nathanaelneveux/global-illumination/issues/3)).

```powershell
git clone https://github.com/nathanaelneveux/global-illumination
cd global-illumination
```

The commands below are PowerShell. In a Linux shell, write `VAR=value command`
instead of `$env:VAR = "value"`.

## 2. The example

```powershell
cargo run -p bevy_voxel_gi --example minimal
```

A small stone room should appear, open on two sides, with a pillar and a
glowing orange lamp on the floor. Sunlight comes through the open sides, and
the lamp and the sun bounce soft light around the room. Note anything black,
flickering or missing. Then save a capture (rendered offscreen, so no window
opens):

```powershell
$env:VOXEL_GI_EXAMPLE_SCREENSHOT = "minimal.png"; $env:VOXEL_GI_EXAMPLE_FRAMES = "600"
cargo run -p bevy_voxel_gi --example minimal
Remove-Item Env:VOXEL_GI_EXAMPLE_SCREENSHOT, Env:VOXEL_GI_EXAMPLE_FRAMES
```

## 3. GPU tests

```powershell
cargo test -p bevy_voxel_gi --lib -- --include-ignored 2>&1 | Tee-Object gpu-tests.txt
cargo test -p bevy_voxel_gi_bvw --lib -- --include-ignored 2>&1 | Tee-Object -Append gpu-tests.txt
```

## 4. Play the lab

```powershell
cargo run --locked
```

This is a streaming world with GI. Fly around, look into caves and dig. The
top-right HUD shows FPS and GI's cost. If the build fails at the link step,
retry with `cargo run --locked --no-default-features`, which turns off Bevy's
dynamic linking.

## 5. Headless measurements

These runs need no window. Each writes a folder under `target/gi-validation/`.

```powershell
$env:RUST_LOG = "warn,bevy_voxel_gi=info"; $env:VOXEL_GI_VALIDATE = "1"
$env:VOXEL_GI_VALIDATE_HEADLESS = "1"; $env:VOXEL_GI_VALIDATE_EXIT = "1"
$env:VOXEL_GI_VALIDATE_FRAME_TIMINGS = "1"
foreach ($p in "streaming", "cave", "house") {
  $env:VOXEL_GI_VALIDATE_PROFILE = $p
  cargo run --locked 2>&1 | Out-File "run-$p.log"
}
```

On Windows, run the loop once per backend, setting it explicitly: wgpu's
default depends on the machine (an RX 7900 XT picked Vulkan,
[windows-first-run](history/windows-first-run.md)). Rename the logs between
runs so the second doesn't overwrite the first.

```powershell
# DX12 must use DXC: FXC, the default compiler, can't build GI's pipelines
# (issue #4). With this override, DXC must be available to the DLL loader
# through PATH or beside the executable (step 1).
$env:WGPU_BACKEND = "dx12"; $env:WGPU_DX12_COMPILER = "dxc"
# ...the loop above, then rename its logs to run-*-dx12.log...
$env:WGPU_BACKEND = "vulkan"; Remove-Item Env:WGPU_DX12_COMPILER
# ...the loop again...
```

DXC's first compile of each build can take 20 s or more before GI's first
frame ([issue #3](https://github.com/nathanaelneveux/global-illumination/issues/3)).
That is not a GI failure: the profile and its captures wait for GI's first
view frame, and `report.json` records the wait as `gi_ready_s` (a run fails
only if GI shows nothing for 120 s). Steps 2 to 4 use the default backend;
the adapter line in their output names it.

## 6. Send back

- The GPU model, driver version, OS and screen resolution.
- `minimal.png`, `gpu-tests.txt`, the `run-*.log` files, and the
  `target/gi-validation` folder zipped.
- Any log lines containing `WARN` or `ERROR`. GI warns once when it falls back:
  no timestamp queries, receiver table overflow, or a failed GPU allocation.
- What the lab felt like: FPS on the HUD, and anything that looked wrong.
