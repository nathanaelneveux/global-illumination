# Testing on another GPU

Voxel GI has been measured only on an Apple M4 (Metal). This guide is for a
first run on Windows (DX12 and Vulkan) or Linux (Vulkan). It takes about an
hour, most of it the first build. Please send back what step 6 asks for.

## 1. Set up

- [Rust](https://rustup.rs) 1.95 or newer (`rustup update stable`).
- Windows: Visual Studio Build Tools with the "Desktop development with C++"
  workload, which rustup offers to install. Linux: the
  [Bevy dependencies](https://github.com/bevyengine/bevy/blob/main/docs/linux_dependencies.md).
- Git, plus about 20 GB of free disk for the first build.

```powershell
git clone https://github.com/nathanaelneveux/global-illumination
cd global-illumination
```

The commands below are PowerShell. In a Linux shell, write `VAR=value command`
instead of `$env:VAR = "value"`.

## 2. The example

```powershell
cargo run -p voxel_gi --example minimal
```

A small stone room should appear, open on two sides, with a pillar and a
glowing orange lamp on the floor. Sunlight comes through the open sides, and
the lamp and the sun bounce soft light around the room. Note anything black,
flickering or missing. Then save a capture (rendered offscreen, so no window
opens):

```powershell
$env:VOXEL_GI_EXAMPLE_SCREENSHOT = "minimal.png"; $env:VOXEL_GI_EXAMPLE_FRAMES = "600"
cargo run -p voxel_gi --example minimal
Remove-Item Env:VOXEL_GI_EXAMPLE_SCREENSHOT, Env:VOXEL_GI_EXAMPLE_FRAMES
```

## 3. GPU tests

```powershell
cargo test -p voxel_gi --lib -- --include-ignored 2>&1 | Tee-Object gpu-tests.txt
cargo test -p voxel_gi_bvw --lib -- --include-ignored 2>&1 | Tee-Object -Append gpu-tests.txt
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
$env:RUST_LOG = "warn,voxel_gi=info"; $env:VOXEL_GI_VALIDATE = "1"
$env:VOXEL_GI_VALIDATE_HEADLESS = "1"; $env:VOXEL_GI_VALIDATE_EXIT = "1"
$env:VOXEL_GI_VALIDATE_FRAME_TIMINGS = "1"
foreach ($p in "streaming", "cave", "house") {
  $env:VOXEL_GI_VALIDATE_PROFILE = $p
  cargo run --locked 2>&1 | Out-File "run-$p.log"
}
```

On Windows, run the loop a second time with `$env:WGPU_BACKEND = "vulkan"`
(DX12 is the default). Rename the logs first so the second run doesn't
overwrite them.

## 6. Send back

- The GPU model, driver version, OS and screen resolution.
- `minimal.png`, `gpu-tests.txt`, the `run-*.log` files, and the
  `target/gi-validation` folder zipped.
- Any log lines containing `WARN` or `ERROR`. GI warns once when it falls back:
  no timestamp queries, receiver table overflow, or a failed GPU allocation.
- What the lab felt like: FPS on the HUD, and anything that looked wrong.
