# Live view-owned far transport

Integrated checkpoint over `0f87c34`, 2026-10-21. The runtime, material binding,
prepass and retention changes are one buildable integration; they were not
independently playable stages. Bounded receiver radius enables the new tier;
the default is still the all-near control until moving gameplay is qualified.

The reduced-resolution view grid owns exact voxel-face identities, raw moments,
path records and unfinished path continuations. Duplicate pixels share one
owner. Exact-identity reprojection preserves these records during camera motion;
it is not color-only TAA or shifted spatial path reuse. A bounded wave resumes
up to three intersections, without imposing a three-bounce path cutoff. Near
and far share traversal, transport, corner sunlight and history-retention rules.
Spatial reconstruction only affects display. Far histories are view-local.

Far gets 25% of the existing transport time share; near gets the remainder.
Discovery, sunlight and reconstruction are additional presentation costs, timed
separately. Asynchronous 160-byte timing/counter receipts enforce the ray budget;
there is no lighting readback or main-thread GPU wait. Optical tables and edit
events are transferred GPU-to-GPU. Material storage bindings are fragment-only
to fit Metal's vertex buffer limit. Prepass layout is Bevy's, not the terrain
forward layout.

## Initial qualification

Apple M4/Metal, dev profile, 2560x1440 offscreen, cave with background loading,
corner sun/indirect, textures on, AO off. Candidate radius16, CPU statistics off;
control radius0, exact `0f87c34` in managed `far-live-control` worktree. Candidate
is the integrated dirty implementation recorded by this checkpoint, before
formatting-only cleanup. All captures are nonblank; final run has clean render
logs. These are offscreen cadence measurements, not windowed FPS.

| >=15s, capture frames excluded | all-near control | far candidate |
| --- | ---: | ---: |
| median interval ms | 22.05 | 21.77 |
| p95 ms | 25.13 | 25.76 |
| p99 ms | 30.89 | 26.78 |

This single pair establishes no small performance win. Candidate showed 5,892
visible owners, all displayed, mean history about 2,374 after one minute.
Representative late GPU costs: trace 0.49ms, presentation 0.48ms. History count
is not a convergence claim. Cave appearance was inspected against the all-near
control: lit ceiling, crystal halos, colored walls and corner sunlight agree
qualitatively. Raw sampler accuracy thresholds are unchanged.

Evidence: `target/far-live-fourth.log` and
`target/gi-validation/cave-1792618765191565000-82319`; control worktree's
`target/baseline-cave.log` and `cave-1792618211014091000-81931`.
Earlier runs exposed and fixed Metal binding limits, quantized prepass normals
and MSAA depth/voxel-plane mismatch; they are not quality passes.

Focused checks passed: both MSAA shader variants, native runtime pipelines,
Bevy material composition, headless material/cache generations, retained-wave
parity and CPU/GPU retention parity. Existing shared-estimator furnace evidence
is reused; not rerun. Camera motion, edits, near/far handoff and windowed frame
pacing remain to qualify. Reconstruction has bounded coplanar support, not
topological connectivity; subpixel surfaces may need better view coverage.
