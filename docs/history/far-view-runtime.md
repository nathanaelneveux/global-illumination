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

## Playable default qualification

The integrated default is radius64, with `config/gi-all-near.json` preserving
the old radius0 control. Four-pixel discovery cells replace the initial eight
at small window sizes; capacity remains 65,536. The turn-and-walk test exposed
a budget deadlock: dividing whole-pass cost by a small path count eventually
rounded every future wave to zero. `0bea5c0` replaces that estimate with the
same bounded feedback controller as near transport. It repays actual GPU
milliseconds and allows an indivisible minimum wave. A focused regression
tests recovery from a costly wave, not just positive throughput at startup.

Windowed gameplay comparison: control `0f87c34`; candidate `10050b9` plus only
the radius64 default and accompanying docs/config diff. Apple M4/Metal, dev,
1280x720, Fifo, identical preferred look and 20% total transport share,
background terrain loading. No competing builds/GPU tests during measured
phases. Both runs completed 80 edits, five moves, five focused/nonblack captures
and clean render logs. Capture/readback cooldown frames are excluded below.

| Windowed result | all-near control | playable near/far |
| --- | ---: | ---: |
| Editing median / p99 ms | 16.66 / 18.73 | 16.66 / 17.34 |
| Streaming median / p99 ms | 16.66 / 19.16 | 16.66 / 17.33 |
| Recovery median / p99 ms | 16.68 / 20.13 | 16.67 / 17.12 |
| Worst frame after startup, excluding captures, ms | 41.38 | 18.93 |
| Peak physical footprint, MB | 2,975 | 939 |

Both have a roughly 0.7-second cold-start frame before play; this is not a
zero-hitch startup claim. The prototype is running at its 60Hz presentation
cadence, not guaranteeing every frame is strictly below 16.67ms. This is one
paired run; the memory reduction is large, while small tail differences need
repeated pairs before optimization claims. Earlier offscreen timing was kept
separate. One offscreen streaming run overlapped computer-use inventory work
and had a large outlier; it is not used as interactive performance evidence.

The final window captures were visually compared. Live-baseline FLIP means
(67 pixels/degree, official FLIP 1.7) were 0.00349 settled, 0.02826 streaming,
0.01990 recovered. Worst regional differences are on distant shaded terrain,
where the near/far candidate is brighter and the all-near control is still
changing. These are appearance differences against a live control, **not**
error against a converged oracle or a new raw accuracy calibration.

Evidence: control worktree's `gameplay-1792619403148040000-83667` and
`target/baseline-gameplay-window-2.log`; candidate
`target/gi-validation/gameplay-1792620156063570000-85343` and
`target/far-default-gameplay-window.log`. The extra 180-degree turn / continuous
64.8-voxel walk passed at `a76ab6a` in
`streaming-1792619970850229000-85012`: all 1,823 visible owners displayed again,
no unmeasured owners at completion, and mean history continued to 379 rather
than freezing at 123. These counts establish liveness, not convergence.

Final focused checks also pass: native runtime pipelines, both prepass shader
variants, headless material/cache generations, far-only source/model updates,
GPU/CPU source/edit parity. Standalone emitting components' direct RGB is now
included in far presentation separately from indirect moments; ordinary Bevy
lights are not counted twice. No BVW dependency changes were made.

The final windowed `structures` route also passed, moving from the house to
the entrance and then the cave with background terrain loading: four valid
captures, clean logs, cave median/p99 16.66/17.15ms, 916MB peak physical
footprint. Its 36-second cave capture shows the lit ceiling, colored emitter
halos and shared-corner sunlight. Evidence:
`target/gi-validation/structures-1792620374920967000-85811` and
`target/far-default-structures-window.log`. It is a scripted camera route,
not a character-controller locomotion benchmark.

Attribution correction: the validation background loader already omits
`Camera3d`, so the explicit active-camera guard is defensive for inactive 3D
views, not the cause of these savings. Bounded receiver storage is the major
memory change. The temporary baseline worktree was archived after its ignored
captures/logs were preserved under `target/far-live-control-evidence/`.

Remaining limits are explicit: histories are view-local; uncovered subpixel
surfaces and coplanar support are not a topologically complete world cache;
near/far overlap still duplicates some work. Spatial path reconnection,
compressed records, and lower-tier hardware qualification remain future work.
The default is playable on the measured machine, not a claim that all research
or all platform/performance work is complete.
