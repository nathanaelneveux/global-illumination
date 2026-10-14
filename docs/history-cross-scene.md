# Cross-scene history response qualification

Branch `codex/gi-history-cross-scene`, preserving the earlier calibration at
`a6b38a9`. Runtime transport, history defaults and BVW are unchanged. This is
an extension of [near-history calibration](near-history-calibration.md), not
a promotion of a new retention floor.

## Acceptance and timing

Use the unchanged `cave-user-2048` limits: mean FLIP <= 0.027400863220808837
and worst 32x32 tile mean <= 0.07296444475650787. House comparisons transfer
these numerical limits as an engineering acceptance gate; the user has not
independently calibrated perceptual invisibility for the house view.

An early pass is not settling. Require every later observed checkpoint to
pass, including at least two later observations. Report the last failed
checkpoint and the beginning of the accepted suffix. The unobserved interval
between those observations remains uncertain. A passing warm control is also
required before labeling a result response from an already settled scene.

Fresh paths measure work to acceptance. The replay's 0.85-second per-face
publication clock is an explicit assumption, **not measured gameplay latency**.
Do not convert the tables below into an interactive seconds/FPS claim. Actual
scheduling, repeated invalidation, newly exposed receivers and mesh publication
still require runtime qualification for a policy that passes these controls.

## Finer cave sweep

Same moderate (8x ordinary power) single off-camera torch and native tape as
the previous study. Observe every 256 fresh paths around the threshold instead
of jumping from 2,048 to 3,072. Windows are independent common-path sequences;
all candidates within a window receive identical samples.

| Retained floor | Removal, window 0 / 1 | Addition, window 0 / 1 |
| --- | ---: | ---: |
| 16 | 2,304 / 3,072 | 3,072 / 3,328 |
| 128 | 2,560 / not repeated | 3,072 / not repeated |
| 256 | 2,560 / 3,072 | 3,072 / 3,328 |
| 512 | 2,560 / 3,072 | 3,328 / 3,584 |

Each nonzero entry is the first sustained passing observation; the immediately
preceding observation, 256 paths earlier, fails. All starts and reference
replica checks pass. The 16-floor removal is one checkpoint faster in window
0, but not in window 1. Thus the older coarse equality was not proof of equal
settling time. The 512-floor addition is one checkpoint slower in both windows.

| Floor | Peak observed worst-tile FLIP: removal, W0 / W1 | Addition, W0 / W1 |
| --- | ---: | ---: |
| 16 | 0.19076 / 0.21648 | 0.17048 / 0.18556 |
| 128 | 0.13243 / — | 0.13743 / — |
| 256 | 0.12091 / 0.12761 | 0.12965 / 0.13980 |
| 512 | 0.11926 / 0.12761 | 0.12666 / 0.12209 |

256 reduces peak local error by 36.6–41.1% on removal and 23.9–24.7% on
addition versus 16. 128 does not improve on 256 in the first window. 512 buys
some additional smoothness but delays brightening. No tested floor dominates
256 on both observed settling work and early error.

The early error integral gives the same cave ordering, so this is not only a
peak-error comparison. Trapezoidal mean FLIP over the first 512 fresh paths
(divided by 512) is 0.04403 / 0.04016 / 0.03828 for removal with floors
16 / 256 / 512, and 0.04764 / 0.04357 / 0.04069 for addition, in window 0.

Floors through 256 use the same 48-voxel near radius and existing far curve.
512 uses a uniform floor across the affected domain (radius 278), avoiding a
nonmonotonic near-512/far-256 policy. It is not a near-only substitution.

## House fixture design

The opt-in `VOXEL_GI_NEAR_FIXTURE` values `house-skylight`, `house-door` and
`house-emitter` use the real authored house with normal background terrain
loading before the offline snapshot. The fixed upper-corner camera is at
(0.5,14,-17.5), aimed at (5.5,17,-12.5), vertical FOV 25 degrees. Its tighter
view excludes edited geometry and the existing low windows. These are retained
surface tests, not new-face/topology tests or the usual turning-camera profile.

The skylight removes/restores (0,17,-18). The west door removes/restores
(-6,11..13,-18). The emitter is at (-4,13,-21), with visible emission
(12,3,0.3), infrared 4 and ionizing 1 in diagnostic material 250. Material 250
must be unused. All changes are private frozen-scene edits, not persistent
world modifications. Offscreen eight-corner/frustum checks guard raster parity.

Each state has two 131,072-path reference replicas, averaged to 262,144 paths.
The tape records the same 4,096-path warm mean and 4,096 fresh paths used by the
cave test. House direct-sun shared corners are solved for each state and saved
with the tape; replay installs the target state's sunlight immediately, including
at time zero. This prevents an opening test from accidentally retaining old
direct sunlight while testing indirect history. Exact sky scheduling is outside
this isolated history comparison.

## House results

First native sequence, sustained passing observations in fresh paths:

| Edit | H16 | H128 | H256 | H512 |
| --- | ---: | ---: | ---: | ---: |
| Open skylight | 1,024 | 768 | 256 | 256 |
| Close skylight | 768 | 768 | 768 | 768 |
| Open west door | 1,024 | 768 | 768 | 256 |
| Close west door | 1,024 | 1,024 | 768 | 768 |
| Add house emitter | 2,560 | 2,560 | >4,096 | >4,096 |
| Remove house emitter | 1,536 | >4,096 | >4,096 | >4,096 |

`>4,096` means **no observation passed**, including the final one, not merely
an insufficient confirmation tail. Every warm control passes. All opening
and closing results have at least two later passing observations. Emitter
addition at H16 first passes at 1,536, fails again at 2,048, and only then
starts its passing suffix at 2,560. H128 addition has the same sustained
checkpoint, but substantially worse early error.

An independent skylight repeat, with finer observations near the crossing,
confirms the direction without pretending the first sequence's 4x opening
gain is universal:

| Edit, window 1 | H16 | H256 | H512 |
| --- | ---: | ---: | ---: |
| Open skylight | 896 | 640 | 512 |
| Close skylight | 768 | 640 | 384 |

Both warm states pass. H256 closing first passes at 192, but fails again
later; only the suffix beginning at 640 passes. This is another reason to
retain the sustained criterion rather than optimize a lucky first crossing.

512 is genuinely better than 256 in the first doorway-opening sequence:
fewer paths to the observed passing suffix and lower early error. On skylight
opening they share the crossing checkpoint, with lower early error at 512.
Higher floors lower observed peak worst-tile error for all four aperture cases:

| Edit | Peak tile, H16 | H256 | H512 |
| --- | ---: | ---: | ---: |
| Open skylight | 0.11705 | 0.09222 | 0.08283 |
| Close skylight | 0.11812 | 0.08722 | 0.07876 |
| Open west door | 0.11546 | 0.08517 | 0.07441 |
| Close west door | 0.12724 | 0.09791 | 0.08625 |

The locally dominant house emitter reverses the trade-off. Its full target
change is mean FLIP 0.30695 / worst tile 0.43120, versus 0.04161 / 0.06887 for the skylight
and 0.04050 / 0.05977 for the door. Reference-replica noise remains well below
the gate (emitter-present 0.00473 / 0.01357), so this is not an uncertain target.
These measured amplitudes explain the result; they are **not** oracle inputs
to the history policies.

At 4,096 fresh paths after emitter removal, H16 is at mean FLIP 0.01317 /
worst tile 0.04027, while H256 is 0.04515 / 0.08120 and H512 is
0.06971 / 0.11734. The latter two remain above the limits. The early
error integral over 0..512 paths also reverses: mean FLIP is
0.12128 / 0.21396 / 0.24541 for removal and 0.10398 / 0.20060 / 0.23495
for addition with H16 / H256 / H512. Retaining more is not just slower here;
it also gives a worse early image because stale lighting dominates noise.

The independent emitter repeat confirms this counterexample: H16 sustains
acceptance from the 1,024-path observation for both addition and removal,
while H256 never passes through 4,096 in either direction. The repeat uses a
coarser 256..1,024 crossing bracket, not a claim of exactly 1,024 paths. Both
warm states pass. At 4,096, H256 removal remains at mean FLIP 0.04209 / worst
tile 0.07746; addition remains at mean 0.04196 / worst tile 0.06155 (the mean
alone fails). This is not a single unlucky sample sequence.

Thus do not promote a universal 256 or 512 floor. They improve ordinary
aperture edits but fail the dominant-source response gate. This is also not
grounds for an emitter-specific exception: the important distinction is
change amplitude versus sampling variance, which geometry and sunlight can
also change. Keep the general baseline while qualifying estimator improvements
against both small-change noise and large-change lag. The earlier repeated-cap
unchanged-cave failures remain relevant and are not reclassified as passes.
This screening rejects blanket higher-floor promotion before spending a full
windowed runtime matrix on it. Actual gameplay seconds to sustained FLIP
acceptance remain unqualified; replay-clock seconds must not fill that gap.

## Distance attribution and a visible sunlight patch

Follow-up on 2026-10-14, native diagnostic commit `9beec66`, analysis
`5a2e15e`. No runtime retention, transport, emitter falloff or BVW changes.
The original small skylight/corner control remains available unchanged.

The original corner view cannot show the floor's direct-sun update. The new
`house-sun-patch` fixture removes/restores nine roof voxels: x=-1..1,
y=17, z=-19..-17. Its camera is (0.5,15.5,-18), aimed at (2.5,11,-16.5),
vertical FOV 40 degrees. All edited roof geometry remains offscreen. Four
unique shared floor corners change from measured black to sunlit:
(3,11,-16), (3,11,-15), (4,11,-16), (4,11,-15). They affect nine floor
faces and 783,994 pixels in this view. Direct-sun-only captures with indirect
GI zeroed independently confirm the patch; the measured full references are
nonblank and retain normal textures and indirect lighting.

As before, the replay installs the target-state direct sunlight at time zero.
This verifies a visible direct-sun change and measures the remaining indirect
history response. It does **not** time the runtime sky worker, secondary
sunlit-face invalidation, or propagation through a streaming/edit pipeline.
Normal background loading occurs before freezing; it does not continue during
the isolated common-path replay.

The opening's full before/after mean FLIP is 0.25460. Even with direct sun
already updated, the old indirect display starts at mean FLIP 0.08480 on
opening and 0.09601 on closing. Thus this is a substantial geometry/light
change, not a disguised emitter-specific test.

Fresh paths to the sustained whole-image gate:

| Floor-view edit | H16, window 0 / 1 | H256, window 0 / 1 |
| --- | ---: | ---: |
| Open 3x3 skylight | 256 / 256 | 1,536 / 1,280 |
| Close 3x3 skylight | 256 / 192 | 1,536 / 1,536 |

The previous failing observations in window 0 are 128 (H16) and 1,024
(H256). In window 1 they are opening 192/1,024 and closing 128/1,280
(H16/H256). Every accepted suffix has at least two later passing observations.

All warm controls and reference replicas pass. In window 0, H16 reduces
both lag and early error: after 256 fresh paths, opening mean/worst-tile FLIP
is 0.01722/0.07181 versus H256's 0.05422/0.14023. Closing H16 is
0.01013/0.03203. Trapezoidal mean FLIP over the first 512 fresh paths is
opening 0.02960/0.05782 and closing 0.02683/0.06177 (H16/H256) in window 0;
window 1 is 0.02656/0.05457 and 0.03409/0.06517 respectively. H16 roughly
halves this early-error measure as well as reaching the gate sooner.
These are observed sample-work crossings, not exact first crossings or
measured seconds. The fixed house engineering gate remains a
transfer of the user's cave calibration, not a new human judgment.

### Error versus source distance

`tools/history_distance_analysis.py` evaluates official FLIP on the original
2560x1440 images, then groups errors by receiver-voxel distance to the edit.
Pixel-center ray/quad intersections map to the nearest audited surface;
distance matches the policy's integer-voxel metric. This maps all pixels in
the house emitter view, 99.69% in the cave, and 99.98% in the tight floor view.
Unmapped pixels are excluded only from regional diagnostics, never from the
whole-image acceptance gate. Regional tile diagnostics require at least 256
in-band pixels per 32x32 tile. Neither regional statistic is an independently
calibrated perceptual gate. Corner reconstruction and FLIP spatial filters
span neighboring receivers, so this is spatial attribution, not identification
of each path's contributing source.

Window-0 mean regional FLIP after 256 fresh paths:

| Visible receiver band | Remove, H16 / H256 | Add, H16 / H256 |
| --- | ---: | ---: |
| House, 9.38 to <12 blocks | 0.08045 / 0.21491 | 0.06416 / 0.19064 |
| House, 12 to 13.15 blocks | 0.06451 / 0.18536 | 0.05138 / 0.17680 |
| Cave, 19.65 to <24 blocks | 0.05539 / 0.03760 | 0.06662 / 0.04808 |
| Cave, 32 to <48 blocks | 0.05046 / 0.04092 | 0.05332 / 0.04118 |

This supports testing a smaller low-history neighborhood. It does **not**
prove perceptual error decreases monotonically with distance. The cave's
full target-change mean is 0.03641 in the 20..24 band and 0.04497 in the
32..48 band. Visibility, material, background lighting and sampling variance
matter, and the house/cave lamps and scenes differ. The house emitter uses
(12,3,0.3) emission, below the normal placeable block's (12,5,1); it is not
an artificially boosted lamp. The cave diagnostic still uses its 8x torch.

### A 16-block low-history neighborhood

The experiment `16:16` retains 16 within 16 blocks, then uses the existing
256-plus-distance-derived curve. It does not truncate emitter range, alter
the distant prior, or add a source-type specialization.

In the cave native replay, sustained crossings are removal 2,560/3,072 and
addition 3,072/3,328 for windows 0/1: the same as `48:256`. At the 24 common
observations per window, mean FLIP differs by less than 8e-8 and worst-tile
FLIP is identical. In every recorded house fixture all receiver distances
are below 16, so `16:16` and `48:16` produce identical cap arrays and replay
states. No redundant house render is needed to establish that identity.
Consequently this candidate preserves the fast house emitter response and
the large floor-opening response while retaining the cave's smoothness.
The original small upper-corner aperture tests still get H16, so their
extra smoothing from H256 is not retained. It is not universally optimal.

**Sixteen is a promising candidate, not an analytically identified optimum.**
The measured house emitter view stops at 13.15 blocks; visible cave receiver
contributors (including corner neighbors) start at 18.97. Cutoffs of 14..18
cannot be distinguished by those two views. Further qualification should
hold scene/source constant and put receivers around this transition in view,
then test runtime edit/streaming response. Runtime defaults remain 48/16.
Emitter falloff/range was deliberately left unchanged.

### Follow-up evidence

- Expanded-opening native tape: `house-1791987058895295000-85469`.
- Floor W0 render: `house-1791987179731036000-85529`, plan/analysis prefix
  `target/history-sun-patch-tight`, 58 planned frames including two direct-only
  controls. Regional analysis: `target/history-sun-patch-distance.json`.
- Floor W1 render: `house-1791987445090808000-85708`, prefix
  `target/history-sun-patch-w1`, 64 planned frames using the independent
  native sequence and finer observations around the passing checkpoints.
- Near-16 cave render: `cave-1791987316132219000-85667`, prefix
  `target/history-cave-near16`, 58 planned frames across two native windows.
- Reused emitter/cave W0 captures from the earlier section; regional output
  `target/history-house-emitter-distance.json`, `target/history-cave-distance.json`.
- Contact sheet: `target/history-sun-patch-overview.png`; unchanged image
  brightness, downsampled only for presentation. Metrics use full resolution.
- The first broad floor framing (`house-1791986805140297000-85287`) showed the
  patch but included 10.5% unaudited outdoor pixels through the doorway.
  Its replay `house-1791986930999523000-85407` was intentionally stopped and
  is not accepted evidence. The tighter view was re-recorded, not silently
  paired with the old camera's captures.

Four focused Rust tests and eleven Python tests pass; native build succeeds.
Successful record/render runs have valid offscreen cameras, target dimensions,
assets and complete geometry-audited captures, with no WARN/ERROR render logs.
As throughout this study, no windowed FPS or interactive latency claim follows.

Reproduction uses the earlier commands with fixture `house-sun-patch` and
`--sky-controls` when making the plan. For distance attribution:

```sh
python tools/history_distance_analysis.py PLAN_JSON RENDER_RUN --output RESULT_JSON
```

Keep the same NumPy/Pillow/FLIP environment as the main analysis. This tool
verifies the native tape hash and capture completion; run the normal `assess`
command too for reference-replica, warm-control and sustained-gate qualification.

## Evidence and reproduction

Apple M4 integrated GPU, Metal, 16 GiB, macOS 26.6.2; Rust 1.95, Bevy 0.19.1,
unchanged BVW 0.17. Dev app opt-level 1, dependencies opt-level 3. Offscreen
2560x1440 LDR sRGB, FLIP 1.7 at 67 PPD, shared-corner sunlight,
corner-reconstructed indirect GI, textures on, baked AO off. One GPU workload
at a time. CPU image analysis may overlap a later workload; no timing claims
are made from those overlapping runs.

- Cave native tape (unchanged): `cave-1791939958286106000-77130`.
- Cave W0 plan: `target/history-cross-cave-plan/plan.json`, 144 frames;
  rendered `cave-1791942419386018000-79023`, analyzer
  `target/history-cross-cave-analysis.json`. Clean `1b49003` at launch, using
  the unchanged native renderer; house diagnostic edits were made after launch.
- Cave W1 plan: `target/history-cross-cave-w1-plan/plan.json`, 110 frames;
  rendered `cave-1791943147539528000-79353`, analyzer
  `target/history-cross-cave-w1-analysis.json`. Native code `0dce328`;
  Python-only reporting commit `6a2aad0` was made during the run.
- House skylight tape: `house-1791942995562898000-79273`; W0 render
  `house-1791943286345808000-79509`, plan/analysis prefix
  `target/history-house-skylight`. The tape ran at clean `0dce328`;
  native renderer unchanged thereafter.
- House doorway tape: `house-1791943476979704000-79555`; W0 render
  `house-1791943816732834000-79691`, plan/analysis prefix
  `target/history-house-door`.
- House emitter tape: `house-1791943668010887000-79619`; W0 render
  `house-1791943986011194000-79741`, plan/analysis prefix
  `target/history-house-emitter`.
- House W0 plans contain 128 frames each. Doorway and emitter recording and
  rendering used native `0dce328` plus Python-only `6a2aad0`, with only this
  untracked documentation draft added during the runs. No algorithm diff.
- Skylight W1 render: `house-1791944165518930000-79781`, 98 plan frames;
  prefix `target/history-house-skylight-w1`.
- Emitter W1 render: `house-1791944385004361000-79822`, 44 plan frames;
  prefix `target/history-house-emitter-w1`. These repeats use the same native
  code and Python analyzer, with only documentation/README edits in progress.

Plans record native tape hashes, moment hashes, policy pairs, view and all
arguments. Analysis records its plan hash. Run folders live under ignored
`target/gi-validation/`. Large binary moments, PNGs and logs are not committed.

Harness checks: four focused Rust tests (`cargo test --locked --offline near_`),
eight Python replay tests, a successful native build and diff review. No routine
toggle stress or full suite was run: runtime algorithms and buffer lifetimes
were unchanged. Replaying the cave through the extended renderer preserves the
earlier matching checkpoint values.

Record another house tape with the same native fixture pipeline:

```sh
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_PROFILE=house \
VOXEL_GI_VALIDATE_BACKGROUND=1 VOXEL_GI_VALIDATE_EXIT=1 \
VOXEL_GI_VALIDATE_HEADLESS=1 VOXEL_GI_NEAR_FIXTURE=house-skylight \
VOXEL_GI_VALIDATE_NEAR_HISTORY=record cargo run --locked --offline
```

Use `tools/near_history_calibration.py plan RUN --output PLAN_DIR --policies
48:16 48:256 278:512 --windows 0 --cases remove add`, optionally supplying
`--checkpoints`. Then rerun the same native command with
`VOXEL_GI_VALIDATE_NEAR_HISTORY` set to the absolute plan.json path, retaining
the same fixture selection. Finally run `assess PLAN_JSON RENDER_RUN --output
RESULT_JSON`. The Python tools require NumPy, Pillow and official FLIP 1.7.
Use `house-door` or `house-emitter` for the other fixtures. Do not pair a house
plan with the cave profile or reuse references from another fixture.
