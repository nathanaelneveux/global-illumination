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

The stronger emitter reverses the trade-off. Its full target change is mean
FLIP 0.30695 / worst tile 0.43120, versus 0.04161 / 0.06887 for the skylight
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
aperture edits but fail the stronger source response gate. This is also not
grounds for an emitter-specific exception: the important distinction is
change amplitude versus sampling variance, which geometry and sunlight can
also change. Keep the general baseline while qualifying estimator improvements
against both small-change noise and large-change lag. The earlier repeated-cap
unchanged-cave failures remain relevant and are not reclassified as passes.
This screening rejects blanket higher-floor promotion before spending a full
windowed runtime matrix on it. Actual gameplay seconds to sustained FLIP
acceptance remain unqualified; replay-clock seconds must not fill that gap.

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
