# Controlled near-edit hallway

This study separates distance from scene/material/source differences in the
earlier house-versus-cave comparisons. Runtime retention, transport, emitter
falloff and display smoothing remain unchanged. See [structures](structures.md)
for the walking route and optional same-world spawn shortcut.

## Fixture and controls

The normal-world hallway has one ordinary warm block emitter, not a boosted
diagnostic torch. All floor receivers use the same rock material. Two entry
baffles block direct sky; the complete chamber is audited against the world
generator before taking a frozen snapshot. Background terrain loads during
the initial gameplay phase, then transport is frozen for matched-path replay.
These replays are **not interactive frame-pacing or edit-latency benchmarks**.
The snapshot retains the rest of the loaded world and its emitter sampling
distribution. "Single emitter" describes the hallway, not an artificially
isolated light sampler with every other loaded source removed. All station
replays share the same tape, so this distribution is held fixed across distance.

Sixteen camera stations sample offsets 1, 2, 3, 4, 6, 8, 12, 14, 16, 18, 20,
24, 32, 40, 48 and 64 along the floor. Each camera has the same 12-degree
vertical FOV and 2.5-block height, filling the 2560x1440 capture with the same
portion of a single floor face. Far receivers do not become artificially
insignificant through perspective. The measured face and every exposed
coplanar corner donor are traced. Block-coordinate distance is sqrt(offset²+1);
corner donors have their own nearby distances, not the central face's distance.

- `hallway-emitter`: switch only the authored lamp's emission off/on; keep
  its diffuse solid geometry. This includes the immediately adjacent receiver
  without confounding it with a newly exposed floor face.
- `hallway-occluder`: insert/remove a five-wide, four-high rock screen one
  block in front of the lamp. Source power stays constant. Stations begin at
  offset 4 so the edited screen and newly covered floor donors are off-camera.
- `null`: apply the proposed retention weight without changing lighting,
  measuring the noise cost of unnecessary invalidation.

The replay's generic `remove` case means before→after, and `add` means the
reverse. For the occluder these are screen insertion/removal, not lamp edits.
Station offsets always refer to the lamp. Invalidation distance still uses
the nearest changed voxel: the offset-8 receiver is sqrt(50) from the screen,
versus sqrt(65) from the lamp voxel.
Existing house/cave physical emitter-removal tests remain necessary before
promoting a policy; this fixture does not replace topology/publication tests.

Each state has two disjoint 131,072-path reference replicas and two matched
4,096-warm + 4,096-fresh path windows. The renderer uses production corner
reconstruction, confidence weighting, material textures and tonemapping.
Direct-sun corners are recorded for both states and installed immediately.

## Literal weights and acceptance

`--literal` tests the exact statistical history weight everywhere. In this
mode 0 discards statistical history, 1 really retains one sample's weight,
and 65536 is the retain-all control. Display history is never reset. Without
this flag, the old curve and h0=retain-all convention remain reproducible.
The original prior-derived bias bound clamps requested small floors to 11
nearby; using that curve would not actually test weights 1, 2, 4 or 8.

Use the unchanged cave-user-2048 FLIP gate (mean <= 0.0274008632, worst 32px
tile mean <= 0.0729644448 at 67 PPD). This is a transferred engineering gate,
not an independently human-calibrated close-up threshold. Settled requires a
passing suffix with at least two later observations, and a passing old-state
warm control. Fresh paths measure work, not elapsed gameplay time. The replay
clock of 0.85 seconds per 16-path publication is an explicit assumption.

The source-off hallway reference is a deliberately dark control. It is not
accepted as black gameplay evidence. It requires zero energy in both native
replicas, resolved-black sky corners, matching tape identity, and nonblank
positive material witnesses of the same geometry before and after the entire
replay. Captured dark reference pixels must be at most one 8-bit code value
(tonemapping dither). General camera/asset/geometry/capture checks stay enabled;
the usual blank-reference rejection remains for other fixtures.

## Reproduction

Record once with normal background loading:

```sh
VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_PROFILE=hallway \
VOXEL_GI_VALIDATE_BACKGROUND=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_EXIT=1 VOXEL_GI_NEAR_FIXTURE=hallway-emitter \
VOXEL_GI_HALLWAY_OFFSET=8 VOXEL_GI_VALIDATE_NEAR_HISTORY=record \
cargo run --locked --offline
```

Use the emitted directory as `TAPE` (a placeholder, not a shell variable
required by the application). The tape contains all authored camera stations:

```sh
python tools/near_history_calibration.py plan TAPE --output PLAN \
  --view d8 --literal --histories 0 1 2 4 8 16 32 64 128 256 65536 \
  --windows 0 1
```

Replay with the same environment, replacing `record` with `PLAN/plan.json`.
Set `VOXEL_GI_HALLWAY_OFFSET` to the selected station; mismatched plan/camera
identity is rejected. Assess with `near_history_calibration.py assess
PLAN/plan.json RUN --output RESULT.json`. FLIP requires the same Python
dependencies as the earlier near-history studies. Shortlist/checkpoint options
avoid repeatedly rendering the full cross product during investigation.

## October 14 qualification: useful low weights, no universal winner

This is an initial controlled qualification, **not a completed fit of all
sixteen stations or a promoted runtime curve**. Screened weights 0, 1, 4, 16,
64 and 256 at offsets 1 and 8; repeated shortlisted candidates on the second
native sequence. Offset 16 and the offset-8 occluder control use window 0.
Unchanged-lighting controls include 65536 (retain all) in the repeated runs.
Weights 2, 8, 32 and 128 and the remaining stations are supported but not yet
visually qualified. No distance cutoff is inferred from these three stations.

Observed sustained acceptance in **fresh paths per face**, not gameplay seconds:

| Offset / edit | H0, W0 / W1 | H16, W0 / W1 | H256, W0 / W1 |
| --- | ---: | ---: | ---: |
| 1, lamp off | 128 / 128 | >4096 / >4096 | >4096 / — |
| 1, lamp on | 1024 / 512 | 1024 / 1536 | >4096 / — |
| 8, lamp off | 128 / 128 | 1024 / 1024 | >4096 / >4096 |
| 8, lamp on | 512 / 1024 | 3072 / 256 | >4096 / >4096 |
| 16, lamp off | 64 / — | 512 / — | 3072 / — |
| 16, lamp on | 2048 / — | 3072 / — | 3072 / — |

Here `>4096` means the final observed image still fails, not an extrapolated
settling estimate; `—` means not repeated. Every listed source-edit response
starts from a passing warm image. Reference-replica differences pass the same
FLIP gate. Offset 1 has the vertical separation described above; it is the
adjacent floor block, not a one-unit center-to-center distance.

Zero is not uniformly better even when the light really changes. At offset 1,
H4 brightening settles at 256 / 1024 paths, versus H0's 1024 / 512: the ordering
reverses between sequences. H4 removal still fails at 4096 in both. At offset 8
in W1, H0 brightening first passes at 128, overshoots, and only stays passing
from 1024. H16 stays passing from 256. Mean **rendered linear-sRGB luminance**
at 256 paths is 109.1% of the reference for H0 versus 102.4% for H16. These are
diagnostic brightness ratios, not replacements for FLIP or physical radiance.
The narrow late-check follow-up confirms that H16's W0 brightening remains
failing at 2048, then passes at 3072, 3584 and 4096. Its earlier first pass
at 1024 was not settling.

The unchanged-lighting control rejects a blanket aggressive reset:

| Offset / window | Peak mean FLIP, H0 | H16 | H256 | Retain all |
| --- | ---: | ---: | ---: | ---: |
| 1 / W1 | 0.05664 | 0.05208 | — | 0.00885 |
| 8 / W1 | 0.11341 | 0.11260 | 0.06945 | 0.01859 |
| 16 / W0 | 0.05662 | 0.03931 | 0.02602 | — |

All these starts pass. At offset 16, H256 stays passing throughout, while H0
and H16 disturb the settled image and do not stay passing again until 2048.
At offset 8, even H256 creates a visible W1 disturbance. This is a measured
cost of invalidating an unchanged integrand, not evidence that retained-all
can respond adequately to an actual large lighting change.

For the **occluding-wall insertion**, H0 and H16 both settle at 1536 paths;
H256 still fails at 4096. H0 improves the early error integral but does not
reduce settling work. Removing that wall shows observed recovery at 512 / 3072
for H0 / H16, but its 4096-path occluded warm image narrowly fails the gate
(mean FLIP 0.02880). That direction is **not qualified settled-start response**.
Do not promote it as a speedup. Both high-sample occluder-state references
qualify (replica mean FLIP 0.01396 / 0.00670).

### Architectural implication

Retained weight affects two things: stale mean bias and the uncertainty used
by display smoothing. We retain the raw second moment as well as the mean.
After a distribution change, the mixture of old bright and new dark samples
can have high variance even when the incoming samples are consistently dark.
Reducing the history count also reduces the estimator's stated confidence.
Consequently, H1 can initially fade more slowly than H16, while H16 leaves a
larger long-term residual. H0 removes this mixed-distribution moment history;
it does **not** discard display history or turn smoothing off.

This follows directly from `replay` / `history.rs::resolve_face`, and explains
why a smooth monotonic distance-to-weight curve cannot be assumed to imply a
smooth monotonic response-time curve. Distance, actual change magnitude,
incoming variance and the display filter interact. Before fine-fitting a
production near curve, retain these controls when evaluating how uncertainty
responds to nonstationary lighting. Do not introduce emitter-only rules from
the especially favorable source-off result, or silently change sampling or
smoothing to make the curve pass. Runtime 48/16 and the qualified distant
retention policy remain unchanged in this branch.

### Evidence and verification

Baseline `d2bf6b2`; literal mode `7c97651`; world hallway `60f83c3`; diagnostic
harness `53af977`; guard/profile-test hardening `1f67435`. The first emitter tape
and W0 replay used the corresponding working harness diff before its checkpoint;
later changes added the occluder fixture, formatting, guards and tests, not
emitter transport/replay behavior. The later runs use the checkpointed harness;
documentation and test-only changes do not alter their renderer.

Apple M4, Metal, 16 GiB, macOS 26.6.2; debug optimized/debuginfo Cargo profile,
Bevy 0.19.1, 2560x1440 offscreen. Default transport, ordinary lamp, shared-corner
sun, reconstructed indirect corners, textures on, baked AO off. Background
terrain streams before the frozen phase. Some CPU image assessments ran while
other offscreen runs rendered; **no frame-time/FPS claims** are made from these
runs. All completed native runs have valid target dimensions/cameras/assets,
complete geometry audits and clean render logs.

Ignored evidence under `target/`:

- Emitter tape: `gi-validation/hallway-1791990327369535000-87051`.
- Occluder tape: `gi-validation/hallway-1791991248477174000-87458`.
- Normal-world overview: `gi-validation/hallway-1791991028355581000-87357`.
- Plans: `hallway-plan-d1`, `hallway-plan-d8`, `hallway-plan-d16`,
  `hallway-plan-d1-repeat`, `hallway-plan-d8-repeat`,
  `hallway-plan-occluder-d8`, and the narrow `hallway-plan-d8-late` follow-up.
- Full metric/plan records: corresponding `hallway-*-assessment.json`; native
  logs use matching `hallway-*.log` names. Runs and plans are cross-checked by
  path and capture count; tapes/moments are SHA-256 identified in each plan.
- Selected W0/W1 source sweep PNGs were compacted after complete assessment to
  save disk. Each affected run's `capture-archive-index.json` records original
  hashes, retained/deleted identities and the completed assessment hash. Native
  tapes, all plans/measurements, references, warm controls and selected response
  captures remain; omitted images can be replayed.

Focused checks: eight structure tests (route, original house preservation,
palette and BVW/direct lookup parity included), two fixture frustum tests,
two preferred-display profile tests, thirteen Python calibration/distance
tests, and successful native builds. No toggle stress or unrelated full suite.
