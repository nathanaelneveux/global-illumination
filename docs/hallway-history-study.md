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
