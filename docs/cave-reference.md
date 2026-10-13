# Cave reference and estimator diagnosis

The real-time cave still has objectionable noise. Earlier convergence reports
sampled 36 wall faces and 144 reconstructed corners; they did not establish that
the entire rendered chamber had converged. A full reference image is needed
before tuning more history or filtering.

## Independent estimator check

`validation/reference_oracle.rs` implements an independent cosine-only Monte
Carlo integral for the current finite-depth transport target. It uses a separate
SplitMix64 random stream, no light tree, no emitter selection, no emitter PDFs,
no MIS and no temporal history. Accumulation uses f64 sums. It shares scene
traversal and source/material snapshots; scalar DDA parity tests cover traversal
separately. It is not independent evidence of the storage or rasterizer.

The target includes analytic light at each of the three diffuse surface hits
and one final emission connection. This matches the production terminal-emission
policy; it is not infinite-bounce physical ground truth. Source visibility,
finite reach and missing-region policy remain relevant to the target.

On Apple M4 / Metal, development test profile, no viewport: eight isolated cave
locations, 262,144 independent CPU paths each, compared with 262,144 production
GPU paths each. GPU means average eight independent 32,768-path blocks in f64
to avoid the 65,536-history ceiling. All 40 band comparisons pass the deliberately
conservative diagnostic tolerance (8 CPU standard errors + 0.5% mean + .001).
The check does not certify 0.5% image accuracy or rule out errors elsewhere.
The independent open-sky analytic check also passes for all six orientations.
Evidence: `target/cave-oracle-{analytic,native}.log`. Tested on the `12a6866`
baseline plus the oracle patch committed with this document. No lighting
algorithm, budget or history policy changed.

## Whole-chamber rendered reference

Run from the repository root in a development build:

```sh
RUST_LOG=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_PROFILE=cave VOXEL_GI_VALIDATE_BACKGROUND=1 \
VOXEL_GI_VALIDATE_REFERENCE_IMAGE=1 VOXEL_GI_VALIDATE_EXIT=1 \
VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD" cargo run --locked --offline
```

After the ordinary 62-second workload, the diagnostic freezes GI system
publication and holds the camera and transport scene. The background loading
camera stops moving. It captures the frozen live image, then computes every
chamber-facing exposed surface, including floor, ceiling, spire and emitters,
with a coplanar halo for the actual corner renderer. Terrain outside this
reference region retains its frozen live values. The rasterizer, textures,
camera, exposure, emission and shared-corner sun remain the regular game path.

The reference fixture also holds a stationary, non-rendering loading camera near
the chamber floor. Before baking, it compares every chamber voxel against the
unedited host generator and writes `cave-geometry-audit.json`. Unknown or wrong
material cells abort the reference. This completes reference-fixture residency
through the existing host API; it changes neither BVW nor normal gameplay.
Do not compare its frame costs directly with the older single-view workload.

Two disjoint sequence ranges each provide 262,144 paths per face. Each estimate
averages independent 32,768-path blocks in f64, below the production history
ceiling. The bake bypasses adaptive scheduling, invalidation, history retention
and temporal display filtering. Equal reconstruction confidence represents the
settled target. Eight independent cosine-only CPU integrals also check the exact
composed scene's floor, ceiling, walls, spire and source surfaces.

The output directory contains `live-frozen.png`, `reference-replica-a.png`,
`reference-replica-b.png`, `cave-reference.png` (their mean) and
`cave-reference-held.png` (the same mean captured again). Compare replicas to
check reference uncertainty, and the held captures to check publication/render
stability. `cave-reference.json` records every reference face, original live
display/history, both estimates, the exact scene generation and oracle checks.
`cave-reference-status.json` is written after all five capture guards complete.
The earlier `report.json` completion describes only the real-time workload.
An unqualified bake returns a nonzero exit after saving its diagnostic images.

An empirical qualification requires eight passing independent estimator checks,
replica RGB difference p99 <=1%, and maximum <=5%, with a .02 RGB norm floor.
These are agreement checks, not mathematical confidence guarantees. A failed
qualification must be reported even if all captures completed. Increase
`VOXEL_GI_REFERENCE_PATHS` (multiples of 32,768 through 1,048,576 per replica)
if replica disagreement is too large. A 32,768-path pilot exercises the capture
workflow cheaply. This offline bake has no real-time performance claim and does
not change normal gameplay resources when its flag is absent.

## First rendered pilot

Clean `3ef744d`, M4/Metal/development, 2560×1440 headless, preferred corner
look and background terrain: `cave-1791896155979103000-48231` bakes 14,572
exposed chamber/halo faces with 32,768 paths per replica (954,990,592 paths
total). All five reference-stage capture guards pass with clean render logs.
The mean and held-mean PNGs have identical SHA-256 hashes, proving this frozen
publication renders the same image again. The mean and live images were
visually inspected. The pilot is a preview, **not a qualified reference**:
replica RGB disagreement p99 1.817%, max 3.294%; one ceiling IR comparison
fails its independent-estimator tolerance.

Review found the newly authored CPU checker initially used visible-geometry
policy on its first ray; production uses conservative missing-region policy
for all diffuse rays (and bounded transport for shortened reach). The checker
is corrected to use those same visibility semantics. This affects the new
diagnostic only. The earlier isolated cave has complete nearby residency, so
its comparison did not expose that mismatch. The high-sample composed-world
run must still qualify the IR comparison; the mismatch is not assumed to
explain it. Evidence: `target/cave-reference-pilot.log` and its output directory.

## Geometry audit supersedes earlier images

The 262,144-path run on `71c8077`,
`cave-1791896665382806000-48456`, passes numerical checks on its frozen GI
snapshot: p99/max replica difference .6229%/1.1808%, all 40 independent CPU
band comparisons pass, held mean PNGs identical. It still is **not** accepted
as the complete authored-cave reference. Its reference-face pages exposed a
residency gap. The follow-up audit on `7d0c303`,
`cave-1791898306410276000-49018`, finds 2,240 unknown chamber cells in page
`(1,-2,-1)`, including part of the lava floor. BVW reports that page absent and
the sampled voxels unset; there are no wrong-material cells. This confirms
incomplete fixture geometry, not the cause of every live-noise symptom.

The user requested no BVW changes. A trial source patch was removed before
commit; `Cargo.toml`, `Cargo.lock` and both existing BVW checkouts are unchanged.
The opt-in fixture loading camera is used instead, with the strict voxel audit
retained. Earlier images are frozen-snapshot previews, superseded for the full
reference. Numerical agreement alone cannot certify complete world geometry.

`tools/cave_reference_analysis.py RUN --pixels` reports full-chamber centers,
58k-plus reconstructed corners, per-orientation histories and image differences.
The optional pixel checks need Pillow. It requires both a passing geometry audit
and the independent numerical qualification before `reference_qualified` can be
true. Sample counts, a still image and workload completion alone cannot pass.

## Qualified complete-chamber reference — October 13

Clean `8c0807d`, Apple M4 / 16 GiB / Metal, macOS 26.6.2, Rust 1.95,
development profile (application opt-level 1, dependencies 3), Bevy 0.19.1 and
unchanged registry BVW 0.17.0. No compiler or second native workload ran during
capture. Subsequent `f752966` changes documentation only. The viewport is
2560×1440 offscreen, textures on, shared-corner sunlight, reconstructed indirect
corners and baked AO off. The live prelude uses the unchanged defaults: 16 paths
per dispatch, 4 ms transport budget, 75 ms display response, three diffuse hits
and 768-voxel reach. Background terrain and the reference residency camera are
enabled. This is an accuracy bake, not interactive frame-pacing evidence.

Run `cave-1791899368791235000-49592` passes the chamber geometry audit: all
125,440 authored cells are known and have the expected material. It bakes
15,163 exposed chamber/halo faces with 262,144 paths per replica, 524,288 in the
mean (7,949,778,944 paths total). All 40 independent CPU band comparisons pass.
Replica relative RGB difference p50/p95/p99/max is
0.1768% / 0.4156% / 0.5625% / 1.1184%. All five reference capture guards pass;
render logs have no warnings or errors. The mean and held-mean PNGs are byte
identical. The reference and live images were visually inspected.

The two rendered replicas differ by an average 0.0447/255 per RGB channel;
99% of pixels differ by at most one 8-bit channel level, and none by more than
two. This gives a stable visual target for the current finite-depth model,
not a claim of infinite-bounce or mathematically exact physical illumination.
Evidence remains under ignored `target/`: `cave-reference-complete.log`,
`cave-reference-complete-analysis.json` and the run's `gi-validation/` directory.
Its final image is `cave-reference.png`; `live-frozen.png` is the matched live
comparison. `cave-reference-status.json` reports a qualified completed bake.

At the 62-second freeze, the 15,156 chamber faces have live center RGB error
p50/p95/p99/max 3.365% / 8.176% / 11.734% / 30.551% versus this reference.
Across 60,624 reconstructed corners those errors are
1.742% / 4.465% / 6.747% / 22.345%; 99.843% are within 10%. A single frozen
image does not measure temporal noise. All 15,163 live face histories are only
272 (12,597 faces) or 288 (2,566 faces) retained samples despite the minute-long
prelude. This is consistent with repeated geometry-revision history reduction
keeping a nonzero noise floor while distant terrain streams. It is a strong
diagnostic lead, not proof that retention is the sole cause, and not a reason
to preserve stale history after every meaningful local edit. Runtime sampling,
history and filtering policies have not changed in this reference branch.
