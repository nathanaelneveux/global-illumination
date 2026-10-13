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

An empirical qualification requires eight passing independent estimator checks,
replica RGB difference p99 <=1%, and maximum <=5%, with a .02 RGB norm floor.
These are agreement checks, not mathematical confidence guarantees. A failed
qualification must be reported even if all captures completed. Increase
`VOXEL_GI_REFERENCE_PATHS` (multiples of 32,768 through 1,048,576 per replica)
if replica disagreement is too large. A 32,768-path pilot exercises the capture
workflow cheaply. This offline bake has no real-time performance claim and does
not change normal gameplay resources when its flag is absent.
