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
