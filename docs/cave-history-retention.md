# Cave history retention experiments

Question: does scene-change history truncation prevent the many-emitter cave
from becoming accurate, and can waking receivers without deleting their history
improve it? Keep the estimator, presentation and work budget fixed first.

## Accuracy instrumentation

`VOXEL_GI_VALIDATE_REFERENCE=1` adds an offline reference **after** a headless,
exiting validation workload has saved its timings and captures. It freezes the
actual loaded scene, materials and sources and independently traces 262,144 paths
per fixed receiver, twice, using disjoint sample sequences. It writes
`reference.json`; absence/failure is not an accuracy pass. Reference work does
not belong to the reported timing interval. The ordinary application continues
running while that asynchronous worker finishes.
Eight independent 32,768-path blocks per replica avoid the production history
ceiling. The first preliminary runs at `c0bf1ec`/`200eed9` used one block instead;
their replica disagreement reached 5% at p95, so finer accuracy conclusions need
the stronger reference.
The corrected reference also begins at sequence 4,194,304, below the path
sampler's 2^24 period. The preliminary 16,777,216 offset aliased early path
directions (though emitter samples differed), so those runs are diagnostic,
not fully independent accuracy evidence. Subsequent references record their
sequence start and stride explicitly.

Use with `VOXEL_GI_VALIDATE_TEMPORAL=1`, then run
`python3 tools/reference_analysis.py target/gi-validation/<run>`.
The analysis reports RGB vector relative error (reference norm floored at 0.02),
replica disagreement, history size and time remaining within 10% until the end
of the run, with a minimum ten-second final dwell. Inspect disagreement before
interpreting accuracy. These are fixed **face measurements/display values**,
not rendered corner/pixel errors. `tools/temporal_analysis.py` separately measures
corner temporal variation and verifies continued accepted sampling.

The reference uses the production estimator and its configured finite path
model. It measures sampling error, not physical unbiasedness. Earlier scene
residency can differ from the frozen final scene: early errors are distance to
that final target, not time-varying ground truth. Reference replicas and matched
run targets should be compared before drawing conclusions from small differences.

Use the regular `cave` profile with `VOXEL_GI_VALIDATE_BACKGROUND=1`; the
optional `VOXEL_GI_VALIDATE_CAVE_LONG=1` extends it from 62 to 182 seconds and
adds 120/180-second captures. Headless app cadence is not interactive FPS.

No history or scheduling policy is changed by this measurement infrastructure.

`VOXEL_GI_VALIDATE_PATH_DIAGNOSTIC=<receiver-index>` additionally replays the
first 2,048 individual paths for one fixed receiver on both CPU and GPU, after
timing, on the same frozen scene. `reference-paths.json` records all five bands.
This distinguishes an estimator outlier from a backend/scene mismatch; it is not
independent accuracy evidence.

## Opt-in ablations

Debug builds accept `VOXEL_GI_EXPERIMENT_RETAIN_HISTORY=1`. This removes
scene/local/source-change statistical caps on compatible faces. It still wakes
stale receivers, rejects obsolete worker geometry, initializes genuinely new
faces, and restarts incompatible transport/material-table models. The ordinary
65,536-sample accumulator ceiling remains. This is not a production default:
old bright history can be stale after a real occlusion/removal.

Combine it with the existing `VOXEL_GI_CONTINUOUS=1` to disable adaptive sleeping
while keeping the same dispatch budget. Thus the comparison is (a) current caps,
(b) retained history with normal wake/audit eligibility, and (c) retained history
with continuous sampling. Turning off refresh entirely would make a frozen wrong
answer look stable and is not a useful convergence experiment.

## Terminal-emission variance experiment

The extended retention run exposed a separate variance source. At receiver
`(99,-18,-56)`, face 1, sample 726 produced RGB `(477.54,136.41,27.27)` on GPU;
exact-scene CPU replay gave `(477.61,136.43,27.27)`. The culprit was **terminal
light-only NEE**, at the third diffuse hit: its red estimate was 867.57 before
0.55 surface reflectance. The emitter proposal can undersample an important
connection and compensate with an enormous weight. Earlier bounces have a
competing cosine path and MIS; the terminal connection did not.

Candidate: draw that one terminal connection from an equal-probability mixture
of the light and cosine proposals. Evaluate `L * p_cos / (0.5*p_light+0.5*p_cos)`
for either proposal. This is the standard [single-sample balance MIS model in
PBRT](https://pbr-book.org/4ed/Monte_Carlo_Integration/Improving_Efficiency#MultipleImportanceSampling),
not a radiance clamp. It estimates the same terminal emission integral and
bounds an individual connection to twice the emitter's energy in each band.
Escaping sky/nonemitting hits still contribute zero to this terminal term; no
extra diffuse bounce, new history policy, spatial filter or per-face storage is
introduced. There is still at most one connection ray, but more proposals can
actually trace rather than reject cheaply, so equal ray *opportunities* do not
imply identical execution cost.

Initial M4/Metal isolated results: two 32,768-path estimates (old/new estimator,
distinct sequences below the path period) differ by 1.86% at all-band p95.
At 64/256/1024 paths, all-band p95 error falls from .4425/.1982/.0912 to
.4192/.1858/.0872. Median compute pass for 1,024 faces ×16 paths rises from
1.1795 to 1.2558 ms (~6.5%); 32 interleaved measured rounds after eight warmups.
These are preliminary fixed-work results, not a gameplay speed claim.

Verification: 13 tracer tests, including an independent area quadrature at all
three terminal depths and a per-band contribution bound; WGSL validation;
native 8,192-path CPU/GPU parity; native exact-dark removal/replacement/occlusion
response remains unchanged. Full streamed-cave evidence follows separately.
