# Cave history retention experiments

Question: does scene-change history truncation prevent the many-emitter cave
from becoming accurate, and can waking receivers without deleting their history
improve it? Keep the estimator, presentation and work budget fixed first.

## Accuracy instrumentation

`VOXEL_GI_VALIDATE_REFERENCE=1` adds an offline reference **after** a headless,
exiting validation workload has saved its timings and captures. It freezes the
actual loaded scene, materials and sources and independently traces 32,768 paths
per fixed receiver, twice, using disjoint sample sequences. It writes
`reference.json`; absence/failure is not an accuracy pass. Reference work does
not belong to the reported timing interval. The ordinary application continues
running while that asynchronous worker finishes.

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
