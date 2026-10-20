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

## Accepted terminal-emission sampler

The extended retention run exposed a separate variance source. At receiver
`(99,-18,-56)`, face 1, sample 726 produced RGB `(477.54,136.41,27.27)` on GPU;
exact-scene CPU replay gave `(477.61,136.43,27.27)`. The culprit was **terminal
light-only NEE**, at the third diffuse hit: its red estimate was 867.57 before
0.55 surface reflectance. The emitter proposal can undersample an important
connection and compensate with an enormous weight. Earlier bounces have a
competing cosine path and MIS; the terminal connection did not.

Draw that one terminal connection from an equal-probability mixture
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
response remains unchanged. Strict all-target Clippy also passes. The streamed
measurements below justify keeping this sampler; history retention remains an
opt-in experiment.

## Streamed-cave measurements

October 10, 2026, Apple M4 (10-core GPU), 16 GiB, Metal, macOS 26.6.2,
Rust 1.95 / Bevy 0.19.1 / registry BVW 0.17. Development profile (root opt1,
dependencies opt3), shared Cargo intermediates, **2560×1440 offscreen**,
60 Hz ScheduleRunner, no window/FPS claim. Normal nearby structure, background
terrain loader on, preferred corner sun/reconstructed indirect, textures on,
baked AO off. Sixteen paths/dispatch, three diffuse hits, reach 768, 4 ms trace
target, 8,192 receiver limit, 2 ms scene budget, 75 ms display response and 50 ms
upload. No concurrent compiler/GPU benchmark during timed phases. Reference
work starts only after captures/reporting. No display smoothing or spatial
filter was changed.

The preliminary 62-second runs were:

| Policy | Build | Artifact under `target/gi-validation/` | 40–60 s corner variation p95 | Minimum fresh paths/site |
| --- | --- | --- | ---: | ---: |
| Current caps | `c0bf1ec`, clean launch | `cave-1791655951663803000-76803` | .08544 | 160 |
| Retain, adaptive wake | `200eed9`, clean | `cave-1791656073725980000-76954` | .03191 | 208 |
| Retain, continuous | `200eed9`, clean | `cave-1791656160562525000-77030` | .02889 | 176 |

The two retention policies had about 1,400 median samples by 60 seconds versus
288 for the caps. Raw RGB p95 error against the **preliminary, partly aliased**
32,768-path references was 11.67% / 8.04% / 8.22%, respectively. Those references
disagreed by ~5% at p95: do not interpret the difference between 8.04 and 8.22 as
a winner. Continuous sampling did not establish an advantage over adaptive
waking; keep the latter. These references were replaced with the stronger,
nonaliased protocol for the decision-quality runs below.

An initial 182-second retained run at `395656f`
(`cave-1791656295331335000-77165`) reduced late variation to .00679 but exposed
the outlier: one receiver was still ~31% too bright. The subsequent exact-scene
replay confirmed it, rather than accepting a quiet but inaccurate cache.

The nonaliased long-run comparisons, each 182 seconds, are:

| Policy | Raw RGB p95 error at 60 / 120 / 180 s | Worst raw error at 180 s | 160–180 s corner variation p95 |
| --- | --- | ---: | ---: |
| Old sampler, current caps | 9.03% / 12.01% / 25.06% | 32.09% | .06203 |
| Old sampler, retain + adaptive wake | 6.40% / 4.20% / 3.09% | 28.38% | .00413 |
| Terminal mixture, current caps | 8.21% / 11.84% / 9.69% | 12.43% | .05732 |
| Terminal mixture, retain + adaptive wake | 5.08% / 4.25% / 3.99% | 4.13% | .00652 |

Artifacts/builds:

- Current caps: `51180d7`, compiled clean; `cave-1791656681673059000-77423`.
  A temporary **test-only** diagnostic was edited while this ran; it was not in
  the running binary. 350.14M paths, 1,581 final loaded pages, app-cadence
  p95/max 26.25/59.33 ms after 15 s and excluding captures.
- Retain/wake: `51180d7` plus `target/cave-history-diagnostic.patch`,
  `cave-1791657063808640000-77749`. Changes only added post-timing CPU/GPU replay
  and temporary CPU term logging; the timed GPU solver was unchanged. 345.75M
  paths, 1,596 pages, cadence p95/max 26.48/74.31 ms. Reference pair disagreement
  p95/max 2.33/3.30%; 35/36 receivers within 10% at 180 s, **not all settled**.
- Terminal mixture: `a76b183`, clean, `cave-1791657616787537000-78143`. 327.29M
  paths, 1,568 pages, cadence p95/max 26.45/47.08 ms. Reference disagreement
  p95/max .413/.448%; old/new reference means differ by p95/max 1.06/1.66%.
  The suspect sample is .44258 red (CPU/GPU agree), and the largest red value
  among those 2,048 paths is 14.228 instead of 477.54. 35/36 raw receivers are
  within 10% at 180 s; repeated caps still prevent robust settling.
- Terminal mixture + retention: `a76b183`, clean code at launch (only this
  document changed during timing), `cave-1791657904944266000-78358`. 343.41M
  paths, 1,574 pages, cadence p95/max 26.40/38.33 ms. Reference disagreement
  p95/max .413/.449%. All 36 raw and displayed face values remained within 10%
  from 46.99 s through the end. Worst raw/display error at 180 s was 4.13/4.14%.
  Median retained history was 4,160 paths, versus 280 with the new sampler and
  current caps. The final capture was valid, nonblank and visually inspected;
  captures, assets, camera/target dimensions and render logs passed validation.

Each long workload's final-scene reference has two 262,144-path replicas at 36
fixed faces. Accuracy is RGB vector error, not tone-mapped pixel difference.
Variation is shader-equivalent connected-corner reconstruction at 144 corners;
it is independent of the reference and does not by itself certify accuracy.
All runs kept sampling: minimum new paths per fixed face over the final 20 s
were 112 / 80 / 144 / 128, respectively. The sampler consumes somewhat more compute
per path; it is a tail-error fix, not a free frame-rate or global throughput win.

### Why not make unconditional retention the default?

The native exact-dark test now includes a 2,048-sample retained history:

| After source removal/replacement/occlusion | Fresh paths for raw residual below 10% | RGB display below 10% |
| --- | ---: | ---: |
| Current local cap of 16 | 148 | 156 |
| Keep all 2,048 old samples | 18,560 | 18,688 |

All five bands follow `H/(H+N)`; the retained case uses 128-path measurement
batches, the spatial case four. These are path counts, **not gameplay seconds**.
The algebraic 10% crossing is around `9*H` fresh paths, so retaining more history
can be disastrous after a real change. The new terminal sampler passes the same
extinction test; it does not solve that history/response tradeoff. Logs:
`target/cave-history-extinction.log`, `target/cave-terminal-extinction.log`.

## Decision and remaining work

Keep the terminal mixture in ordinary CPU/GPU transport. It fixes a real
extreme-contribution failure mode without clipping energy, introducing a
source-specific exception or adding spatial blur. Isolated GPU work costs ~6.5%
more; the streamed headless runs do not establish an interactive FPS improvement.
Default history caps and adaptive wake/audit scheduling remain unchanged.

The combined sampler/retention experiment is the strongest static-cave result:
all tracked faces reached and sustained the 10% band by ~47 seconds; the original
control did not sustain it for every face within three minutes. Late corner
variation was ~9.5× lower than that control. This is evidence for these 36 faces,
this streamed scene and the configured finite-bounce model, not proof of whole-
world convergence. The independent reference replicas provide an uncertainty
check, not physical ground truth. A quiet retained cache with the old sampler
still had a 28% wrong face: do not choose policies on smoothness alone.

Unconditional retention is **not** accepted as the default because a real change
can leave it carrying obsolete energy for ~125× more fresh paths in the exact-
dark control. Continuous sampling did not establish a benefit over adaptive
waking. The next history experiment should discriminate meaningful changes from
unrelated streaming before discarding evidence, then validate both static-cave
accuracy and changed-scene response. Simply increasing history or never resetting
exchanges one failure for another. Spatial filtering may subsequently reduce
display variance, but cannot make stale transport accurate and must not count
interpolated values as fresh simulation samples.

Runtime/harness verification is recorded in the scoped commits through
`a76b183`. This closeout changes documentation only and reuses that verification;
large native logs, captures and replay data remain under ignored `target/`.
