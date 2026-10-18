# Edit retention: theory, prior work and cache re-derivation

October 17, branch `gi-surface-cache`. Question: the whole-range curve
`H(d) = (d/12)^4` was fitted to the old estimator's settling speed. Can it be
updated for the faster-settling cache, and is the idea already known?

## What the curve is

[distance-history](history/distance-history.md) derives it. After an edit,
a face keeps `H` samples of old mean `m0` and adds `N` fresh ones. Stale bias
is `H/(H+N) * D`, where `D` is the true change at the face. With per-sample
variance `sigma^2`, mean squared error is minimized at

```text
H* = sigma^2 / (D^2 + V0)
```

A changed patch of area `A` and radiance change `L` at distance `d` changes the
face's light by at most `D <= L*A / (pi d^2)`. Substituting gives `H ~ d^4`, which
is the curve's shape. The scale is then

```text
s ~ (L*A / sigma)^(1/2)
```

so it carries both the estimator's noise and the strength of a typical change.
[whole-range-history](history/whole-range-history.md) fitted `s = 12` over
hallway, house and cave edits with the old estimator.

## Prior work

The ingredients are established. Closest relatives:

- **Temporal radiance caching** (Gautron, Bouatouch, Pattanaik, IEEE TVCG 2007;
  [SIGGRAPH 2008 course notes](https://cgg.mff.cuni.cz/~jaroslav/papers/2008-irradiance_caching_class/09-pascal-temporal-notes.pdf)).
  A record's temporal weight is the inverse of its estimated change rate. The
  rate is estimated by reprojecting future lighting, and validity is capped at
  5–10 frames. This is the same "reuse ∝ 1/change" principle, with the change
  *measured*, not predicted from geometry.
- **Perception-guided GI for animation** (Myszkowski, Tawara, Akamine, Seidel,
  SIGGRAPH 2001; [ACM](https://dx.doi.org/10.1145/383259.383284)). A local
  energy-based metric stops photons from being reused over time where lighting
  changes quickly. This is per-region temporal reuse driven by the change.
- **Selective photon tracing** (Dmitriev, Brabec, Myszkowski, Seidel, EGWR 2002;
  [project](https://resources.mpi-inf.mpg.de/anim/spt/)). Pilot photons find the
  regions an edit affects; the Halton sequence's periodicity re-traces only those
  paths. This localizes updates; it does not weight retained history.
- **Change localization:** Drettakis & Sillion's line-space hierarchy
  (SIGGRAPH 1997; [abstract](https://history.siggraph.org/?p=117251)) finds
  radiosity links crossed by a moving object. Tawara et al. localize final
  gathering (VMV 2002, CGI 2004; [dissertation](https://publikationen.sulb.uni-saarland.de/handle/20.500.11880/25935)).
  Instant caching (Debattista et al., CGF 2009; [EG](https://diglib.eg.org/items/13214630-c8d3-4481-a622-77335fb7816b))
  recomputes invalidated paths.
- **Corrections instead of resets:** radiosity redistribution with negative
  energy (George, Sillion, Greenberg 1990; [Cornell](https://en-cg-web.coecis.cornell.edu/pubs/1990/GSG90.html)).
- **Measured temporal change:** A-SVGF matched-seed gradients
  ([HPG 2018](https://momentsingraphics.de/HPG2018.html)) and DDGI hysteresis
  ([JCGT 2021](https://jcgt.org/published/0010/02/01/paper-lowres.pdf)).
  Both were tested here and rejected as defaults
  ([adaptive history](history/adaptive-history-qualification.md)).
- **Optimal forgetting:** adaptive filtering's forgetting factor balances lag
  against noise, with `1 - lambda*` scaling as drift/noise. `H* = sigma^2/D^2` is
  the same trade-off for one step change.

Not found: an *a priori* solid-angle change bound, taken from the edit itself
with no extra rays, that sets an MSE-optimal per-surface history weight in a
persistent world-space cache. Several full texts (MPI, ACM) were not readable
here, so this is a gap in a limited search, not a novelty claim.

## Experiment

`retention_curve_study` / `retention_curve_study_cave` (`gpu/retention_study.rs`)
replay real edits on every exposed face of a sealed scene: 3,175 hallway faces
and 15,047 cave faces. Each arm:

1. Converge the pre-edit scene.
2. Apply the edit.
3. Cap each face's history by its distance to the change.
4. Track RMS error relative to that estimator's own converged post-edit answer,
   at 1, 2, 4 … 256 batches of 16 paths.

Score is the mean over those log-spaced checkpoints (early response and settled
noise weigh equally). The cache and a cache-disabled explicit shader run the
same arms. Analysis: `tools/retention_curve_analysis.py`. Apple M4, native, no app.

Edits:
- **Hallway lamp off/on:** the corridor's only light.
- **Cave torch off/on:** one wall torch among lava, crystals, spire, the other
  torches and a skylight.

Measured noise: on the cave's 15,163 faces at 2,048 paths, the cache's
per-sample relative variance is 0.42x the old estimator's (median; p10–p90
0.27–0.65). Predicted scale shift: `0.42^(-1/4)` = 1.24x.

Best distance scale per case (lower score is better):

| Case | Explicit | Cache |
| --- | --- | --- |
| Hallway lamp on | s = 9 (.295), 12 (.349) | s = 18 (.254), 15 (.256) |
| Hallway lamp off | grows without bound (scene goes exactly dark; reset is free) | same |
| Cave torch off | retain everything (.020); s = 6 .025 | s = 6 (.013); 12 .025 |
| Cave torch on | retain everything (.020); s = 6 .027 | s = 6 (.016); 12 .026 |

Findings:

1. **The method reproduces the old fit.** The explicit estimator prefers
   9–12 on the hallway lamp; the curve was independently fitted at 12.
2. **The cache shifts the best scale up about 1.5x in both scenes**, in line with
   the variance prediction (1.24x) and the perceptual one (1.41x). It settles
   faster, so it should forget more aggressively near an edit.
3. **One distance scale cannot fit both scenes.** The best scale differs about 3x
   between "the only light switched" and "one light among many". A distance-only
   curve is a compromise because `D` depends on how strong the change is relative
   to each face's existing light.

### Physically based retention

Compute both terms of `H* = c * sigma^2 / D^2` per face instead of fitting `s`:

- `D = flux / (pi d^2)`, with flux the edited emitter's luminance times its
  exposed faces (unit area).
- `sigma^2` from the face's own stored second moment.

A signed variant predicts the new state's noise as the face's relative noise
times its predicted new mean `max(M + signed D, 0)`.

| Case (cache) | (d/12)^4 now | Best distance | Physical, c = 1 | Signed, c = 1 |
| --- | ---: | ---: | ---: | ---: |
| Hallway lamp off | .209 | .036 (s = 32, still falling) | .016 | .016 |
| Hallway lamp on | .270 | **.254** (s = 18) | .339 | .331 |
| Cave torch off | .025 | .013 (s = 6) | **.009** | **.009** |
| Cave torch on | .026 | .016 (s = 6) | .012 | **.011** |
| Mean of four | .133 | .107 (s = 24) | .094 | **.092** |

The physical rule has no distance-scale parameter. It cuts error 2.3–2.7x
relative to today's curve wherever lit regions change, and wins the four-case
mean. It loses one case: lighting a previously **dark** region. There, the optimum is
`H ≈ r_new^2`, the squared relative noise of the newly arriving light. That is
low near the lamp (direct light) and high far down the corridor (rare bounce
paths), and nothing measured before the edit predicts it. Flat history floors
and a dark-noise fallback were tested and do not fix it; they only trade cases.
The distance curve encodes "far is noisier" for this case by accident.

## Measured noise with cache read-gating

Implemented in the resident passes, off by default (`TraceWork.physical_retention`
= c; per-face `change_prior` = predicted D):

1. **CPU.** For each face near an edit, predict `D = flux / (pi d^2)`. Flag the
   face only if its predicted history `c * sigma^2 / D^2` is below its current
   sample count, or if it was dark relative to D. Distant or small changes
   predict huge H, so they cost nothing and change nothing.
2. **Prepare pass.** A flagged face sets the high bit of its sample count. For
   that batch the trace treats it as uncached, so rays that hit it continue
   explicitly through the *new* scene instead of reading its stale cached light.
3. **Accumulate pass.** Keep `H = c * sigma^2 / D^2` samples. sigma^2 is the
   retained variance, or the merged batch's own variance if the face was dark
   relative to D (no retained noise to go on). Then clear the flag.

No extra rays or passes. Gating reads turned out to be essential with a cache.
Without it, the first batch after an edit reads neighbours' stale light and
keeps it: hallway lamp-off scored .30 instead of .013. Bounding by the
predicted D is what the global change detector lacked. Far, small changes can
never trigger a reset, which removes the false alarms on rare bright paths.

| Case (cache) | (d/12)^4 now | Best distance | Physical (sigma before) | **Measured + gating, c = 1** |
| --- | ---: | ---: | ---: | ---: |
| Hallway lamp off | .209 | .036 | .016 | **.013** |
| Hallway lamp on (into dark) | .270 | .254 | .339 | **.198** |
| Cave torch off | .025 | .013 | .009 | .010 |
| Cave torch on | .026 | .016 | .012 | .013 |
| Mean of four | .133 | .107 | .094 | **.059** |

c = 4 gives .065 (better cave, worse lamp-off). It is the first rule here
that beats every distance curve on the dark-room case as well.

## Recommendation

- Do **not** just retune `s`. The data reject a single scale.
- Measured noise with cache read-gating (above) is the candidate rule. Before
  promoting it, give ordinary block edits and terrain loads a flux bound. For
  loads, only terrain rising above a receiver can replace sky it used to see.
  Then qualify it in gameplay against the current curve, with streaming.
- For gameplay integration, journal events need a flux bound for ordinary
  blocks too. The cache already holds the outgoing light of the edited voxel's
  surfaces and their neighbours, and changed-sunlight origins already exist.

Paper potential: the derivation (`H ∝ d^4`, `s ∝ (flux/sigma)^(1/2)`), the
cross-estimator validation, and a parameter-free physical rule on a persistent
world-space cache form a coherent short contribution. It would need a proper
literature search (Google Scholar citations of Gautron 2007 and Myszkowski
2001), the dark-region fix, and more scenes before anyone claims it is new.

## Gameplay integration and validation

`measured_retention` (c, default 1 on this branch; `config/gi-retention-curve.json`
restores the whole-range curve). Journal events carry a flux bound per change:

- **Edits:** emission delta, plus the brightest outgoing light around the edit
  times about two faces per changed voxel.
- **Loads:** sky (or reflected sun) per unit of new terrain height *above* a
  receiver. Flat or lower distant terrain predicts no change.
- **Unloads:** zero.
- **Sunlight:** changed shared corners.
- **Unknown changes, and probe faces not selected in the batch:** the curve cap.

Same build, headless, background loading, Apple M4. Edit timings are the mean
of three pairs; stability is one pair each.

| Measure | Curve | Measured |
| --- | ---: | ---: |
| `response` emitter added, settle p90 | 2.49 s | **1.64 s** |
| `response` emitter removed, settle p90 | 1.06 s | 1.17 s (run spread 0.82–1.35) |
| `response` summed p90 | 3.55 s | **2.81 s** |
| `edits` summed p90 (1–2 faces per event) | 6.6 s | 5.6 s |
| `cave` settle p50 / flicker p95 | 6.9 s / .0061 | **5.0 s / .0052** |
| `live-hallway` settle p50 / flicker p95 | 9.6 s / .0486 | 9.6 s / **.0419** |
| `cave` traced paths | 225M | 196M |
| Frame p95 cave / hallway (offscreen) | 30.0 / 27.7 ms | 31.2 / 28.3 ms |

Streaming stability holds or improves, added light settles about 35% faster,
removals are unchanged within noise, and frame time is unchanged within run
noise. The cost is about 13% fewer paths in the busy cave: a gated face traces
full explicit paths for one batch. Edit probes track only 1–7 faces, so those
timings are directional. The first integration panicked on an air-only loaded
region (no top voxel); that is fixed and unit-tested.

## Offshoot: A-SVGF and DDGI on the cache (branch `gi-cache-asvgf`)

Same harness, rules ported from the earlier studies (`codex/history-ddgi`,
`codex/history-asvgf`):

- **DDGI:** relative hysteresis, applied on every merge. A batch moving > 80%
  resets history; > 25% trims 0.15 of the old fraction.
- **A-SVGF:** re-trace the last 16 or 64 paths with identical random numbers on
  the old and new scene, reading the live cache. Optionally average the
  gradients over coplanar faces within 14 voxels. Map the result to history by
  (1 - lambda)^4 or a 25% threshold, or plug the measured change into the MSE
  optimum.

| Edit (cache) | Measured + gating | Best A-SVGF | DDGI |
| --- | ---: | ---: | ---: |
| Hallway lamp off | **.013** | .30 | .26 |
| Hallway lamp on | **.198** | .34 | .44 |
| Cave torch off | **.010** | .014 | .056 |
| Cave torch on | .013 | **.012** | .058 |
| Extra paths per face per edit | **0** | 32–128 | 0 |

Without the cache, A-SVGF is excellent: hallway lamp-off .0001. With the cache,
its replays read neighbours' stale cached light, so it underestimates the change
and keeps stale history. Fixing that needs read-gating, which needs the edit
prediction to know what to gate. That is the measured rule plus 32–128 rays per
face, for one small gain. DDGI's per-batch disagreement under sampling noise
keeps trimming good history, even far from the edit. **Decision: measured noise
with read-gating; neither offshoot is integrated.**
