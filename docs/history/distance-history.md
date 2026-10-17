# Distance-aware history qualification — October 13, 2026

## Decision

Repeated distant caps really can prevent a nearly settled cache from staying
settled. Retaining compatible history is a substantial win in the remote cave
control. The canyon supports much gentler reduction at long distances, including
beyond 384 blocks. **A universal 384-block cutoff is rejected**, however: a roof
roughly 500 blocks away can remove all the sunlight reflected from a nearby
floor into a receiver. That is indirect transport, not merely the separate
direct-sun display cache.

Keep production defaults unchanged while qualifying a dependency-aware policy.
This experiment changes neither BVW nor the renderer, transport kernel, history
defaults or scheduler. The previous [reach study](transport-horizon.md) is a
different experiment: limiting each path segment does not bound the effect of
loading arbitrary distant geometry.

Follow-up: the [sun-aware retention study](sun-aware-retention.md) combines
actual shared-corner sunlight changes with sampling-calibrated history curves.
Its conservative arm preserves the remote-cave win and baseline shadow/edit
responses; aggressive resets still regress noise. Runtime integration is pending.

## Current policy, precisely

For scene edits, loads and unloads, `geometry.rs` uses changed voxel bounds
(whole 16³ regions for residency changes). Existing means and sampling sequences
survive; on refresh their **effective statistical weight** is capped to:

- 16 within 48 blocks of a change;
- 256 everywhere else, with no outer radius;
- at most 64 if the eight-transition journal cannot reconstruct all changes,
  or if a stale face has no earlier probe revision to query. A known tighter
  local cap still wins.

These are not stored individual rays. Reducing a 4,096-sample mean's weight to
256 makes new samples more influential; it does not immediately change the mean.
All receivers remain eligible. Source/model changes have their own existing
revision semantics. Distance-based retention alone will not fix journal eviction:
a receiver can miss eight unrelated stream updates and lose the spatial evidence
that they were remote.

## Analytic relationship

Let `H` be retained weight, `N` subsequent fresh paths, `m0` the old stored mean,
`mu1` the new true value, and `mN` the mean of those fresh paths. Until runtime
history saturation, with no intervening cap:

```
estimate = (H*m0 + N*mN) / (H + N)
error    = H/(H+N) * (m0-mu1) + N/(H+N) * (mN-mu1)
```

This identity does not require independent samples. If the old estimate was
accurate and the true change magnitude is `D`, its stale contribution after `N`
paths is `H/(H+N)*D`. For an allowed stale contribution `epsilon`:

```
D <= epsilon: keep all available history
D > epsilon:  H <= N * epsilon / (D - epsilon)
```

This is a **bias budget**, not a bound on total sample error. Old uncertainty and
new sampling noise still count. For an ideal scalar independent-sample model, with old
mean variance `V0` and new single-sample variance `sigma1²`, mean squared error is
`[H²*(D²+V0) + N*sigma1²] / (H+N)²`. Its unconstrained optimum is
`H = sigma1²/(D²+V0)`, capped to available weight. Do not substitute our correlated
QMC samples and empirical moments and call that a certified confidence interval.
The native replays below measure actual tails separately. The standard
[Monte Carlo bias/variance treatment](https://www.pbr-book.org/4ed/Monte_Carlo_Integration/Monte_Carlo_Basics)
explains the statistical assumptions; the formulas here apply it to our weighted
history update.

Distance enters through a bound or estimate of `D`, not through that identity.
For a bounded diffuse patch of area `A` at minimum distance `d`, projected solid
angle gives a single-connection absolute change bound of
`Lmax * min(1, A/(pi*d²))`, where `Lmax` bounds the radiance difference over that
patch. Receiver/emitter orientation tightens it. Fixed area and bounded radiance
therefore motivate inverse-square change, and approximately fourth-power history
growth in the ideal MSE expression. Larger changed areas, different source
strengths, shadowed secondary surfaces and multiple connections violate a simple
distance-only model. A distant directional shadow need not diminish with distance.
This follows the geometry and visibility terms of the
[light transport equation](https://www.pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/The_Light_Transport_Equation),
not a new physical attenuation rule or permission to clamp sampled energy.

## Protocol and provenance

Native tape recorded on clean **`df3115d`**, branched from `d64fb90` as
`codex/gi-distance-history`. Apple M4 integrated GPU, 16 GiB, Metal,
macOS 26.6.2, Rust 1.95, Bevy 0.19.1, registry BVW 0.17. Development profile
root opt1/dependencies opt3, shared Cargo build directory. No viewport: native
compute only, no concurrent compiler during measurements. Runtime model remains
reach 768 / three diffuse hits. No gameplay-FPS or CPU-stall improvement is claimed.

For each before/after scene:

- two reference windows starting at 574,117 and 1,853,903, each 65,536 paths;
  f64 averages of bounded 32,768-path blocks, not the runtime history EMA;
- two live sequences starting at 17 and 131,119, each 4,096 warm paths followed
  by 512 independent batch records of 16 new paths (8,192 paths total);
- the same production path samples replayed for every policy and both change
  directions; reference means are never policy inputs;
- all five transport bands, with no spatial/display filter or direct-sun pixels.

The QMC windows are disjoint sensitivity checks, not independent random scrambles
or formal confidence intervals. "Independent batch records" means each record
starts with no accumulated history, not statistically independent random draws.

Fixtures:

1. A black 32×32 roof inserted at heights 64/128/256/384/512 above an open floor.
   Sixteen floor receivers measure diffuse-sky occlusion. A separate sun-only
   version uses sixteen vertical receivers facing the floor, measuring its lost
   sun bounce. Intervening air is resident; the 512-high roof's shadow ray is
   checked independently by a CPU geometry test.
2. The shared fully resident canyon, loading/unloading the complete last 64-block
   slab (`z=448..511`). Its 478 persistent receivers span near and remote geometry.
3. The shared many-emitter cave, loading/unloading a 64×64 roof at `y=512`.
   All 121 cave receivers are 469–503 blocks from the changed residency bounds.
   This is real geometry, not an event marker with an unchanged scene pointer.
   The repeat control alternates the roof every 128 fresh paths.

These are sparse native fixtures, not the complete composed-world cave reference
or a rendered whole-canyon comparison. They isolate policy effects cheaply.

Candidate event caps, always limited to available weight:

| Policy | Cap beyond 48 blocks |
| --- | --- |
| Baseline | 256 |
| Quadratic | `clamp(256*(d/96)^2, 256, 65536)` |
| Quartic | `clamp(256*(d/96)^4, 256, 65536)` |
| Cutoff384 | 256 below 384; retain above it |
| Retain control | Retain everywhere, including near changes |

The first four keep the existing cap of 16 within 48 blocks. Replays assume the
spatial change is known: they do not model journal eviction, asynchronous updates,
adaptive sampling, corner reconstruction or display smoothing.
Distances use each fixture's changed-volume bounding box, a conservative lower
bound on distance to the changed voxel/region union, not an exact replay of the
production planner's individual bins. The canyon's highest empty regions do not
themselves alter transport; enclosing the slab makes the spatial policy stricter,
not more permissive.

RGB error divides vector difference by the norm of the componentwise brighter
before/after reference, with a 0.02 norm floor. Each band uses its own brighter
reference with a 0.02 floor. This fixed normalization handles both addition and
removal without dividing by a newly dark target. It differs from the symmetric
reach-study normalization and is not a perceptual score.

Late tails pool faces and batches from 4,112 through 8,192 fresh paths within
each run; tables report the worst of the two live prefixes and both directions.
Settling means every tracked face stays within 5% RGB error through the remainder
of the tape, with at least 16 final good batches. An already accurate retained
cache can settle at zero additional paths. Neither result establishes seconds
of gameplay latency.

## How much did distant geometry actually change lighting?

Canyon results, with a conservative-for-this-measurement cap calculated from the
largest measured change in each bin. The last column permits no more than **2%
stale-change contribution after 256 fresh paths**, starting from 4,096 old paths:

| Distance to loaded slab | Faces | Mean change | Maximum change | Analytic retained-weight ceiling |
| --- | ---: | ---: | ---: | ---: |
| 0–48 | 58 | 14.068% | 42.810% | 12 |
| 48–96 | 52 | 3.221% | 22.192% | 25 |
| 96–192 | 125 | 1.274% | 5.403% | 150 |
| 192–256 | 60 | .427% | 2.043% | Keep 4,096 |
| 256–384 | 135 | .167% | 1.356% | Keep 4,096 |
| 384–512 | 48 | .035% | .212% | Keep 4,096 |

Bins are lower-inclusive/upper-exclusive; these are measured sites, not continuous
coverage of each interval. The table is an offline diagnostic, **not universal
safe caps or recommended production constants**. It excludes sampling noise and
uses the mean reference, not an upper confidence bound. Changing the acceptable
error, response budget, geometry or emitter power changes the answer. The strong
48–96 changes also show that today's cap of 256 can be too slow, not just too noisy
at long range.

Reference-window disagreement reaches 1.094% at the worst canyon face. Common
random paths make the *change* more precise: maximum disagreement between the
two paired change vectors is .0324% beyond 384, versus the measured .212% maximum
change. It reaches .601% in the nearest bin. These are repeatability diagnostics,
not guaranteed error bars.

In the cave, the maximum measured remote-roof change is only .00837%; paired
change disagreement reaches .01513%, so do not claim that tiny change has been
precisely resolved. Full reference disagreement is at most .983%. The important
result is that reducing history creates much larger errors than this change.

## Settled tails and overshoot

Repeated remote cave changes, equal fresh sampling:

| Quantity | Current cap of 256 | Retain beyond 384 |
| --- | ---: | ---: |
| Late RGB error p95 | 6.784% | 1.657% |
| Late RGB error p99 | 8.785% | 2.179% |
| Worst late RGB error | 14.958% | 2.984% |
| Largest bright excursion | 16.802% | 1.685% |
| Largest dark excursion | 12.063% | 1.948% |
| Late accumulated variation p99 | 2.94233 | .09369 |
| All faces sustainably within 5% | Not reached | Already inside; remain inside |

That is **4.03× lower p99 error** and **31.4× less late variation** in this
controlled test. Variation is accumulated RGB movement after the fixed
`v/(.25+v)` compression; it is not rendered flicker or an error percentage.
Overshoot/undershoot is relative luminance excursion outside both true endpoints
and the initial measured cache, using a .02 floor. Already-present initial error
is not charged as newly introduced overshoot.

Five-band late p99 errors (R/G/B/IR/ionizing) are
`8.63/12.48/29.00/3.59/25.77%` for baseline and
`2.26/2.90/6.50/.814/3.70%` for retention. Visible RGB agreement alone does not
qualify a simulation band.

Even one remote cave event causes baseline to leave the 5% band and need
1,280–1,840 new paths to stay back inside it; retention stays inside throughout.
After only one event both eventually have similar late tails (p99 2.407% versus
2.179%). Repeated caps are what make the much larger persistent difference.

For the real canyon load/unload, the quartic ramp preserves comparable late RGB
tails: p99 1.780% versus baseline 1.885%; all-face settling ranges 2,080–2,800
versus 2,080–2,880 fresh paths. This small difference is **not a qualified speedup**.
Worst early bright/dark excursions remain 26.80/25.29% with either policy because
near caps are unchanged. Unconditional retention everywhere fails: late p99
12.348%, with no all-face settling inside the tape. Retaining far history is not
an excuse to keep obsolete nearby lighting.

## Why not promote a distance-only curve?

Diffuse roof occlusion behaves as expected: its worst relative change falls
from 11.956% at height 64 to 3.242/0.819/0.372/0.211% at 128/256/384/512. The
analysis also checks these changes against independent midpoint area quadrature
of the sky gradient over the roof, not the path sampler. The maximum native versus
quadrature difference is .0249% of the brighter reference, across all five heights.

The sun-bounce control has a **100% change at every roof height**. At height 512,
all receivers are beyond 384. The roof blocks the sun reaching the floor that
they see. The cutoff and graded candidates keep all 4,096 old samples, leaving
about one third of that obsolete light even after 8,192 fresh paths. Their late
p99 error is 49.72%, versus baseline 6.23%. Baseline reaches the 5% band in
4,912–5,584 fresh paths; retention does not. A small sky-solid-angle bound around
the receiver would miss this large change to a nearby secondary surface.

No candidate is a universal winner. The results justify the intended architecture:

1. Preserve compatible remote evidence instead of imposing a global 256 ceiling.
2. Bound the **change in transport**, accounting for changed extent, radiance,
   and visibility dependencies through secondary surfaces. Use that bound and a
   response/error budget to choose history weight. A distance ramp is a prior,
   not proof. Do not solve this with gameplay material or scene-specific rules.
3. Make spatial change summaries survive unrelated streaming revisions (bounded
   per-region summaries or conservative compaction). Otherwise journal eviction
   will silently reinstate global caps of 64. Retention and freshness stay separate;
   genuinely changed receivers must still wake and old-scene jobs remain rejected.
4. Requalify the same paired tapes and actual streamed cave/canyon presentation
   before promotion. Add larger occluders/emitting areas and multiple simultaneous
   changes before calling an analytic bound general. Existing local emitter and
   skylight tapes remain useful response controls.

## Reproduction and verification

```
VOXEL_GI_DISTANCE_OUTPUT=target/distance-history-tape.json \
cargo test --locked --offline record_distance_history_tape -- --ignored --nocapture

PYTHONPATH=tools python3 -m unittest distance_history_analysis -v
PYTHONPATH=tools python3 tools/distance_history_analysis.py \
  target/distance-history-tape.json --output target/distance-history-analysis.json \
  --figure target/distance-history-overview.png
```

Python needs NumPy and Pillow (the optional overview uses the system Arial font).
The native test took 50.23 seconds, with no warnings/render errors. The geometry
control unit test and strict `cargo clippy --locked --offline --tests -- -D warnings`
passed. Five Python tests cover the bias law, cap boundaries, exact-dark/null
replay behavior and far-field area quadrature. The overview was visually checked
for readable labels and correct values. No unnecessary gameplay/stress run:
production behavior did not change, and these are numerical-policy results.

Native build and first analysis: `df3115d`, clean. Later analysis-only additions
record paired-change repeatability, independent roof quadrature, additional unit
tests and the overview; they do not change the replay policies or paths. Evidence
under ignored `target/`: `distance-history-native.log`,
`distance-history-tape.json` (SHA-256
`ed6ad0b71b212feb9bc6ce4f5492eaf08de42291056035c53fa004eabfd0cd40`),
`distance-history-analysis.json`, `distance-history-analysis.log`, and
`distance-history-overview.png`. Large artifacts remain local and ignored.
