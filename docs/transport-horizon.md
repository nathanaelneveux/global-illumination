# Transport horizon study

The user's accepted cave appearance is around 2,048–4,096 fresh samples per
face. Reducing integrand complexity and avoiding unnecessary invalidation are
separate opportunities to reach that appearance sooner. A smaller sample count
or a full coverage counter alone does not establish equivalent quality.

## Long-range daylight fixture

`VOXEL_GI_SCENE=canyon cargo run --locked --offline` selects a finite
256×256×512 canyon: rough ground, approximately 220–240-voxel rims, stepped
isolated towers and a camera sightline longer than 256 voxels. Integer geometry
queries and the optical palette are shared with the fully resident numerical
fixture. The ordinary world remains the default; BVW is unchanged.

`VOXEL_GI_VALIDATE_PROFILE=canyon` selects its fixed-view headless validation
profile (15/30/60-second captures). These ordinary live captures are not a
settled reference. The native research fixture independently constructs every
page, including air, so unloaded geometry cannot masquerade as a shorter ray
limit. Its receivers cover ground, high rims and tower walls, including sites
beyond 400 voxels down the canyon.

## What the current knobs mean

`TransportSettings::reach` limits **each diffuse ray and voxel-emitter
connection**, not total path length or receiver-to-source distance. The current
baseline is 768 voxels and three explicit diffuse hits after the receiver, plus
the terminal emission connection. Directional visibility has a separate
768-voxel horizon; analytic point lights have their own source-defined ranges.
At a shortened diffuse cutoff, environment contributes only when the remaining
coarse occupancy frontier is empty. That test can also depend on distant pages.

Consequently, choosing reach 256 is not sufficient justification for a
256-voxel geometry-invalidation radius. Several legs can cross that radius, a
nearby bounce can receive distant sunlight, and the sky frontier can change.
A true receiver/source or total-path horizon would be a different transport
model and would need its own image qualification. Sun/sky occlusion dependencies
must remain explicit rather than being silently treated as local.

Current geometry changes use a 48-voxel near region with history capped to 16;
other receivers can still be capped to 256 on a transport revision, regardless
of distance. An evicted/untracked revision transition uses the conservative
64-history fallback. Repeated remote streaming can therefore prevent nearby
faces from accumulating the user's preferred amount of compatible history.
This study does not yet change that policy.

## Emitter geometry is already part of the estimator

An exposed voxel face is a one-unit-square area light. Explicit light sampling
uses its position, orientation and squared distance. Its solid-angle PDF is
`p_select * distance_squared / cos_emitter` (unit area); the receiver cosine PDF
is `cos_receiver / pi`. Balance MIS combines explicit light selection with
cosine directions, with matching weights when a diffuse path hits an emitter.
The spatial light tree also uses emitter bounds, angular extent and orientation
when proposing lights. Distance attenuation is therefore already present; it
is not missing from an emitter hit just because emitted radiance is constant.

These are standard [area-light sampling](https://www.pbr-book.org/4ed/Light_Sources/Area_Lights)
and [spatial light-selection](https://www.pbr-book.org/4ed/Light_Sources/Light_Sampling)
principles. Radiance along an unobstructed ray does not itself decay as inverse
square: the emitter's subtended angle shrinks. The point-light inverse-square
approximation is appropriate far from the source, not as an extra attenuation
factor on top of the area-light estimator.

The cache's empirical moments do **not** yet use a geometric upper bound on a
particular emitter's integrated contribution to estimate confidence. That is
worth investigating, especially to distinguish a rare direct-light hit from
large uncertainty. It must bound the actual estimator and account for all
emitters, visibility and reflected paths. Clamping a multi-bounce sample to
one emitter's direct-light bound would bias valid energy. A large lava pool
also contains many faces: a one-face bound is not a bound on the pool.

## Fixed-work matrix

The opt-in native `horizon_reach_depth_study` compares the unchanged production
kernel against the 768/3 baseline. Default reaches are 16, 32, 64, 128, 192,
256, 384, 512, 768; depths are 1, 2, 3. It covers analytic sky, analytic colored
emitters, a skylit room's floor and ceiling, the many-emitter cave, and canyon.
Two disjoint unaligned sample windows use matched sequences between models.
Means average blocks in f64 below the runtime history ceiling. No adaptive
scheduler, history cap, display filter, streaming or wall-clock-dependent
invalidation participates.

```sh
VOXEL_GI_HORIZON_PATHS=4096 \
VOXEL_GI_HORIZON_OUTPUT=target/horizon-screen.json \
cargo test --locked --offline horizon_reach_depth_study -- --ignored --nocapture
```

`VOXEL_GI_HORIZON_FIXTURES` optionally selects comma-separated fixture names.
`VOXEL_GI_HORIZON_MODELS=128:3,256:3,768:2` restricts confirmation to selected
reach/depth pairs; the 768/3 baseline is always measured. Increase paths for
candidate/boundary confirmation. Outputs include all five-band means, sites,
paired errors and baseline replica disagreement. Linear transport uses a
symmetric relative norm with a .02 floor. It is not a perceptual metric or
proof that every unmeasured voxel looks unchanged. Native GPU timestamp sums
are isolated compute work, **not interactive frame pacing**; a missing/invalid
timestamp makes the aggregate null, not zero. Candidate images
and whole-field checks remain necessary before changing defaults.

## Boundary confirmation (2026-10-13)

Clean executable/test commit `a461037`, Apple M4 / Metal, macOS 26.6.2,
development test profile, no viewport. The 27-model screen used 4,096 paths per
window; boundary confirmation used 65,536, two matched windows, 1,054 receiver
faces across the six fixtures. Evidence: `target/horizon-{screen,confirm}.json`
and their `.log` files. Both studies complete without render/compute errors.
The initial timestamp-abort is a harness failure, not quality evidence.

Worst across the two windows, **indirect RGB**, symmetric relative error:

| Scene / model | Mean difference | p99 | Maximum |
| --- | ---: | ---: | ---: |
| Isolated cave, 64 / 3 | 1.752% | 6.977% | 7.196% |
| Isolated cave, 80 / 3 | <.001% | .0012% | .0019% |
| Isolated cave, 96 / 3 | 0% | 0% | 0% |
| Isolated cave, 768 / 2 | 5.878% | 9.744% | 10.314% |
| Canyon, 256 / 3 | 2.354% | 8.233% | 10.503% |
| Canyon, 288 / 3 | 1.021% | 3.856% | 5.446% |
| Canyon, 320 / 3 | .510% | 2.343% | 3.560% |
| Canyon, 384 / 3 | .131% | 1.085% | 1.570% |
| Canyon, 512 / 3 | .0022% | .0669% | .1562% |
| Canyon, 768 / 2 | 9.746% | 29.403% | 34.300% |

The other four controls match exactly at reach 64 and above with three hits.
At reach 32 the skylit room floor loses 5.617% on average (20.847% p99); the
ceiling loses .917% (1.895% p99). Two hits loses 20.087% on average on the
room floor and 11.180% on its ceiling. Lower depth is not a general replacement
for reducing reach. This compares against the finite three-hit production
target, not an infinite-bounce ground truth.

The five-band screen matters: cave reach 64 has ionizing p99/max differences
14.23%/26.64%, despite its much smaller RGB error. Canyon reach 256 reaches
11.01% p99 in blue. Reach 288 is the lowest **tested** pair passing a practical
5%-p99/10%-maximum engineering gate in every band of every native fixture;
384 passes the stricter 2%/5% gate. These are sparse numerical gates, not a
claim of visual indistinguishability or universal safe ranges. The canyon
baseline's replica RGB disagreement at 65,536 is .839% p99 / 1.057% maximum;
the cave's is .975% / .983%. Matched paths strongly reduce noise in differences
at the same depth. Skylit room replicas remain noisier (6.94% maximum), but
same-depth models at reach 64+ match path-for-path there.

Shorter rays did not establish a compute win: finite-mode frontier checking
adds work, and empty space is already cheap. The main value could be a bounded
dependency model that preserves history, not a lower raw cost per path. No
default, history rule or transport kernel has been changed by this study.

### Composed-world images

The sparse isolated cave is not the cave embedded in the normal terrain. To
check that distinction without rebaking an unchanged high-sample oracle:

```sh
RUST_LOG=info VOXEL_GI_VALIDATE=1 VOXEL_GI_VALIDATE_HEADLESS=1 \
VOXEL_GI_VALIDATE_PROFILE=cave VOXEL_GI_VALIDATE_BACKGROUND=1 \
VOXEL_GI_VALIDATE_MODEL_COMPARISON=1 VOXEL_GI_VALIDATE_EXIT=1 \
VOXEL_GI_VALIDATE_EXPECT_BUILD_ROOT="$PWD" cargo run --locked --offline
```

After the usual workload, this freezes the actual composed snapshot and uses
the full authored-geometry audit and chamber/halo receiver set from the existing
reference harness. It measures 64/96/128/256/384 at three hits, plus 768/2 and
the 768/3 baseline, each with the same two 8,192-path windows. Images display
each pair's mean through ordinary corner reconstruction/textures/tonemapping.
`VOXEL_GI_MODEL_PATHS` supports 1,024–32,768 per window in 1,024 multiples.
The JSON retains every five-band face mean and both windows. This is a
matched model comparison, **not a new perfect-reference qualification**;
completion and numerical/visual acceptance are explicitly separate. Neither
ordinary live capture nor offscreen cadence establishes gameplay FPS.

Completed on clean binary `fe1ff32` (analysis-only Python tool added while it
ran), M4/Metal/development, headless 2560×1440, preferred corner sun and
corner-reconstructed indirect, textures on, baked AO off, background terrain
enabled. `cave-1791913761637216000-54996`: all 125,440 authored cells match;
15,163 face/halo sites, 60,624 reconstructed corners. All eight comparison-stage
and three workload capture guards pass; no warning/error logs. Each model uses
two matched 8,192-path windows, displayed as their mean. The baseline windows'
raw RGB disagreement is 3.638% p99 / 6.703% maximum; they are not a new
high-sample reference.

| Composed cave model | Corner RGB p99 / maximum | Rendered FLIP mean / worst 32px tile |
| --- | ---: | ---: |
| 64 / 3 | 6.580% / 14.426% | .02822 / .11094 |
| 96 / 3 | .01058% / .10539% | .000289 / .00559 |
| 128 / 3 | .00368% / .02756% | .0000694 / .00187 |
| 256 / 3 | 0% / .00125% | .00000147 / .000266 |
| 384 / 3 | 0% / 0% | 0 / 0 |
| 768 / 2 | 13.783% / 19.290% | .07748 / .11588 |

Reach 384's rendered PNG is pixel-identical to the matched baseline. The
two-bounce image has OKLab delta p99 .01538; 64/3 has .00931. FLIP uses the
official 1.7 implementation at 67 pixels/degree, a viewing assumption rather
than universal invisibility threshold. Transport metrics here use symmetric
normalization with a .02 floor. The entire cave's 64/3 ionizing differences are
larger than RGB (p99 12.84%, maximum 42.21%); visible equivalence alone cannot
qualify simulation accuracy.

Evidence: `target/horizon-composed-cave.log`,
`target/horizon-composed-analysis.json`, the run's JSON/PNGs, and
`transport-model-overview.png`. `tools/transport_model_analysis.py` validates
completion/audit/schema, reuses the sample study's tested corner topology and
OKLab/FLIP routines, and makes the contact sheet from original captures without
per-image brightness adjustment. Its new symmetric metric passes analytic
identity/symmetry/dark-floor checks. Focused native tuning CPU/GPU parity and
finite-frontier/no-sky-leak regressions pass after integration. The `tuning_`
filter also incidentally ran the UI-only offscreen test; no toggle stress or
whole-suite repetition was needed.

The separate `canyon-1791914038253548000-55066` live preview on `fe1ff32`
completes all three 2560×1440 headless capture guards with assets verified and
clean logs. It shows the fixture, not a settled or fully resident canyon
reference. Its cadence is not interactive FPS evidence. The canyon quality
numbers above come from the independent fully resident native fixture, not
from this live preview.

## Conclusion and next dependency experiment

Keep three hits. The lowest tested practical numerical candidate across these
fixtures is 288/3; 384/3 is the stronger quality candidate, while 512/3 makes
the measured canyon difference very small. No range is promoted yet: the
whole-canyon image, other composed-world views/source changes and a true
influence bound still need qualification. In particular, this study does not
claim that 256/3 or any smaller radius works for every procedural world.

The next architectural experiment should explicitly separate bounded diffuse
transport from sun/sky/point visibility dependencies. A total-path or
receiver-influence budget can bound the former; it changes the integrand and
must be measured against this baseline. Visibility dependencies may still
extend beyond it and need their own conservative tests/caches. The existing
direct-sun cache already uses spatial shadow sweeps rather than whole-world
invalidation. Equivalent dependency reasoning—not just a smaller number in the
distance-cap rule—is needed before preserving all history outside a radius.
The target remains retaining compatible thousands-of-samples history during
remote streaming while responding correctly to nearby edits and changed shadows.

Emitter size-aware proposal weighting is already implemented; size-aware
uncertainty/error bounds are not. [Lightcuts](https://www.cs.cornell.edu/projects/lightcuts/)
is a relevant precedent for bounding groups of lights to trade approximation
error against work. It is not the same as our stochastic light-tree proposal,
and applying its reasoning to cache confidence would be a new qualified
experiment, not a justification for clamping path samples.
