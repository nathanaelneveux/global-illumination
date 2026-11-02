# Cave sample requirements and perceptual accuracy

> Historical: the sample sweep and its analysis tools were removed on 2026-10-23
> with the world display (last present at `3cd98b2`).

This is an offline fixed-work accuracy study, not a gameplay scheduling, edit
response or frame-pacing benchmark. BVW and normal transport defaults are
unchanged. Its purpose is to determine how much *retained useful history* the
current estimator needs, then provide a common target for variance-reduction
experiments. More submitted paths do not replace history discarded by resets.

**Current visual target:** subsequent user review accepts the 2,048-path
appearance, with 1,536 noticeable and 1,024 very noticeable. Use the versioned
[human-calibrated profile](#user-calibrated-visual-target) for appearance/performance
experiments. Original engineering gates/results remain separate diagnostics.

## Exact target and workload

Use the [complete geometry-audited cave reference](cave-reference.md), adding
`VOXEL_GI_VALIDATE_SAMPLE_SWEEP=1` to its command. The same frozen scene produces
both the low-sample prefixes and a new qualified 524,288-path reference. Never
compare a new scene against an old frozen scene without checking the target.
The bake checks all authored chamber cells before sampling and independently
checks eight CPU integrals afterward. A failed reference invalidates the sweep.

Each chamber/halo face receives 16, 32, 64, 128, 256, 384, 512, 768, 1,024,
1,536, 2,048, 3,072, 4,096, 6,144, 8,192, 12,288, 16,384, 24,576 and 32,768
paths. Four disjoint sequence windows start at 0, 574,117, 1,853,903 and
3,102,719, away from the reference ranges and below the sampler period. Include
unaligned windows: a history restart can occur anywhere in a stream. These are
not independent random scrambles and do not establish confidence bounds.

The ordinary GPU kernel, sources, materials and finite three-hit estimator are
unchanged. Prefixes accumulate without resets, scheduling, adaptive sleeping or
temporal display filtering, below the statistical history ceiling. All are
rendered with equal corner confidence through the game's actual textures,
material, exposure and tonemapping. This isolates sampling quality, not startup
bootstrap, temporal lag or the quality of an IR/radiation simulation model.

`cave-sample-sweep.json` keeps all five bands at every face/checkpoint/window in
the same order as `cave-reference.json`. PNGs are named
`samples-<paths>-window-<index>.png`. The final status must pass all 81 reference
and prefix capture guards, including the frozen live capture. The harness
retains one publication probe map, not 80 cloned maps, while rendering frames.

## What accuracy means

The previous RGB metric is `length(testRGB-referenceRGB) /
max(length(referenceRGB), .02)`, evaluated in linear transport units. It checks
energy/color error, not what a viewer sees. Preserve it for solver correctness
and report per-band IR/radiation errors separately. Connected four-face corner
reconstruction uses the same geometric connectivity as the shader. It does not
add another blur to the reference.

HSL separates useful editing controls, but is not perceptually uniform: equal
HSL lightness or hue distances do not imply equal perceived differences.
[W3C CSS Color 4](https://www.w3.org/TR/css-color-4/#the-hsl-notation) documents
this explicitly. We instead use [OKLab](https://bottosson.github.io/posts/oklab/)
for per-pixel lightness and color differences after sRGB decoding, and the
official [LDR-FLIP implementation](https://github.com/NVlabs/flip) for perceptual
rendered-image differences. FLIP models color and spatial features rather than
just independently comparing RGB channels. Apply LDR-FLIP to the actual
tone-mapped PNGs; HDR-FLIP's exposure sweep would answer a different question.

Display sampling is explicitly fixed at FLIP's standard 67 pixels per visual
degree, independent of the *camera's* FOV. This is a reproducible assumed viewing
condition, not a measurement of the user's monitor. Keep image dimensions,
camera and display assumptions fixed. Perception at one view does not certify
every unseen cave surface, and 8-bit captures have a quantization floor.

For real-time flicker, [ColorVideoVDP (SIGGRAPH 2024)](https://www.cl.cam.ac.uk/research/rainbow/projects/colorvideovdp/)
is the more relevant extension: it jointly models spatial/temporal luminance
and color sensitivity under explicit display/viewing conditions. It is not run
by this prefix study. Static FLIP or OKLab scores alone cannot certify temporal
stability; runtime edits/streaming must still be qualified separately.

## Predeclared thresholds

These are engineering tolerances, not universal definitions of visual
invisibility. Report the curves as well as the threshold crossings.

- Practical whole-cave RGB: 99% of reconstructed chamber corners within 5% of
  reference, no corner above 10%, in every sequence window.
- Strict whole-cave RGB: 99% within 2%, no corner above 5%, every window.
- Near-reference captured view: OKLab difference p99 <= .01 and max <= .02;
  FLIP mean <= .01 and worst 32×32-pixel tile mean <= .05, every window.

The .02 OKLab difference is a common approximate JND scale used by
[W3C's gamut-mapping procedure](https://www.w3.org/TR/css-color-4/#binsearch);
it is not a visibility guarantee for images or temporal noise. FLIP has no
universal JND cutoff; its tolerances above are conservative comparison gates
that still need visual confirmation. Worst-tile pooling prevents a localized
lighting artifact from disappearing in a whole-image average. The reported
minimum is the **lowest tested checkpoint** satisfying its criteria at itself
and all larger checkpoints, across all four windows, not an exact universal
sample count. Dark bands use a .02 normalization floor in solver statistics.

## Analysis

`tools/cave_sample_analysis.py RUN --perceptual` needs NumPy, Pillow and the
official `flip-evaluator==1.7`. Keep optional analysis packages out of the game
and shared runtime; a local ignored installation is sufficient:

```sh
python3 -m pip install --target target/perceptual-python --no-deps flip-evaluator==1.7
PYTHONPATH=target/perceptual-python python3 tools/cave_sample_analysis.py RUN --perceptual
```

Use a Python environment already supplying NumPy/Pillow. The tool validates
reference qualification, geometry, scene generation, transport settings and all
four windows before analysis. `tools/test_cave_sample_analysis.py` checks known
OKLab colors, official FLIP identity/brightness response, connected corners and
the all-windows/persistent threshold rule. Large data/captures stay in `target/`.

## Full-resolution visual review

Analysis writes a small `sample-gallery.json` manifest beside the captures, so
the browser never needs to parse the 100+ MiB scientific face dataset. From the
repository root, serve locally with `python3 -m http.server 8765 --bind 127.0.0.1`
and open `http://127.0.0.1:8765/tools/cave_sample_gallery.html?run=<run-name>`.
The gallery includes every checkpoint and all four windows, original PNG links,
reference hold/wipe comparison and native-pixel zoom corrected for display
pixel ratio. It neither recompresses originals nor changes exposure. The local
server does not publish anything remotely. Stop it when review is finished.

`tools/cave_sample_contact_sheet.py RUN --window 0` creates a labeled overview
of every checkpoint and the reference. That overview is downsampled for
navigation only; evaluate subtle noise/gradients with the original captures.

## Measured baseline — October 13, 2026

Executable built/launched clean at `3331e20`; subsequent `a8ee3cc` changes only
Python LDR metadata handling. Apple M4 / 16 GiB / Metal, macOS 26.6.2,
Rust 1.95, Bevy 0.19.1, unchanged registry BVW 0.17.0. Development application
opt1/dependencies opt3, 2560×1440 headless, preferred corner sun/reconstructed
indirect, textures on, baked AO off. Background terrain and the reference-only
residency camera are enabled. The 62-second live prelude uses unchanged
16-path/4 ms/75 ms defaults; the sweep has no time budget, reset or temporal
display filtering. No compiler or second native GPU workload overlaps the run.
No interactive FPS or speed-to-convergence claim is made.

Run `cave-1791907675248389000-51327` traces 9,937,223,680 paths in total and
audits all 125,440 authored chamber cells
with zero unknowns/material mismatches. It samples 15,163 chamber/halo faces in
four windows and compares 15,156 centers / 60,624 corners against the same-scene
524,288-path mean. All 40 independent CPU comparisons pass; reference-replica
RGB disagreement p99/max is .5625%/1.1184%. All 81 reference-stage captures
(plus the three timed prelude captures) pass with clean logs. Repeated reference
mean PNGs are byte-identical. Full-resolution sample/reference images and the
contact sheet were visually inspected; Safari sample selection, window
selection, wipe and native-pixel zoom controls were exercised.

The **lowest tested practical threshold is 3,072 paths per face**. At 2,048,
the worst corner is still 11.97% off; at 3,072, every window has corner p99
<=2.63% and worst <=8.97%. Raw face-center RGB also passes the practical 5%/10%
gate at 3,072 (p99/max <=4.513%/9.926%). The **strict corner threshold is 8,192**:
p99/max <=1.539%/4.792%. Strict raw face centers require 16,384. These are finite
tested checkpoints across four windows, not universal guarantees.

Worst across the four windows at each count:

| Paths/face | Corner RGB p99 | Worst corner RGB | Image OKLab p99 | Image FLIP mean |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 10.764% | 39.292% | .02103 | .06029 |
| 512 | 7.143% | 29.265% | .01179 | .04625 |
| 1,024 | 4.832% | 18.030% | .00937 | .03668 |
| 2,048 | 3.266% | 11.969% | .00660 | .02740 |
| 3,072 | 2.629% | 8.971% | .00587 | .02318 |
| 4,096 | 2.271% | 7.227% | .00463 | .02104 |
| 8,192 | 1.538% | 4.792% | .00352 | .01704 |
| 32,768 | .752% | 2.368% | .00316 | .01013 |

The OKLab-only component passes at 1,024 (p99 <=.00937, max <=.01409), but
**the combined predeclared perceptual gate does not pass all windows through
32,768**: worst-window FLIP mean .010129 exceeds .01. Do not move the gate to
manufacture a pass. At 32,768, worst tile mean is .03051 and OKLab max .00614.
The two high-sample reference replicas themselves have FLIP mean .005983 and
OKLab p99/max .002816/.005125; 8-bit quantization and residual reference noise
matter at these levels. FLIP .01 is an engineering choice, not a published
visibility threshold. This failure does **not** establish that gameplay needs
more than 32k paths to look good. The gallery and human review should calibrate
the desired appearance, and a temporal study must still test runtime flicker.

Per-band raw face-center tolerances differ. With the same practical 5% p99 / 10%
max rule and .02 floor, IR first passes at 1,024; ionizing first passes at
16,384. At 3,072, IR p99/max is 2.414%/5.580%, ionizing 9.686%/30.130%.
Never use perceived RGB agreement to certify a simulation channel.

Fresh prefixes are not equivalent to the live cache's displayed sample count:
its capped statistical history and separate display filter retain discounted
older information. This study intentionally measures actual fresh accumulated
paths without those filters. Retaining history should eliminate one source of
persistent noise, but the frozen low-sample prefixes prove intrinsic sampling
variance also needs work. Increasing the work budget alone is not the result.

Next sampling candidates should aim to halve the *same-target* practical/strict
requirements to <=1,536 / <=4,096 paths, preserving all bands and source
gradients. Keep the common reference and thresholds fixed when comparing a
candidate; also inspect early 16–512-path errors and tails, not just the final
average. No sampler, filter, history or gameplay default is promoted here.

Evidence: `target/cave-sample-sweep-run.log`,
`target/cave-sample-sweep-analysis.json`,
`target/cave-sample-reference-analysis.json` and the run directory. Python
analysis uses the bundled Python with NumPy/Pillow and isolated FLIP 1.7 under
`target/perceptual-python`. Focused Rust tests, strict all-target Clippy and dev
build passed before launch; four analytic Python tests pass. No solved toggle
stress or broad native suite was rerun. All BVW source/dependency files remain
unchanged; these are diagnostic-only additions.

## User-calibrated visual target

On October 13 the user accepted **2,048 paths as visually indistinguishable
from the reference**, 1,536 as noticeable, and 1,024 as very noticeable. This is
an appearance target, **not a cap on useful history**. Live sample count/coverage
does not certify equivalent accuracy after resets or changing geometry.

`tools/quality_profiles/cave-user-2048.json` freezes the measured four-window
envelope of this qualified sweep. The user did not identify a sequence window;
the envelope is an automated robustness choice, not a claim they reviewed all
four. Keep these error limits fixed for subsequent samplers: a 2× efficiency win
now means reaching this same target in <=1,024 paths. Preserve older practical/
strict gates as diagnostics and qualify simulation channels independently.

Worst across the four windows:

| Paths | User assessment | Corner RGB p99 / max | Raw-face RGB p99 / max | FLIP mean / worst tile |
| ---: | --- | --- | --- | --- |
| 1,024 | Very noticeable | 4.832% / 18.030% | 8.409% / 19.679% | .03668 / .10453 |
| 1,536 | Noticeable | 3.843% / 14.686% | 6.643% / 15.090% | .03061 / .08128 |
| 2,048 | Accepted | 3.266% / 11.969% | 5.636% / 11.969% | .02740 / .07296 |

The combined gate uses raw-face/corner p99/max and FLIP mean/worst 32×32 tile.
All four 2,048 windows and every larger measured checkpoint pass; all 1,536 and
1,024 windows fail. FLIP is not a percentage RGB error. A 12% raw outlier does not
permit every surface to be 12% wrong or establish an invisibility threshold in
other scenes. This static target does not certify flicker, lag, overshoot, unseen
views, different exposure, or IR/ionizing simulation values.

OKLab stays diagnostic: its worst-window p99 at 1,536 is .006494, slightly lower
than .006601 at 2,048 despite the user's preference. One single-pixel maximum at
3,072 is .010543, slightly above .010362 at 2,048. A strict monotonic OKLab gate
would misrepresent this accepted appearance; the spatial/transport gates above
distinguish the observed levels without that assumption.

```sh
PYTHONPATH=tools python3 tools/visual_quality.py \
  target/cave-sample-sweep-analysis.json \
  --profile tools/quality_profiles/cave-user-2048.json --verify-anchor \
  --output target/cave-user-2048-acceptance.json
```

For a candidate use its qualified same-target analysis without `--verify-anchor`;
this evaluates fixed limits, not new limits learned from the candidate. That
flag specifically checks the original baseline hash, accepted envelope and
visible-failure controls. Generating reference/geometry/capture checks remain
mandatory. The older engineering gates and their historical results are not
overwritten or retroactively declared passes.

Verification at `d38c251` / `7ecb0c7`: 17 focused profile/history Python tests
pass. All 12 original 2560×1440 images for 1,024/1,536/2,048 were re-evaluated
with official FLIP 1.7 and OKLab; metrics reproduce to 1e-12. Geometry/reference
qualification, the 81 capture guards, nonblank images and identical held-reference
pixels were checked. Reference and 2,048 images were inspected. This reuses
existing qualified captures, not a new reference bake. Evidence:
`target/cave-user-2048-{acceptance.json,capture-check.log}`.

The [retention follow-up](sun-aware-retention.md#user-calibrated-follow-up) supports
this target while distinguishing a new goal from a better sampler. Runtime
integration remains separate.
