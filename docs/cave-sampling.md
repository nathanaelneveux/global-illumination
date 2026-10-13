# Cave sample requirements and perceptual accuracy

This is an offline fixed-work accuracy study, not a gameplay scheduling, edit
response or frame-pacing benchmark. BVW and normal transport defaults are
unchanged. Its purpose is to determine how much *retained useful history* the
current estimator needs, then provide a common target for variance-reduction
experiments. More submitted paths do not replace history discarded by resets.

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
