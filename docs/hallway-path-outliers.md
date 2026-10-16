# Repeating hallway outliers: diagnosis and guiding experiment

## Reproduced failure (2026-10-16)

Production baseline: `c02d99a`, Apple M4 / Metal, macOS 26.6.2, development
profile, preferred corner display with uncapped 2048-evidence reconstruction.
Three saved 182-second live hallway runs repeat the same high-energy batch at
floor receiver `(-54,-12,-20)`, face 2, sequence 2240–2255. In the latest run
(`live-hallway-1792161245235760000-71464`), that batch contributes about 82% of
the receiver's accumulated red energy at the end. Its mean is about 2.9 times
the two-million-path reference, despite continuing to receive new samples.
The adjacent edit does not fix a deterministic undersampled path distribution.

`hallway_outlier_paths` reproduces the batch independently on CPU and GPU.
Path 2243 travels from the distant floor to its nearby ceiling, then 74 voxels
back toward the lamp. A surviving roulette step leads to a terminal connection
near the lamp. The contribution is large but physically valid. CPU/GPU agree;
the batch merge uses the ordinary 16/new-history weight. This is evidence of
severe variance, not evidence that all reported bad faces have the same cause.
The original isolated fixture has a different global emitter distribution, so
its exact contribution differs from the streamed world's.

## Experimental proposal, not a lighting-model change

The disabled-by-default `emitter_guiding` setting mixes cosine sampling with a
smooth source-directed spherical lobe. A task chooses a geometry/source-only
anchor; every path uses its explicit PDF in throughput and emission MIS,
including terminal connections. No radiance clipping, borrowed samples,
history resets, or emitter falloff changes are involved. All five bands use
the same unbiased weights. A nonzero cosine component preserves support.

The lobe uses the [Henyey–Greenstein distribution](https://pbr-book.org/4ed/Volume_Scattering/Phase_Functions)
as a proposal, **not** as a physical medium. This follows the importance-sampling
principle described in [PBRT's path tracer](https://pbr-book.org/4ed/Light_Transport_I_Surface_Reflection/A_Better_Path_Tracer).
It is source-informed analytic guiding, not the learned SD-tree algorithm in
[Practical Path Guiding](https://jannovak.info/publications/PathGuide/index.html).

Anchor selection uses the existing light hierarchy but never changes its
physical light-selection probabilities. A bounded search tests up to eight
candidate voxel centers for visibility. Concentration inside a 16-voxel source
region controls proposal strength; distant, occluded sources must not dilute an
isolated lamp simply because they share a global tree. The small flat-light
distribution uses its strongest projected source directly. The fixed radius
and concentration/focus scales are measured proposal heuristics, not physical
lighting distances or analytically established universal optima.

Rejected stages remain in experimental commits: secondary-bounce-only guiding,
hard cones, and global concentration each failed important comparisons. The
loaded hallway fixture includes the disconnected cave's emitters: without it,
an apparently good guide chose an occluded cave torch instead of the hallway
lamp. An isolated-room win is not a gameplay win.

## Raw evidence so far

`conditional_guiding_prefix_study` uses four disjoint sequence windows, no
display smoothing or spatial borrowing, and two independent 1,048,576-path
control references. References are block averaged below the history cap.
The `hallway_loaded` fixture has 237 floor receivers; the cave samples its
existing research wall/ceiling receivers. These are sparse raw-field metrics,
**not FLIP**, full-image certification, or windowed FPS.

For the radius-2 candidate with distance focus denominator `256 * radius²`:

| Fixture / paths | Relative RGB norm error p95, control → candidate | p99 |
| --- | --- | --- |
| Loaded hallway / 2048 | .991 → .273 | 3.394 → .505 |
| Loaded hallway / 8192 | .729 → .166 | 1.611 → .220 |
| Cave / 2048 | .028 → .029 | .037 → .035 |
| Cave / 8192 | .014 → .014 | .018 → .019 |

Evidence: `target/conditional-guiding-focus256-study.json`, shader diff from
`c341b1c` changes only focus 64 → 256. References reused explicitly from
`target/conditional-guiding-loaded-study.json`; bad supplied reference files
fail rather than silently regenerating them. The cave is approximately neutral,
not a claimed sampler improvement. Dark hallway relative errors remain material.

CPU/GPU matched 16-path prefixes pass for hallway-with-competing-lights and cave
receivers across three windows. Proposal integration against constant incident
light and the terminal emitter's numerical area integral pass, including all
five bands. Runtime opt-in: `VOXEL_GI_SETTINGS=config/gi-guiding.json`.
Actual-dispatch throughput and live streamed captures must be qualified before
changing defaults. The first live run was concurrent with a test build: use it
for lighting diagnosis, not comparative performance claims.

## Gameplay qualification and rejected shortcut

The initial live guided hallway (`live-hallway-1792166631031324000-74640`)
reduces the 180-second p95 raw relative RGB norm error .489 → .182 against the
saved frozen reference. Its neighboring cave run (`live-cave-1792166920149121000-74833`,
`1f828bd`) has clean captures but traces only 309M paths versus the saved 483M
control. Neutral cave accuracy per path does not justify that global throughput
cost. These results did not qualify a default.

`80d8633` tried stopping the anchor search when a candidate was weak. **Rejected**:
the full-world hallway run `live-hallway-1792167337797769000-75058` stops before
finding its lamp and reproduces the bright batch. The queued cave run was
terminated once this failure was known; it is not completed evidence.
Weakness of one candidate cannot certify that remaining candidates are weak.

This exposed a fixture limitation: the original `hallway_loaded` includes only
part of the competing cave. The corrected `hallway_loaded_full` includes its
east side and ceiling/crystals. Its reference is newly computed, not relabeled
from the partial scene. `target/conditional-guiding-front-study.json` gives:

| Full competing cave / paths | p95 relative RGB norm, control → guided | p99 |
| --- | --- | --- |
| Hallway / 2048 | 1.022 → .278 | 2.934 → .600 |
| Hallway / 8192 | .743 → .172 | 1.930 → .229 |

The revised optimization rejects the entire guide query only when the light
tree's bounds lie behind the receiver's hemisphere. That cannot suppress the
hallway lamp search; the bounded visible-source search remains intact. Weak
mixtures are disabled only **after** selecting an anchor. Each face shares its
anchor across cooperative lanes, and zero-guide PDFs avoid lobe arithmetic.
This is still proposal selection, never occlusion culling of transported energy.

The exact offending 16-path batch in the full fixture changes from red .05073
to .000114; a focused regression records this, alongside native CPU/GPU parity
including the offending receiver. Constant-illumination and area-integral tests
protect the PDF/energy contract. This single batch is not an accuracy certificate;
the disjoint-window and full-world comparisons remain necessary.
