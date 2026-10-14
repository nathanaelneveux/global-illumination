# Whole-range history decision

Baseline `3e6170f`, branch `codex/gi-whole-range-history`. This supersedes the
near-only recommendation in `history-decision.md`: that recommendation joined
a new near curve to a flat inherited floor and was not the whole-range solution
requested. Preserve its measurements, but do not promote that joined policy.

## Final decision

Select **one fourth-power law over the whole distance range**:

```text
H(d) = round((d / 12)^4)
retained_weight = min(existing_weight, H(d))
```

Keep the existing 65,536 statistical ceiling. The implementation saturates H
at that ceiling to avoid overflow; this cannot discard additional supported
history. Preserve the mean, second moment, sample sequence and displayed value
when reducing weight. Zero statistical weight is not a command to paint black:
the next incoming batch replaces the estimator, while the existing temporal
display filter remains active.

This is the selected **data-calibrated tradeoff**, not a physical theorem or a
proof of a globally optimal coefficient. Both parameters were free in the
120-curve screen, with final selection by native material images and distant
transport controls. No more coefficient sweep is required to make this decision.
The code in this branch implements and verifies the replay policy; **normal
gameplay defaults have not changed**. Live event/journal integration is a
separate implementation checkpoint, not a claimed result of offline rendering.

| Distance in blocks | Maximum retained statistical weight |
| ---: | ---: |
| 8 | 0 |
| 16 | 3 |
| 24 | 16 |
| 32 | 51 |
| 48 | 256 |
| 64 | 809 |
| 96 | 4,096 |
| 128 | 12,945 |
| 192 | 65,536 |

**48/256 is a result, not an anchor.** The tested alternative `(d/16)^4`
retains 81 there, and the other shortlisted curves differ again. There is no
branch, flat floor or inherited far equation at 48; the selected law continues
to 809 at 64 rather than staying at 256. Neither 278 nor 384 appears in it.
For a receiver holding weight W, history is unaffected at approximately
`d >= 12 * W^(1/4)`: about 80.7 blocks for W=2,048 and 96 for W=4,096.
That is an evidence-dependent retention distance, **not a light/ray cutoff**.

Distance means receiver-to-change distance, taking the nearest known change
for a multi-voxel event. Actual changed sunlight surfaces remain relevant
spatial evidence: a remote blocker can change nearby direct illumination.
Do not add separate emitter, wall, brightening or darkening curves. The distant
controls below exercise one law for geometry and changed-sunlight origins;
the native sunlight-patch replays install target sunlight directly, so they
do not certify live notification timing.

Keep 2,048 as the original cave's human-calibrated quality reference, not a
universal static sample guarantee or a perpetual history cap. The earlier
[cap and reference-sensitivity measurements](history-decision.md#reject-a-universal-2048-statistical-ceiling)
still apply. Coefficients should be requalified when sampling accuracy changes;
automatic rescaling from a sample-count target alone is not established.

## Decision protocol

- Fit a single monotonic function over the entire measured distance range.
  Neither 48/256, 384, nor the old heuristic cutoff is a constraint.
- Start with the two-parameter family `H(d) = (d / s)^p`, with both `s` and `p`
  free. Integer rounding and `min(existing_weight, H)` are representation and
  evidence limits, not spatial bands. Add model complexity only for a measured
  failure, not to preserve old constants.
- Reuse native path tapes. A cheap display-space error screen may shortlist
  coefficients; it cannot declare a FLIP pass or gameplay improvement.
- Use actual native-material FLIP for the final near/mid-distance decision,
  with the fixed human-calibrated gate, sustained acceptance, both edit
  directions, null-change controls and second-sequence validation. Use existing
  distant roof/canyon/streaming tapes as separate transport controls, not
  substitute them for perceptual measurements.
- Rank qualified sustained response first, report worst regressions, and favor
  lower early error when response work is similar. Keep scene families balanced.
  Do not hide missing tails or failing warm starts in aggregate timings.
- Keep the current large statistical ceiling, transport, display smoothing,
  source strengths and preferred corner look fixed. No new emitter-specific
  curve. Changed sunlight remains spatial evidence; distant geometry can alter
  nearby sunlit surfaces, so geometry distance alone is not enough.

This round finalizes the equation selection, not an unmeasured runtime/FPS
claim. Any required native-material checks use headless offscreen capture with
the existing strict geometry/reference/asset guards.

## Screening evidence

`whole_history_fit.py` evaluates 120 free coefficient pairs: scales
4/6/8/10/12/14/16/20/24/32/48/64 and exponents
1/1.5/2/2.5/3/3.5/4/5/6/8. A scale is the distance at which unrounded weight
equals one, not a radius. There is no constraint that the curve cross 48/256.
The seven existing native tapes cover hallway source/occluder, house
emitter/door/visible sunlight patch and ordinary/moderate cave sources.
Window 0 supplies the fit; window 1 is reserved for confirmation.

The cheap screen preserves native moment and display-filter updates but uses
compressed RGB mean/p90 error integrals on a deterministic distance-stratified
subset (at most 512 sites). It balances early/late error, cases and scene
families. It is **not** FLIP, actual corner/material reconstruction, a visual
pass, or the final selection criterion. These limitations matter: its best
score alone favors nearly clearing the entire hallway while worsening cave
early error.

The distant screen reuses all 12 roof/canyon/remote-cave cases, their native
paths and actual changed-sunlight distances. It rejects a candidate if an
existing baseline-settled control acquires a failing raw 5% tail. Sixty-eight
of the 120 curves survive window 0. Raw transport acceptance is deliberately
separate from the fixed perceptual gate. Vectorized results match the original
scalar replay exactly on 182 previously evaluated arms (final p99 and settled
classification).

Examples of informative rejections: `(d/64)^3` re-noises the repeatedly updated
remote cave; `(d/10)^8`, `(d/12)^8` and `(d/8)^6` retain too much obsolete canyon
lighting. These failures are preserved, not discarded by fitting only nearby
source cases. No reference values enter any candidate's runtime weight rule.

The native-image shortlist is `(d/64)^4`, `(d/14)^6`, `(d/12)^4`, `(d/16)^4`:
the aggressive-screen leader, the best surviving more-retentive family, and
two lower-staleness alternatives. The accidental equality of `(48/12)^4` to
256 is not a constraint; the other candidates do not share it. Their complete
curves, not just near sections, are applied to every recorded corner donor.

Evidence: `target/whole-fit-screen.json`, `whole-fit-far-filter.json`,
`whole-far-screen.json`, `whole-far-shortlist.json` and their logs. Metadata and
moment hashes identify reused native tapes. The mean/display screen is
committed at `6f8ef47`; the distant replay/screen at `6948894`. Runtime Rust,
transport, shaders and BVW are unchanged from the preserved baseline.

## Native screening and exact reuse

The first whole-curve cave run rejects the proxy-only ranking. `(d/64)^4`
does not finish cave edits sooner but raises peak mean FLIP on removal/addition
from .05132/.05360 to .07509/.07753. `(d/14)^6` lowers early error but delays
the sustained checkpoints from 2,560/3,072 to 3,584/3,584. Its large diffuse-roof
control also needs 6,048 rather than 480 fresh paths to the existing raw 5%
transport criterion. That is not a FLIP timing measurement, but is a real
stale-light response weakness. Neither extreme is the selected balance.

The two remaining whole curves `(d/12)^4` and `(d/16)^4` preserve the cave's
window-0 2,560/3,072 checkpoints. Their null-control peak mean FLIP is
.03772 / .05051, respectively, versus baseline .04766. At hallway offset 16,
the more aggressive `(d/16)^4` removes light in 256 rather than 512 paths, but
both brighten in 2,048 rather than the baseline's 2,560. The final comparison
keeps this speed/noise tradeoff explicit.

No new house captures are needed for these finalists. At **every** recorded
house receiver, the whole-curve frame inputs exactly match previously rendered
ones: `(d/12)^4` matches the old near-power4 arm, and `(d/16)^4` matches the old
near-power6 arm. `whole_history_reuse.py` regenerates every observed full
RGB/confidence buffer and requires byte equality with the file supplied to the
native renderer. Across five house assessments, all 390 buffers match.
The original plans, image names, metrics and provenance are retained alongside
explicit alias proofs. This is reused evidence, not a claimed rerun.

Negative checks reject both an incorrect house alias and a hallway alias whose
distant donor buffers differ. Thus reuse does not rely on guessing which
receivers might be visible. The source house checks cover emitter removal/addition
in both windows, the visible sunlight patch in both windows, and the doorway.

New native executables hash to
`b783a60f8703b6fbddbc52292c62fbc5fb4f006722c6f661a065918e7682cc78`
after rebuilding the shared Cargo cache. No Rust/shader source changed.
Old/new cave references differ by at most one 8-bit channel step, with mean
FLIP about .000014; new runs include their own paired baseline. Hardware and
presentation remain M4/16 GiB, Metal, macOS 26.6.2, optimized development,
2560x1440 offscreen, corner sun/reconstructed indirect, textures on, AO off.

## Completed native decision

The fixed gate is mean FLIP <= .027400863220808837 **and** worst 32px-tile mean
<= .07296444475650787, at 67 pixels per degree. It is the original cave's
human-calibrated 2,048-path reference, transferred to other fixtures as an
engineering gate, not separately human-calibrated for every camera.
Every included old-state warm control passes. Sustained acceptance requires
all later recorded observations to pass, with at least two later observations.
New plans add a 3,840-path observation before 4,096, allowing a 3,584 crossing
to be confirmed; this is better tail observation, not faster transport.

| Policy | Qualified edit cases with sustained tail | Equal-family geometric work ratio | Worst native slowdown |
| --- | ---: | ---: | ---: |
| Existing 48/16 baseline | 23 / 24 | 1.0000 | 1.0x |
| **Whole `(d/12)^4`** | **24 / 24** | **.7578** | 4.0x |
| Whole `(d/16)^4` | 24 / 24 | .7715 | 4.0x |

The aggregate compares the same 23 jointly settled cases, balancing hallway,
house and cave equally. It does not assign an invented duration to baseline's
failure. Selected family ratios are .51795 hallway, .93303 house and .90053
cave: about **24% less offline settling work** overall, not 24% higher FPS or
a measured gameplay latency reduction. The small aggregate separation between
finalists is not the main reason to choose 12; its repeated cave smoothness
and ordinary-strength cave response are more convincing than that small score
difference. Screen-rejected curves are not ranked using incomplete holdouts.

Fresh paths per receiver to sustained acceptance, shown as **remove / add**
of the fixture's edited blocks (roof removal opens the skylight):

| Fixture / sequence | Baseline | Selected `(d/12)^4` | Alternative `(d/16)^4` |
| --- | ---: | ---: | ---: |
| Hallway offset 1, W0 | >4,096 / 1,024 | 128 / 1,024 | 128 / 1,024 |
| Hallway offset 8, W0 | 1,024 / 2,560 | 128 / 512 | 128 / 512 |
| Hallway offset 8, W1 | 1,024 / 256 | 128 / 1,024 | 128 / 1,024 |
| Hallway offset 16, W0 | 512 / 2,560 | 512 / 2,048 | 256 / 2,048 |
| House emitter, W0 | 1,536 / 2,560 | 1,024 / 2,560 | 1,024 / 2,560 |
| House emitter, W1 | 1,024 / 1,024 | 512 / 1,536 | 512 / 1,536 |
| House doorway, W0 | 1,024 / 1,024 | 1,024 / 1,024 | 1,024 / 1,024 |
| House sunlight patch, W0 | 256 / 256 | 256 / 256 | 256 / 256 |
| House sunlight patch, W1 | 256 / 256 | 256 / 256 | 256 / 256 |
| Cave source 8x, W0 | 2,560 / 3,072 | 2,560 / 3,072 | 2,560 / 3,072 |
| Cave source 8x, W1 | 3,072 / 3,584 | 3,072 / 3,584 | 3,072 / 3,584 |
| Cave source 1x, W0 | 2,560 / 1,536 | 2,048 / 1,024 | 2,560 / 2,048 |

These are observed checkpoints, not exact minima between observations. The
adjacent hallway baseline still fails at 4,096; that is not merely an
unconfirmed final crossing. The changed 8x source is one edited source in the
many-emitter cave, not an eightfold increase to every cave light.

The selected curve reduces unchanged-lighting reset peak mean FLIP in the
8x cave from .04766 to .03772 (W0) and .04977 to .03982 (W1), about 20-21%.
Its first-512-path mean-error integral improves about 18% on those null
controls and 6.6% on their actual edits. At ordinary strength, peak null error
falls .04468 -> .03466, and sustained null recovery improves 1,536 -> 1,024.
The `(d/16)^4` alternative instead raises ordinary-cave null error to .04720
and delays recovery to 2,048. This is why simply discarding more history is not
the winner. House doorway-opening peak error is .04051 for baseline/selected,
versus .05382 for the more aggressive alternative, with equal settling work.

Keep the counterexamples explicit:

- Hallway offset-8 W1 brightening takes 1,024 rather than baseline's 256 paths
  (4x); house-emitter W1 takes 1,536 rather than 1,024 (1.5x). Both finalists
  share these regressions. The other sequence improves or ties, so one favorable
  sequence cannot establish universally quicker brightening.
- Unchanged hallway/house lighting can be re-noised more than baseline.
  For example, selected house-emitter null peak error in W1 is .05812 versus
  .04879; the aggressive alternative is worse again at .06311. Null controls
  are reported separately, not counted as edit-speed wins or omitted as noise.
- Less FLIP error does not imply zero brightness overshoot. For the 8x cave
  removal in W1, the maximum 32px-tile positive excursion beyond the brighter
  reference endpoint is .03041 instead of baseline .02897 in rendered linear
  luminance (difference .00144). These are absolute luminance units, not a
  percentage. The alternative reaches .03126. The existing static/zero-history
  comparisons identify early sampling noise as a separate remaining problem.

The occluded-hall tape participates in the cheap screen, but its previously
failing warm control cannot certify settled-start native response. Dim distant
hallway cameras that failed the existing reference-variation guard in the
previous round are still excluded; no guard was relaxed to obtain this result.
This is not a certification of arbitrary scenes, source strengths or an
always-sufficient 2,048 sample count.

## Distant transport result and measured weaknesses

The selected curve has **no newly unresolved raw tails** in the 12 distant
cases, both windows and directions, including repeated remote-cave events
(52 arms per policy). These controls evaluate all five transport bands.
Their existing raw 5% acceptance rule is not the perceptual FLIP threshold.

| Control | Baseline fresh paths | Selected fresh paths | Qualification |
| --- | ---: | ---: | --- |
| Diffuse roof at 64 | 480 | 1,328 | Measured 2.77x slower response |
| Bounced sunlight, roofs 64 through 512 | 5,584 | 1,104 | Actual changed-sun surface distances included |
| Canyon remote residency | 2,880 | 2,560 | Both settle; selected late error is slightly higher |
| Remote cave, single and repeated updates | 0 | 0 | Already within raw gate; histories/results unchanged |

Values are worst observed work across each control's windows/directions, not
native-rendered timing. The diffuse roof at 128 remains inside the raw gate
throughout, but selected late p99 error is .01580 versus baseline .00307.
At 64 the respective late errors are .02038 and .00934. More retained history
does delay some genuine diffuse-sky changes; this weakness is not erased by
passing the eventual tail. The `(d/16)^4` alternative matches baseline's
480-path roof-64 response, another real tradeoff against its higher cave noise.

The canyon's selected late/final raw RGB p99 is .02132/.01826, versus baseline
.01813/.01614. Remote-cave late/final p99 remains exactly .02179/.01834.
These results support tightening retention, not claiming that all distant
changes are negligible or that this equation dominates baseline everywhere.

## Runtime integration boundary

This branch finalizes the equation and evidence; it does not silently switch
the normal game to an offline-tested policy. Integration must replace the old
flat-near-zone `bounds_limit` early exit, not multiply a new near curve onto the
old plateau. Apply a cap to existing weight, never assign synthetic evidence.
Keep the current estimator, display smoothing, transport and BVW unchanged.

Exact events and overflow summaries must enforce the same distance function.
The journal's distance bands are a conservative storage approximation, not
policy bands. Their lower-distance bounds can discard more history than exact
per-receiver replay, especially with a steep curve, so event delivery and
compressed-summary enforcement need their own paired gameplay check. The
existing actual-sun-change notifications must remain spatially relevant; no
remote geometry cutoff can replace them.

Near/material tests have the standard background-loading prelude, then render
controlled recorded buffers. Far transport controls explicitly replay changed
residency. Neither is a measurement of live editing while new events arrive,
main-thread stalls, FPS, GPU savings, memory savings or simulation accuracy.
Do not convert their synthetic replay cadence into gameplay latency. A live
integration comparison is bounded verification of this selected policy, not
an excuse to restart the coefficient search.

## Evidence identity and verification

Final comparison: `target/whole-final-comparison.json`, SHA-256
`ec847541e672ba972dde2e694f43bf147a124cf7fdb60f212341c3a8e669ca3b`.
It contains every input assessment's full hash, per-arm early/tile/endpoint
errors, null controls, qualification flags and pairwise response-work ratios.
There are six new completed native assessments and five explicitly reused
house assessments. New native launches all have a clean tracked diff; their
provenance records full commits, plan hashes, GI environment and executable.

| New assessment prefix in `target/` | Launch commit | Native directory under `target/gi-validation/` |
| --- | --- | --- |
| `whole-screen-cave` | `6948894` | `cave-1792013162705471000-95971` |
| `whole-screen-d16` | `3e5b079` | `hallway-1792013485746167000-97676` |
| `whole-confirm-cave` | `3e5b079` | `cave-1792013776387648000-97833` |
| `whole-confirm-d8` | `2fa9253` | `hallway-1792014000101103000-97995` |
| `whole-ordinary-cave` | `1943357` | `cave-1792014283319161000-98268` |
| `whole-confirm-d1` | `1943357` | `hallway-1792014507021043000-98311` |

Each prefix has `-assessment.json`, `-provenance.json`, `.log` and
`-assessment.log`; its plan is `<prefix>-plan/plan.json`. Source tape metadata
and moment-file hashes are embedded in the plans. Runtime source is unchanged
from this round's `3e6170f` baseline. New runs use the executable hash and
hardware/settings above; Rust 1.95, Bevy 0.19.1, optimized development
(project opt-level 1, dependencies 3, debug info), macOS build 25G83.

| Verified reuse prefix in `target/` | Original assessment prefix | Original launch commit |
| --- | --- | --- |
| `whole-reuse-house-emitter` | `decision-curves-house-emitter` | `6a61e35` |
| `whole-reuse-house-emitter-w1` | `decision-repeat-house-emitter` | `aee920f` |
| `whole-reuse-house-door` | `decision-curves-house-door` | `aee920f` |
| `whole-reuse-sun-patch` | `decision-curves-sun-patch` | `6a61e35` |
| `whole-reuse-sun-patch-w1` | `decision-repeat-sun-patch` | `b5be0e1` + documented diff |

The final reused source's dirty diff only corrects two numerical values in
`docs/history-decision.md`; the complete diff remains in its provenance.
Original house executables hash to
`3d150537301f1d422b02a8ab887df344cb1c71d764a5020db512e65c7ea6f7f8`.
Original paired controls/references are retained. No new run, executable
parity, or fresh FLIP measurement is invented for these aliases.

Reproduction with NumPy, Pillow and official FLIP 1.7 available, and `tools`
on `PYTHONPATH`:

```sh
python3 tools/whole_history_fit.py target/whole-fit-manifest.json --output target/new-fit.json
python3 tools/whole_history_far.py target/distance-history-tape.json target/sun-distance-complete-metadata.json --curves 12:4 16:4 --output target/new-far.json
python3 tools/near_history_run.py target/whole-confirm-d8-plan/plan.json --prefix target/new-whole-d8 --compact
python3 tools/history_decision_analysis.py compare target/whole-*-assessment.json --output target/new-whole-comparison.json
python3 -m unittest near_history_calibration.Tests whole_history_fit.Tests history_decision_analysis.Tests history_distance_analysis.Tests distance_history_analysis.Tests
```

Focused verification passes **27 tests**, plus the 182-arm scalar/vectorized
parity check, 390 exact reused frame buffers and two rejecting alias controls.
All six new renders completed strict native geometry/capture audits, valid
offscreen targets, reference agreement checks and clean render logs. Known
zero-energy hallway controls require bracketing positive material witnesses;
they are not accepted as unexplained black gameplay captures. Native rendering
also verifies the unchanged Rust application builds; the unrelated full Rust
suite and toggle stress were not rerun.

After completed assessment, reproducible intermediate PNGs were hashed and
compacted. Native tapes, binary frame inputs, plans, reference images, metrics
and selected trajectory images remain. No extra coefficient search, transport
change, BVW change or unmeasured default switch is included in this decision.
