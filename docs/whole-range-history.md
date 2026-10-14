# Whole-range history decision

Baseline `3e6170f`, branch `codex/gi-whole-range-history`. This supersedes the
near-only recommendation in `history-decision.md`: that recommendation joined
a new near curve to a flat inherited floor and was not the whole-range solution
requested. Preserve its measurements, but do not promote that joined policy.

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
