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
