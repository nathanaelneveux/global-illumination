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
