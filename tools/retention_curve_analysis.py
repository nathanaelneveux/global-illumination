"""Score retention-curve arms from target/retention-curve-study.json.

Usage: python3 -I tools/retention_curve_analysis.py [PATH ...]  (rows are merged)

Error is each face's luminance minus its estimator's converged post-edit
value, relative to the brighter of its before/after levels. Score = mean
over log-spaced checkpoints (1..256 batches of 16 paths) of RMS error, so
early response and settled noise count equally. Lower is better.
"""
import json, math, sys
from collections import defaultdict

rows = [r for path in (sys.argv[1:] or ["target/retention-curve-study.json"]) for r in json.load(open(path))]
BANDS = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9)]

def rms(xs):
    return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else float("nan")

def order(label):
    if label in ("retain", "reset"):
        return (0, label, 0.0)
    try:
        return (1, "", float(label))
    except ValueError:
        return (2, label, 0.0)

table = defaultdict(dict)
for r in rows:
    d = r["distances"]
    per_check = [rms(e) for e in r["errors"]]
    band_scores = []
    for lo, hi in BANDS:
        idx = [i for i, x in enumerate(d) if lo <= x < hi]
        band_scores.append(sum(rms([e[i] for i in idx]) for e in r["errors"]) / len(r["errors"]) if idx else float("nan"))
    table[(r["case"], r["mode"])][r["scale"]] = (sum(per_check) / len(per_check), per_check, band_scores, len(d))

for (case, mode), arms in table.items():
    n = next(iter(arms.values()))[3]
    print(f"\n{case} / {mode}  ({n} faces)   score = mean RMS over checkpoints 1..256 batches")
    print(f"{'scale':>7s} {'score':>7s}   " + " ".join(f"{a}-{b if b < 1e9 else '':<3}" .rjust(7) for a, b in BANDS) + "   RMS@1  @16  @256")
    best = min(arms.items(), key=lambda kv: kv[1][0])[0]
    for scale, (score, per, bands, _) in sorted(arms.items(), key=lambda kv: order(kv[0])):
        mark = " <- best" if scale == best else ""
        print(f"{scale:>7s} {score:7.4f}   " + " ".join(f"{b:7.4f}" for b in bands) + f"   {per[0]:.3f} {per[4]:.3f} {per[-1]:.3f}{mark}")

print("\nCombined over cases (mean score):")
for mode in sorted({m for _, m in table}):
    scales = defaultdict(list)
    for (case, m), arms in table.items():
        if m == mode:
            for s, v in arms.items():
                scales[s].append(v[0])
    ranked = sorted(scales.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
    print(f"  {mode:9s} " + ", ".join(f"{s}={sum(v)/len(v):.4f}" for s, v in ranked))
