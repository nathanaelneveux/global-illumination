"""Smooth-wall noise of a frozen surface-cache sweep, against its own reference.

Usage: PYTHONPATH=tools:target/perceptual-python python3 tools/cache_wall_noise.py RUN

Same metric, regions and provisional limit as cave-spatial-noise-2048.json (the
user's preferred 2048-evidence smoothness). That profile pins the old reference
image's hash; here the reference is the cache estimator's own frozen reference,
so the residual measures noise relative to the right target. Lower is smoother;
<= 1 is within the preferred envelope. Not a fidelity check: pair with FLIP.
"""
import json
import sys
from pathlib import Path

from cave_sample_analysis import load_rgb
from spatial_noise_quality import residual_bands, summarize_windows

run = Path(sys.argv[1])
profile = json.loads((Path(__file__).parent / "quality_profiles/cave-spatial-noise-2048.json").read_text())
regions = profile["view"]["regions"]
limit = profile["provisional_wall_noise_limit"]
reference = load_rgb(run / "cave-reference.png")
labels = [("samples", "raw"), ("evidence128", "128"), ("evidence256", "256"),
          ("evidence", "512"), ("evidence1024", "1024")]
counts = [4, 16, 64, 128, 256, 512, 1024, 2048]
print("normalized wall noise (<= 1 matches the preferred 2048-evidence smoothness)")
print(f"{'target':>7s} " + " ".join(f"{c:>6d}" for c in counts) + "   worst")
for label, name in labels:
    row = []
    for count in counts:
        groups = {r: [] for r in regions}
        for window in range(4):
            image = load_rgb(run / f"{label}-{count}-window-{window}.png")
            for r, (x0, y0, x1, y1) in regions.items():
                groups[r].append(residual_bands(image[y0:y1, x0:x1], reference[y0:y1, x0:x1])[0])
        row.append(max(summarize_windows(w)["wall_noise_max"] for w in groups.values()) / limit)
    print(f"{name:>7s} " + " ".join(f"{v:6.2f}" for v in row) + f"   {max(row):5.2f}")
