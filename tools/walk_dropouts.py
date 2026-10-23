"""Single-frame black flashes in a `motion` walk run (VOXEL_GI_VALIDATE_MOTION_WALK).

Usage: uv run --with numpy --with pillow python tools/walk_dropouts.py RUN...
Counts pixels black (all channels <= 2/255) in one consecutive burst frame but
lit (> .05 luminance) in both neighbours. Use the indirect light component:
textured appearance shimmers in motion. Flicker evidence, not accuracy."""
import sys, json
from pathlib import Path
import numpy as np
from PIL import Image
def load(p): return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255
w = np.array([.2126, .7152, .0722])
out = {}
for run in sys.argv[1:]:
    r = Path(run); counts = []
    for burst in "ab":
        f = [load(r / f"walk-{burst}-{i:02}.png") for i in range(16)]
        for i in range(1, 15):
            black = f[i].max(axis=2) <= 2 / 255
            lit = ((f[i-1] @ w) > .05) & ((f[i+1] @ w) > .05)
            counts.append(int((black & lit).sum()))
    out[r.name] = {"per_frame": counts, "frames_with_dropouts": sum(c > 0 for c in counts), "total": sum(counts)}
print(json.dumps(out))
