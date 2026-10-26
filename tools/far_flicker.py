"""Frame-to-frame light change inside the far tier's pixels during a walk.

Usage: PYTHONPATH=target/perceptual-python python tools/far_flicker.py COVERAGE_RUN RUN...
   or: ... tools/far_flicker.py LIST COVERAGE_LABEL LABEL...  (a tools/runs.sh list)
Runs are `motion` walks (VOXEL_GI_VALIDATE_MOTION_WALK, walk-{a,b}-NN.png).
COVERAGE_RUN is the same walk with VOXEL_GI_VALIDATE_VIEW_COVERAGE=1: its
cyan pixels are the far tier's, green the receivers' (the control region).
Prints, per run and region, the mean and p99 absolute luminance change
between consecutive frames, and the mean relative to mean luminance. Use
VOXEL_GI_VALIDATE_LIGHT_COMPONENT=indirect runs for the indirect signal,
plain runs for the final image (texture and raster motion dominate there).
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

W = np.array([.2126, .7152, .0722], dtype=np.float32)


def load(run, name):
    return np.asarray(Image.open(f"{run}/{name}.png").convert("RGB"), dtype=np.float32) / 255


def luminance(image):
    return (image * W).sum(axis=-1)


def region_masks(coverage):
    far = (coverage[..., 0] < .1) & (coverage[..., 1] > .9) & (coverage[..., 2] > .9)
    receivers = (coverage[..., 0] < .1) & (coverage[..., 1] > .9) & (coverage[..., 2] < .1)
    return {"far": far, "receivers": receivers}


def main(coverage_run, runs):
    for run in runs:
        changes = {"far": [], "receivers": []}
        levels = {"far": [], "receivers": []}
        for burst in "ab":
            frames = [luminance(load(run, f"walk-{burst}-{i:02}")) for i in range(16)]
            for i in range(1, 16):
                for region, mask in region_masks(load(coverage_run, f"walk-{burst}-{i:02}")).items():
                    if mask.any():
                        changes[region].append(np.abs(frames[i] - frames[i - 1])[mask])
                        levels[region].append(frames[i][mask])
        parts = []
        for region in ("far", "receivers"):
            if not changes[region]:
                parts.append(f"{region}: none")
                continue
            d, level = np.concatenate(changes[region]), np.concatenate(levels[region])
            parts.append(f"{region}: mean {d.mean():.4f} p99 {np.percentile(d, 99):.4f}"
                         f" relative {d.mean() / max(level.mean(), 1e-6):.4f}")
        print(f"{run.rstrip('/').rsplit('/', 1)[-1]} | " + " | ".join(parts))


if __name__ == "__main__":
    args = sys.argv[1:]
    if Path(args[0]).is_file():
        runs = {}
        for line in open(args[0]):
            if line.startswith("done"):
                continue
            label, rest = line.strip().split(" ", 1)
            runs[label] = re.findall(r"/.*?(?= /|$)", rest)[0]
        args = [runs[label] for label in args[1:]]
    main(args[0], args[1:])
