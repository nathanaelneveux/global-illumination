"""Count view-face lookup failures, including partial MSAA pixel contributions.

Usage: python tools/view_validity.py RUN...
Requires numpy and Pillow. Use the motion profile's face-validity diagnostic:
red = missing indirect support; green = not an exposed transport face. Unlike
the green/yellow/red coverage palette, these independent channels cannot cancel
one another during MSAA resolve. The blue sky is excluded; mixed sky/silhouette
pixels are excluded too. These are lookup diagnostics, not lighting accuracy.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def evaluate(run):
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "completed" and report["view_diagnostic"] == 6
    assert report["capture_mode"] == "offscreen"
    assert report["capture_windows"] and all(
        c["target_valid"] and c["varied"] and c["width"] == 2560 and c["height"] == 1440
        for c in report["capture_windows"]
    )
    captures = {}
    for name in ["before", "base-1", "turned-1", "base-2", "turned-2", "returned"]:
        pixels = np.asarray(Image.open(run / f"{name}.png").convert("RGB"))
        assert pixels.shape == (1440, 2560, 3)
        terrain = pixels[..., 2] == 0
        missing, invalid = (terrain & (pixels[..., channel] > 0) for channel in [0, 1])
        captures[name] = {
            "terrain_pixels": int(terrain.sum()),
            "missing_support_pixels": int(missing.sum()),
            "invalid_face_pixels": int(invalid.sum()),
            "both_flags_pixels": int((missing & invalid).sum()),
            "missing_support_resolved_pixel_equivalents": float(pixels[..., 0][terrain].sum() / 255),
            "invalid_face_resolved_pixel_equivalents": float(pixels[..., 1][terrain].sum() / 255),
        }
    # Both flags in a resolved pixel need not belong to the same MSAA sample.
    return {"run": str(run), "build": report["build"], "captures": captures}


if __name__ == "__main__":
    print(json.dumps([evaluate(Path(arg)) for arg in sys.argv[1:]], indent=2))
