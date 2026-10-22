"""Evaluate the fixed-eye motion profile, not raw transport accuracy.

Usage: PYTHONPATH=target/perceptual-python python tools/motion_quality.py RUN...
The known 0.15-degree turn is removed analytically before comparing lighting.
Two-pixel low-pass filtering suppresses texture/raster resampling differences;
these metrics are NOT the calibrated raw-reference FLIP accuracy threshold.
Run outside timed gameplay. Requires Pillow, numpy and flip-evaluator.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from view_cache_quality import compare, image


def aligned_turn(base_path, turned_path, turn_degrees=.15, appearance=True):
    base = np.asarray(Image.open(base_path).convert("RGB").filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255
    turned = np.asarray(Image.open(turned_path).convert("RGB").filter(ImageFilter.GaussianBlur(2)), dtype=np.float32) / 255
    h, w, _ = base.shape
    assert base.shape == turned.shape == (1440, 2560, 3)
    y, x = np.indices((h, w), dtype=np.float32)
    focal = h / (2 * np.tan(np.pi / 8))
    ray = np.stack(((x + .5 - w / 2) / focal, -(y + .5 - h / 2) / focal, -np.ones_like(x)), axis=-1)

    def rx(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

    angle = -np.deg2rad(turn_degrees)
    c, s = np.cos(angle), np.sin(angle)
    ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    ray = ray @ (rx(.12) @ ry @ rx(-.12)).T
    u = focal * ray[..., 0] / -ray[..., 2] + w / 2 - .5
    v = -focal * ray[..., 1] / -ray[..., 2] + h / 2 - .5
    valid = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    u, v = np.clip(u, 0, w - 1.001), np.clip(v, 0, h - 1.001)
    ix, iy = u.astype(int), v.astype(int)
    fx, fy = (u - ix)[..., None], (v - iy)[..., None]
    aligned = ((turned[iy, ix] * (1 - fx) + turned[iy, ix + 1] * fx) * (1 - fy)
               + (turned[iy + 1, ix] * (1 - fx) + turned[iy + 1, ix + 1] * fx) * fy)
    # The green-terrain heuristic only applies to textured appearance, never
    # white/blue irradiance. Components use the complete valid overlap, including
    # unchanged sky; compare like component captures, not their mean to appearance.
    terrain = valid & (base[..., 1] > base[..., 2]) & (aligned[..., 1] > aligned[..., 2]) if appearance else valid
    error = np.abs((aligned - base) @ np.array([.2126, .7152, .0722]))[terrain]
    assert error.size, "No terrain pixels in the fixed-eye motion comparison"
    return {"mask": "green terrain" if appearance else "valid image overlap", "compared_pixels": int(terrain.sum()), "mean_abs_display_luminance": float(error.mean()),
            "p95_p99_p999_abs_display_luminance": np.percentile(error, [95, 99, 99.9]).tolist(),
            "pixels_over_002": int((error > .02).sum())}


def evaluate(run):
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "completed" and report["profile"] == "motion" and report["capture_mode"] == "offscreen"
    captures = report["capture_windows"]
    assert captures and all(c["target_valid"] and c["varied"] and c["width"] == 2560
                            and c["height"] == 1440 for c in captures)
    mode = report.get("view_diagnostic", 0)
    assert mode != 2, "Coverage colors are not lighting-stability evidence"
    return {"run": str(run), "build": report["build"], "view_diagnostic": mode,
            "return_flip": compare(image(run / "base-1.png"), image(run / "base-2.png")),
            "turned_repeat_flip": compare(image(run / "turned-1.png"), image(run / "turned-2.png")),
            "turn_degrees": report.get("motion_turn_degrees", .15),
            "aligned_small_turn": aligned_turn(run / "base-1.png", run / "turned-1.png",
                turn_degrees=report.get("motion_turn_degrees", .15), appearance=mode == 0)}


if __name__ == "__main__":
    print(json.dumps([evaluate(Path(p)) for p in sys.argv[1:]], indent=2))
