#!/usr/bin/env python3
"""Qualify footprint replays against the saved, preferred target-2048 images.

Reuse paths, not a new noisy reference. Require renderer/control equivalence
before applying the existing provisional wall-noise calibration and FLIP guard.
The one-code-value allowance covers rare final-render rounding, not view drift.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cave_sample_analysis import load_rgb, perceptual_difference
from spatial_noise_quality import residual_bands, summarize_windows


def equivalent_render(a, b):
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("replay shape/values differ")
    delta = np.abs(a - b)
    result = dict(max=float(delta.max()), mean=float(delta.mean()),
                  changed_pixel_fraction=float(np.mean(np.any(delta != 0, axis=-1))))
    if result["max"] > 1.01 / 255 or result["mean"] > 1e-6 or result["changed_pixel_fraction"] > .001:
        raise ValueError(f"replay does not reproduce saved rendering: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("replay", type=Path)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    profile = read(Path(__file__).parent / "quality_profiles/cave-spatial-noise-2048.json")
    source = read(args.source / "cave-reference.json")
    assert source["qualification"]["qualified"]
    for folder, status_name in [(args.source, "cave-reference-status.json"),
                                (args.replay, "cave-model-comparison-status.json")]:
        assert read(folder / status_name)["status"] == "completed"
        audit = read(folder / "cave-geometry-audit.json")
        assert audit["unknown_voxels"] == audit["mismatched_voxels"] == 0
    path = args.source / "cave-reference.png"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == profile["view"]["reference_sha256"]
    reference = load_rgb(path)
    assert reference.shape == (1440, 2560, 3)
    equivalence = {"reference": equivalent_render(reference, load_rgb(args.replay / path.name))}
    counts = [4, 16, 64, 128]
    for count in counts:
        for window in range(4):
            name = f"evidence2048-{count}-window-{window}.png"
            equivalence[name] = equivalent_render(load_rgb(args.source / name), load_rgb(args.replay / name))
    rows = []
    for label in ["evidence2048", "evidence2048cap32", "evidence2048cap64", "evidence2048cap128"]:
        for count in counts:
            bands = {name: [] for name in profile["view"]["regions"]}
            bias = {name: [] for name in bands}
            errors = []
            for window in range(4):
                image = load_rgb(args.replay / f"{label}-{count}-window-{window}.png")
                for name, (x0, y0, x1, y1) in profile["view"]["regions"].items():
                    b, lightness = residual_bands(image[y0:y1, x0:x1], reference[y0:y1, x0:x1])
                    bands[name].append(b)
                    bias[name].append(lightness)
                errors.append(perceptual_difference(reference, image, 67))
            regions = {name: dict(**summarize_windows(b), signed_lightness=bias[name]) for name, b in bands.items()}
            noise = max(r["wall_noise_max"] for r in regions.values())
            rows.append(dict(label=label, paths=count, normalized_wall_noise=noise / profile["provisional_wall_noise_limit"],
                regions=regions, windows=errors, flip=max(e["flip"]["mean"] for e in errors),
                worst_tile=max(e["flip_32px_tile_means"]["max"] for e in errors)))
    print(json.dumps(dict(source=str(args.source), replay=str(args.replay), profile=profile,
        note="Matched path/display study. Noise envelope is provisional, not a quality or settling pass. No interactive timing claim.",
        render_equivalence=equivalence, measurements=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
