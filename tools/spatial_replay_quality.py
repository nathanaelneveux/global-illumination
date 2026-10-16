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


def aggregate_errors(errors):
    """Keep window-average fidelity distinct from worst-window/local failures."""
    means = np.array([e["flip"]["mean"] for e in errors])
    tiles = np.array([e["flip_32px_tile_means"]["max"] for e in errors])
    if len(errors) != 4 or not np.isfinite(means).all() or not np.isfinite(tiles).all():
        raise ValueError("require four finite window measurements")
    return dict(average_flip=float(means.mean()), flip=float(means.max()),
                average_worst_tile=float(tiles.mean()), worst_tile=float(tiles.max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--labels", nargs="+", default=[
        "evidence2048", "evidence2048cap32", "evidence2048cap64", "evidence2048cap128"])
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 16, 64, 128])
    parser.add_argument("--fresh-reference", action="store_true",
                        help="analyze variants from one newly qualified native sweep, not a replay of old images")
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text())
    profile = read(Path(__file__).parent / "quality_profiles/cave-spatial-noise-2048.json")
    source = read(args.source / "cave-reference.json")
    assert source["qualification"]["qualified"]
    if args.fresh_reference and args.source.resolve() != args.replay.resolve():
        raise ValueError("fresh-reference mode requires the same source and image directory")
    for folder, status_name in [(args.source, "cave-reference-status.json"),
                                (args.replay, "cave-reference-status.json" if args.fresh_reference
                                 else "cave-model-comparison-status.json")]:
        assert read(folder / status_name)["status"] == "completed"
        audit = read(folder / "cave-geometry-audit.json")
        assert audit["unknown_voxels"] == audit["mismatched_voxels"] == 0
    path = args.source / "cave-reference.png"
    reference_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    calibration_match = reference_hash == profile["view"]["reference_sha256"]
    if not args.fresh_reference and not calibration_match:
        raise ValueError("replay source does not match the saved calibration reference")
    reference = load_rgb(path)
    assert reference.shape == (1440, 2560, 3)
    equivalence = {}
    counts = args.counts
    if not args.fresh_reference:
        equivalence["reference"] = equivalent_render(reference, load_rgb(args.replay / path.name))
        for count in counts:
            for window in range(4):
                name = f"evidence2048-{count}-window-{window}.png"
                equivalence[name] = equivalent_render(load_rgb(args.source / name), load_rgb(args.replay / name))
    rows = []
    for label in args.labels:
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
            rows.append(dict(label=label, paths=count, wall_noise=noise,
                normalized_wall_noise=noise / profile["provisional_wall_noise_limit"] if calibration_match else None,
                regions=regions, windows=errors, **aggregate_errors(errors)))
    print(json.dumps(dict(source=str(args.source), replay=str(args.replay), profile=profile,
        reference_sha256=reference_hash, calibration_reference_matches=calibration_match,
        fresh_reference=args.fresh_reference,
        note="Matched path/display study. Noise envelope is provisional, not a quality or settling pass. No interactive timing claim.",
        render_equivalence=equivalence, measurements=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
