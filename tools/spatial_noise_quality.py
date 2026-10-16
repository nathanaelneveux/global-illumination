#!/usr/bin/env python3
"""Diagnostic smooth-wall noise score, separate from full-image FLIP fidelity.

Compare reference-relative OKLab band energy on two fixed cave wall interiors.
The regions exclude emitter halos, sunlight edges and silhouettes: full-image
FLIP must still protect those features. This is a project-specific multiscale
diagnostic, not FLIP, NLPD, a universal perceptual threshold or a settling gate.
Four sequence windows expose sensitivity to the sampling sequence; their
dispersion is not a temporal-flicker measurement or a confidence interval.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from cave_sample_analysis import load_rgb, srgb_to_oklab


SCALES = (4, 8, 16, 32)


def lowpass(value):
    """Separable binomial blur; reflection avoids an artificial crop border."""
    for axis in (0, 1):
        pad = [(0, 0)] * value.ndim
        pad[axis] = (2, 2)
        extended = np.pad(value, pad, mode="reflect")
        value = sum(weight * np.take(extended, np.arange(value.shape[axis]) + offset,
                                     axis=axis)
                    for offset, weight in enumerate((1/16, 4/16, 6/16, 4/16, 1/16)))
    return value


def residual_bands(image, reference):
    if (image.shape != reference.shape or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) < 64 or not np.isfinite(image).all()
            or not np.isfinite(reference).all()):
        raise ValueError("require matching finite RGB crops at least 64 pixels across")
    residual = srgb_to_oklab(image) - srgb_to_oklab(reference)
    brightness = float(residual[..., 0].mean())
    h, w = residual.shape[:2]
    # Average residuals, not sRGB, to suppress material texels before measuring
    # voxel-scale lighting variations. Broad bias is reported separately.
    current = residual[:h // 4 * 4, :w // 4 * 4].reshape(
        h // 4, 4, w // 4, 4, 3).mean(axis=(1, 3))
    bands = []
    for _ in SCALES:
        coarse = lowpass(current)
        bands.append(current - coarse)
        current = coarse[::2, ::2]
    return bands, brightness


def band_energy(bands):
    return [float(np.mean(np.sum(b * b, axis=-1))) for b in bands]


def summarize_windows(windows):
    if len(windows) != 4:
        raise ValueError("require all four sequence windows")
    energies = np.array([band_energy(w) for w in windows])
    # This companion term isolates sequence-dependent variation from fixed
    # structure/bias. Four windows are too few to certify estimator variance.
    dispersion = [float(np.mean(np.sum(np.var(np.stack(b), axis=0, ddof=1), axis=-1)))
                  for b in zip(*windows)]
    return dict(wall_noise_max=float(np.sqrt(energies.sum(axis=1)).max()),
                wall_noise_each=np.sqrt(energies.sum(axis=1)).tolist(),
                band_rms_max=np.sqrt(energies.max(axis=0)).tolist(),
                sequence_dispersion=float(np.sqrt(sum(dispersion))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--counts", nargs="+", type=int, default=[4, 16, 64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--labels", nargs="+",
                        default=["samples", "evidence", "evidence1024", "evidence2048", "evidence4096"])
    parser.add_argument("--profile", type=Path, default=Path(__file__).parent /
                        "quality_profiles/cave-spatial-noise-2048.json")
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    if profile["metric"] != "oklab-wall-band-rms-v1":
        raise ValueError("unsupported noise metric")
    regions = profile["view"]["regions"]
    limit = profile["provisional_wall_noise_limit"]
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("require a finite positive calibration limit")
    source = json.loads((args.run / "cave-reference.json").read_text())
    status = json.loads((args.run / "cave-reference-status.json").read_text())
    audit = json.loads((args.run / "cave-geometry-audit.json").read_text())
    if (not source["qualification"]["qualified"] or status["status"] != "completed"
            or audit["unknown_voxels"] or audit["mismatched_voxels"]):
        raise ValueError("require a complete, qualified reference and matching geometry")
    reference_path = args.run / "cave-reference.png"
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != profile["view"]["reference_sha256"]:
        raise ValueError("this provisional calibration requires the exact saved reference")
    reference = load_rgb(reference_path)
    if reference.shape != (profile["view"]["height"], profile["view"]["width"], 3):
        raise ValueError("fixed cave regions require the calibrated view dimensions")
    rows = []
    for label in args.labels:
        for count in args.counts:
            groups = {name: [] for name in regions}
            brightness = {name: [] for name in regions}
            for window in range(4):
                image = load_rgb(args.run / f"{label}-{count}-window-{window}.png")
                if image.shape != reference.shape:
                    raise ValueError("candidate view dimensions differ")
                for name, (x0, y0, x1, y1) in regions.items():
                    bands, bias = residual_bands(image[y0:y1, x0:x1], reference[y0:y1, x0:x1])
                    groups[name].append(bands)
                    brightness[name].append(bias)
            measured = {name: dict(**summarize_windows(windows),
                                   signed_lightness_each=brightness[name])
                        for name, windows in groups.items()}
            noise = max(r["wall_noise_max"] for r in measured.values())
            rows.append(dict(label=label, paths=count, regions=measured,
                             wall_noise_max=noise, normalized_wall_noise=noise / limit,
                             within_preferred_noise_envelope=noise <= limit))
    print(json.dumps(dict(run=str(args.run), profile=profile, source_pixel_scales=SCALES,
        note="Provisional preference calibration, not an overall quality pass. Keep full-image FLIP, brightness and lighting-detail checks. Validate on held-out images before promotion.",
        measurements=rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
