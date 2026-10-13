#!/usr/bin/env python3
"""Analyze audited paired model images; make a remote-friendly contact sheet.

Shares the already-tested corner topology and perceptual implementation with
the sample-count study. Comparison completion is never a quality pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from cave_sample_analysis import (
    corner_members, load_rgb, perceptual_difference, reconstruct, stats,
)


def symmetric_error(a, b):
    return np.linalg.norm(a[..., :3] - b[..., :3], axis=-1) / np.maximum(
        (np.linalg.norm(a[..., :3], axis=-1) + np.linalg.norm(b[..., :3], axis=-1)) * .5, .02)


def analyze(folder, pixels=True):
    data = json.loads((folder / "cave-model-comparison.json").read_text())
    audit = json.loads((folder / "cave-geometry-audit.json").read_text())
    status = json.loads((folder / "cave-model-comparison-status.json").read_text())
    if (data["kind"] != "paired_model_comparison" or status["status"] != "completed"
            or not status["geometry_audit_passed"] or audit["unknown_voxels"]
            or audit["mismatched_voxels"] or status["qualification"]["quality_pass_claimed"]):
        raise ValueError("missing audit/completion, or comparison mislabeled as a quality pass")
    models = data["models"]
    if models[0]["model"] != dict(reach=768., bounces=3):
        raise ValueError("comparison does not use the current baseline")
    sites = data["sites"]
    patch = corner_members(sites, np.array(data["origin"]))
    reference = np.array(models[0]["mean"])
    windows = [np.array(v) for v in models[0]["windows"]]
    if reference.shape != (len(sites), 5) or len(windows) != 2:
        raise ValueError("invalid face/window schema")
    target = load_rgb(folder / (models[0]["name"] + ".png")) if pixels else None
    if pixels and target.shape != (1440, 2560, 3):
        raise ValueError("unexpected camera dimensions")
    results = []
    for model in models:
        values = np.array(model["mean"])
        if values.shape != reference.shape or not np.all(np.isfinite(values)):
            raise ValueError("missing/nonfinite face values")
        measured_windows = [np.array(v) for v in model["windows"]]
        if len(measured_windows) != len(windows) or any(v.shape != reference.shape for v in measured_windows):
            raise ValueError("missing paired windows")
        row = dict(model=model["model"], image=model["name"] + ".png",
            raw_rgb=stats(symmetric_error(values, reference)),
            corners_rgb=stats(symmetric_error(reconstruct(values, patch), reconstruct(reference, patch))),
            paired_windows=[stats(symmetric_error(a, b)) for a, b in zip(measured_windows, windows)],
            bands=[stats(np.abs(values[:, b] - reference[:, b]) /
                np.maximum((np.abs(values[:, b]) + np.abs(reference[:, b])) * .5, .02)) for b in range(5)])
        if pixels:
            image = load_rgb(folder / row["image"])
            if image.shape != target.shape or image.max() - image.min() < .05:
                raise ValueError("blank or wrong-sized image")
            row["perceptual"] = perceptual_difference(target, image, 67.)
        results.append(row)
    return dict(kind=data["kind"], geometry_audit_passed=True, quality_pass_claimed=False,
        paths_per_window=data["paths_per_window"], sequences=data["sequences"],
        face_count=len(sites), corner_count=len(patch[1]),
        baseline_replica_rgb_difference=stats(symmetric_error(*windows)), models=results,
        note="Paired same-snapshot model differences, not convergence or visual-equivalence certification. Images average two windows; perceptual scores use rendered LDR RGB, FLIP at 67 pixels/degree and OKLab. Transport errors are symmetric with a .02 floor. No interactive FPS claim.")


def contact_sheet(folder, report, canyon_preview=None, horizon_results=None):
    width, height, label = 768, 432, 50
    extra = int(canyon_preview is not None) + int(horizon_results is not None)
    rows = (len(report["models"]) + extra + 2) // 3
    canvas = Image.new("RGB", (width * 3, rows * (height + label) + 100), "#14191f")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 23)
    small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 19)
    for i, row in enumerate(report["models"]):
        x, y = i % 3 * width, i // 3 * (height + label)
        model = row["model"]
        text = f"Reach {model['reach']:g} / {model['bounces']} hits"
        if i == 0:
            text += " — baseline"
        else:
            text += f"  |  corner p99 {row['corners_rgb']['p99'] * 100:.2f}%"
        draw.text((x + 12, y + 13), text, font=font, fill="#f1f3f5")
        with Image.open(folder / row["image"]) as image:
            canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, y + label))
    panel = len(report["models"])
    if canyon_preview is not None:
        x, y = panel % 3 * width, panel // 3 * (height + label)
        draw.text((x + 12, y + 13), "Canyon fixture — live preview, NOT reference", font=font, fill="#f1f3f5")
        with Image.open(canyon_preview) as image:
            canvas.paste(image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS), (x, y + label))
        panel += 1
    if horizon_results is not None:
        x, y = panel % 3 * width, panel // 3 * (height + label)
        study = json.loads(horizon_results.read_text())
        canyon = next(f for f in study["fixtures"] if f["fixture"] == "daylight_canyon")
        draw.text((x + 16, y + 13), "Canyon: paired indirect RGB differences", font=font, fill="#f1f3f5")
        draw.text((x + 16, y + 63), f"{len(canyon['sites']):,} native faces, 2 × {study['paths_per_window']:,} paths", font=small, fill="#ccd4de")
        draw.text((x + 16, y + 100), "Reach / hits       Mean           p99         Maximum", font=font, fill="#f1f3f5")
        candidates = [r for r in canyon["runs"] if r["model"]["reach"] in (256, 288, 320, 384, 512)
                      and r["model"]["bounces"] == 3]
        for index, run in enumerate(candidates):
            error = [100 * max(w["paired_difference"]["rgb"][k] for w in run["windows"])
                     for k in ("mean", "p99", "max")]
            draw.text((x + 16, y + 145 + index * 43),
                f"{run['model']['reach']:g} / 3              {error[0]:.3f}%       {error[1]:.3f}%       {error[2]:.3f}%",
                font=font, fill="#ccd4de")
        draw.text((x + 16, y + 390), "Relative to 768 / 3. Fully resident numerical fixture.", font=small, fill="#ccd4de")
        draw.text((x + 16, y + 422), "Not a whole-canyon image equivalence or FPS pass.", font=small, fill="#ccd4de")
    footer = rows * (height + label)
    draw.text((12, footer + 12),
        f"Actual composed cave, frozen/audited geometry. Two matched {report['paths_per_window']:,}-path windows per model; ordinary corner renderer.",
        font=small, fill="#ccd4de")
    detail = ("Canyon preview is live; its table is a separate fully resident numerical study. "
              if extra else "")
    draw.text((12, footer + 39), detail + "No brightness adjustment to individual panels.",
        font=small, fill="#ccd4de")
    draw.text((12, footer + 66), "Model comparison, not a new perfect-reference, whole-canyon visual-equivalence or interactive FPS qualification.",
        font=small, fill="#ccd4de")
    path = folder / "transport-model-overview.png"
    canvas.save(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-pixels", action="store_true")
    parser.add_argument("--contact-sheet", action="store_true")
    parser.add_argument("--canyon-preview", type=Path)
    parser.add_argument("--horizon-results", type=Path)
    args = parser.parse_args()
    report = analyze(args.run, not args.no_pixels)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.contact_sheet:
        print(contact_sheet(args.run, report, args.canyon_preview, args.horizon_results))
    for row in report["models"]:
        print(row["model"], "raw/corner p99", row["raw_rgb"]["p99"], row["corners_rgb"]["p99"])


if __name__ == "__main__":
    main()
