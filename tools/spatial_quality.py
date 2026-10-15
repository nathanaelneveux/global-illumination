#!/usr/bin/env python3
"""Compare matched spatial-reconstruction captures with the fixed cave gate.

Do not mistake smooth bias for accuracy: retain FLIP mean/worst tile and measure
both fine-scale residual roughness and coarse residual patches against truth.
This is a presentation study, never a transport or simulation qualification.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cave_sample_analysis import load_rgb, perceptual_difference, srgb_to_oklab


def residual_scales(image, reference):
    residual = srgb_to_oklab(image) - srgb_to_oklab(reference)
    result = {}
    for size in (1, 4, 16, 64):
        h, w = residual.shape[:2]
        blocks = residual[:h // size * size, :w // size * size].reshape(
            h // size, size, w // size, size, 3).mean(axis=(1, 3))
        # Coarse residual energy catches smooth blotches that roughness misses.
        roughness = np.concatenate([
            np.linalg.norm(np.diff(blocks, axis=0), axis=-1).ravel(),
            np.linalg.norm(np.diff(blocks, axis=1), axis=-1).ravel()])
        result[str(size)] = dict(rms=float(np.sqrt(np.mean(blocks ** 2))),
                                neighbor_difference_p95=float(np.quantile(roughness, .95)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--counts", type=int, nargs="+", default=[16, 64, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--labels", nargs="+", default=["samples", "spatial"])
    parser.add_argument("--runtime-run", type=Path, action="append", default=[],
                        help="compare cave-15/30/60s captures; reference scene equivalence is not implied")
    args = parser.parse_args()
    source = json.loads((args.run / "cave-reference.json").read_text())
    status = json.loads((args.run / "cave-reference-status.json").read_text())
    audit = json.loads((args.run / "cave-geometry-audit.json").read_text())
    if (not source["qualification"]["qualified"] or status["status"] != "completed"
            or audit["unknown_voxels"] or audit["mismatched_voxels"]):
        raise ValueError("require qualified reference, complete captures and geometry")
    profile = json.loads((Path(__file__).parent / "quality_profiles/cave-user-2048.json").read_text())
    reference = load_rgb(args.run / "cave-reference.png")
    if reference.shape != (profile["view"]["height"], profile["view"]["width"], 3):
        raise ValueError("view dimensions differ from calibrated gate")
    if args.runtime_run:
        rows = []
        for folder in args.runtime_run:
            report = json.loads((folder / "report.json").read_text())
            if report["status"] != "completed" or report["captures"] != 3:
                raise ValueError("require complete cave workload and captures")
            for seconds in (15, 30, 60):
                image = load_rgb(folder / f"cave-{seconds}s.png")
                rows.append(dict(run=str(folder), seconds=seconds,
                    residual_scales=residual_scales(image, reference),
                    **perceptual_difference(reference, image, profile["view"]["ppd"])))
        print(json.dumps(dict(reference=str(args.run), measurements=rows,
            note="Live streamed terrain differs from frozen reference residency; comparative image errors, not certified settled gates."),
            allow_nan=False, indent=2))
        return
    rows = []
    for label in args.labels:
        for n in args.counts:
            for window in range(4):
                image = load_rgb(args.run / f"{label}-{n}-window-{window}.png")
                error = perceptual_difference(reference, image, profile["view"]["ppd"])
                passed = (error["flip"]["mean"] <= profile["limits"]["flip"]["mean"]
                          and error["flip_32px_tile_means"]["max"] <= profile["limits"]["flip_32px_tile_means"]["max"])
                rows.append(dict(label=label, paths=n, window=window, image_pass=passed,
                                 residual_scales=residual_scales(image, reference), **error))
    summary = []
    for label in args.labels:
        for n in args.counts:
            group = [r for r in rows if r["label"] == label and r["paths"] == n]
            summary.append(dict(label=label, paths=n, passed=all(r["image_pass"] for r in group),
                flip=max(r["flip"]["mean"] for r in group),
                worst_tile=max(r["flip_32px_tile_means"]["max"] for r in group),
                roughness=max(r["residual_scales"]["1"]["neighbor_difference_p95"] for r in group),
                coarse_rms=max(r["residual_scales"]["64"]["rms"] for r in group)))
    print(json.dumps(dict(run=str(args.run), profile=profile["id"],
        minimum_sustained_image_paths={label: next((n for n in sorted(args.counts)
            if all(r["image_pass"] for r in rows if r["label"] == label and r["paths"] >= n)), None)
            for label in args.labels}, summary=summary, measurements=rows), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
