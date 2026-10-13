#!/usr/bin/env python3
"""Frozen cave sample requirements: transport, reconstructed corners, perception.

Perceptual checks use actual sRGB captures, not a guessed tonemapping curve.
RGB error remains a solver diagnostic, never a perceptual distance. Four
sequence windows are a robustness check, not independent confidence bounds.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


NORMALS = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                    (0, -1, 0), (0, 0, 1), (0, 0, -1)])


def stats(values):
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(v) or not np.all(np.isfinite(v)):
        raise ValueError("missing/nonfinite measurements")
    p = np.quantile(v, [.5, .95, .99, .999], method="higher")
    return dict(count=len(v), mean=float(v.mean()), p50=float(p[0]),
                p95=float(p[1]), p99=float(p[2]), p999=float(p[3]), max=float(v.max()))


def rgb_error(values, reference):
    return np.linalg.norm(values[..., :3] - reference[..., :3], axis=-1) / np.maximum(
        np.linalg.norm(reference[..., :3], axis=-1), .02)


def corner_members(rows, origin):
    lookup = {(tuple(row["voxel"]), row["face"]): i for i, row in enumerate(rows)}
    centers, patches = [], []
    for i, row in enumerate(rows):
        voxel, face = np.array(row["voxel"]), row["face"]
        air = voxel + NORMALS[face] - origin
        if not np.all((air >= 5) & (air <= [58, 42, 58])):
            continue
        centers.append(i)
        u, v = ((1, 2), (0, 2), (0, 1))[face // 2]

        def neighbor(x, y):
            p = voxel.copy()
            p[u] += x
            p[v] += y
            return lookup.get((tuple(p), face))

        for x, y in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
            a, b = neighbor(x, 0), neighbor(0, y)
            patches.append([i, a, b, neighbor(x, y) if a is not None or b is not None else None])
    counts = np.array([sum(i is not None for i in p) for p in patches])
    indices = np.array([[i if i is not None else 0 for i in p] for p in patches])
    weights = np.array([[i is not None for i in p] for p in patches], dtype=np.float64)
    return np.array(centers), indices, weights, counts


def reconstruct(values, patch):
    _, indices, weights, counts = patch
    return (values[indices] * weights[..., None]).sum(axis=1) / counts[:, None]


def transport_analysis(reference, sweep):
    if reference["scene_generation"] != sweep["scene_generation"]:
        raise ValueError("sweep and reference scene generations differ")
    if reference["face_count"] != sweep["face_count"] or reference["transport"] != sweep["transport"]:
        raise ValueError("sweep and reference targets differ")
    rows = reference["faces"]
    if len(sweep["windows"]) != 4:
        raise ValueError("require all four sequence windows")
    counts = [p["paths"] for p in sweep["windows"][0]["prefixes"]]
    if not counts or counts != sorted(set(counts)):
        raise ValueError("invalid sample checkpoints")
    if any([p["paths"] for p in w["prefixes"]] != counts for w in sweep["windows"]):
        raise ValueError("sequence windows have different checkpoints")
    truth = np.array([r["mean"] for r in rows])
    patch = corner_members(rows, np.array(reference["origin"]))
    center_indices = patch[0]
    corners = reconstruct(truth, patch)
    results = []
    for window, data in enumerate(sweep["windows"]):
        previous = None
        for prefix in data["prefixes"]:
            measured = np.array(prefix["means"])
            if measured.shape != truth.shape:
                raise ValueError("missing sweep face measurements")
            display = reconstruct(measured, patch)
            error = rgb_error(display, corners)
            summary = stats(error)
            worst = int(error.argmax())
            row = rows[int(center_indices[worst // 4])]
            results.append(dict(window=window, sequence_start=data["sequence_start"], paths=prefix["paths"],
                centers=stats(rgb_error(measured[center_indices], truth[center_indices])),
                corners=summary,
                corners_within_5_percent=float(np.mean(error <= .05)),
                corners_within_10_percent=float(np.mean(error <= .1)),
                worst_corner=dict(voxel=row["voxel"], face=row["face"], corner=worst % 4),
                bands=[stats(np.abs(measured[center_indices, b] - truth[center_indices, b]) /
                             np.maximum(np.abs(truth[center_indices, b]), .02)) for b in range(5)],
                practical_pass=summary["p99"] <= .05 and summary["max"] <= .1,
                strict_pass=summary["p99"] <= .02 and summary["max"] <= .05,
                previous_prefix_corner_change=(stats(rgb_error(display, previous)) if previous is not None else None)))
            previous = display
    return results


def srgb_to_oklab(rgb):
    # Björn Ottosson's linear-sRGB matrices (2021 update):
    # https://bottosson.github.io/posts/oklab/ . No tone mapping in this tool.
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    lms = linear @ np.array([
        [.4122214708, .5363325363, .0514459929],
        [.2119034982, .6806995451, .1073969566],
        [.0883024619, .2817188376, .6299787005],
    ]).T
    return np.cbrt(lms) @ np.array([
        [.2104542553, .7936177850, -.0040720468],
        [1.9779984951, -2.4285922050, .4505937099],
        [.0259040371, .7827717662, -.8086757660],
    ]).T


def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def perceptual_difference(first, second, ppd):
    import flip_evaluator as flip
    if first.shape != second.shape:
        raise ValueError("capture dimensions differ")
    a, b = srgb_to_oklab(first), srgb_to_oklab(second)
    delta = np.linalg.norm(a - b, axis=-1)
    error_map, _, params = flip.evaluate(first, second, "LDR", applyMagma=False,
                                       parameters={"ppd": float(ppd)})
    error_map = np.asarray(error_map).reshape(first.shape[:2])
    h, w = error_map.shape
    tiles = [error_map[y:y + 32, x:x + 32].mean()
             for y in range(0, h, 32) for x in range(0, w, 32)]
    return dict(oklab_delta_e=stats(delta),
        oklab_absolute_lightness_difference=stats(np.abs(a[..., 0] - b[..., 0])),
        fraction_pixels_over_delta_e_002=float(np.mean(delta > .02)),
        flip=stats(error_map), flip_32px_tile_means=stats(tiles), flip_parameters=params)


def perceptual_analysis(folder, results, ppd):
    reference = load_rgb(folder / "cave-reference.png")
    images = []
    for r in results:
        path = folder / f"samples-{r['paths']}-window-{r['window']}.png"
        difference = perceptual_difference(reference, load_rgb(path), ppd)
        # Engineering gates declared before measuring this sweep. These are
        # conservative near-reference tolerances, NOT calibrated invisibility
        # probabilities. FLIP has no universal JND threshold.
        passed = (difference["oklab_delta_e"]["p99"] <= .01
                  and difference["oklab_delta_e"]["max"] <= .02
                  and difference["flip"]["mean"] <= .01
                  and difference["flip_32px_tile_means"]["max"] <= .05)
        images.append(dict(paths=r["paths"], window=r["window"],
                           image=path.name, near_reference_pass=passed, **difference))
    return dict(view=dict(width=reference.shape[1], height=reference.shape[0], ppd=ppd,
                    encoding="8-bit sRGB, game material/texture/tonemapping, fixed camera"),
        reference_replica_difference=perceptual_difference(
            load_rgb(folder / "reference-replica-a.png"), load_rgb(folder / "reference-replica-b.png"), ppd),
        images=images)


def minimum_sustained(rows, key):
    counts = sorted({r["paths"] for r in rows})
    passes = [all(r[key] for r in rows if r["paths"] == n) for n in counts]
    return next((n for i, n in enumerate(counts) if all(passes[i:])), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--perceptual", action="store_true", help="requires official flip-evaluator 1.7")
    parser.add_argument("--ppd", type=float, default=67, help="display pixels/degree, not camera FOV")
    args = parser.parse_args()
    reference = json.loads((args.run / "cave-reference.json").read_text())
    audit = json.loads((args.run / "cave-geometry-audit.json").read_text())
    status = json.loads((args.run / "cave-reference-status.json").read_text())
    if (not reference["qualification"]["qualified"] or audit["unknown_voxels"]
            or audit["mismatched_voxels"] or status["status"] != "completed"):
        raise ValueError("unqualified or incomplete reference")
    sweep = json.loads((args.run / "cave-sample-sweep.json").read_text())
    results = transport_analysis(reference, sweep)
    output = dict(criteria=dict(practical="Every window: 99% of chamber corners within 5% RGB, worst <=10%",
        strict="Every window: 99% of chamber corners within 2% RGB, worst <=5%",
        perceptual="Every window at captured view: OKLab p99<=.01/max<=.02, FLIP mean<=.01/worst 32px tile mean<=.05",
        persistence="Lowest tested count passing at itself and all larger checkpoints, all four windows"),
        minimum_practical_paths=minimum_sustained(results, "practical_pass"),
        minimum_strict_paths=minimum_sustained(results, "strict_pass"), measurements=results)
    if args.perceptual:
        output["perception"] = perceptual_analysis(args.run, results, args.ppd)
        output["minimum_captured_view_paths"] = minimum_sustained(
            output["perception"]["images"], "near_reference_pass")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
