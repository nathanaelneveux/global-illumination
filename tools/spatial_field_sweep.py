#!/usr/bin/env python3
"""Screen spatial strength on saved raw prefixes before spending another GPU run.

This is a field diagnostic, not a visual gate. Render promising candidates and
qualify the native shader separately. Uses the same connected topology/formula
as spatial.rs, varying only the uncertainty scale at which blending fades.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cave_sample_analysis import corner_members, reconstruct, rgb_error


def donor_indices(rows):
    lookup = {(tuple(r["voxel"]), r["face"]): i for i, r in enumerate(rows)}
    result = np.full((len(rows), 20), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        p = np.array(row["voxel"])
        axes = ((1, 2), (0, 2), (0, 1))[row["face"] // 2]
        directions = [np.eye(3, dtype=int)[a] * s for a in axes for s in (1, -1)]
        for axis, offset in enumerate(directions):
            for d in range(1, 5):
                n = lookup.get((tuple(p + offset * d), row["face"]), -1)
                result[i, axis * 4 + d - 1] = n
                if n < 0:
                    break
        for k, (a, b) in enumerate(((0, 2), (0, 3), (1, 2), (1, 3))):
            if result[i, a * 4] >= 0 or result[i, b * 4] >= 0:
                result[i, 16 + k] = lookup.get((tuple(p + directions[a] + directions[b]), row["face"]), -1)
    return result


def filtered(mean, second, count, donors, scale):
    sigma = np.sqrt(np.maximum(second - mean * mean, 0) / count)
    noise = np.max(sigma / (.002 + np.abs(mean)), axis=1)
    blend = noise ** 2 / (noise ** 2 + scale ** 2)
    other = mean[np.maximum(donors, 0)]
    tolerance = (.002 + .1 * np.maximum(np.abs(mean[:, None]), np.abs(other))
                 + 2 * (sigma[:, None] + sigma[np.maximum(donors, 0)]))
    delta = np.max(np.abs(mean[:, None] - other) / tolerance, axis=2)
    distance = np.array([1, 4, 9, 16] * 4 + [2] * 4)
    weights = np.exp(-.125 * distance - .5 * delta ** 2) * count / (count + 8)
    weights *= donors >= 0
    average = (mean + (other * weights[..., None]).sum(axis=1)) / (1 + weights.sum(axis=1))[:, None]
    return mean + (average - mean) * np.where(blend > .001, blend, 0)[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    reference = json.loads((args.run / "cave-reference.json").read_text())
    sweep = json.loads((args.run / "cave-sample-sweep.json").read_text())
    if not reference["qualification"]["qualified"] or reference["scene_generation"] != sweep["scene_generation"]:
        raise ValueError("require qualified matching scene")
    rows = reference["faces"]
    donors = donor_indices(rows)
    patches = corner_members(rows, np.array(reference["origin"]))
    truth = reconstruct(np.array([r["mean"] for r in rows])[:, :3], patches)
    output = []
    for window, data in enumerate(sweep["windows"]):
        for prefix in data["prefixes"]:
            mean = np.array(prefix["means"])[:, :3]
            second = np.array(prefix["second_moments"])
            for scale in (None, .15, .075, .05, .025):
                value = mean if scale is None else filtered(mean, second, prefix["paths"], donors, scale)
                corners = reconstruct(value, patches)
                error = rgb_error(corners, truth)
                output.append(dict(window=window, paths=prefix["paths"], scale=scale,
                    p95=float(np.quantile(error, .95)), p99=float(np.quantile(error, .99)),
                    signed_rgb_error=(corners - truth).mean(axis=0).tolist()))
    print(json.dumps(output, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
