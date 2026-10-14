#!/usr/bin/env python3
"""Attribute full-render FLIP to distance bands on audited voxel receivers.

Ray/quad intersection supplies a pixel-to-receiver map, not replacement lighting.
FLIP is evaluated on the original full images before masking, preserving its
spatial filters. Distances use voxel coordinates, matching the retention policy.
Regional scores are diagnostics, not newly calibrated perceptual pass gates.
"""
import argparse
import json
from pathlib import Path
import unittest

import numpy as np

from cave_sample_analysis import load_rgb
from near_history_calibration import digest, edit_distances, flip_map, flip_statistics


def receiver_map(sites, view, width=2560, height=1440):
    """Nearest front-facing unit quad, at pixel centers, in a perspective view.

    Greedy meshing changes topology but not these axis-aligned surface planes.
    Unmapped pixels remain -1; never assign sky or omitted geometry a distance.
    This is geometric attribution, not per-sample contributor attribution:
    corner reconstruction and FLIP's filters span neighboring receivers.
    """
    eye = np.asarray(view["eye"], dtype=float)
    forward = np.asarray(view["target"], dtype=float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 1., 0.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tangent = np.tan(np.deg2rad(view.get("fov_degrees", 65.)) / 2)
    xs = (2 * (np.arange(width) + .5) / width - 1) * tangent * width / height
    ys = (1 - 2 * (np.arange(height) + .5) / height) * tangent
    depth = np.full((height, width), np.inf)
    identity = np.full((height, width), -1, dtype=np.int32)
    for index, (voxel, face) in enumerate(sites):
        axis, sign = face // 2, 1 if face % 2 == 0 else -1
        base = np.asarray(voxel, dtype=float).copy()
        base[axis] += int(sign > 0)
        if (eye[axis] - base[axis]) * sign <= 0:
            continue
        u, v = [i for i in range(3) if i != axis]
        corners = np.tile(base, (4, 1))
        corners[:, u] += [0, 1, 0, 1]
        corners[:, v] += [0, 0, 1, 1]
        delta = corners - eye
        z = delta @ forward
        if z.max() <= 0:
            continue
        if z.min() <= 1e-8:
            x0, x1, y0, y1 = 0, width, 0, height
        else:
            sx = (delta @ right / z / (tangent * width / height) + 1) * width / 2
            sy = (1 - delta @ up / z / tangent) * height / 2
            x0, x1 = np.clip([np.floor(sx.min()), np.ceil(sx.max())], 0, width).astype(int)
            y0, y1 = np.clip([np.floor(sy.min()), np.ceil(sy.max())], 0, height).astype(int)
        if x1 <= x0 or y1 <= y0:
            continue
        rays = (forward + xs[None, x0:x1, None] * right
                + ys[y0:y1, None, None] * up)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (base[axis] - eye[axis]) / rays[:, :, axis]
            hit = eye + t[:, :, None] * rays
        valid = ((t > 0) & (t < depth[y0:y1, x0:x1])
                 & (hit[:, :, u] >= base[u]) & (hit[:, :, u] <= base[u] + 1)
                 & (hit[:, :, v] >= base[v]) & (hit[:, :, v] <= base[v] + 1))
        depth[y0:y1, x0:x1][valid] = t[valid]
        identity[y0:y1, x0:x1][valid] = index
    return identity


def regions(error, distance, edges):
    rows = []
    for low, high in zip(edges, edges[1:]):
        mask = (distance >= low) & (distance < high)
        count = int(mask.sum())
        if not count:
            continue
        # Avoid a tiny sliver of a distance bin pretending to be a whole tile.
        counts = mask.reshape(45, 32, 80, 32).sum((1, 3))
        sums = np.where(mask, error, 0).reshape(45, 32, 80, 32).sum((1, 3))
        eligible = counts >= 256
        rows.append(dict(min_distance=low, max_distance=high, pixels=count,
                         mean=float(error[mask].mean(dtype=np.float64)),
                         tile_mean_max=float((sums[eligible] / counts[eligible]).max())
                         if eligible.any() else None))
    return rows


def analyze(args):
    plan = json.loads(args.plan.read_text())
    tape_path = Path(plan["tape"]) / "near-history.json"
    if digest(tape_path) != plan["tape_sha256"]:
        raise ValueError("native tape hash differs from replay")
    tape = json.loads(tape_path.read_text())
    status = json.loads((args.run / "cave-model-comparison-status.json").read_text())
    if status["status"] != "completed" or not status["geometry_audit_passed"]:
        raise ValueError("incomplete rendering or invalid geometry")
    ids = receiver_map(tape["sites"], tape["view"])
    mapped = ids >= 0
    distances = edit_distances(tape)
    distance = np.where(mapped, distances[np.maximum(ids, 0)], np.nan)
    # Fail instead of drawing conclusions from a substantially different view.
    if mapped.mean() < .9:
        raise ValueError(f"only {mapped.mean():.1%} of pixels map to audited receivers")
    refs = {f"near-{state}-reference": load_rgb(args.run / f"near-{state}-reference.png")
            for state in ("before", "after")}
    results = []

    def measure(name, reference, candidate, metadata):
        error = flip_map(reference, candidate)
        row = dict(metadata, name=name, **flip_statistics(error),
                   bands=regions(error, distance, args.edges))
        results.append(row)
        print(name, row["flip"], row["bands"], flush=True)

    measure("full-state-change", refs["near-after-reference"], refs["near-before-reference"], {})
    for frame in plan["frames"]:
        if (frame.get("case") not in ("remove", "add")
                or frame["fresh_paths"] not in args.paths
                or frame["history"] not in args.histories):
            continue
        measure(frame["name"], refs[frame["target"]], load_rgb(args.run / f"{frame['name']}.png"), frame)
    sun = None
    if tape["states"][0].get("sky") is not None:
        before, after = [np.asarray(s["sky"], dtype=np.uint32) for s in tape["states"]]
        faces = np.asarray([s[1] for s in tape["sites"]])
        changed = np.any(before != after, axis=1)
        floor = faces == 2
        gained = np.any(((before >> 24) == 0) & ((after >> 24) != 0), axis=1)
        visible = np.bincount(ids[mapped], minlength=len(faces))
        sun = dict(changed_faces=int(changed.sum()), changed_floor_faces=int((changed & floor).sum()),
                   newly_lit_floor_faces=int((gained & floor).sum()),
                   visible_changed_floor_pixels=int(visible[changed & floor].sum()),
                   visible_newly_lit_floor_pixels=int(visible[gained & floor].sum()))
        if tape.get("fixture") == "house-sun-patch" and (
                sun["newly_lit_floor_faces"] < 4 or sun["visible_newly_lit_floor_pixels"] < 10000):
            raise ValueError(f"sun-patch fixture does not exercise visible floor corners: {sun}")
    output = dict(plan_sha256=digest(args.plan), tape_sha256=digest(tape_path),
                  render_run=str(args.run.resolve()), mapped_pixels=int(mapped.sum()),
                  unmapped_pixels=int((~mapped).sum()), distance_min=float(np.nanmin(distance)),
                  distance_max=float(np.nanmax(distance)), sun_corner_audit=sun, measurements=results,
                  note="Full-image FLIP before masking; receiver-voxel distance as in retention. Regional means are diagnostic, not calibrated pass gates. Pixel-center analytic quad mapping omits antialiasing at silhouettes; corner reconstruction and perceptual filters span neighboring receivers. Frozen captures do not measure gameplay latency.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


class Tests(unittest.TestCase):
    def test_nearest_quad_and_unmapped_pixels(self):
        view = dict(eye=[.5, .5, 3], target=[.5, .5, 0], fov_degrees=60)
        ids = receiver_map([[[0, 0, 0], 4], [[0, 0, -1], 4]], view, 32, 32)
        self.assertEqual(ids[16, 16], 0)
        self.assertEqual(ids[0, 0], -1)
        self.assertFalse(np.any(ids == 1))
        self.assertTrue(np.all(receiver_map([[[0, 0, 0], 5]], view, 32, 32) == -1))

    def test_distance_bands_do_not_double_count_boundaries(self):
        distance = np.full((1440, 2560), 16.)
        distance[:, :1280] = 8.
        rows = regions(np.full_like(distance, .1), distance, [0, 16, 32])
        self.assertEqual(sum(r["pixels"] for r in rows), distance.size)
        self.assertAlmostEqual(rows[0]["mean"], .1)
        self.assertAlmostEqual(rows[1]["tile_mean_max"], .1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--edges", nargs="+", type=float, default=[0, 4, 8, 12, 16, 24, 32, 48, 64, 96])
    parser.add_argument("--paths", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--histories", nargs="+", type=int, default=[16, 256])
    args = parser.parse_args()
    if len(args.edges) < 2 or not np.all(np.isfinite(args.edges)) or any(
            b <= a for a, b in zip(args.edges, args.edges[1:])):
        parser.error("distance edges must be finite and strictly increasing")
    analyze(args)
