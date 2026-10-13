#!/usr/bin/env python3
"""Whole-chamber reference agreement and live error, including rendered pixels.

Solver error and 8-bit display differences are separate metrics. This does not
turn replica agreement into a confidence certificate or a convergence timer.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


NORMALS = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def summary(values):
    ordered = sorted(values)
    if not ordered:
        return dict(count=0)
    return dict(count=len(values), p50=statistics.median(ordered),
                p95=ordered[math.ceil(len(ordered) * .95) - 1],
                p99=ordered[math.ceil(len(ordered) * .99) - 1], max=ordered[-1])


def error(a, b):
    return math.dist(a, b) / max(math.hypot(*b), .02)


def corners(rows, row, source):
    face = row["face"]
    u, v = ((1, 2), (0, 2), (0, 1))[face // 2]

    def sample(x, y):
        p = row["voxel"].copy()
        p[u] += x
        p[v] += y
        neighbor = rows.get((tuple(p), face))
        if neighbor is None:
            return [0, 0, 0], 0
        if source == "live":
            return neighbor["live_display"][:3], neighbor["live_display"][3]
        return neighbor[source][:3], 1

    result = []
    for x, y in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        members = [sample(0, 0), sample(x, 0), sample(0, y)]
        if members[1][1] > 0 or members[2][1] > 0:
            members.append(sample(x, y))
        weight = sum(w for _, w in members)
        result.append([sum(rgb[b] * w for rgb, w in members) / max(weight, 1e-6)
                       for b in range(3)])
    return result


def analyze(document):
    rows = {(tuple(r["voxel"]), r["face"]): r for r in document["faces"]}
    if len(rows) != document["face_count"]:
        raise ValueError("duplicate or missing reference faces")
    origin = document["origin"]
    chamber = []
    for row in rows.values():
        normal = NORMALS[row["face"]]
        air = [p + n - o for p, n, o in zip(row["voxel"], normal, origin)]
        if all(5 <= v <= high for v, high in zip(air, [58, 42, 58])):
            chamber.append(row)
    center_errors = [error(r["live_display"][:3], r["mean"][:3]) for r in chamber]
    corner_errors = [error(a, b) for r in chamber
                     for a, b in zip(corners(rows, r, "live"), corners(rows, r, "mean"))]
    return dict(qualification=document["qualification"], paths_per_replica=document["paths_per_replica"],
                reference_faces=len(rows), chamber_faces=len(chamber),
                live_center_rgb_error=summary(center_errors),
                live_corner_rgb_error=summary(corner_errors),
                corner_fraction_within_10_percent=sum(v <= .1 for v in corner_errors) / len(corner_errors),
                by_orientation=[dict(face=f, normal=n,
                                     center_error=summary([e for r, e in zip(chamber, center_errors) if r["face"] == f]),
                                     live_history=summary([r["live_history"] for r in chamber if r["face"] == f]))
                                for f, n in enumerate(NORMALS)],
                note="Live cached RGB versus finite-depth reference. This cave profile has no custom-visible analytic source; such a source needs its direct RGB added to these face references. Halo faces excluded as centers; their values still reconstruct chamber corners. No pixel perceptual or convergence-time claim.")


def pixel_comparison(folder):
    from PIL import Image, ImageChops, ImageStat

    def compare(a, b):
        with Image.open(folder / a) as first, Image.open(folder / b) as second:
            if first.size != second.size:
                raise ValueError("reference image dimensions differ")
            diff = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
            bands = diff.split()
            maximum = ImageChops.lighter(ImageChops.lighter(bands[0], bands[1]), bands[2])
            histogram = maximum.histogram()
            pixels = sum(histogram)
            cumulative = 0
            p99 = 0
            for value, count in enumerate(histogram):
                cumulative += count
                if cumulative >= pixels * .99:
                    p99 = value
                    break
            return dict(width=first.width, height=first.height,
                        mean_absolute_rgb_byte_difference=statistics.mean(ImageStat.Stat(diff).mean),
                        p99_max_channel_byte_difference=p99,
                        max_channel_byte_difference=max(i for i, n in enumerate(histogram) if n),
                        fraction_pixels_differing_by_more_than_two_bytes=sum(histogram[3:]) / pixels)

    held = [hashlib.sha256((folder / name).read_bytes()).hexdigest()
            for name in ["cave-reference.png", "cave-reference-held.png"]]
    return dict(replica_a_vs_b=compare("reference-replica-a.png", "reference-replica-b.png"),
                live_vs_reference=compare("live-frozen.png", "cave-reference.png"),
                held_mean_png_byte_identical=held[0] == held[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--pixels", action="store_true", help="also compare captures; requires Pillow")
    args = parser.parse_args()
    result = analyze(json.loads((args.run / "cave-reference.json").read_text()))
    audit_path = args.run / "cave-geometry-audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    result["geometry_audit"] = (dict(available=True,
        unknown_voxels=audit["unknown_voxels"], mismatched_voxels=audit["mismatched_voxels"])
        if audit else dict(available=False))
    result["reference_qualified"] = bool(result["qualification"]["qualified"] and audit
        and audit["unknown_voxels"] == 0 and audit["mismatched_voxels"] == 0)
    if args.pixels:
        result["pixels"] = pixel_comparison(args.run)
    print(json.dumps(result, indent=2))
