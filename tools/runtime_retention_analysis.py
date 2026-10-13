#!/usr/bin/env python3
"""Pair live cave observations with each run's own frozen-scene reference.

The raw 2048-path limits are useful diagnostics on these sparse receivers, not
a whole-image perceptual pass. Reference sampling and image capture happen
outside timed gameplay. Earlier residency can differ from the final scene.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from temporal_analysis import summarize


def distribution(values):
    a = np.asarray(values, dtype=float)
    if not a.size or not np.all(np.isfinite(a)):
        raise ValueError("missing or nonfinite measurement")
    return dict(median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), max=float(np.max(a)))


def relative(values, reference):
    return np.linalg.norm(values-reference, axis=-1) / np.maximum(
        np.linalg.norm(reference, axis=-1), .02)


def validate_capture(folder, report):
    status = json.loads((folder / "status.json").read_text())
    captures = report["capture_windows"]
    if (status["status"] != "completed" or report["status"] != "completed"
            or not captures or len(captures) != report["captures"]):
        raise ValueError("incomplete workload/captures")
    for capture in captures:
        if not capture["varied"] or (capture["width"], capture["height"]) != (2560, 1440):
            raise ValueError("blank or mismatched capture")
        if report["capture_mode"] == "window" and not (capture["focused"] and capture["visible"]):
            raise ValueError("unfocused/hidden window")
        if report["capture_mode"] == "offscreen" and not capture["target_valid"]:
            raise ValueError("invalid offscreen target")


def analyze(folder):
    report = json.loads((folder / "report.json").read_text())
    trace = json.loads((folder / "temporal.json").read_text())
    validate_capture(folder, report)
    if not (folder / "reference.json").exists():
        if report["capture_mode"] != "window":
            raise ValueError("headless accuracy run is missing its reference")
        return dict(run=str(folder), policy=report["spatial_retention"],
                    capture_mode="window", phases=report["measurements"]["phases"],
                    traced_paths=report["traced_paths"], max_loaded_pages=report["max_loaded_pages"],
                    background=report["background_terrain"],
                    stability=[summarize(trace,a,b,"corners") for a,b in ((30,60),(60,120),(120,180))],
                    note="Windowed stability/frame pacing only; no accuracy reference for this run.")
    ref = json.loads((folder / "reference.json").read_text())
    if report["status"] != "completed" or trace.get("overflow", False):
        raise ValueError("incomplete workload")
    identities = [(v, f) for v, f in zip(trace["sites"], trace["site_faces"])]
    if identities != [(s["voxel"], s["face"]) for s in ref["sites"]]:
        raise ValueError("reference receiver identity/order differs")
    centers = np.asarray(ref["references"], dtype=float)[..., :3]
    corners = np.asarray(ref["corner_references"], dtype=float)[..., :3]
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(corners)):
        raise ValueError("nonfinite reference")
    bank = [[] for _ in identities]
    for event in trace["events"]:
        bank[event["site"]].append(event)

    def snapshot(at):
        states = [next((e["state"] for e in reversed(es) if e["at_s"] <= at), None)
                  for es in bank]
        if any(s is None or s.get("corners") is None for s in states):
            return dict(at_s=at, complete=False)
        measured = np.array([s["measured"] for s in states])
        display = np.array([s["corners"] for s in states])
        a = relative(measured, centers.mean(axis=0))
        b = relative(display, corners.mean(axis=0))
        return dict(at_s=at, complete=True, centers=distribution(a), corners=distribution(b),
                    effective_history=distribution([s["samples"] for s in states]),
                    sequence=distribution([s["sequence"] for s in states]))

    times = [t for t in (15, 30, 60, 120, 180) if t <= trace["end_s"]]
    snapshots = [snapshot(t) for t in times]
    intervals = []
    for start, end in ((30, 60), (60, 120), (120, 180)):
        if end > trace["end_s"]:
            continue
        samples = [snapshot(t) for t in range(start, end+1)]
        if not all(s["complete"] for s in samples):
            raise ValueError("missing receiver or display in analysis interval")
        intervals.append(dict(interval_s=[start, end],
            stability=summarize(trace, start, end, "corners"),
            worst_center_p99=max(s["centers"]["p99"] for s in samples),
            worst_corner_p99=max(s["corners"]["p99"] for s in samples),
            median_corner_p99=float(np.median([s["corners"]["p99"] for s in samples]))))
    return dict(run=str(folder), policy=report["spatial_retention"],
                capture_mode=report["capture_mode"], phases=report["measurements"]["phases"],
                traced_paths=report["traced_paths"], transport_batches=report["transport_batches"],
                max_loaded_pages=report["max_loaded_pages"], background=report["background_terrain"],
                reference_paths_each=ref["paths_per_replicate"],
                reference_disagreement=dict(centers=distribution(relative(centers[0], centers[1])),
                                            corners=distribution(relative(corners[0], corners[1]))),
                snapshots=snapshots, intervals=intervals,
                note=__doc__.strip())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([analyze(folder) for folder in args.runs], indent=2))
