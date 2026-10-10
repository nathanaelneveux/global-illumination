#!/usr/bin/env python3
"""Analyze fixed-receiver, zero-order-held temporal.json observations.

RGB is compressed as x/(0.25+x) before measuring variation. This is a
solver-space sensitivity measure, NOT a perceptual metric or rendered pixels.
Noisy and smooth fades with identical endpoints differ in excess variation.
An optional independent reference enables sustained-error/overshoot metrics;
the last frame of a noisy run is deliberately never used as ground truth.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import unittest


def compress(rgb):
    return [v / (0.25 + v) for v in rgb]


def series_metrics(series, start, end, reference=None, tolerance=0.02, dwell=2.0):
    """series: sorted (time, RGB-or-None). Missing spans break continuity."""
    if end <= start:
        raise ValueError("empty analysis interval")
    value = None
    points = []
    for at, rgb in series:
        if at <= start:
            value = rgb
        elif at < end:
            points.append((at, rgb))
    points = [(start, value)] + points + [(end, None)]
    variation = [0.0] * 3
    excess = [0.0] * 3
    missing = integral = max_step = overshoot = 0.0
    last_bad_end = start
    first = previous = None
    reference = compress(reference) if reference is not None else None

    def finish_segment():
        if first is not None:
            for c in range(3):
                excess[c] += max(0.0, variation[c] - abs(previous[c] - first[c]))
                variation[c] = 0.0

    for (at, rgb), (until, _) in zip(points, points[1:]):
        if rgb is None:
            missing += until - at
            last_bad_end = until
            finish_segment()
            first = previous = None
            continue
        current = compress(rgb)
        if first is None:
            first = current
        if previous is not None:
            delta = [abs(a - b) for a, b in zip(current, previous)]
            max_step = max(max_step, max(delta))
            variation = [a + b for a, b in zip(variation, delta)]
        previous = current
        if reference is not None:
            error = max(abs(a - b) for a, b in zip(current, reference))
            integral += error * (until - at)
            overshoot = max(overshoot, max(a - b for a, b in zip(current, reference)))
            if error > tolerance:
                last_bad_end = until
    finish_segment()
    result = dict(excess_variation=max(excess), max_step=max_step, missing_s=missing)
    if reference is not None:
        result.update(mean_error=integral / max(1e-9, end - start - missing),
                      overshoot=overshoot,
                      sustained_settle_s=last_bad_end - start if end - last_bad_end >= dwell else None,
                      reference_tolerance=tolerance, minimum_final_dwell_s=dwell)
    return result


def summarize(document, start, end, key, references=None):
    end = min(end, document["end_s"])
    by_site = [[] for _ in document["sites"]]
    for event in document["events"]:
        by_site[event["site"]].append(event)
    rows = []
    for site, events in enumerate(by_site):
        voxel = document["sites"][site]
        channels = range(4) if key == "corners" else [None]
        for corner in channels:
            values = []
            for event in events:
                state = event["state"]
                value = state.get(key) if state else None
                if value is not None:
                    value = value[corner] if corner is not None else value[:3]
                values.append((event["at_s"], value))
            label = ",".join(map(str, voxel)) + (f":{corner}" if corner is not None else "")
            reference = references.get(label) if references else None
            row = series_metrics(values, start, end, reference)
            row.update(site=label)
            relevant = [e["state"] for e in events if start <= e["at_s"] < end and e["state"]]
            row["history_drops"] = sum(b["samples"] < a["samples"] for a, b in zip(relevant, relevant[1:]))
            row["last_history"] = relevant[-1]["samples"] if relevant else None
            rows.append(row)
    complete = [r for r in rows if r["missing_s"] == 0]
    ordered = sorted(r["excess_variation"] for r in complete)
    return dict(interval_s=[start, end], field=key, complete_sites=len(complete), sites=len(rows),
                median_excess=statistics.median(ordered) if ordered else None,
                p95_excess=ordered[min(len(ordered)-1, math.ceil(len(ordered)*0.95)-1)] if ordered else None,
                worst=sorted(rows, key=lambda r: r["excess_variation"], reverse=True)[:8])


class MetricsTests(unittest.TestCase):
    def test_monotone_fade_is_not_oscillation(self):
        smooth = [(i, [i / 5] * 3) for i in range(6)]
        noisy = [(i, [v] * 3) for i, v in enumerate([0, 0.8, 0.2, 0.9, 0.4, 1])]
        self.assertAlmostEqual(series_metrics(smooth, 0, 6)["excess_variation"], 0)
        self.assertGreater(series_metrics(noisy, 0, 6)["excess_variation"], 0.5)

    def test_duplicated_frames_do_not_change_score(self):
        series = [(0, [0]*3), (1, [1]*3), (2, [0.2]*3)]
        duplicate = [(t+dt, v) for t, v in series for dt in [0, 0.5]]
        self.assertEqual(series_metrics(series, 0, 4, [0.2]*3), series_metrics(duplicate, 0, 4, [0.2]*3))

    def test_last_excursion_not_first_threshold_crossing(self):
        series = [(0, [0]*3), (1, [1]*3), (2, [0]*3), (3, [1]*3)]
        self.assertIsNone(series_metrics(series, 0, 4, [1]*3)["sustained_settle_s"])
        self.assertEqual(series_metrics(series, 0, 6, [1]*3)["sustained_settle_s"], 3)

    def test_missing_history_is_not_black_or_a_stable_result(self):
        series = [(0, [1]*3), (2, None)]
        result = series_metrics(series, 0, 6, [1]*3)
        self.assertEqual(result["missing_s"], 4)
        self.assertIsNone(result["sustained_settle_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="?", type=Path)
    parser.add_argument("--interval", action="append", default=[], help="start:end seconds; repeatable")
    parser.add_argument("--field", choices=["measured", "display", "mirror", "corners"], default="corners")
    parser.add_argument("--reference", type=Path, help='Independent RGB map: "x,y,z[:corner]": [r,g,b]')
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__])
    elif args.trace:
        data = json.loads(args.trace.read_text())
        refs = json.loads(args.reference.read_text()) if args.reference else None
        intervals = args.interval or ["10:30", "30:50", "50:70", "70:90"]
        print(json.dumps([summarize(data, *map(float, span.split(":")), args.field, refs)
                          for span in intervals], indent=2))
    else:
        parser.error("provide a trace or --test")
