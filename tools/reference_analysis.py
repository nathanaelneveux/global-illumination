#!/usr/bin/env python3
"""Compare event-complete receiver histories to an independent final-scene reference.

RGB relative error is Euclidean error / max(reference norm, 0.02). The floor
keeps near-black receivers meaningful. Reference replicas use disjoint paths
but the same transport model: agreement measures sampling uncertainty, not
physical correctness. Earlier residency may differ from the final scene.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import unittest


def relative_error(value, reference):
    return math.dist(value[:3], reference[:3]) / max(math.hypot(*reference[:3]), 0.02)


def percentiles(values):
    ordered = sorted(values)
    if not ordered:
        return None
    return dict(median=statistics.median(ordered),
                p95=ordered[math.ceil(len(ordered) * 0.95) - 1], max=ordered[-1])


def analyze(trace, reference, times, tolerance=0.1, dwell=10.0):
    if len(trace["sites"]) != len(trace["site_faces"]):
        raise ValueError("incomplete receiver identities")
    sites = [(tuple(v), f) for v, f in zip(trace["sites"], trace["site_faces"])]
    ref_sites = [(tuple(s["voxel"]), s["face"]) for s in reference["sites"]]
    if sites != ref_sites:
        raise ValueError("reference receiver identity/order differs from temporal trace")
    a, b = reference["references"]
    if len(a) != len(sites) or len(b) != len(sites):
        raise ValueError("incomplete reference")
    refs = [[(x + y) * 0.5 for x, y in zip(aa[:3], bb[:3])]
            for aa, bb in zip(a, b)]
    events = [[] for _ in sites]
    for event in trace["events"]:
        events[event["site"]].append(event)
    snapshots = []
    for at in times:
        if at > trace["end_s"]:
            continue
        states = [next((e["state"] for e in reversed(es) if e["at_s"] <= at), None) for es in events]
        row = dict(at_s=at, present=sum(s is not None for s in states), sites=len(sites))
        for field in ["measured", "display"]:
            errors = [relative_error(s[field], ref) for s, ref in zip(states, refs) if s]
            row[field] = dict(relative_error=percentiles(errors),
                              within_tolerance=sum(e <= tolerance for e in errors))
        row["samples"] = percentiles([s["samples"] for s in states if s])
        snapshots.append(row)
    settling = {}
    for field in ["measured", "display"]:
        times_settled = []
        for es, ref in zip(events, refs):
            last_bad = 0.0
            for e, until in zip(es, [e["at_s"] for e in es[1:]] + [trace["end_s"]]):
                state = e["state"]
                if state is None or relative_error(state[field], ref) > tolerance:
                    last_bad = until
            if not es or es[0]["at_s"] > 0.0:
                last_bad = max(last_bad, es[0]["at_s"] if es else trace["end_s"])
            times_settled.append(last_bad if trace["end_s"] - last_bad >= dwell else None)
        settling[field] = dict(settled=sum(t is not None for t in times_settled),
                               sites=len(sites), seconds=percentiles([t for t in times_settled if t is not None]))
    return dict(reference_paths_each=reference["paths_per_replicate"],
                reference_pair_disagreement=percentiles([relative_error(aa, bb) for aa, bb in zip(a, b)]),
                end_s=trace["end_s"], relative_tolerance=tolerance, minimum_final_dwell_s=dwell,
                snapshots=snapshots, sustained_until_end=settling,
                caveat="Final-scene target, not time-varying ground truth; same-estimator reference, not a physical unbiasedness certificate.")


class Tests(unittest.TestCase):
    def test_floor_and_rgb_norm(self):
        self.assertEqual(relative_error([0, 0, 0], [0, 0, 0]), 0)
        self.assertAlmostEqual(relative_error([0.001, 0, 0], [0, 0, 0]), 0.05)
        self.assertAlmostEqual(relative_error([1.1] * 3, [1] * 3), 0.1)

    def test_last_bad_and_missing_are_not_a_pass(self):
        trace = dict(sites=[[0, 0, 0]], site_faces=[0], end_s=30, events=[
            dict(site=0, at_s=t, state=None if v is None else dict(measured=[v]*3, display=[v]*3, samples=10))
            for t, v in [(0, None), (1, 1), (20, 0), (22, 1)]])
        ref = dict(sites=[dict(voxel=[0, 0, 0], face=0)], references=[[[1]*5], [[1]*5]], paths_per_replicate=100)
        result = analyze(trace, ref, [0, 1, 29])
        self.assertEqual(result["snapshots"][0]["present"], 0)
        self.assertEqual(result["sustained_until_end"]["measured"]["settled"], 0)
        trace["end_s"] = 35
        result = analyze(trace, ref, [])
        self.assertEqual(result["sustained_until_end"]["measured"]["seconds"]["max"], 22)
        ref["sites"][0]["face"] = 1
        with self.assertRaises(ValueError):
            analyze(trace, ref, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="?", type=Path, help="directory with temporal.json and reference.json")
    parser.add_argument("--times", default="15,30,45,60,90,120,150,180")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=[__file__])
    elif args.run:
        print(json.dumps(analyze(json.loads((args.run / "temporal.json").read_text()),
                                 json.loads((args.run / "reference.json").read_text()),
                                 list(map(float, args.times.split(",")))), indent=2))
    else:
        parser.error("provide a run directory or --test")
