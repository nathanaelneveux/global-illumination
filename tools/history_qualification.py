#!/usr/bin/env python3
"""Evaluate history policies against fixed native-GPU sample tapes.

Experiment branches supply history_candidate.policies(); baseline controls need
only this file. Policies never receive reference values, future samples or an
oracle telling them whether an edit matters. Audit queries are restricted to
the last four batches that existed before each event, and every replay is billed.
"""
import argparse
import copy
import json
import math
import unittest
from pathlib import Path


def percentile(values, fraction=0.95):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)]


def mean_reference(replicas):
    return [[(a + b) / 2 for a, b in zip(x, y)]
            for x, y in zip(*replicas)]


def error(estimate, target, scale):
    rgb = math.sqrt(sum((a - b) ** 2 for a, b in zip(estimate[:3], target[:3])))
    rgb /= max(0.02, math.sqrt(sum(x * x for x in scale[:3])))
    bands = max(abs(a - b) / max(0.002, s) for a, b, s in zip(estimate, target, scale))
    return rgb, bands


class History:
    def __init__(self, moment, weight):
        self.mean = moment["mean"][:]
        self.second = moment["second"][:]
        self.weight = float(weight)
        self.reductions = 0

    def integrate(self, fresh, paths, retained):
        retained = max(0.0, min(self.weight, retained, 65536 - paths))
        if retained < self.weight - 0.01:
            self.reductions += 1
        self.weight = retained + paths
        alpha = paths / self.weight
        self.mean = [a + (b - a) * alpha for a, b in zip(self.mean, fresh["mean"])]
        self.second = [a + (b - a) * alpha for a, b in zip(self.second, fresh["second"])]


class Audit:
    """Only old, already-seen batches are legal paired replay inputs."""
    def __init__(self, run, event_index, was_after, now_after, batch_paths):
        self.run = run
        self.event_index = event_index
        self.was_after = was_after
        self.now_after = now_after
        self.paths = 0
        self.batch_paths = batch_paths
        self.used = set()

    def replay(self, site, age=1):
        assert 1 <= age <= 4
        key = site, age
        assert key not in self.used, "a replay must not be billed/reused as fresh evidence twice"
        self.used.add(key)
        self.paths += self.batch_paths
        index = self.event_index + 4 - age
        old = self.run["after" if self.was_after else "before"][index][site]
        new = self.run["after" if self.now_after else "before"][index][site]
        return old, new


class Fixed:
    def __init__(self, name, cap):
        self.name, self.cap = name, cap

    def event(self, histories, sites, audit):
        self.pending = True

    def weights(self, histories, fresh, paths):
        cap = self.cap if self.pending else float("inf")
        self.pending = False
        return [min(h.weight, cap) for h in histories]


def policies():
    result = [Fixed("retain", 65536), Fixed("cap16", 16), Fixed("cap64", 64), Fixed("cap256", 256)]
    try:
        import history_candidate
    except ModuleNotFoundError as exc:
        if exc.name != "history_candidate":
            raise
    else:
        result.extend(history_candidate.policies())
    return result


def sustained(curve, key, threshold, steps):
    last_bad = max((i for i, c in enumerate(curve) if c[key] > threshold), default=-1)
    first = last_bad + 1
    if first > steps - 16:
        return None
    return curve[first]["total_paths_per_site"]


def evaluate(case, run, policy, batch_paths, warm_paths, steps):
    histories = [History(v, warm_paths) for v in run["warm"]]
    before, after = [mean_reference(case[key]) for key in ("references_before", "references_after")]
    scales = [[max(a, b) for a, b in zip(x, y)] for x, y in zip(before, after)]
    variation = [0.0] * len(histories)
    curves, replay_paths = [], 0
    now_after = False
    for step in range(steps):
        event = step == 0 or (case["repeated"] and step % 8 == 0)
        if event:
            was_after, now_after = now_after, not now_after
            audit = Audit(run, step, was_after, now_after, batch_paths)
            policy.event(histories, case["sites"], audit)
            replay_paths += audit.paths
        fresh = run["after" if now_after else "before"][step + 4]
        target = after if now_after else before
        weights = policy.weights(histories, fresh, batch_paths)
        rgb, bands = [], []
        for i, (h, f, w, reference, scale) in enumerate(zip(histories, fresh, weights, target, scales)):
            old = h.mean[:]
            h.integrate(f, batch_paths, w)
            r, b = error(h.mean, reference, scale)
            rgb.append(r)
            bands.append(b)
            if step >= steps // 2:
                # Same fixed compression for every policy; variation is not
                # reference accuracy and not rendered/corner variation.
                variation[i] += math.sqrt(sum((a / (0.25 + a) - b / (0.25 + b)) ** 2
                                              for a, b in zip(old[:3], h.mean[:3])))
        curves.append({"fresh_paths":(step + 1) * batch_paths,
                       "total_paths_per_site":(step + 1) * batch_paths + replay_paths / len(histories),
                       "rgb_p95":percentile(rgb), "rgb_max":max(rgb),
                       "all_band_p95":percentile(bands), "all_band_max":max(bands)})
    result = {"policy":policy.name, "seed":run["seed"], "case":case["name"],
              "final":curves[-1], "history_reductions":sum(h.reductions for h in histories),
              "weight_p50":percentile([h.weight for h in histories], 0.5),
              "late_face_variation_p95":percentile(variation),
              "replay_paths_per_site":replay_paths / len(histories),
              "sustained_rgb_p95_10":sustained(curves,"rgb_p95",0.1,steps),
              "sustained_all_band_p95_10":sustained(curves,"all_band_p95",0.1,steps),
              "sustained_all_band_max_20":sustained(curves,"all_band_max",0.2,steps),
              "checkpoints":[c for c in curves if c["fresh_paths"] in (64,128,256,512,1024,2048,4096,8192)],
              "curve":curves}
    return result


def analyze(tape):
    results, uncertainty = [], []
    for case in tape["cases"]:
        for which in ("references_before", "references_after"):
            ref = case[which]
            avg = mean_reference(ref)
            errors = [error(a,b,s) for a,b,s in zip(ref[0],ref[1],avg)]
            uncertainty.append({"case":case["name"],"target":which,
                                "rgb_p95":percentile([e[0] for e in errors]),
                                "all_band_p95":percentile([e[1] for e in errors])})
        for run in case["runs"]:
            for policy in policies():
                result = evaluate(case,run,policy,tape["batch_paths"],tape["warm_paths"],tape["steps"])
                results.append(result)
                brief = {k:v for k,v in result.items() if k not in ("curve","checkpoints")}
                print(json.dumps(brief))
    return {"uncertainty":uncertainty,"results":results,
            "note":"Native frozen-scene tapes, virtual batches; not gameplay FPS/seconds. Correlated replay is billed but never added as independent samples."}


class Tests(unittest.TestCase):
    def test_extinction_and_reweighting(self):
        h = History({"mean":[1]*5,"second":[1]*5},4096)
        zero = {"mean":[0]*5,"second":[0]*5}
        h.integrate(zero,16,16)
        self.assertEqual(h.mean,[0.5]*5)
        self.assertEqual(h.second,[0.5]*5)
        self.assertEqual(h.weight,32)

    def test_replay_only_consumes_pre_event_samples(self):
        old = [[i] for i in range(20)]
        new = [[i+100] for i in range(20)]
        audit = Audit({"before":old,"after":new},8,False,True,16)
        self.assertEqual(audit.replay(0,1),(11,111))
        self.assertEqual(audit.replay(0,4),(8,108))
        self.assertEqual(audit.paths,32)
        with self.assertRaises(AssertionError): audit.replay(0,1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape",nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["history_qualification.py"])
    else:
        report = analyze(json.loads(Path(args.tape).read_text()))
        if args.output:
            Path(args.output).write_text(json.dumps(report))
