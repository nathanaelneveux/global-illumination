#!/usr/bin/env python3
"""Qualify distance-only retention on native before/after path tapes.

Policies see distance, not reference light changes. Reference-informed caps are
reported separately as an offline diagnostic, never passed to a candidate.
Metrics are linear indirect face energy, not perceptual/display/FPS evidence.
"""
import argparse
import json
import math
import unittest
from pathlib import Path

import numpy as np

POLICIES = ("baseline", "quadratic", "quartic", "cutoff384", "retain")


def permitted_history(delta, fresh, tolerance, available=4096):
    """Bias budget only: |bias| = H/(H+N)*|delta|, before runtime saturation.

    Delta/tolerance must use the same fixed units. Sampling noise is NOT
    included; native tail checks are still necessary. No finite radius is
    implied by this algebra.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (delta, fresh, tolerance, available)):
        raise ValueError("finite nonnegative inputs required")
    if delta <= tolerance:
        return available
    return min(available, fresh * tolerance / (delta - tolerance))


def caps(policy, distance):
    distance = np.asarray(distance, dtype=float)
    if policy == "retain":
        return np.full_like(distance, 65536)
    if policy not in POLICIES:
        raise ValueError(policy)
    far = np.full_like(distance, 256)
    if policy in ("quadratic", "quartic"):
        exponent = 2 if policy == "quadratic" else 4
        far = np.clip(256 * (distance / 96) ** exponent, 256, 65536)
    elif policy == "cutoff384":
        far = np.where(distance >= 384, 65536, 256)
    return np.where(distance <= 48, 16, far)


def summary(values):
    values = np.asarray(values, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError("empty/nonfinite measurement")
    return {"mean": float(np.mean(values)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)), "max": float(np.max(values))}


def rgb_error(a, b, scale):
    return np.linalg.norm((a - b)[..., :3], axis=-1) / scale


def moments(run, key):
    value = np.asarray([[m["mean"] for m in batch] for batch in run[key]], dtype=float)
    if not np.isfinite(value).all() or np.min(value) < 0:
        raise ValueError("invalid path tape")
    return value


def replay(case, index, reverse, repeated, policy, batch_paths, warm_paths):
    before, after = [np.asarray(case["references_" + side], dtype=float).mean(axis=0)
                     for side in ("before", "after")]
    refs = [before, after]
    sides = ("before", "after")
    side = int(reverse)
    run = case[sides[side]][index]
    mean = np.asarray([m["mean"] for m in run["warm"]], dtype=float)
    scale = np.maximum(0.02, np.linalg.norm(np.maximum(before, after)[:, :3], axis=1))
    band_scale = np.maximum(0.02, np.maximum(before, after))
    weight = np.full(len(mean), float(warm_paths))
    limit = caps(policy, [s["distance"] for s in case["sites"]])
    bank = [moments(case[s][index], "batches") for s in sides]
    if bank[0].shape != bank[1].shape or bank[0].shape[1:] != mean.shape:
        raise ValueError("inconsistent path tape")
    steps = len(bank[0])
    luminance = np.array([0.2126, 0.7152, 0.0722])
    endpoints = np.stack([mean[:, :3] @ luminance, before[:, :3] @ luminance,
                          after[:, :3] @ luminance])
    low, high = endpoints.min(axis=0), endpoints.max(axis=0)
    brightness_scale = np.maximum(0.02, high)
    bright, dark = np.zeros(len(mean)), np.zeros(len(mean))
    variation = np.zeros(len(mean))
    curves, late, band_late = [], [], []
    # Include the initial state: a retained accurate cache can be settled at
    # zero additional paths. Require at least 16 final good batches.
    target = refs[1 - side]
    initial_error = rgb_error(mean, target, scale)
    last_bad = np.where(initial_error > 0.05, 0, -1)
    for step in range(steps):
        if step == 0 or (repeated and step % 8 == 0):
            side = 1 - side
            weight = np.minimum(weight, limit)
        old = mean.copy()
        retained = np.minimum(weight, 65536 - batch_paths)
        weight = retained + batch_paths
        mean += (bank[side][step] - mean) * (batch_paths / weight[:, None])
        error = rgb_error(mean, refs[side], scale)
        last_bad = np.where(error > 0.05, step + 1, last_bad)
        brightness = mean[:, :3] @ luminance
        bright = np.maximum(bright, np.maximum(0, brightness - high) / brightness_scale)
        dark = np.maximum(dark, np.maximum(0, low - brightness) / brightness_scale)
        if step >= steps // 2:
            variation += np.linalg.norm((mean / (0.25 + mean) - old / (0.25 + old))[:, :3], axis=1)
            late.append(error)
            band_late.append(np.abs(mean - refs[side]) / band_scale)
        if (step + 1) in (1,4,8,16,32,64,128,256,512):
            curves.append({"fresh_paths": (step + 1) * batch_paths,
                           "rgb":summary(error)})
    settling = np.where(last_bad == -1, 0, (last_bad + 1) * batch_paths)
    settled = last_bad < steps - 16
    return {"policy":policy,"seed":run["seed"],"reverse":reverse,"repeated":repeated,
            "late_rgb":summary(late),"late_bands":[summary(np.asarray(band_late)[...,i]) for i in range(5)],
            "late_variation":summary(variation),"bright_excursion":summary(bright),
            "dark_excursion":summary(dark),"final_rgb":summary(error),
            "settled_fraction_5pct":float(np.mean(settled)),
            "all_settled_paths_5pct":int(settling.max()) if settled.all() else None,
            "weight":summary(weight),"curve":curves}


def analyze(tape):
    if tape["schema"] != 1 or tape["warm_paths"] + tape["steps"] * tape["batch_paths"] > 65536:
        raise ValueError("unexpected schema or saturation invalidates simple bias model")
    reports = []
    for case in tape["cases"]:
        refs = [np.asarray(case["references_" + s], dtype=float) for s in ("before","after")]
        before, after = [r.mean(axis=0) for r in refs]
        scale = np.maximum(0.02,np.linalg.norm(np.maximum(before,after)[:,:3],axis=1))
        delta = rgb_error(before,after,scale)
        distance = np.asarray([s["distance"] for s in case["sites"]])
        bins = []
        for lo, hi in zip((0,48,96,192,256,384,512),(48,96,192,256,384,512,float("inf"))):
            mask = (distance >= lo) & (distance < hi)
            if not mask.any():
                continue
            allowed = [permitted_history(d,256,0.02,tape["warm_paths"]) for d in delta[mask]]
            bins.append({"distance":[lo,hi if math.isfinite(hi) else None],"sites":int(mask.sum()),
                         "change_rgb":summary(delta[mask]),
                         "reference_difference_rgb":[summary(rgb_error(r[0],r[1],scale)[mask]) for r in refs],
                         "oracle_bias_cap_2pct_after_256":summary(allowed)})
        runs = []
        for index in range(len(case["before"])):
            if case["before"][index]["seed"] != case["after"][index]["seed"]:
                raise ValueError("unpaired sequences")
            for reverse in (False,True):
                for repeated in ((False,True) if case["name"] == "cave_far_residency" else (False,)):
                    for policy in POLICIES:
                        runs.append(replay(case,index,reverse,repeated,policy,tape["batch_paths"],tape["warm_paths"]))
        reports.append({"name":case["name"],"bounds":case["changed_bounds"],"bins":bins,"runs":runs})
    return {"schema":1,"reference_paths_per_window":tape["reference_paths_per_window"],"cases":reports,
            "note":"Offline distance-only policies; oracle caps are bias-budget diagnostics, not policy inputs. Tail metrics pool late time/sites per run; two QMC prefixes are sensitivity checks, not confidence intervals. No runtime promotion or perceptual/FPS claim."}


class Tests(unittest.TestCase):
    def test_bias_budget(self):
        for delta in (0,0.01,0.02,0.05,0.1,1.0):
            h = permitted_history(delta,256,0.02)
            self.assertLessEqual(h / (h+256) * delta,0.02+1e-12)
        self.assertEqual(permitted_history(0,256,0.02),4096)
        self.assertAlmostEqual(permitted_history(1,256,0.02),256/49)
        with self.assertRaises(ValueError):
            permitted_history(float("nan"),256,0.02)

    def test_policy_boundaries(self):
        np.testing.assert_array_equal(caps("baseline",[0,48,49,384,1000]),[16,16,256,256,256])
        np.testing.assert_array_equal(caps("quartic",[48,96,192,384,768]),[16,256,4096,65536,65536])
        np.testing.assert_array_equal(caps("cutoff384",[383.99,384]),[256,65536])

    def test_runtime_mean_blend_obeys_bias_law(self):
        from history_qualification import History
        for retained in (16,256,1024,4096):
            h = History({"mean":[1.0]*5,"second":[1.0]*5},4096)
            fresh = {"mean":[0.0]*5,"second":[0.0]*5}
            for step in range(16):
                h.integrate(fresh,16,retained if step == 0 else h.weight)
            self.assertAlmostEqual(h.mean[0],retained/(retained+256))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.tape.read_text()))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for case in report["cases"]:
        print(case["name"],case["bounds"])
        for row in case["bins"]:
            print(" ",row["distance"],row["sites"],"change",row["change_rgb"])
        for policy in POLICIES:
            runs = [r for r in case["runs"] if r["policy"] == policy and not r["repeated"]]
            print(" ",policy,"worst late p99",max(r["late_rgb"]["p99"] for r in runs),
                  "settled",[r["all_settled_paths_5pct"] for r in runs])


if __name__ == "__main__":
    main()
