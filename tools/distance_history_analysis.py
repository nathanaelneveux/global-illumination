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


def roof_sky_loss(voxel, height, resolution=256):
    """Independent midpoint area quadrature for the black horizontal roof.

    Integrate the production environment gradient over its projected solid
    angle, not the path sampler or a retained cache. The intervening 32-high
    vertical wall does not intersect rays from the floor to this roof.
    """
    x,y,z = voxel
    u = (np.arange(resolution) + 0.5) * (32 / resolution)
    dy = height - (y + 1.001)
    if dy <= 0:
        raise ValueError("roof must be above the receiver")
    r2 = (u[:,None] - (x+0.5))**2 + (u[None,:] - (z+0.5))**2 + dy**2
    gradient = 0.65 + 0.35 * dy / np.sqrt(r2)
    return float(np.sum(gradient * dy**2 / (math.pi * r2**2)) * (32/resolution)**2)


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
                         "paired_change_difference_rgb":summary(rgb_error(
                             refs[1][0]-refs[0][0],refs[1][1]-refs[0][1],scale)[mask]),
                         "oracle_bias_cap_2pct_after_256":summary(allowed)})
        runs = []
        for index in range(len(case["before"])):
            if case["before"][index]["seed"] != case["after"][index]["seed"]:
                raise ValueError("unpaired sequences")
            for reverse in (False,True):
                for repeated in ((False,True) if case["name"] == "cave_far_residency" else (False,)):
                    for policy in POLICIES:
                        runs.append(replay(case,index,reverse,repeated,policy,tape["batch_paths"],tape["warm_paths"]))
        result = {"name":case["name"],"bounds":case["changed_bounds"],"bins":bins,"runs":runs}
        if case["name"] == "roof_diffuse_sky":
            loss = np.asarray([roof_sky_loss(s["voxel"],case["changed_bounds"][0][1])
                               for s in case["sites"]])
            expected = np.repeat(loss[:,None],5,axis=1)
            result["independent_roof_quadrature"] = {
                "absolute_loss":summary(loss),
                "normalized_native_difference":summary(rgb_error(before-after,expected,scale))}
        reports.append(result)
    return {"schema":1,"reference_paths_per_window":tape["reference_paths_per_window"],"cases":reports,
            "note":"Offline distance-only policies; oracle caps are bias-budget diagnostics, not policy inputs. Tail metrics pool late time/sites per run; two QMC prefixes are sensitivity checks, not confidence intervals. No runtime promotion or perceptual/FPS claim."}


def figure(report, output):
    """One remote-friendly numerical comparison; not a rendered game capture."""
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new("RGB", (1500, 1110), "#101923")
    draw = ImageDraw.Draw(canvas)
    font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
    small, regular, title = [ImageFont.truetype(font_path, size) for size in (22,27,38)]
    def text(x,y,value,font=regular,color="#e7eef5"):
        draw.text((x,y),value,font=font,fill=color)
    text(42,28,"Distance helps retention — but does not bound shadows",title)
    text(42,85,"Paired native path study · 4,096 warm paths · 2 × 65,536-path reference windows",small)
    text(42,116,"Raw indirect face energy; not screenshots, perceptual scores or gameplay frame timings.",small)
    draw.rounded_rectangle((30,167,1470,589),radius=12,fill="#1b2938")
    text(54,185,"Canyon: loading the far 64-block slab",title)
    text(54,241,"Distance to new geometry",small)
    text(510,241,"Largest measured change",small)
    text(1040,241,"Bias-budget history cap*",small)
    canyon = next(c for c in report["cases"] if c["name"] == "canyon_far_residency")
    for i, row in enumerate(canyon["bins"]):
        lo,hi = row["distance"]
        delta = row["change_rgb"]["max"]
        allowed = permitted_history(delta,256,0.02)
        text(54,281+43*i,f"{lo}–{hi} blocks ({row['sites']} faces)")
        text(555,281+43*i,f"{delta*100:.2f}%")
        text(1080,281+43*i,"Keep 4,096" if allowed == 4096 else f"{math.floor(allowed):,}")
    text(54,550,"*2% residual change after 256 fresh paths; offline bias only, excludes sampling noise.",small)
    draw.rounded_rectangle((30,608,1470,872),radius=12,fill="#1b2938")
    text(54,626,"Cave: repeated remote geometry changes",title)
    text(54,680,"Policy",small)
    text(475,680,"Late error p99",small)
    text(850,680,"Maximum bright overshoot",small)
    cave = next(c for c in report["cases"] if c["name"] == "cave_far_residency")
    for i, (policy,label) in enumerate((("baseline","Current 256 cap"),("cutoff384","Retain beyond 384"))):
        runs = [r for r in cave["runs"] if r["policy"] == policy and r["repeated"]]
        error = max(r["late_rgb"]["p99"] for r in runs)
        excursion = max(r["bright_excursion"]["max"] for r in runs)
        text(54,719+45*i,label)
        text(520,719+45*i,f"{error*100:.2f}%")
        text(965,719+45*i,f"{excursion*100:.2f}%")
    text(54,824,"Worst of two path prefixes and both event directions; 121 faces, equal fresh work.",small)
    text(42,902,"Counterexample: a roof ~500 blocks away can remove 100% of a sun bounce.",regular,"#ffd59b")
    text(42,948,"Keeping 4,096 old samples leaves ~33% stale energy after 8,192 fresh paths.",regular,"#ffd59b")
    text(42,994,"Conclusion: retain distant compatible history, but also account for shadow dependencies.",regular)
    text(42,1046,"Experimental measurement only. Runtime history and transport defaults remain unchanged.",small)
    canvas.save(output)


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

    def test_roof_quadrature_far_field_and_area_bound(self):
        for height in (64,128,256,384,512,100000):
            loss = roof_sky_loss([16,0,16],height,32)
            bound = 32**2 / (math.pi*(height-1.001)**2)
            self.assertGreater(loss,0)
            self.assertLessEqual(loss,bound)
        self.assertAlmostEqual(loss/bound,1,places=7)

    def test_runtime_mean_blend_obeys_bias_law(self):
        from history_qualification import History
        for retained in (16,256,1024,4096):
            h = History({"mean":[1.0]*5,"second":[1.0]*5},4096)
            fresh = {"mean":[0.0]*5,"second":[0.0]*5}
            for step in range(16):
                h.integrate(fresh,16,retained if step == 0 else h.weight)
            self.assertAlmostEqual(h.mean[0],retained/(retained+256))

    def test_replay_null_and_exact_dark_response(self):
        def side(value):
            moment = {"mean":[value]*5,"second":[value*value]*5}
            return [{"seed":17,"warm":[moment],"batches":[[moment]]*512}]
        case = {"sites":[{"distance":512}],"before":side(1),"after":side(0),
                "references_before":[[[1]*5]]*2,"references_after":[[[0]*5]]*2}
        for policy,retained in (("baseline",256),("cutoff384",4096)):
            result = replay(case,0,False,False,policy,16,4096)
            self.assertAlmostEqual(result["final_rgb"]["max"],retained/(retained+8192))
            self.assertEqual(result["bright_excursion"]["max"],0)
        case["after"] = side(1)
        case["references_after"] = case["references_before"]
        result = replay(case,0,False,True,"baseline",16,4096)
        self.assertEqual(result["final_rgb"]["max"],0)
        self.assertEqual(result["late_variation"]["max"],0)
        self.assertEqual(result["all_settled_paths_5pct"],0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--figure",type=Path)
    args = parser.parse_args()
    report = analyze(json.loads(args.tape.read_text()))
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    if args.figure:
        figure(report,args.figure)
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
