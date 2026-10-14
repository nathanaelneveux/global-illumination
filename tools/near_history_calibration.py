#!/usr/bin/env python3
"""Matched-ray near-edit history sweeps rendered by the actual game material.

Record native tapes with VOXEL_GI_VALIDATE_NEAR_HISTORY=record. `plan` creates
RGB/confidence frames; pass its plan.json as that variable to render them. Then
`assess` uses the unchanged human-calibrated FLIP gate. Replay time is an explicit
per-face publication cadence, NOT measured gameplay latency or FPS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from cave_sample_analysis import load_rgb, perceptual_difference
from visual_quality import within

PROFILE = Path(__file__).parent / "quality_profiles/cave-user-2048.json"
SINGLE_CHECKPOINTS = (0, 16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096)
# Include the early part of consecutive late cycles, not only their best ends.
REPEAT_CHECKPOINTS = (0, 16, 64, 128, 512, 1024, 2048, 2064, 2080, 2112,
                      2176, 2192, 2208, 2240, 2304, 3072, 4096)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_tape(folder):
    data = json.loads((folder / "near-history.json").read_text())
    if data["schema"] != 1 or data["batch_paths"] != 16:
        raise ValueError("unsupported native tape")
    n, steps, chunk = len(data["sites"]), data["steps"], data["chunk_faces"]
    for state in data["states"]:
        raw = np.memmap(folder / state["moments"], dtype="<f4", mode="r")
        expected = len(data["starts"]) * (steps + 1) * n * 6
        if raw.size != expected:
            raise ValueError("incomplete native moments file")
        values = np.empty((len(data["starts"]), steps + 1, n, 6), dtype=np.float32)
        offset = 0
        for w in range(len(data["starts"])):
            for first in range(0, n, chunk):
                count = min(chunk, n - first)
                size = (steps + 1) * count * 6
                values[w, :, first:first + count] = raw[offset:offset + size].reshape(steps + 1, count, 6)
                offset += size
        if not np.all(np.isfinite(values)) or np.min(values) < 0:
            raise ValueError("nonfinite/negative native moments")
        state["data"] = values
    return data


def limits(distances, radius, near):
    """Vary near floor/radius only; keep the qualified distant prior unchanged."""
    prior = np.minimum(1., 48. ** 2 / np.maximum(distances ** 2, 48. ** 2))
    bias = np.ceil((2048. / 6) * .03 / np.maximum(prior - .03, 1e-15))
    floor = np.where(distances <= radius, near, 256.)
    return np.where(prior <= .03, 65536., np.maximum(bias, floor))


def relative_error(mean, second, weight, floor):
    variance = np.maximum(second - mean * mean, 0.)
    return np.minimum(8., np.max(np.sqrt(variance / weight[:, None]) /
                                (floor + np.abs(mean)), axis=1))


def display_step(display, confidence, mean, second, weight, dt):
    """Vectorized history.rs::resolve_face; parity checked against Rust fixtures."""
    uncertainty = relative_error(mean, second, weight, .02)
    noise = relative_error(mean, second, weight, .002)
    evidence = weight / (weight + 32.)
    alpha = np.minimum(1. - np.exp(-dt / .075), evidence / (1. + (4. * noise) ** 2))
    old = (1. - alpha) / (.25 + display.max(axis=1))
    new = alpha / (.25 + mean.max(axis=1))
    blend = np.minimum(alpha, new / (old + new))
    display = display + (mean - display) * blend[:, None]
    target_confidence = np.clip(weight / (weight + 8.) / (1. + uncertainty), .125, 1.)
    confidence = confidence + (target_confidence - confidence) * alpha
    return display, confidence


def replay(tape, window, case, radius, near, dt, every):
    before, after = [s["data"][window] for s in tape["states"]]
    sides = (before, after) if case != "add" else (after, before)
    if case == "null":
        sides = (before, before)
    batch, warm = tape["batch_paths"], tape["warm_paths"]
    n = len(tape["sites"])
    mean, second = np.split(sides[0][0].astype(np.float64).copy(), 2, axis=1)
    weight = np.full(n, float(warm))
    display = mean.copy()
    confidence = np.clip(weight / (weight + 8.) /
                         (1. + relative_error(mean, second, weight, .02)), .125, 1.)
    distance = np.linalg.norm(np.array([s[0] for s in tape["sites"]]) - tape["edit"], axis=1)
    cap = limits(distance, radius, near) if near else np.full(n, 65536.)
    # Zero is the explicitly named retain control, not a zero-history reset.
    now = 0
    checkpoints = REPEAT_CHECKPOINTS if every else SINGLE_CHECKPOINTS
    yield 0, sides[0] is after, mean.copy(), display.copy(), confidence.copy(), weight.copy()
    for step in range(tape["steps"]):
        event = step == 0 or (every and (step * batch) % every == 0)
        if event:
            now = 1 - now
            weight = np.minimum(weight, cap)
        incoming = sides[now][step + 1].astype(np.float64)
        retained = np.minimum(weight, 65536. - batch)
        weight = retained + batch
        alpha = batch / weight[:, None]
        mean += (incoming[:, :3] - mean) * alpha
        second += (incoming[:, 3:] - second) * alpha
        display, confidence = display_step(display, confidence, mean, second, weight, dt)
        paths = (step + 1) * batch
        if paths in checkpoints:
            yield paths, sides[now] is after, mean.copy(), display.copy(), confidence.copy(), weight.copy()


def make_plan(args):
    tape = load_tape(args.tape)
    if args.every and (args.every < 16 or args.every % 16):
        raise ValueError("event spacing must be a positive multiple of the native batch")
    args.output.mkdir(parents=True, exist_ok=False)
    frames = []

    def save(name, values, confidence, **metadata):
        values = np.column_stack((np.asarray(values)[:, :3], confidence)).astype("<f4")
        values.tofile(args.output / f"{name}.bin")
        frames.append(dict(name=name, **metadata))

    for state in tape["states"]:
        save(f"near-{state['name']}-reference", state["mean"], np.ones(len(tape["sites"])), reference=True)
        for i, replica in enumerate(state["replicas"]):
            save(f"near-{state['name']}-replica-{i}", replica, np.ones(len(tape["sites"])), reference=True)
    for w in args.windows:
        if not 0 <= w < len(tape["starts"]):
            raise ValueError("missing native sequence window")
        for case in args.cases:
            for radius in args.radii:
                for history in args.histories:
                    for paths, after, raw, display, confidence, weight in replay(
                            tape, w, case, radius, history, args.dt, args.every):
                        name = f"{case}-r{radius}-h{history}-w{w}-p{paths}"
                        target = "after" if after else "before"
                        save(name, display, confidence, case=case, radius=radius, history=history,
                             window=w, fresh_paths=paths, replay_seconds=paths / 16 * args.dt,
                             target=f"near-{target}-reference", weight_p50=float(np.median(weight)),
                             raw_rgb_rmse=float(np.sqrt(np.mean((raw - np.asarray(tape["states"][int(after)]["mean"])[:, :3]) ** 2))))
    metadata = dict(schema=1, sites=tape["sites"], frames=frames,
                    tape=str(args.tape.resolve()), tape_sha256=digest(args.tape / "near-history.json"),
                    moments_sha256={s["moments"]:digest(args.tape / s["moments"]) for s in tape["states"]},
                    arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    note="Frozen edited integrands, retained existing receivers, actual production display resolve at explicit uniform per-face cadence. Warm display starts at the 4096-path mean, not a perfectly converged oracle. No scheduling/new-face/interactive latency claim. h0 is retain control. Radius changes only the floor boundary, not the distant prior.")
    (args.output / "plan.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(plan=str(args.output.resolve() / "plan.json"), frames=len(frames))))


def sustained(rows):
    """Suffix pass, with at least two later observations; never first crossing."""
    ordered = sorted(rows, key=lambda r:r["fresh_paths"])
    for i, row in enumerate(ordered[:-2]):
        if all(r["accepted"] for r in ordered[i:]):
            return dict(paths=row["fresh_paths"], replay_seconds=row["replay_seconds"])
    return None


def assess(args):
    plan = json.loads(args.plan.read_text())
    profile = json.loads(PROFILE.read_text())
    status = json.loads((args.run / "cave-model-comparison-status.json").read_text())
    if status["status"] != "completed" or not status["geometry_audit_passed"]:
        raise ValueError("incomplete rendering or failed geometry audit")
    gates = {s:profile["limits"][s] for s in ("flip","flip_32px_tile_means")}
    references = {}
    for state in ("before","after"):
        path = args.run / f"near-{state}-reference.png"
        rgb = load_rgb(path)
        if rgb.shape != (profile["view"]["height"],profile["view"]["width"],3) or np.std(rgb) < .01:
            raise ValueError("invalid/blank capture dimensions")
        references[f"near-{state}-reference"] = rgb
    # Check residual reference noise against the SAME fixed perceptual limits.
    replicas = {}
    for state in ("before","after"):
        replicas[state] = perceptual_difference(load_rgb(args.run / f"near-{state}-replica-0.png"),
                                               load_rgb(args.run / f"near-{state}-replica-1.png"),67)
        if not within(replicas[state], gates):
            raise ValueError("reference replicas disagree beyond the visual gate")
    results = []
    for frame in plan["frames"]:
        if frame.get("reference"):
            continue
        image = load_rgb(args.run / (frame["name"] + ".png"))
        difference = perceptual_difference(references[frame["target"]], image,67)
        row = dict(frame, **difference, accepted=within(difference,gates))
        results.append(row)
        print(frame["name"], round(difference["flip"]["mean"],5),
              round(difference["flip_32px_tile_means"]["max"],5), row["accepted"], flush=True)
        # Checkpoint expensive analysis so a later interruption is recoverable.
        args.output.write_text(json.dumps(dict(profile_id=profile["id"], complete=False,
                                               measurements=results),allow_nan=False))
    keys = sorted({(r["case"],r["radius"],r["history"],r["window"]) for r in results})
    summary = []
    for key in keys:
        rows = [r for r in results if (r["case"],r["radius"],r["history"],r["window"]) == key]
        summary.append(dict(case=key[0],radius=key[1],history=key[2],window=key[3],
                            sustained=sustained(rows), failed_captures=sum(not r["accepted"] for r in rows),
                            peak_flip=max(r["flip"]["mean"] for r in rows),
                            peak_tile=max(r["flip_32px_tile_means"]["max"] for r in rows)))
    output = dict(complete=True, profile_id=profile["id"], plan_sha256=digest(args.plan),
                  plan=plan, reference_replica_difference=replicas, measurements=results, summary=summary,
                  note="Fixed FLIP goal, actual 2560x1440 material captures at 67 PPD. Sustained means all captured later checkpoints pass, with two later checkpoints required; unobserved intervals are not certified. Replay seconds assume the declared per-face cadence, not measured gameplay response.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary,indent=2))


class Tests(unittest.TestCase):
    def test_existing_near_and_distant_curve_is_reproduced(self):
        got = limits(np.array([0.,48.,96.,128.,192.,256.,384.]),48,16)
        np.testing.assert_equal(got,[16,16,256,256,316,1986,65536])

    def test_display_is_convex_and_does_not_change_measurement(self):
        mean=np.array([[.01,.02,.03],[4.,2.,1.]])
        second=mean**2*100
        before=mean.copy()
        display=np.array([[1.,1.,1.],[0.,0.,0.]])
        out,c=display_step(display,np.ones(2),mean,second,np.array([32.,256.]),.1)
        self.assertTrue(np.all(out>=np.minimum(display,mean)))
        self.assertTrue(np.all(out<=np.maximum(display,mean)))
        self.assertTrue(np.all((c>=.125)&(c<=1)))
        np.testing.assert_array_equal(mean,before)
        np.testing.assert_allclose(out,[[.98887787,.98899021,.98910256],
                                        [.03296449,.01648225,.00824112]],atol=1e-7)
        np.testing.assert_allclose(c,[.99313831,.95036899],atol=1e-7)

    def test_sustained_rejects_late_failure_and_short_tail(self):
        rows=[dict(fresh_paths=i*16,replay_seconds=i*.1,accepted=True) for i in range(6)]
        self.assertEqual(sustained(rows)["paths"],0)
        rows[-2]["accepted"]=False
        self.assertIsNone(sustained(rows))
        rows[1]["accepted"]=False
        rows[-2]["accepted"]=True
        self.assertEqual(sustained(rows)["paths"],32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("plan")
    p.add_argument("tape",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--histories",type=int,nargs="+",default=[16,256])
    p.add_argument("--radii",type=int,nargs="+",default=[48])
    p.add_argument("--windows",type=int,nargs="+",default=[0])
    p.add_argument("--cases",choices=["null","remove","add"],nargs="+",default=["null","remove","add"])
    p.add_argument("--dt",type=float,default=.1)
    p.add_argument("--every",type=int,default=0,help="repeat edit every N fresh paths; 0 is one edit")
    a = sub.add_parser("assess")
    a.add_argument("plan",type=Path)
    a.add_argument("run",type=Path)
    a.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="plan": make_plan(args)
    else: assess(args)


if __name__=="__main__":
    main()
