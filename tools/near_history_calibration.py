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

from cave_sample_analysis import load_rgb
from visual_quality import within

PROFILE = Path(__file__).parent / "quality_profiles/cave-user-2048.json"
SINGLE_CHECKPOINTS = (0, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 3584, 4096)
# Include the early part of consecutive late cycles, not only their best ends.
REPEAT_CHECKPOINTS = (0, 16, 64, 128, 512, 1024, 2048, 2064, 2080, 2112,
                      2176, 2192, 2208, 2240, 2304, 3072, 4096)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def flip_map(reference, candidate):
    """Evaluate the full image before selecting any spatial diagnostic masks."""
    import flip_evaluator as flip
    if reference.shape != candidate.shape or reference.shape != (1440,2560,3):
        raise ValueError("capture dimensions differ from the calibrated view")
    error, _, _ = flip.evaluate(reference,candidate,"LDR",applyMagma=False,
                               parameters={"ppd":67.})
    error = np.asarray(error).reshape(1440,2560)
    if not np.all(np.isfinite(error)):
        raise ValueError("nonfinite FLIP map")
    return error


def flip_statistics(error):
    tiles = error.reshape(45,32,80,32).mean((1,3))
    return dict(flip=dict(mean=float(np.mean(error,dtype=np.float64))),
                flip_32px_tile_means=dict(max=float(tiles.max())))


def flip_difference(reference, candidate):
    return flip_statistics(flip_map(reference, candidate))


def luminance(rgb):
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    return linear @ np.array([.2126, .7152, .0722], dtype=np.float32)


def brightness_statistics(image, reference, old_reference):
    """Rendered linear-sRGB diagnostics, not physical flux or FLIP thresholds.

    Above a dark new target during a falling transition is not overshoot. Only
    excursions outside BOTH endpoint references count as transition overshoot.
    Cold/static cases pass their target as both endpoints for sampling error.
    """
    y = luminance(image)
    target, old = luminance(reference), luminance(old_reference)
    upper = np.maximum(y - np.maximum(target, old), 0.)
    lower = np.maximum(np.minimum(target, old) - y, 0.)
    target_mean = float(target.mean(dtype=np.float64))
    mean = float(y.mean(dtype=np.float64))
    return dict(mean=mean, reference_mean=target_mean,
                mean_ratio=mean / target_mean if target_mean > 1e-6 else None,
                positive_target_error_p99=float(np.quantile(np.maximum(y-target,0),.99)),
                upper_endpoint_excursion_p99=float(np.quantile(upper,.99)),
                upper_endpoint_tile_max=float(upper.reshape(45,32,80,32).mean((1,3)).max()),
                lower_endpoint_tile_max=float(lower.reshape(45,32,80,32).mean((1,3)).max()))


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


def power_limits(distances, power):
    """Continuous near equation; preserve the qualified far prior exactly.

    Rounding only quantizes to the runtime's integer statistical weight. This
    is a policy experiment, not an inverse-square law for light transport.
    """
    if not np.isfinite(power) or power <= 0:
        raise ValueError("curve power must be positive and finite")
    far = np.minimum(limits(distances, 0, 256), 65536.)
    curved = np.floor(far * np.minimum(distances / 48., 1.) ** power + .5)
    return np.where(distances ** 2 >= 76800., 65536., curved)


def edit_distances(tape):
    sites = np.asarray([s[0] for s in tape["sites"]])
    edits = np.asarray(tape.get("edits", [tape["edit"]]))
    return np.linalg.norm(sites[:, None, :] - edits[None, :, :], axis=2).min(axis=1)


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


def replay(tape, window, case, radius, near, dt, every, checkpoints=None, literal=False,
           power=None, maximum=65536):
    before, after = [s["data"][window] for s in tape["states"]]
    sides = (before, after) if case != "add" else (after, before)
    if case == "null" or case.endswith("before"):
        sides = (before, before)
    if case.endswith("after"):
        sides = (after, after)
    batch, warm = tape["batch_paths"], tape["warm_paths"]
    n = len(tape["sites"])
    mean, second = np.split(sides[0][0].astype(np.float64).copy(), 2, axis=1)
    weight = np.full(n, float(min(warm, maximum)))
    display = mean.copy()
    confidence = np.clip(weight / (weight + 8.) /
                         (1. + relative_error(mean, second, weight, .02)), .125, 1.)
    if case.startswith("cold-"):
        mean.fill(0.)
        second.fill(0.)
        weight.fill(0.)
        display.fill(0.)
        confidence.fill(.125)
    distance = edit_distances(tape)
    cap = (power_limits(distance, power) if power is not None else
           np.full(n, float(near)) if literal else
           limits(distance, radius, near) if near else np.full(n, 65536.))
    # Legacy h0 is retain-all. Literal mode bypasses BOTH the prior-derived
    # lower bound (11 nearby) and distance curve; h0 then truly retains nothing.
    now = 0
    if checkpoints is None:
        checkpoints = REPEAT_CHECKPOINTS if every else SINGLE_CHECKPOINTS
    # Time zero is immediately AFTER the event: compare the retained old
    # display to the NEW target, not to the previously correct scene.
    yield 0, sides[1] is after, mean.copy(), display.copy(), confidence.copy(), weight.copy()
    for step in range(tape["steps"]):
        event = not case.startswith(("cold-", "static-")) and (
            step == 0 or (every and (step * batch) % every == 0))
        if event:
            now = 1 - now
            weight = np.minimum(weight, cap)
        incoming = sides[now][step + 1].astype(np.float64)
        retained = np.minimum(weight, maximum - batch)
        weight = retained + batch
        alpha = batch / weight[:, None]
        mean += (incoming[:, :3] - mean) * alpha
        second += (incoming[:, 3:] - second) * alpha
        display, confidence = display_step(display, confidence, mean, second, weight, dt)
        paths = (step + 1) * batch
        if paths in checkpoints:
            yield paths, sides[now] is after, mean.copy(), display.copy(), confidence.copy(), weight.copy()


def policy_pairs(histories=None, radii=None, policies=None):
    """Allow an explicit shortlist instead of rendering a wasteful cross product."""
    if policies is not None:
        if histories is not None or radii is not None:
            raise ValueError("use --policies or --histories/--radii, not both")
        pairs = [tuple(map(int, value.split(":"))) for value in policies]
    else:
        pairs = [(r, h) for r in (radii or [48]) for h in (histories or [16,256])]
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError("policies must be radius:history pairs")
    if len(set(pairs)) != len(pairs):
        raise ValueError("duplicate policies would overwrite frame identities")
    if any(r < 0 or h < 0 or h > 65536 for r,h in pairs):
        raise ValueError("invalid retained history/radius")
    return pairs


def native_zero_control(tape, skies):
    if tape.get("fixture") != "hallway-emitter" or len(tape["states"]) != 2:
        return False
    after = tape["states"][1]
    n = len(tape["sites"])
    return (after["name"] == "after" and len(after["replicas"]) == 2
            and all(np.asarray(v).shape == (n,5) and np.count_nonzero(v) == 0
                    for v in [after["mean"], *after["replicas"]])
            and np.asarray(skies.get("after", [])).shape == (n,4)
            and np.all(np.asarray(skies.get("after", [])) == 1)
            and n > 0)


def make_plan(args):
    tape = load_tape(args.tape)
    view = tape.get("view")
    if args.view is not None:
        view = (tape.get("views") or {}).get(args.view)
        if view is None:
            raise ValueError("requested view is not an authored tape station")
    if args.equations:
        if args.histories is not None or args.radii is not None or args.policies is not None or args.literal:
            raise ValueError("equations cannot be combined with literal/legacy history options")
        if len(set(args.equations)) != len(args.equations):
            raise ValueError("duplicate equation")
        pairs = []
        for equation in args.equations:
            if equation == "baseline":
                pairs.append((48,16,None,"baseline-"))
            elif equation == "near16":
                pairs.append((16,16,None,"near16-"))
            elif equation.startswith("power"):
                power = int(equation[5:])
                power_limits(np.array([0.]), power)
                pairs.append((48,256,power,equation+"-"))
            else:
                raise ValueError("equations: baseline, near16 or power<positive integer>")
    else:
        pairs = [(r,h,None,"") for r,h in policy_pairs(args.histories,args.radii,args.policies)]
    if args.maximum < tape["batch_paths"] or args.maximum > 65536:
        raise ValueError("maximum must be in [native batch size, 65536]")
    if args.checkpoints is not None and (len(set(args.checkpoints)) != len(args.checkpoints)
            or any(n < 0 or n > tape["steps"] * tape["batch_paths"] or n % tape["batch_paths"]
                   for n in args.checkpoints)):
        raise ValueError("checkpoints must be distinct native batch boundaries within the tape")
    if not np.isfinite(args.dt) or args.dt <= 0:
        raise ValueError("publication cadence must be positive and finite")
    for values in (args.windows,args.cases):
        if len(set(values)) != len(values):
            raise ValueError("duplicate sweep arguments would overwrite frame identities")
    for spacing in (args.every,args.null_every):
        if spacing and (spacing < 16 or spacing % 16):
            raise ValueError("event spacing must be a positive multiple of the native batch")
    args.output.mkdir(parents=True, exist_ok=False)
    frames = []
    skies = {s["name"]:s["sky"] for s in tape["states"] if s.get("sky") is not None}
    # A fully enclosed unlit hallway has a known zero-energy target. This is
    # NOT a black gameplay capture pass: bracket offline replay with positive
    # material witnesses, and require both native reference replicas and sky
    # corners to certify zero before treating the dark image as a control.
    dark_control = native_zero_control(tape, skies)

    def save(name, values, confidence, **metadata):
        values = np.column_stack((np.asarray(values)[:, :3], confidence)).astype("<f4")
        values.tofile(args.output / f"{name}.bin")
        # Opening/closing geometry changes deterministic direct sunlight too.
        # Use the target state's sky at t=0, before indirect history responds.
        state = metadata.get("target", name).split("-")[1]
        if state in skies:
            metadata.setdefault("sky_state", state)
        frames.append(dict(name=name, **metadata))

    if dark_control:
        save("witness-before", np.ones((len(tape["sites"]),3)), np.ones(len(tape["sites"])),
             reference=True, sky_state="after")
    for state in tape["states"]:
        save(f"near-{state['name']}-reference", state["mean"], np.ones(len(tape["sites"])), reference=True)
        for i, replica in enumerate(state["replicas"]):
            save(f"near-{state['name']}-replica-{i}", replica, np.ones(len(tape["sites"])), reference=True)
    if args.sky_controls:
        if len(skies) != 2:
            raise ValueError("direct-sun controls require both recorded sky states")
        for state in tape["states"]:
            save(f"direct-{state['name']}", np.zeros((len(tape["sites"]), 3)),
                 np.ones(len(tape["sites"])), reference=True)
    if args.analytic:
        distance=edit_distances(tape)
        for case, old, new in (("remove",0,1),("add",1,0)):
            a,b=[np.asarray(tape["states"][i]["mean"])[:,:3] for i in (old,new)]
            target=f"near-{tape['states'][new]['name']}-reference"
            for radius in (0,16,24,32,48,64):
                values=np.where((distance<=radius)[:,None],b,a)
                save(f"spatial-{case}-r{radius}",values,np.ones(len(distance)),
                     kind="spatial",case=case,radius=radius,target=target)
            for fraction in (1.,.5,.25,.125,.0625,.03125):
                save(f"bias-{case}-f{int(fraction*100000)}",a*fraction+b*(1-fraction),np.ones(len(distance)),
                     kind="bias",case=case,stale_fraction=fraction,target=target)
    for w in args.windows:
        if not 0 <= w < len(tape["starts"]):
            raise ValueError("missing native sequence window")
        for state in tape["states"]:
            warm = state["data"][w,0].astype(np.float64)
            weight = np.full(len(tape["sites"]),float(tape["warm_paths"]))
            confidence = np.clip(weight/(weight+8)/(1+relative_error(warm[:,:3],warm[:,3:],weight,.02)),.125,1.)
            save(f"warm-{state['name']}-w{w}",warm[:,:3],confidence,
                 case=f"warm-{state['name']}",radius=0,history=tape["warm_paths"],window=w,
                 fresh_paths=0,replay_seconds=0,target=f"near-{state['name']}-reference")
        for case in args.cases:
            every = args.null_every if case == "null" and args.null_every is not None else args.every
            # Cold/steady controls do not apply edit equations; render once.
            for radius, history, power, prefix in (pairs[:1] if case.startswith(("cold-","static-")) else pairs):
                for paths, after, raw, display, confidence, weight in replay(
                        tape, w, case, radius, history, args.dt, every, args.checkpoints, args.literal,
                        power, args.maximum):
                    name = f"{prefix}{case}-r{radius}-h{history}-w{w}-p{paths}"
                    target = "after" if after else "before"
                    save(name, display, confidence, case=case, radius=radius, history=history,
                         policy=prefix.rstrip('-') or f"r{radius}-h{history}",
                         curve_power=power, maximum_weight=args.maximum,
                         window=w, fresh_paths=paths, replay_seconds=paths / 16 * args.dt,
                         event_interval_paths=every,
                         target=f"near-{target}-reference", weight_p50=float(np.median(weight)),
                         raw_rgb_rmse=float(np.sqrt(np.mean((raw - np.asarray(tape["states"][int(after)]["mean"])[:, :3]) ** 2))))
    if dark_control:
        save("witness-after", np.ones((len(tape["sites"]),3)), np.ones(len(tape["sites"])),
             reference=True, sky_state="after")
    metadata = dict(schema=1, fixture=tape.get("fixture", "cave-torch"), dark_zero_control=dark_control,
                    view=view, skies=skies, sites=tape["sites"], frames=frames,
                    tape=str(args.tape.resolve()), tape_sha256=digest(args.tape / "near-history.json"),
                    moments_sha256={s["moments"]:digest(args.tape / s["moments"]) for s in tape["states"]},
                    policies=pairs, literal_history=args.literal,
                    arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                    note="Frozen integrands, actual production display resolve at explicit uniform per-face cadence. Warm display starts at the recorded warm mean, not an oracle. Cold controls start with zero RGB/weight and .125 reconstruction confidence, use fresh tape batches only, and are not a simulation of runtime neighbor seeding. Static controls retain the recorded warm mean with weight clamped to maximum; the warm prefix itself was not capped. No scheduling/new-face/interactive latency claim. Literal h0 discards statistical history; legacy h0 retains all. Equations use each donor's own distance. Maximum is independent of the edit cap.")
    (args.output / "plan.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(plan=str(args.output.resolve() / "plan.json"), frames=len(frames))))


def sustained(rows):
    """Suffix pass, with at least two later observations; never first crossing."""
    ordered = sorted(rows, key=lambda r:r["fresh_paths"])
    for i, row in enumerate(ordered[:-2]):
        if all(r["accepted"] for r in ordered[i:]):
            return dict(paths=row["fresh_paths"], replay_seconds=row["replay_seconds"],
                        previous_failure_paths=ordered[i-1]["fresh_paths"] if i else None)
    return None


def summarize(results):
    """Keep an observed recovery separate from a qualified settled-start edit.

    A noisy pre-edit image can recover, but cannot establish response latency
    from an already settled scene. Warm controls use their old-state target.
    """
    temporal = [r for r in results if "kind" not in r]
    warm = {(r["case"], r["window"]): r["accepted"] for r in temporal
            if r["case"].startswith("warm-")}
    def identity(r):
        return (r["case"], r.get("policy", "legacy"), r["radius"], r["history"], r["window"])
    keys = sorted({identity(r) for r in temporal})
    summary = []
    for case, policy, radius, history, window in keys:
        rows = [r for r in temporal if identity(r) == (case,policy,radius,history,window)]
        recovery = sustained(rows)
        passes = [r["fresh_paths"] for r in rows if r["accepted"]]
        settled_start = warm.get(("warm-after" if case == "add" else "warm-before", window), False)
        edit = case in ("remove", "add", "null")
        summary.append(dict(case=case, policy=policy, radius=radius, history=history, window=window,
                            sustained=recovery,
                            first_pass_paths=min(passes) if passes else None,
                            settled_start=settled_start if not case.startswith("warm-") else None,
                            response_qualified=edit and settled_start and recovery is not None,
                            failed_captures=sum(not r["accepted"] for r in rows),
                            peak_flip=max(r["flip"]["mean"] for r in rows),
                            peak_tile=max(r["flip_32px_tile_means"]["max"] for r in rows)))
    return summary


def assess(args):
    plan = json.loads(args.plan.read_text())
    profile = json.loads(PROFILE.read_text())
    status = json.loads((args.run / "cave-model-comparison-status.json").read_text())
    if status["status"] != "completed" or not status["geometry_audit_passed"]:
        raise ValueError("incomplete rendering or failed geometry audit")
    if (Path(status["qualification"]["plan"]).resolve() != args.plan.resolve()
            or status["captures"] != len(plan["frames"]) + 1):
        raise ValueError("completed run does not match this replay plan/capture count")
    gates = {s:profile["limits"][s] for s in ("flip","flip_32px_tile_means")}
    if plan.get("dark_zero_control"):
        # Never waive camera/assets/target/geometry checks or classify a blank
        # gameplay screenshot as a pass. This narrowly qualifies a native-zero
        # offline material control, with a bright same-geometry witness pair.
        tape = load_tape(Path(plan["tape"]))
        if (not native_zero_control(tape,plan["skies"])
                or digest(Path(plan["tape"]) / "near-history.json") != plan["tape_sha256"]
                or plan["skies"]["after"] != tape["states"][1]["sky"]):
            raise ValueError("dark control lacks matching native zero-energy evidence")
        witness = [load_rgb(args.run / f"witness-{s}.png") for s in ("before","after")]
        if any(np.std(im) < .01 or np.mean(im) < .05 for im in witness):
            raise ValueError("dark control has no positive material witness")
        if not within(flip_difference(*witness), gates):
            raise ValueError("bracketing material witnesses disagree")
    references = {}
    for state in ("before","after"):
        path = args.run / f"near-{state}-reference.png"
        rgb = load_rgb(path)
        known_dark = plan.get("dark_zero_control") and state == "after" and np.max(rgb) <= 1. / 255. + 1e-7
        if rgb.shape != (profile["view"]["height"],profile["view"]["width"],3) or (np.std(rgb) < .01 and not known_dark):
            raise ValueError("invalid/blank capture dimensions")
        references[f"near-{state}-reference"] = rgb
    # Check residual reference noise against the SAME fixed perceptual limits.
    replicas = {}
    for state in ("before","after"):
        replicas[state] = flip_difference(load_rgb(args.run / f"near-{state}-replica-0.png"),
                                         load_rgb(args.run / f"near-{state}-replica-1.png"))
        if not within(replicas[state], gates):
            raise ValueError("reference replicas disagree beyond the visual gate")
    results = []
    for frame in plan["frames"]:
        if frame.get("reference"):
            continue
        image = load_rgb(args.run / (frame["name"] + ".png"))
        difference = flip_difference(references[frame["target"]], image)
        target = references[frame["target"]]
        old_target = (references["near-after-reference" if frame.get("case") == "add" else "near-before-reference"]
                      if frame.get("case") in ("add","remove") else target)
        row = dict(frame, **difference, accepted=within(difference,gates),
                   luminance=brightness_statistics(image,target,old_target))
        results.append(row)
        print(frame["name"], round(difference["flip"]["mean"],5),
              round(difference["flip_32px_tile_means"]["max"],5), row["accepted"], flush=True)
        # Checkpoint expensive analysis so a later interruption is recoverable.
        args.output.write_text(json.dumps(dict(profile_id=profile["id"], complete=False,
                                               measurements=results),allow_nan=False))
    summary = summarize(results)
    output = dict(complete=True, profile_id=profile["id"], plan_sha256=digest(args.plan),
                  plan=plan, reference_replica_difference=replicas, measurements=results, summary=summary,
                  note="Fixed cave-calibrated FLIP goal, transferred to other fixtures as an engineering gate, not independent human calibration. Actual 2560x1440 material captures at 67 PPD. Sustained means all captured later checkpoints pass, with two later checkpoints required; unobserved intervals are not certified. Previous failure brackets observed settling. Replay seconds assume the declared per-face cadence, not measured gameplay response.")
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary,indent=2))


class Tests(unittest.TestCase):
    def test_power_equation_anchors_and_unchanged_far_policy(self):
        distances = np.array([0.,1.,8.,16.,24.,32.,48.,96.,192.,256.,384.])
        np.testing.assert_equal(power_limits(distances,4),[0,0,0,3,16,51,256,256,316,1986,65536])
        for power in (2,4,6):
            curve = power_limits(np.arange(0,385.),power)
            self.assertTrue(np.all(np.diff(curve) >= 0))
            np.testing.assert_equal(curve[48:],np.minimum(limits(np.arange(48,385.),0,256),65536))
        for power in (0,-1,float('nan')):
            with self.assertRaises(ValueError):
                power_limits(distances,power)

    def test_cold_matches_zero_history_raw_samples_and_zero_display_seed(self):
        a = np.ones((1,257,1,6))
        b = np.zeros_like(a)
        tape = dict(states=[dict(data=a),dict(data=b)],batch_paths=16,warm_paths=4096,
                    sites=[[[0,0,0],0]],edit=[0,0,0],steps=256)
        cold = list(replay(tape,0,"cold-before",48,0,.85,0,[16,2048,4096],literal=True))
        edit = list(replay(tape,0,"add",48,0,.85,0,[16,2048,4096],literal=True))
        for fresh, reset in zip(cold,edit):
            np.testing.assert_equal(fresh[2],reset[2])
            np.testing.assert_equal(fresh[3],reset[3])
            if fresh[0]:
                np.testing.assert_equal(fresh[5],reset[5])
        self.assertEqual(cold[0][4][0],.125)
        self.assertGreater(edit[0][4][0],.99)

    def test_maximum_is_ongoing_ema_not_a_periodic_reset(self):
        a = np.ones((1,257,1,6))
        b = np.zeros_like(a)
        tape = dict(states=[dict(data=a),dict(data=b)],batch_paths=16,warm_paths=4096,
                    sites=[[[0,0,0],0]],edit=[0,0,0],steps=256)
        rows = list(replay(tape,0,"remove",48,65536,.85,0,[16,2048,4096],literal=True,maximum=2048))
        for row in rows[1:]:
            np.testing.assert_equal(row[5],[2048])
            np.testing.assert_allclose(row[2],(1-16/2048)**(row[0]/16))
        default = list(replay(tape,0,"remove",48,65536,.85,0,[16],literal=True))
        np.testing.assert_allclose(default[1][2],4096/(4096+16))

    def test_zero_control_requires_complete_native_evidence_not_a_blank_image(self):
        tape = dict(fixture="hallway-emitter",sites=[[[0,0,0],2]],states=[{},
                    dict(name="after",mean=[[0]*5],replicas=[[[0]*5],[[0]*5]])])
        self.assertTrue(native_zero_control(tape,{"after":[[1]*4]}))
        self.assertFalse(native_zero_control(tape,{}))
        self.assertFalse(native_zero_control(tape,{"after":[[0]*4]}))
        self.assertFalse(native_zero_control(tape,{"after":[[1]*3]}))
        tape["states"][1]["replicas"][1][0][0] = .001
        self.assertFalse(native_zero_control(tape,{"after":[[1]*4]}))

    def test_literal_weights_bypass_the_bias_floor_and_keep_display_history(self):
        a = np.ones((1,257,1,6))
        b = np.zeros_like(a)
        tape = dict(states=[dict(data=a),dict(data=b)],batch_paths=16,warm_paths=4096,
                    sites=[[[0,0,0],0]],edit=[0,0,0],steps=256)
        for history in (0,1,2,4,8,16,32,64,128,256,65536):
            rows = list(replay(tape,0,"remove",48,history,.85,0,[16],literal=True))
            retained = min(history,4096)
            self.assertEqual(rows[1][5][0],retained+16)
            np.testing.assert_allclose(rows[1][2],retained/(retained+16))
            self.assertTrue(np.all(rows[1][3] > rows[1][2]))
        legacy = list(replay(tape,0,"remove",48,0,.85,0,[16]))
        self.assertEqual(legacy[1][5][0],4112)
        np.testing.assert_equal(limits(np.array([1.]),48,1),[11])

    def test_extra_observations_do_not_change_the_replayed_estimator(self):
        before = np.ones((1,257,1,6))
        after = np.zeros_like(before)
        tape = dict(states=[dict(data=before), dict(data=after)], batch_paths=16,
                    warm_paths=4096, sites=[[[0,0,0],0]], edit=[0,0,0], steps=256)
        original = list(replay(tape,0,"remove",48,256,.85,0))
        extra = dict((r[0],r) for r in replay(tape,0,"remove",48,256,.85,0,
                                            tuple(SINGLE_CHECKPOINTS)+(2304,2560,2816)))
        for row in original:
            for value, other in zip(row[2:],extra[row[0]][2:]):
                np.testing.assert_array_equal(value,other)
        self.assertIn(2304,extra)

    def test_multi_voxel_edit_uses_the_closest_changed_voxel(self):
        tape = dict(sites=[[[0,0,0],0], [[10,0,0],0]],edit=[0,0,0])
        np.testing.assert_equal(edit_distances(tape),[0,10])
        tape["edits"] = [[0,0,0],[9,0,0]]
        np.testing.assert_equal(edit_distances(tape),[0,1])

    def test_explicit_shortlist_preserves_legacy_defaults_and_frame_identity(self):
        self.assertEqual(policy_pairs(), [(48,16), (48,256)])
        self.assertEqual(policy_pairs(policies=["48:16", "278:2048"]), [(48,16), (278,2048)])
        for args in (dict(histories=[16], policies=["48:16"]),
                     dict(policies=["48:16", "48:16"]),
                     dict(policies=["48:16:32"]), dict(policies=["48:-1"])):
            with self.assertRaises(ValueError):
                policy_pairs(**args)

    def test_noisy_warm_image_cannot_qualify_settled_start_response(self):
        base = dict(radius=48, history=16, window=0, replay_seconds=0.,
                    flip=dict(mean=.01), flip_32px_tile_means=dict(max=.02))
        rows = [dict(base, case="warm-before", fresh_paths=0, accepted=False),
                dict(base, case="warm-after", fresh_paths=0, accepted=True)]
        for case in ("remove", "add"):
            rows.extend(dict(base, case=case, fresh_paths=n, accepted=True) for n in (0,16,32))
        arms = {r["case"]: r for r in summarize(rows)}
        self.assertIsNotNone(arms["remove"]["sustained"])
        self.assertFalse(arms["remove"]["response_qualified"])
        self.assertTrue(arms["add"]["response_qualified"])

    def test_zero_time_compares_old_display_to_new_target(self):
        a=np.ones((1,257,1,6))
        b=np.zeros((1,257,1,6))
        tape=dict(states=[dict(data=a),dict(data=b)],batch_paths=16,warm_paths=4096,
                  sites=[[[0,0,0],0]],edit=[0,0,0],steps=256)
        start=next(replay(tape,0,"remove",48,16,.1,0))
        self.assertTrue(start[1])
        np.testing.assert_array_equal(start[3],[[1,1,1]])
        start=next(replay(tape,0,"add",48,16,.1,0))
        self.assertFalse(start[1])
        np.testing.assert_array_equal(start[3],[[0,0,0]])

    def test_existing_near_and_distant_curve_is_reproduced(self):
        got = limits(np.array([0.,48.,96.,128.,192.,256.,384.]),48,16)
        np.testing.assert_equal(got,[16,16,256,256,316,1986,65536])

    def test_small_near_reset_preserves_the_existing_far_curve(self):
        distances = np.array([0., 16., 16.001, 48., 192., 384.])
        np.testing.assert_equal(limits(distances,16,16),[16,16,256,256,316,65536])

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
        self.assertEqual(sustained(rows)["previous_failure_paths"],16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("plan")
    p.add_argument("tape",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--histories",type=int,nargs="+",help="default: 16 256")
    p.add_argument("--literal",action="store_true",
                   help="use exact history weights everywhere, bypassing the prior floor; 0 discards, 65536 retains")
    p.add_argument("--view",help="select a named equal-footprint camera station recorded in the tape")
    p.add_argument("--radii",type=int,nargs="+",help="default: 48")
    p.add_argument("--policies",nargs="+",help="explicit radius:history shortlist, instead of histories/radii")
    p.add_argument("--equations",nargs="+",help="baseline, near16, power2, power4, power6; no literal/history flags")
    p.add_argument("--maximum",type=int,default=65536,help="statistical weight ceiling, independent of edit caps")
    p.add_argument("--windows",type=int,nargs="+",default=[0])
    p.add_argument("--cases",choices=["null","remove","add","cold-before","cold-after",
                                    "static-before","static-after"],nargs="+",default=["null","remove","add"])
    p.add_argument("--dt",type=float,default=.85,
                   help="uniform seconds per 16-path face publication, not frame time")
    p.add_argument("--every",type=int,default=0,help="repeat edit every N fresh paths; 0 is one edit")
    p.add_argument("--null-every",type=int,help="separate event interval for unchanged-lighting pressure control")
    p.add_argument("--analytic",action="store_true",
                   help="render noise-free residual-bias and ideal update-radius controls")
    p.add_argument("--sky-controls",action="store_true",
                   help="also render each direct-sun state with indirect GI zeroed")
    p.add_argument("--checkpoints",type=int,nargs="+",
                   help="explicit fresh-path observations; zero is always captured")
    a = sub.add_parser("assess")
    a.add_argument("plan",type=Path)
    a.add_argument("run",type=Path)
    a.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.command=="plan": make_plan(args)
    else: assess(args)


if __name__=="__main__":
    main()
