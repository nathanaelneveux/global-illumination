"""Compare actual view-owned presentation, not the CPU world-field mirror.

Usage: PYTHONPATH=target/perceptual-python python tools/view_cache_quality.py
    PROFILE BASELINE_RUN CANDIDATE_RUN OUTPUT_DIRECTORY
Requires numpy, Pillow and flip-evaluator. FLIP uses 67 pixels/degree. Early
captures are compared with each run's own later image: remaining change, NOT
accuracy against a converged oracle. Cross-run final differences are separate.
"""
import argparse
import json
from pathlib import Path

import flip_evaluator as flip
import numpy as np
from PIL import Image, ImageDraw


def image(path, ceiling=False):
    im = Image.open(path).convert("RGB")
    if ceiling:
        im = im.crop((0, 0, im.width, int(im.height * .4)))
    return np.asarray(im, dtype=np.float32) / 255


def compare(actual, reference):
    assert actual.shape == reference.shape
    errors, mean, _ = flip.evaluate(reference, actual, "LDR", applyMagma=False,
                                   parameters={"ppd": 67.0})
    weights = np.array([.2126, .7152, .0722])
    lum, ref_lum = actual @ weights, reference @ weights
    # Image luminance is display-referred. Do not interpret it as irradiance.
    return {"flip_mean": float(mean), "flip_p95": float(np.quantile(errors, .95)),
            "display_luminance_ratio": float(lum.mean() / max(ref_lum.mean(), 1e-8))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", choices=["house", "response", "cave", "handoff"])
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    pairs = {
        "house": [("turned-0400ms", "second-corner"), ("return-0400ms", "first-corner-returned")],
        "response": [(f"emitter-{action}-{ms:04}ms", final)
                     for action, offsets, final in [("add", [200, 1000], "emitter-settled"),
                         ("remove", [100, 300, 600, 1000], "room-recovered")]
                     for ms in offsets],
        "cave": [(f"cave-{s}s", "cave-60s") for s in [15, 30]],
        "handoff": [(f"admit-{ms:04}ms", "near-later") for ms in [100, 400, 1000, 2000]]
                   + [("return-0400ms", "far-later"), ("far-before", "far-later")],
    }[args.profile]
    results = {"profile": args.profile, "interpretation": "self change, not reference accuracy"}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, run in [("baseline", args.baseline), ("candidate", args.candidate)]:
        report = json.loads((run / "report.json").read_text())
        assert report["capture_mode"] == "offscreen"
        captures = report["capture_windows"]
        assert captures and all(c["target_valid"] and c["varied"] and c["width"] == 2560
                                and c["height"] == 1440 for c in captures)
        frames = [f["interval_ms"] for f in report["measurements"]["frame_timeline"]
                  if f["at_s"] >= 15 and not f["capture_excluded"]]
        results[name] = {"run": str(run), "build": report["build"], "settings": report["settings"],
                         "peak_mb": report["memory"]["peak_mb"],
                         "offscreen_cadence_p50_p95_p99_ms": np.percentile(frames, [50, 95, 99]).tolist(),
                         "self_change": {a: compare(image(run / f"{a}.png", args.profile == "house"),
                                                   image(run / f"{b}.png", args.profile == "house"))
                                         for a, b in pairs}}
    finals = sorted({b for _, b in pairs})
    results["final_cross_run"] = {b: compare(image(args.candidate / f"{b}.png", args.profile == "house"),
                                           image(args.baseline / f"{b}.png", args.profile == "house"))
                                  for b in finals}
    (args.output / f"{args.profile}.json").write_text(json.dumps(results, indent=2))
    selected = list(dict.fromkeys([a for a, _ in pairs] + finals))
    for capture in selected:
        sheet = Image.new("RGB", (1536, 464), "#202020")
        draw = ImageDraw.Draw(sheet)
        for column, (label, run) in enumerate([("baseline", args.baseline), ("two-way", args.candidate)]):
            im = Image.open(run / f"{capture}.png").convert("RGB")
            sheet.paste(im.resize((768, 432), Image.Resampling.LANCZOS), (column * 768, 32))
            draw.text((column * 768 + 8, 8), f"{label}: {capture}", fill="white")
        sheet.save(args.output / f"{args.profile}-{capture}.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
