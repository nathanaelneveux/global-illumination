#!/usr/bin/env python3
"""Compare final spatial presentation through streamed geometry/emitter edits.

The control's final frames per state are observed tails, NOT qualified truth.
Also report each run's distance to its own tail and cross-run tail disagreement:
otherwise a different settled answer can masquerade as a faster response.
CPU temporal traces deliberately do not stand in for final spatial images.
"""
import argparse
import json
from pathlib import Path

from cave_sample_analysis import load_rgb, perceptual_difference
from spatial_quality import residual_scales


PROFILES = {
    "house-edits": (9, {
        "skylight": ("skylight-0400ms", "skylight-10s", "skylight-29s"),
        "west-door": ("west-door-0400ms", "west-door-29s"),
        "east-door-turn": ("east-door-turn-0400ms", "east-door-turn-10s", "east-door-turn-29s"),
    }),
    "response": (14, {
        "cavity": ("dig-0100ms", "dig-0300ms", "dig-0600ms", "dig-1000ms", "cavity-settled"),
        "emitter-add": ("emitter-add-0200ms", "emitter-add-1000ms", "emitter-settled"),
        "emitter-remove": ("emitter-remove-0100ms", "emitter-remove-0300ms",
                           "emitter-remove-0600ms", "emitter-remove-1000ms", "room-recovered"),
    }),
}


def validate_run(folder):
    report = json.loads((folder / "report.json").read_text())
    expected = PROFILES.get(report["profile"], (0, {}))[0]
    if (not expected or report["status"] != "completed"
            or report["captures"] != expected or not report["background_terrain"]["enabled"]):
        raise ValueError(f"require complete streamed edit run: {folder}")
    captures = report["capture_windows"]
    if len(captures) != expected or any(not c["target_valid"] or not c["varied"]
            or (c["width"], c["height"]) != (2560, 1440)
            or (c["capture_mode"] != "offscreen" and not c["focused"]) for c in captures):
        raise ValueError(f"invalid capture evidence: {folder}")
    if report["profile"] == "house-edits":
        house = json.loads((folder / "house.json").read_text())
        edits = {e["name"]: e["removed_solid_voxels"] for e in house["events"]}
        if edits != {"skylight": 1, "west-door": 3, "east-door-turn": 3}:
            raise ValueError(f"require all three geometry edits: {folder}")
    else:
        response = json.loads((folder / "response.json").read_text())
        if {s["event"] for s in response["samples"] if "event" in s} != {1, 2, 3}:
            raise ValueError(f"require cavity, emitter addition and removal: {folder}")
    return report


def difference(reference, image):
    return dict(**perceptual_difference(reference, image, 67),
                residual_scales=residual_scales(image, reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidates", type=Path, nargs="+")
    args = parser.parse_args()
    reports = {str(p): validate_run(p) for p in [args.control, *args.candidates]}
    profile = reports[str(args.control)]["profile"]
    if any(r["profile"] != profile for r in reports.values()):
        raise ValueError("cannot compare different edit profiles")
    rows = []
    for phase, checkpoints in PROFILES[profile][1].items():
        common_tail = load_rgb(args.control / f"{checkpoints[-1]}.png")
        for folder in [args.control, *args.candidates]:
            own_tail = load_rgb(folder / f"{checkpoints[-1]}.png")
            for checkpoint in checkpoints:
                image = load_rgb(folder / f"{checkpoint}.png")
                common = difference(common_tail, image)
                own = common if folder == args.control else difference(own_tail, image)
                rows.append(dict(run=str(folder), phase=phase, checkpoint=checkpoint,
                                 common_tail=common, own_tail=own))
    print(json.dumps(dict(
        note="Observed tails, not ground truth or a certified settling threshold. "
             "Background scheduling varies between runs; inspect cross-run tail differences.",
        control=str(args.control), measurements=rows,
        settings={p: r["settings"] for p, r in reports.items()},
    ), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
