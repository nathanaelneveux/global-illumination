"""Apply the user-calibrated cave FLIP gate to a frozen surface-cache sweep.

Usage: PYTHONPATH=tools:target/perceptual-python python3 tools/cache_flip_gate.py RUN

RUN is a cave reference run with VOXEL_GI_VALIDATE_CACHE_SWEEP=1. Each raw
prefix (samples-N) and each evidence reconstruction (evidenceT-N) is compared
with cave-reference.png. The gate is tools/quality_profiles/cave-user-2048.json's
FLIP limits: mean <= .0274008632 and worst 32px tile mean <= .0729644448, in every
window. Raw pass means sampler noise alone is acceptable; an evidence target
passing at low N means that reconstruction hides the remaining noise.
"""
import json
import re
import sys
from pathlib import Path

from cave_sample_analysis import load_rgb, perceptual_difference

MEAN, TILE = 0.027400863220808837, 0.07296444475650787

def main(run):
    run = Path(run)
    reference = json.loads((run / "cave-reference.json").read_text())
    if not reference["qualification"]["qualified"]:
        raise SystemExit(f"unqualified reference: {reference['qualification']}")
    status = json.loads((run / "cave-reference-status.json").read_text())
    if status.get("status") != "completed":
        raise SystemExit(f"incomplete capture: {status}")
    truth = load_rgb(run / "cave-reference.png")
    replica = perceptual_difference(load_rgb(run / "reference-replica-a.png"),
                                    load_rgb(run / "reference-replica-b.png"), 67)
    rows = {}
    for path in sorted(run.glob("*-window-*.png")):
        match = re.fullmatch(r"(samples|evidence\d*)-(\d+)-window-(\d+)", path.stem)
        if not match:
            continue
        label, paths, window = match.group(1), int(match.group(2)), int(match.group(3))
        d = perceptual_difference(truth, load_rgb(path), 67)
        rows.setdefault((label, paths), []).append(
            (window, d["flip"]["mean"], d["flip_32px_tile_means"]["max"]))
    out = {"replica_flip_mean": replica["flip"]["mean"],
           "replica_worst_tile": replica["flip_32px_tile_means"]["max"],
           "gate": {"flip_mean": MEAN, "worst_tile": TILE}, "rows": []}
    for (label, paths), windows in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        mean = max(w[1] for w in windows)
        tile = max(w[2] for w in windows)
        out["rows"].append({"variant": label, "paths": paths, "windows": len(windows),
                            "worst_flip_mean": mean, "worst_tile": tile,
                            "pass": len(windows) == 4 and mean <= MEAN and tile <= TILE})
    print(f"reference replicas: FLIP mean {out['replica_flip_mean']:.5f}, worst tile {out['replica_worst_tile']:.5f}")
    print(f"{'variant':12s} {'paths':>6s} {'flip mean':>10s} {'worst tile':>10s}  pass")
    for r in out["rows"]:
        print(f"{r['variant']:12s} {r['paths']:6d} {r['worst_flip_mean']:10.5f} {r['worst_tile']:10.5f}  {'PASS' if r['pass'] else 'fail'}")
    (run / "cache-flip-gate.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main(sys.argv[1])
