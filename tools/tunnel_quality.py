"""Compare visible tunnel edit transients; later frames are NOT an accuracy oracle.

Usage: PYTHONPATH=target/perceptual-python python tools/tunnel_quality.py
    BASELINE_RUN CANDIDATE_RUN OUTPUT_DIRECTORY
The 400ms/1700ms pair uses the same camera and requested geometry (both blocks
removed). Review the images for late mesh publication before interpreting FLIP.
50ms captures still contain the lower block and are not compared as lighting.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from view_cache_quality import compare, image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    depths = [1, 2, 4, 8, 12, 20]
    results = {"interpretation": "post-edit self change, not reference accuracy"}
    sheet = Image.new("RGB", (1536, len(depths) * 240), "#202020")
    draw = ImageDraw.Draw(sheet)
    for label, run in [("baseline", args.baseline), ("candidate", args.candidate)]:
        report = json.loads((run / "report.json").read_text())
        assert report["status"] == "completed" and report["profile"] == "tunnel"
        assert report["capture_mode"] == "offscreen"
        assert report["view_diagnostic"] == 0, "Diagnostic masks are not lighting quality"
        assert report["settings"]["receiver_strategy"]["mode"] == "view"
        captures = report["capture_windows"]
        assert len(captures) == 20 and all(c["target_valid"] and c["varied"]
            and c["width"] == 2560 and c["height"] == 1440 for c in captures)
        events = json.loads((run / "tunnel.json").read_text())["events"]
        assert len(events) == 40 and all(e["was_solid"] for e in events)
        frames = [f["interval_ms"] for f in report["measurements"]["frame_timeline"]
                  if f["at_s"] >= 15 and not f["capture_excluded"]]
        result = {"run": str(run), "build": report["build"], "settings": report["settings"],
                  "offscreen_cadence_p50_p95_p99_ms": np.percentile(frames, [50, 95, 99]).tolist(),
                  "depths": {}}
        for row, depth in enumerate(depths):
            early_path = run / f"depth{depth:02}-0400ms.png"
            later_path = run / f"depth{depth:02}-1700ms.png"
            early, later = image(early_path), image(later_path)
            lum = np.array([.2126, .7152, .0722])
            a, b = early @ lum, later @ lum
            stats = compare(early, later)
            stats["black_then_lit_pixels"] = int(np.count_nonzero((early.max(axis=2) <= 2 / 255) & (b > .02)))
            stats["bright_then_dimmer_pixels"] = int(np.count_nonzero(a > b * 1.5 + .02))
            result["depths"][depth] = stats
            for offset, path in enumerate([early_path, later_path]):
                column = (0 if label == "baseline" else 2) + offset
                sheet.paste(Image.open(path).convert("RGB").resize((384, 216), Image.Resampling.LANCZOS),
                            (column * 384, row * 240 + 24))
                draw.text((column * 384 + 5, row * 240 + 5),
                          f"{label} depth {depth}: {'400' if offset == 0 else '1700'} ms", fill="white")
        results[label] = result
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "tunnel.json").write_text(json.dumps(results, indent=2))
    sheet.save(args.output / "tunnel.png")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
