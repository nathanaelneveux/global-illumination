"""Compare two tools/gates.sh labels: cost and what the player sees.

usage: uv run --python 3.12 --with numpy --with pillow --with flip-evaluator \
    python tools/gates.py <baseline-label> <candidate-label>

Cost: view-tier GPU receipts (median after the first quarter of the run),
headless frame-time percentiles and GI main-thread time per frame (streaming),
peak process footprint. Offscreen cadence is not windowed frame pacing.
Visible: static-pose pixels darker/brighter than baseline by >.1 luminance
(combined, sun, indirect), single-frame black flashes on two walks, black
pixels per frame after a turn, tunnel black-then-lit, response FLIP.
"""
import json, re, statistics, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
W = np.array([.2126, .7152, .0722])
ANSI = re.compile(r"\x1b\[[0-9;]*m")
GPU_KEYS = ["far_trace_ms", "far_presentation_ms", "far_discovery_mean_ms", "far_pool_scan_ms"]


def runs(label):
    out = {}
    for line in open(ROOT / f"target/gates/{label}.txt"):
        name, d, log = line.split()
        out[name] = (None if d == "-" else ROOT / d, ROOT / log)
    return out


def lum(path):
    return (np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255) @ W


def gpu(log):
    rows = []
    for line in open(log):
        if "GI view transport" in line:
            rows.append(dict(re.findall(r"(far_\w+)=([-\d.e]+)", ANSI.sub("", line))))
    rows = rows[len(rows) // 4:]
    return {k: statistics.median(float(r[k]) for r in rows) if rows else float("nan") for k in GPU_KEYS}


def cost(d):
    r = json.load(open(d / "report.json"))
    m = r["measurements"]
    phase = [p for p in m["phases"] if p["frame_ms"].get("count")][-1]
    gi = sorted(sum(f["previous_gi_body_ms"]) for f in m["frame_timeline"][len(m["frame_timeline"]) // 4:])
    peak = r["memory"].get("peak_mb", float("nan"))
    return {"frame_p50": phase["frame_ms"]["p50"], "frame_p95": phase["frame_ms"]["p95"],
            "frame_p99": phase["frame_ms"]["p99"],
            "gi_main_p95": gi[int(len(gi) * .95)] if gi else float("nan"), "peak_mb": peak}


def static(base, cand):
    out = []
    for name in ["base-1", "turned-1", "returned"]:
        d = lum(cand / f"{name}.png") - lum(base / f"{name}.png")
        out.append((int((d < -.1).sum()), int((d > .1).sum())))
    return out


def flashes(d):
    f = [np.asarray(Image.open(d / f"walk-{b}-{i:02}.png").convert("RGB"), dtype=np.float32) / 255
         for b in "ab" for i in range(16)]
    total = 0
    for b in range(2):
        s = f[b * 16:(b + 1) * 16]
        for i in range(1, 15):
            total += int(((s[i].max(axis=2) <= 2 / 255) & (s[i - 1] @ W > .05) & (s[i + 1] @ W > .05)).sum())
    return total


def turn(d):
    final = lum(d / "turn-3s.png") > .08
    return [int(((lum(d / f"turn-{i:02}.png") < .02) & final).sum()) for i in range(8)]


def tool(args, out):
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, *args, str(out)], check=True, capture_output=True)
    return json.load(open(next(out.glob("*.json"))))


def main(base_label, cand_label):
    base, cand = runs(base_label), runs(cand_label)
    missing = [n for n in base if base[n][0] is None or cand.get(n, (None,))[0] is None]
    if missing:
        print("MISSING/FAILED runs:", missing)
    rows = []
    for name in ["streaming", "cave"]:
        if name in missing: continue
        b, c = gpu(base[name][1]), gpu(cand[name][1])
        rows += [(f"{name} {k.replace('far_', '').replace('_ms', '')} ms", b[k], c[k]) for k in GPU_KEYS]
    if "streaming" not in missing:
        b, c = cost(base["streaming"][0]), cost(cand["streaming"][0])
        rows += [(f"streaming {k}", b[k], c[k]) for k in b]
    for label, base_v, cand_v in rows:
        delta = cand_v - base_v if base_v == base_v and cand_v == cand_v else float("nan")
        print(f"{label:34} {base_v:9.3f} {cand_v:9.3f} {delta:+9.3f}")
    for name in ["static", "static-sun", "static-indirect"]:
        if name not in missing:
            print(f"{name:34} darker/brighter >.1 per capture: {static(base[name][0], cand[name][0])}")
    if not {"walk1", "walk2"} & set(missing):
        print(f"{'walk black flashes':34} {flashes(base['walk1'][0])}, {flashes(base['walk2'][0])} -> "
              f"{flashes(cand['walk1'][0])}, {flashes(cand['walk2'][0])}")
    if "turn" not in missing:
        print(f"{'turn black px, frames 0-7':34} {turn(base['turn'][0])}")
        print(f"{'':34} {turn(cand['turn'][0])}")
    scratch = ROOT / f"target/gates/compare-{base_label}-{cand_label}"
    if "tunnel" not in missing:
        t = tool([str(ROOT / "tools/tunnel_quality.py"), str(base["tunnel"][0]), str(cand["tunnel"][0])], scratch / "tunnel")
        for side in ["baseline", "candidate"]:
            depths = t[side]["depths"]
            print(f"tunnel {side:27} black-then-lit {sum(v['black_then_lit_pixels'] for v in depths.values())}, "
                  f"FLIP by depth {[round(v['flip_mean'], 4) for v in depths.values()]}")
    if "response" not in missing:
        r = tool([str(ROOT / "tools/view_cache_quality.py"), "response", str(base["response"][0]), str(cand["response"][0])], scratch / "response")
        for side in ["baseline", "candidate"]:
            print(f"response {side:25} " + ", ".join(f"{k.replace('emitter-', '')} {v['flip_mean']:.3f}" for k, v in r[side]["self_change"].items()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
