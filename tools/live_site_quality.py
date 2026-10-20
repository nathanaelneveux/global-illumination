"""Gameplay-facing quality metrics from a live validation temporal.json.

Usage: python3 -I tools/live_site_quality.py RUN_DIR [RUN_DIR ...]

For each run (fixed receiver sites, event-compressed states):
- blotch@t: median relative difference between adjacent sites (distance 1,
  same face) at time t, for raw measured and displayed RGB. Smooth lighting
  has small neighbour differences; noisy sampling has large ones.
- settle: per site, first time after which the displayed luminance stays
  within 15% of its own mean over the last sixth of the run. p50/p90 over sites.
- flicker: per site, relative std of displayed luminance over the last third.
These are self-consistency metrics, not accuracy against a reference.
"""
import json, os, statistics, sys

def lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def timelines(run):
    t = json.load(open(os.path.join(run, "temporal.json")))
    sites = [tuple(s) for s in t["sites"]]
    series = [[] for _ in sites]
    for e in t["events"]:
        st = e["state"]
        if st is None:
            series[e["site"]].append((e["at_s"], None, None))
        else:
            series[e["site"]].append((e["at_s"], st["measured"], st["display"]))
    return sites, t["site_faces"], series, t["end_s"]

def value_at(points, time, index):
    current = None
    for at, measured, display in points:
        if at > time:
            break
        current = (measured, display)[index] if measured is not None else None
    return current

def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")

def analyse(run):
    sites, faces, series, end = timelines(run)
    pairs = [(i, j) for i in range(len(sites)) for j in range(i + 1, len(sites))
             if faces[i] == faces[j] and sum(abs(a - b) for a, b in zip(sites[i], sites[j])) == 1]
    out = {"run": os.path.basename(run), "sites": len(sites), "pairs": len(pairs), "end_s": round(end)}
    for time in [t for t in (5, 15, 30, 60, 120, 180) if t <= end]:
        for index, name in ((0, "raw"), (1, "display")):
            diffs = []
            for i, j in pairs:
                a, b = value_at(series[i], time, index), value_at(series[j], time, index)
                if a is None or b is None:
                    continue
                la, lb = lum(a), lum(b)
                if max(la, lb) > 0:
                    diffs.append(abs(la - lb) / max(la, lb))
            out[f"blotch_{name}@{time}s"] = round(statistics.median(diffs), 4) if diffs else None
    settle, flicker, final = [], [], []
    for points in series:
        samples = [(at, lum(d)) for at, m, d in points if d is not None]
        tail = [v for at, v in samples if at >= end * 5 / 6]
        if not tail or statistics.mean(tail) <= 0:
            continue
        target = statistics.mean(tail)
        final.append(target)
        settled_at = 0.0
        for at, v in samples:
            if abs(v - target) > 0.15 * target:
                settled_at = at
        settle.append(settled_at)
        window = [v for at, v in samples if at >= end * 2 / 3]
        if len(window) > 2:
            flicker.append(statistics.pstdev(window) / target)
    out["settle_p50_s"] = round(quantile(settle, 0.5), 1)
    out["settle_p90_s"] = round(quantile(settle, 0.9), 1)
    out["flicker_p50"] = round(quantile(flicker, 0.5), 4)
    out["flicker_p95"] = round(quantile(flicker, 0.95), 4)
    out["final_lum_p50"] = quantile(final, 0.5)
    return out

if __name__ == "__main__":
    rows = [analyse(run) for run in sys.argv[1:]]
    keys = list(rows[0].keys())
    for key in keys:
        print(f"{key:22s} " + " ".join(f"{str(r.get(key)):>28s}" for r in rows))
