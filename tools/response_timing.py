"""Edit response from a `response`/`edits` run's response.json, or the
`house-edits` run's house.json (same capture and sample layout).

Usage: python3 -I tools/response_timing.py RUN [RUN ...]

Events come from capture names `<event>-<ms>ms` (event start = capture time minus
its offset). For each tracked face, displayed luminance is followed from one
event to the next. Settle = the last time after the event that the face was more
than 10% of its change away from its value just before the next event. Faces
whose change is under 5% of the larger level are skipped (not reached). Also
reports the summed settle p90 across events as one response score.
"""
import json, os, re, sys

def lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def load(run):
    name = "response.json" if os.path.exists(os.path.join(run, "response.json")) else "house.json"
    data = json.load(open(os.path.join(run, name)))
    starts = {}
    for c in data["captures"]:
        m = re.fullmatch(r"(.+)-(\d+)ms", c["name"])
        if m:
            start = c["requested_s"] - int(m.group(2)) / 1000
            starts[m.group(1)] = min(starts.get(m.group(1), start), start)
    end = max(c["requested_s"] for c in data["captures"])
    faces = {}
    for sample in data["samples"]:
        for f in sample.get("faces", []):
            if f["display"] is not None:
                faces.setdefault((tuple(f["voxel"]), f.get("face")), []).append((sample["at_s"], lum(f["display"])))
    return sorted(starts.items(), key=lambda kv: kv[1]), end, faces

def settle(points, start, stop):
    before = [v for t, v in points if t < start]
    window = [(t, v) for t, v in points if start <= t <= stop]
    if not before or len(window) < 3:
        return None
    initial, target = before[-1], window[-1][1]
    if abs(target - initial) < 0.05 * max(abs(target), abs(initial), 1e-9):
        return None
    last = start
    for t, v in window:
        if abs(v - target) > 0.1 * abs(target - initial):
            last = t
    return last - start

def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]

for run in sys.argv[1:]:
    events, end, faces = load(run)
    print(os.path.basename(run))
    score = 0.0
    for i, (name, start) in enumerate(events):
        stop = events[i + 1][1] if i + 1 < len(events) else end
        times = [s for p in faces.values() if (s := settle(p, start, stop)) is not None]
        if times:
            score += quantile(times, 0.9)
            print(f"  {name:14s} faces {len(times):2d}  settle p50 {quantile(times, .5):5.2f}s  p90 {quantile(times, .9):5.2f}s")
    print(f"  summed p90 {score:.2f}s")
