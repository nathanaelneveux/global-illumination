"""Edit response from a `response` validation run's response.json.

Usage: python3 -I tools/response_timing.py RUN [RUN ...]

For each tracked face, the displayed luminance is followed across the scripted
emitter add (~20 s) and removal (~25 s). Reports how long after each event the
face first stays within 10% (of its settled value) of where it ends up, using
the value just before the next event as the target. Faces whose total change is
under 5% of the larger level are skipped (the event did not reach them).
"""
import json, os, statistics, sys

def lum(rgb):
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]

def series(run):
    data = json.load(open(os.path.join(run, "response.json")))
    caps = {c["name"]: c["requested_s"] for c in data["captures"]}
    faces = {}
    for sample in data["samples"]:
        for f in sample.get("faces", []):
            if f["display"] is None:
                continue
            key = (tuple(f["voxel"]), f["face"])
            faces.setdefault(key, []).append((sample["at_s"], lum(f["display"])))
    return caps, faces

def settle_after(points, start, end):
    before = [v for t, v in points if t < start]
    window = [(t, v) for t, v in points if start <= t <= end]
    if not before or len(window) < 3:
        return None
    initial, target = before[-1], window[-1][1]
    if abs(target - initial) < 0.05 * max(abs(target), abs(initial), 1e-9):
        return None
    settled = start
    for t, v in window:
        if abs(v - target) > 0.1 * abs(target - initial):
            settled = t
    return settled - start

for run in sys.argv[1:]:
    caps, faces = series(run)
    add = caps["emitter-add-0200ms"] - 0.2
    remove = caps["emitter-remove-0100ms"] - 0.1
    end = caps["room-recovered"]
    out = []
    for name, (start, stop) in {"emitter add": (add, remove), "emitter remove": (remove, end)}.items():
        times = [s for p in faces.values() if (s := settle_after(p, start, stop)) is not None]
        if times:
            times.sort()
            out.append(f"{name}: faces {len(times)}, settle p50 {times[len(times)//2]:.2f}s p90 {times[int(len(times)*.9)]:.2f}s max {times[-1]:.2f}s")
    print(os.path.basename(run))
    for line in out:
        print("  " + line)
