"""Day-cycle analysis: day_cycle_analysis.py label=run_dir ...

Prints each run's day.json series (history percentiles, traced paths, sky
retraces), FLIP between the paired captures one second apart, and, for runs
not labelled ref*, FLIP of the 60/120/180 s captures against fixed-sun
references labelled ref18/ref36/ref54 (their settled 180 s captures).
Run with: uv run --python 3.12 --with numpy --with pillow --with flip-evaluator."""
import sys, json, os
sys.path.insert(0, 'tools')
from cave_sample_analysis import load_rgb, perceptual_difference
runs = dict(a.split('=', 1) for a in sys.argv[1:])
def flip(a, b):
    d = perceptual_difference(load_rgb(a), load_rgb(b), 67)
    return d['flip']['mean'], d['flip_32px_tile_means']['max']
for label, r in runs.items():
    d = json.load(open(os.path.join(r, 'day.json')))
    s = d['samples']
    print(f"== {label} {d['settings']}")
    prev = None
    for x in s:
        t = x['at_s']
        if prev is None or int(t) % 15 == 0 or x is s[-1]:
            if prev is not None:
                dt = t - prev['at_s']
                paths = (x['traced_paths'] - prev['traced_paths']) / dt / 1e6
                sky = (x['sky_corners_traced'] - prev['sky_corners_traced']) / dt / 1e3
            else:
                paths = sky = 0
            print(f"  t={t:6.1f} angle={x['angle_deg']:5.1f} faces={x['faces']:7d} samples p10/p50/p90={x['samples_p10']}/{x['samples_p50']}/{x['samples_p90']} paths={paths:6.2f}M/s sky={sky:7.1f}k/s src_rev={x['source_revision']}")
            prev = x
    pairs = []
    for t in (30, 60, 90, 120, 150, 180):
        m, tile = flip(f"{r}/day-{t:03d}s.png", f"{r}/day-{t+1:03d}s.png")
        pairs.append(f"{t}:{m:.4f}/{tile:.2f}")
    print('  flicker (t vs t+1, mean/tile max):', ' '.join(pairs))
refs = {k: v for k, v in runs.items() if k.startswith('ref')}
for label, r in runs.items():
    if label.startswith('ref'): continue
    out = []
    for t, ref in ((60, 'ref18'), (120, 'ref36'), (180, 'ref54')):
        if ref in refs:
            m, tile = flip(f"{r}/day-{t:03d}s.png", f"{refs[ref]}/day-180s.png")
            out.append(f"{t}s vs {ref}: {m:.4f}/{tile:.2f}")
    print(f"lag {label}:", ' | '.join(out))
# reference self-noise: ref at 120 vs its own 180 (same sun, settled)
for label, r in refs.items():
    m, tile = flip(f"{r}/day-120s.png", f"{r}/day-180s.png")
    print(f"ref noise {label} 120 vs 180: {m:.4f}/{tile:.2f}")
