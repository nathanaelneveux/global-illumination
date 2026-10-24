"""High-skylight response: FLIP of each capture against the same-time capture
of a run whose skylight was open from startup (cave profile,
VOXEL_GI_VALIDATE_CAVE_SKYLIGHT=pre). The reference is settled, not an
accuracy oracle; its own 10 s vs 30 s difference is printed as the noise floor.

Usage: uv run --python 3.12 --with numpy --with pillow --with flip-evaluator
    tools/skylight_quality.py PRE_RUN label=OPEN_RUN [label=OPEN_RUN ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from view_cache_quality import compare, image

NAMES = ["before", "open-0100ms", "open-0300ms", "open-0600ms", "open-1000ms",
         "open-3000ms", "open-10s", "open-30s"]


def tile_max(a, b, size=32):
    weights = np.array([.2126, .7152, .0722])
    d = np.abs(a @ weights - b @ weights)
    h, w = d.shape[0] // size * size, d.shape[1] // size * size
    return float(d[:h, :w].reshape(h // size, size, w // size, size).mean(axis=(1, 3)).max())


def check(run):
    report = json.loads((run / "report.json").read_text())
    assert report["status"] == "completed" and report["profile"] == "cave", run
    return report.get("view_diagnostic", 0)


def main():
    reference = Path(sys.argv[1])
    diagnostic = check(reference)
    floor = compare(image(reference / "open-10s.png"), image(reference / "open-30s.png"))
    print(f"diagnostic {diagnostic}; reference 10s vs 30s flip {floor['flip_mean']:.4f}")
    print(f"{'':10}" + "".join(f"{n.removeprefix('open-'):>15}" for n in NAMES))
    for argument in sys.argv[2:]:
        label, run = argument.split("=", 1)
        run = Path(run)
        assert check(run) == diagnostic, "compare like diagnostics only"
        cells = []
        for name in NAMES:
            a, b = image(run / f"{name}.png"), image(reference / f"{name}.png")
            cells.append(f"{compare(a, b)['flip_mean']:.4f}/{tile_max(a, b):.2f}")
        print(f"{label:10}" + "".join(f"{c:>15}" for c in cells))


if __name__ == "__main__":
    main()
