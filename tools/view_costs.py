"""View-tier cost per run: frame times, GPU stages, receivers and far tier.

Usage: PYTHONPATH=target/perceptual-python python tools/view_costs.py LIST
LIST is tools/runs.sh output ("<label> <run> <log>" lines). Frame
percentiles come from the run's report (offscreen cadence, not windowed
pacing); the rest is the mean of the second half of the log's "GI view
transport" receipts. "fixed" is discovery + presentation + far tier: the GPU
cost the transport budget does not control (trace is budgeted).
"""
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import gates  # noqa: E402

ANSI = re.compile(r"\x1b\[[0-9;]*m")
KEYS = ["far_discovery_ms", "far_trace_ms", "far_presentation_ms", "far_pixel_ms", "far_active",
        "far_pixel_owned", "far_pixel_traced", "far_pixel_history", "far_history"]


def main(path):
    print(f"{'run':10} {'p50':>6} {'p95':>6} {'p99':>6} | {'disc':>5} {'pres':>5} {'far':>5} {'fixed':>6} {'trace':>5}"
          f" | {'recv':>6} {'owned':>7} {'traced':>7} {'cellH':>5} {'recvH':>6}")
    for line in open(path):
        if line.startswith("done"):
            continue
        label, rest = line.strip().split(" ", 1)
        run, log = re.findall(r"/.*?(?= /|$)", rest)
        cost = gates.cost(Path(run))
        rows = [dict(re.findall(r"(far_\w+)=([-\d.e]+)", ANSI.sub("", l))) for l in open(log) if "GI view transport" in l]
        rows = rows[len(rows) // 2:] or rows
        m = {k: statistics.mean(float(r.get(k, 0)) for r in rows) for k in KEYS}
        fixed = m["far_discovery_ms"] + m["far_presentation_ms"] + m["far_pixel_ms"]
        print(f"{label:10} {cost['frame_p50']:6.1f} {cost['frame_p95']:6.1f} {cost['frame_p99']:6.1f} |"
              f" {m['far_discovery_ms']:5.2f} {m['far_presentation_ms']:5.2f} {m['far_pixel_ms']:5.2f} {fixed:6.2f}"
              f" {m['far_trace_ms']:5.2f} | {m['far_active']:6.0f} {m['far_pixel_owned']:7.0f}"
              f" {m['far_pixel_traced']:7.0f} {m['far_pixel_history']:5.1f} {m['far_history']:6.1f}")


if __name__ == "__main__":
    main(sys.argv[1])
