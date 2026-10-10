#!/usr/bin/env python3
"""Run the exact finite-window policy and a single-null-edit sensitivity case.

The additional case reuses the unchanged native cave tape, marking it suspect
only once instead of every 128 paths. This distinguishes a false alarm from
repeated invalidation. It is a synthetic event, not an actual geometry edit or
a streaming/gameplay benchmark. Retaining history without any suspect events
is already represented by the retain control.
"""
import argparse
import json
from pathlib import Path

from history_qualification import analyze


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    tape = json.loads(Path(args.tape).read_text())
    cave = next(case for case in tape["cases"] if case["name"] == "cave-null-streaming")
    single = dict(cave, name="cave-single-null-edit", repeated=False,
                  runs=[dict(run, **{
                      key: {"0": run[key]["0"]}
                      for key in ("individual_before", "individual_after")
                  }) for run in cave["runs"]])
    tape["cases"].append(single)
    report = analyze(tape)
    Path(args.output).write_text(json.dumps(report))
