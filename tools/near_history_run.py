#!/usr/bin/env python3
"""Render/assess recorded near-history plans sequentially, with bounded disk use.

No transport benchmark is run and no interactive performance is claimed. Each
plan uses the normal background-loading prelude and strict native capture audit.
Optional compaction hashes reproducible intermediates only after full assessment.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from near_history_calibration import digest


def compact(plan, run, assessment, keep_paths=(0, 512, 2048, 4096)):
    """Only delete exact generated captures identified by a completed assessment."""
    result = json.loads(assessment.read_text())
    if not result.get("complete") or result["plan"] != plan:
        raise ValueError("compaction requires this plan's complete assessment")
    root = Path("target/gi-validation").resolve()
    if run.resolve().parent != root:
        raise ValueError("compaction is restricted to immediate validation runs")
    archive = run / "capture-archive-index.json"
    if archive.exists():
        return
    records = []
    for frame in plan["frames"]:
        name = frame["name"]
        if not re.fullmatch(r"[A-Za-z0-9-]+",name):
            raise ValueError("invalid generated frame name")
        path = run / (name + ".png")
        keep = (frame.get("reference") or name.startswith("warm-")
                or frame.get("fresh_paths") in keep_paths)
        records.append(dict(name=path.name,bytes=path.stat().st_size,
                            sha256=digest(path),retained=bool(keep)))
    archive.write_text(json.dumps(dict(
        assessment=str(assessment.resolve()),assessment_sha256=digest(assessment),
        note="Generated intermediate PNGs compacted after complete assessment. Native tapes, replay plans, metrics, references and selected review images remain; omitted PNGs are reproducible.",
        captures=records),indent=2)+"\n")
    for record in records:
        if not record["retained"]:
            (run / record["name"]).unlink()
    print("compacted bytes",sum(r["bytes"] for r in records if not r["retained"]),flush=True)


def run_plan(path, prefix, compact_after):
    plan = json.loads(path.read_text())
    tape = json.loads((Path(plan["tape"]) / "near-history.json").read_text())
    fixture = plan["fixture"]
    env = {k:v for k,v in os.environ.items() if not k.startswith("VOXEL_GI_")}
    env["RUST_LOG"] = "info"
    env.update(VOXEL_GI_VALIDATE="1", VOXEL_GI_VALIDATE_BACKGROUND="1",
               VOXEL_GI_VALIDATE_HEADLESS="1", VOXEL_GI_VALIDATE_EXIT="1",
               VOXEL_GI_NEAR_FIXTURE=fixture, VOXEL_GI_VALIDATE_NEAR_HISTORY=str(path.resolve()),
               VOXEL_GI_NEAR_EMITTER_SCALE=str(tape.get("emitter_scale",1)))
    env["VOXEL_GI_VALIDATE_PROFILE"] = ("hallway" if fixture.startswith("hallway") else
                                        "house" if fixture.startswith("house") else "cave")
    if fixture.startswith("hallway"):
        views = tape["views"]
        station = next((key for key,view in views.items() if view == plan["view"]),None)
        if station is None:
            raise ValueError("no matching hallway station")
        env["VOXEL_GI_HALLWAY_OFFSET"] = station.removeprefix("d")
    log = Path(str(prefix)+".log")
    assessment = Path(str(prefix)+"-assessment.json")
    if log.exists() or assessment.exists():
        raise ValueError("refuse to overwrite existing run evidence")
    before = subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    provenance = dict(commit=before,dirty_diff=subprocess.check_output(["git","diff"],text=True),
                      plan_sha256=digest(path),environment={k:v for k,v in env.items() if k.startswith("VOXEL_GI_")})
    print("render",path,flush=True)
    with log.open("w") as output:
        subprocess.run(["cargo","run","--locked","--offline"],env=env,stdout=output,
                       stderr=subprocess.STDOUT,check=True,timeout=1800)
    text = log.read_text()
    if re.search(r"\b(?:WARN|ERROR)\b|panicked|Validation Error",text):
        raise ValueError("native replay has warnings/errors; inspect log")
    matched = re.search(r"GI validation initialized .*output=(\S+)",text)
    if matched is None:
        raise ValueError("missing run identity")
    run = Path(matched[1])
    provenance.update(run=str(run.resolve()),executable_sha256=digest(Path("target/debug/prototype_voxel_template")))
    Path(str(prefix)+"-provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print("assess",run,flush=True)
    with Path(str(prefix)+"-assessment.log").open("w") as output:
        subprocess.run([sys.executable,"tools/near_history_calibration.py","assess",str(path),
                        str(run),"--output",str(assessment)],stdout=output,
                       stderr=subprocess.STDOUT,check=True,timeout=1800)
    if compact_after:
        compact(plan,run,assessment)
    data = json.loads(assessment.read_text())
    for row in data["summary"]:
        if row["case"].startswith("warm-"):
            continue
        print(row["case"],row["policy"],"window",row["window"],
              "sustained",row["sustained"],"peak",round(row["peak_flip"],5),flush=True)
    print("complete",assessment,flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan",type=Path)
    p.add_argument("--prefix",type=Path,required=True)
    p.add_argument("--compact",action="store_true")
    args = p.parse_args()
    run_plan(args.plan,args.prefix,args.compact)
