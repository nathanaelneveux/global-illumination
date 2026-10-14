#!/usr/bin/env python3
"""Decision summaries for complete native-material history-policy replays.

Keep censored failures, unresolved warm starts, and scene families explicit.
Do not convert an offline publication clock into gameplay time or hide a
regression inside an average of many nearby hallway camera stations.
"""
import argparse
import json
from pathlib import Path
import unittest

import numpy as np

from cave_sample_analysis import load_rgb
from near_history_calibration import brightness_statistics, digest


def integral(rows, field, end=512):
    selected = sorted((r for r in rows if r["fresh_paths"] <= end),key=lambda r:r["fresh_paths"])
    if not selected or selected[0]["fresh_paths"] != 0 or selected[-1]["fresh_paths"] != end:
        raise ValueError("early integral needs observations at both endpoints")
    return float(np.trapezoid([field(r) for r in selected],
                             [r["fresh_paths"] for r in selected]) / end)


def arm_rows(assessment):
    data = json.loads(assessment.read_text())
    if not data.get("complete"):
        raise ValueError("incomplete assessment")
    plan = data["plan"]
    fixture = plan["fixture"]
    family = fixture.split("-")[0]
    view = json.dumps(plan.get("view"),sort_keys=True)
    output = []
    for arm in data["summary"]:
        if arm["case"] not in ("remove","add","null"):
            continue
        policy = arm["policy"]
        rows = [r for r in data["measurements"] if r["case"] == arm["case"]
                and r.get("policy") == policy and r["window"] == arm["window"]]
        tail = arm["sustained"]
        output.append(dict(family=family,fixture=fixture,view=view,policy=policy,
            case=arm["case"],window=arm["window"],
            tape_sha256=plan["tape_sha256"],assessment=str(assessment),
            settled_start=arm["settled_start"],
            sustained_paths=tail["paths"] if tail else None,
            last_observed_paths=max(r["fresh_paths"] for r in rows),
            final_pass=max(rows,key=lambda r:r["fresh_paths"])["accepted"],
            early_flip=integral(rows,lambda r:r["flip"]["mean"]),
            early_tile=integral(rows,lambda r:r["flip_32px_tile_means"]["max"]),
            peak_flip=arm["peak_flip"],peak_tile=arm["peak_tile"],
            upper_endpoint_tile=max(r["luminance"]["upper_endpoint_tile_max"] for r in rows if r["fresh_paths"]),
            lower_endpoint_tile=max(r["luminance"]["lower_endpoint_tile_max"] for r in rows if r["fresh_paths"])))
    return output


def key(row):
    return (row["fixture"],row["view"],row["tape_sha256"],row["case"],row["window"])


def paired_summary(rows):
    baseline = {key(r):r for r in rows if r["policy"] == "baseline"}
    required = {k for k,r in baseline.items() if r["case"] != "null" and r["settled_start"]}
    output = []
    for policy in sorted({r["policy"] for r in rows}):
        available = {key(r) for r in rows if r["policy"] == policy and r["settled_start"]}
        missing = required - available
        paired = []
        for r in rows:
            if r["policy"] != policy or key(r) not in baseline or r["case"] == "null":
                continue
            b = baseline[key(r)]
            if not r["settled_start"] or not b["settled_start"]:
                continue
            rt, bt = r["sustained_paths"],b["sustained_paths"]
            paired.append(dict(family=r["family"],fixture=r["fixture"],view=r["view"],
                case=r["case"],window=r["window"],candidate=rt,baseline=bt,
                work_ratio=(max(rt,16)/max(bt,16) if rt is not None and bt is not None else None),
                early_error_ratio=r["early_flip"]/max(b["early_flip"],1e-15),
                extra_upper_endpoint_tile=r["upper_endpoint_tile"]-b["upper_endpoint_tile"]))
        families = []
        for family in sorted({p["family"] for p in paired}):
            group = [p for p in paired if p["family"] == family]
            ratios = [p["work_ratio"] for p in group if p["work_ratio"] is not None]
            families.append(dict(family=family,cases=len(group),
                unresolved_fraction=sum(p["candidate"] is None for p in group)/len(group),
                baseline_unresolved_fraction=sum(p["baseline"] is None for p in group)/len(group),
                geometric_work_ratio=float(np.exp(np.log(ratios).mean())) if ratios else None,
                largest_observed_slowdown=max(ratios) if ratios else None,
                resolved_baseline_failures=sum(p["baseline"] is None and p["candidate"] is not None for p in group),
                newly_unresolved=sum(p["baseline"] is not None and p["candidate"] is None for p in group)))
        valid = [f["geometric_work_ratio"] for f in families if f["geometric_work_ratio"] is not None]
        output.append(dict(policy=policy,families=families,pairs=paired,
            comparison_complete=not missing,missing_baseline_cases=len(missing),
            equal_family_unresolved_fraction=float(np.mean([f["unresolved_fraction"] for f in families])) if families and not missing else None,
            equal_family_geometric_work_ratio=float(np.exp(np.log(valid).mean())) if valid and not missing else None))
    return output


def compare(args):
    rows = [r for file in args.assessments for r in arm_rows(file)]
    identities = [(key(r),r["policy"]) for r in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate case/window/policy would bias the decision")
    output = dict(assessments={str(f):digest(f) for f in args.assessments},
                  rows=rows,policies=paired_summary(rows),
                  note="Equal scene-family aggregation. Work ratios use only jointly settled, qualified starts, with zero-work cases floored at one 16-path batch; unresolved cases remain separate, not invented settling times. Null-change costs are reported in rows but do not veto cold-start-equivalent re-noising.")
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    for policy in output["policies"]:
        print(policy["policy"],policy["equal_family_unresolved_fraction"],policy["equal_family_geometric_work_ratio"])
        for family in policy["families"]:
            print(family)


def static_sweep(args):
    source = json.loads(args.analysis.read_text())
    target = load_rgb(args.run / "cave-reference.png")
    records = []
    for row in source["perception"]["images"]:
        image = load_rgb(args.run / row["image"])
        records.append(dict(paths=row["paths"],window=row["window"],
            flip=row["flip"]["mean"],tile=row["flip_32px_tile_means"]["max"],
            luminance=brightness_statistics(image,target,target)))
    output = dict(source=str(args.analysis),source_sha256=digest(args.analysis),
                  note="Original native static cave prefixes, no display history, same frozen reference. FLIP reused from the qualified completed assessment; endpoint luminance diagnostics newly measured from retained full-resolution images.",
                  records=records)
    args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    for count in sorted({r["paths"] for r in records}):
        rows = [r for r in records if r["paths"] == count]
        print(count,"FLIP",max(r["flip"] for r in rows),"tile",max(r["tile"] for r in rows),
              "brightness range",min(r["luminance"]["mean_ratio"] for r in rows),
              max(r["luminance"]["mean_ratio"] for r in rows),
              "positive tile",max(r["luminance"]["upper_endpoint_tile_max"] for r in rows),flush=True)


class Tests(unittest.TestCase):
    def test_integral_is_normalized_and_requires_both_endpoints(self):
        rows = [dict(fresh_paths=n,error=n/512) for n in (0,16,128,512)]
        self.assertEqual(integral(rows,lambda r:r["error"]),.5)
        with self.assertRaises(ValueError):
            integral(rows[:-1],lambda r:r["error"])

    def test_unresolved_cases_are_not_assigned_fake_work(self):
        base = dict(fixture="hallway-emitter",view="one",tape_sha256="same",case="remove",
                    window=0,family="hallway",settled_start=True,early_flip=.1,upper_endpoint_tile=.01)
        rows = [dict(base,policy="baseline",sustained_paths=None),
                dict(base,policy="power4",sustained_paths=128)]
        summary = paired_summary(rows)[1]
        self.assertEqual(summary["equal_family_unresolved_fraction"],0)
        self.assertIsNone(summary["equal_family_geometric_work_ratio"])
        self.assertEqual(summary["families"][0]["resolved_baseline_failures"],1)

    def test_missing_hard_cases_cannot_improve_the_aggregate(self):
        base = dict(fixture="hallway-emitter",view="one",tape_sha256="same",case="remove",
                    window=0,family="hallway",settled_start=True,early_flip=.1,upper_endpoint_tile=.01)
        rows = [dict(base,policy="baseline",sustained_paths=1024),
                dict(base,policy="baseline",sustained_paths=1024,window=1),
                dict(base,policy="power4",sustained_paths=128)]
        summary = paired_summary(rows)[1]
        self.assertFalse(summary["comparison_complete"])
        self.assertEqual(summary["missing_baseline_cases"],1)
        self.assertIsNone(summary["equal_family_geometric_work_ratio"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("compare")
    p.add_argument("assessments",type=Path,nargs="+")
    p.add_argument("--output",type=Path,required=True)
    p = sub.add_parser("static-sweep")
    p.add_argument("run",type=Path)
    p.add_argument("analysis",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    (compare if args.command == "compare" else static_sweep)(args)
