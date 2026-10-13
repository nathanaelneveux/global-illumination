#!/usr/bin/env python3
"""Join existing native samples to sunlight-change evidence and quality calibration.

The held-out composed cave sample sweep supplies a sampler work scale. Policy
selection receives distances and that calibration, never transition references.
No synthetic accuracy improvement is reported as an actual faster sampler.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import unittest

import numpy as np

from distance_history_analysis import caps, replay, summary
from retention_budget import ChangePrior, QualityGoal, ResponseBudget, SamplingCalibration, select_limit


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def joined_cases(tape, metadata):
    if tape["schema"] != 1 or metadata["schema"] != 1 or tape["transport"] != metadata["transport"]:
        raise ValueError("incompatible evidence schemas/models")
    if len(tape["cases"]) != len(metadata["cases"]):
        raise ValueError("missing change evidence")
    for case,evidence in zip(tape["cases"],metadata["cases"]):
        if case["name"] != evidence["name"] or case["changed_bounds"] != evidence["changed_bounds"]:
            raise ValueError("fixture mismatch")
        if len(case["sites"]) != len(evidence["sites"]):
            raise ValueError("missing receiver evidence")
        for a,b in zip(case["sites"],evidence["sites"]):
            if a["voxel"] != b["voxel"] or a["face"] != b["face"] or not math.isclose(a["distance"],b["geometry_distance"],abs_tol=1e-9):
                raise ValueError("receiver mismatch")
        yield case,evidence


def analyze(tape,metadata,calibration,budget,prior):
    reports = []
    for case,evidence in joined_cases(tape,metadata):
        distances = np.asarray([s["geometry_distance"] for s in evidence["sites"]])
        effective = np.asarray([min(s["geometry_distance"],s["sun_distance"])
                                if s["sun_distance"] is not None else s["geometry_distance"]
                                for s in evidence["sites"]])
        policies = {
            "baseline":caps("baseline",distances),
            "geometry-quartic":caps("quartic",distances),
            "sun-quartic":caps("quartic",effective),
            "geometry-budget":[select_limit(s["geometry_distance"],None,calibration,budget,prior)
                               for s in evidence["sites"]],
            "sun-budget":[select_limit(s["geometry_distance"],s["sun_distance"],calibration,budget,prior)
                          for s in evidence["sites"]],
        }
        runs = []
        for index in range(len(case["before"])):
            for reverse in (False,True):
                for repeated in ((False,True) if case["name"] == "cave_far_residency" else (False,)):
                    for name,limits in policies.items():
                        result = replay(case,index,reverse,repeated,name,tape["batch_paths"],tape["warm_paths"],limits)
                        runs.append(result)
        reports.append({"name":case["name"],"bounds":case["changed_bounds"],
                        "sun_targets":evidence["sun_targets"],"sun_lowered_distance_sites":int(np.sum(effective<distances)),
                        "limits":{k:summary(v) for k,v in policies.items()},"runs":runs})
    return {"schema":1,"calibration":asdict(calibration),"budget":asdict(budget),"prior":asdict(prior),
            "cases":reports,"note":"Fixed-work offline combined-policy experiment. Calibration from separate composed-cave quality sweep; exact native transition samples reused. No unknown/journal fallback, production event pipeline, display or FPS qualification."}


def sensitivity(calibration,budget,prior):
    """Counterfactual calibration inputs only; no fictional new path samples."""
    return [{"distance":distance,"caps":[min(4096,select_limit(distance,None,
                replace(calibration,paths_for_quality=calibration.paths_for_quality*factor),budget,prior))
                for factor in (1,0.5,0.25)]}
            for distance in (0,48,96,128,192,256,384,512)]


class Tests(unittest.TestCase):
    @staticmethod
    def sweep():
        return {"measurements":[{"window":w,"paths":n,"centers":{"p99":error/2,"max":error}}
                    for w in range(4) for n,error in ((512,0.2),(1024,0.05),(2048,0.11),(4096,0.05))]}

    def test_calibration_requires_persistent_complete_evidence(self):
        report = self.sweep()
        self.assertEqual(SamplingCalibration.from_sweep(report,"sampler").paths_for_quality,4096)
        for rows in (report["measurements"][:-1],report["measurements"]+[report["measurements"][0]]):
            with self.assertRaises(ValueError):
                SamplingCalibration.from_sweep({"measurements":rows},"sampler")
        with self.assertRaises(ValueError):
            SamplingCalibration.from_sweep(report,"sampler",QualityGoal(.001,.002))

    def test_recalibration_rescales_weights_without_distance_table(self):
        report = self.sweep()
        old = SamplingCalibration.from_sweep(report,"old")
        for row in report["measurements"]:
            row["paths"] /= 2
        new = SamplingCalibration.from_sweep(report,"new")
        self.assertEqual(new.paths_for_quality,old.paths_for_quality/2)
        for distance in (0,48,96,192,256):
            self.assertAlmostEqual(select_limit(distance,None,new),select_limit(distance,None,old)/2)
        self.assertEqual(select_limit(512,None,new),65536)

    def test_local_shadow_overrides_far_geometry_and_is_reversal_invariant(self):
        calibration = SamplingCalibration(3072,"test")
        self.assertEqual(select_limit(512,None,calibration),65536)
        self.assertEqual(select_limit(512,10,calibration),select_limit(10,None,calibration))
        self.assertLess(select_limit(512,10,calibration),16)
        with self.assertRaises(ValueError):
            select_limit(512,float("nan"),calibration)

    def test_bias_cap_and_existing_history_never_increase(self):
        quality = SamplingCalibration(3072,"test")
        budget = ResponseBudget()
        for distance in (0,48,96,128,192,256,384):
            delta = ChangePrior().magnitude(distance)
            for available in (0,1,16,256,4096,65536):
                h = budget.retained(available,delta,quality)
                self.assertLessEqual(h,available)
                self.assertLessEqual(h/(h+512)*delta,budget.stale_error+1e-12)

    def test_default_replay_stays_bit_identical_with_explicit_baseline(self):
        moment = {"mean":[1.0]*5,"second":[2.0]*5}
        run = {"seed":1,"warm":[moment],"batches":[[moment]]*32}
        case = {"sites":[{"distance":512}],"before":[run],"after":[run],
                "references_before":[[[1]*5]]*2,"references_after":[[[1]*5]]*2}
        a = replay(case,0,False,False,"baseline",16,4096)
        b = replay(case,0,False,False,"baseline",16,4096,[256])
        self.assertEqual(a,b)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tape",type=Path)
    p.add_argument("metadata",type=Path)
    p.add_argument("--quality-report",type=Path,required=True)
    p.add_argument("--sampler-id",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stale-error",type=float,default=0.03)
    p.add_argument("--response-fraction",type=float,default=1/6)
    p.add_argument("--change-distance",type=float,default=48)
    p.add_argument("--change-power",type=float,default=2)
    args = p.parse_args()
    quality,quality_hash = load(args.quality_report)
    calibration = SamplingCalibration.from_sweep(quality,args.sampler_id)
    tape,tape_hash = load(args.tape)
    metadata,metadata_hash = load(args.metadata)
    budget = ResponseBudget(args.stale_error,args.response_fraction)
    prior = ChangePrior(args.change_distance,args.change_power)
    report = analyze(tape,metadata,calibration,budget,prior)
    report["inputs"] = dict(tape_sha256=tape_hash,metadata_sha256=metadata_hash,quality_sha256=quality_hash)
    report["calibration_sensitivity_only"] = sensitivity(calibration,budget,prior)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for c in report["cases"]:
        print(c["name"],c["bounds"],"sun targets",c["sun_targets"])
        for name in c["limits"]:
            for repeated in (False,True):
                runs = [r for r in c["runs"] if r["policy"] == name and r["repeated"] == repeated]
                if runs:
                    print(name,"repeated",repeated,"late p99",max(r["late_rgb"]["p99"] for r in runs),
                          "variation",max(r["late_variation"]["p99"] for r in runs),
                          "settled",[r["all_settled_paths_5pct"] for r in runs])


if __name__ == "__main__":
    main()
