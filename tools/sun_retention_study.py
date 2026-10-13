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
from retention_budget import ChangePrior, QualityGoal, QualifiedFloor, ResponseBudget, SamplingCalibration, select_limit


class VarianceGuard:
    """Experimental first-innovation noise floor over a known geometry event.

    If the new batch is exactly equal to history, do not discard evidence yet.
    Otherwise compare the bias cap with history needed to bound the estimated
    standard deviation of this update. Sample moments are an IID proxy, not a
    certified QMC tail bound. Conflicting budgets are reported, not concealed.
    """
    def __init__(self,limits,update_noise=0.05):
        if not math.isfinite(update_noise) or update_noise <= 0:
            raise ValueError("positive finite update-noise target required")
        self.limits = np.asarray(limits,dtype=float)
        self.update_noise = update_noise
        self.pending = np.zeros(len(limits),dtype=bool)
        self.conflicts = 0

    def weights(self,mean,available,fresh,second,paths,event):
        if event:
            self.pending[:] = True
        decision = self.pending & np.any(fresh != mean,axis=1)
        variance = np.maximum(0,second-fresh*fresh)
        scale = np.maximum(0.02,np.maximum(np.abs(mean),np.abs(fresh)))
        noise_history = np.maximum(0,np.max(np.sqrt(variance*paths)/scale,axis=1)/self.update_noise-paths)
        self.conflicts += int(np.sum(decision & (noise_history > self.limits)))
        chosen = np.maximum(self.limits,noise_history)
        self.pending[decision] = False
        return np.where(decision,np.minimum(available,chosen),available)


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
        floor = np.asarray([QualifiedFloor().limit(d,calibration) for d in distances])
        policies["sun-retention"] = np.maximum(policies["sun-budget"],floor)
        runs = []
        for index in range(len(case["before"])):
            for reverse in (False,True):
                for repeated in ((False,True) if case["name"] == "cave_far_residency" else (False,)):
                    for name,limits in policies.items():
                        result = replay(case,index,reverse,repeated,name,tape["batch_paths"],tape["warm_paths"],limits)
                        runs.append(result)
                    guard = VarianceGuard(policies["sun-budget"])
                    runs.append(replay(case,index,reverse,repeated,"sun-budget-variance",
                        tape["batch_paths"],tape["warm_paths"],policies["sun-budget"],guard))
        reports.append({"name":case["name"],"bounds":case["changed_bounds"],
                        "sun_targets":evidence["sun_targets"],"sun_lowered_distance_sites":int(np.sum(effective<distances)),
                        "retention_bias_floor_conflicts":int(np.sum(floor>policies["sun-budget"])),
                        "limits":{k:summary(v) for k,v in policies.items()},"runs":runs})
    return {"schema":1,"calibration":asdict(calibration),"budget":asdict(budget),"prior":asdict(prior),
            "qualified_floor":asdict(QualifiedFloor()),
            "cases":reports,"note":"Fixed-work offline combined-policy experiment. Calibration from separate composed-cave quality sweep; exact native transition samples reused. No unknown/journal fallback, production event pipeline, display or FPS qualification."}


def sensitivity(calibration,budget,prior):
    """Counterfactual calibration inputs only; no fictional new path samples."""
    return [{"distance":distance,"caps":[min(4096,select_limit(distance,None,
                replace(calibration,paths_for_quality=calibration.paths_for_quality*factor),budget,prior))
                for factor in (1,0.5,0.25)]}
            for distance in (0,48,96,128,192,256,384,512)]


def legacy_controls(tape,calibration,budget,prior):
    """Reuse existing local emitter/skylight tapes without tracing them again.

    A verified baseline cap16 proves distance <=48, where this prior is flat.
    For remote events require ALL actual edit targets in the saved diagnostics;
    never fill missing geometry with a guessed distance. No-edit controls stay
    no-edit, and sun changes can only reduce distance below the local flat band.
    """
    if prior.full_change_distance < 48:
        raise ValueError("legacy local caps only certify distances within 48 blocks")
    from history_qualification import evaluate
    class Policy:
        def __init__(self,name,limits,has_edit,guarded):
            self.name,self.limits,self.has_edit = name,limits,has_edit
            self.guard = VarianceGuard(limits) if guarded else None
            self.pending = False

        def event(self,histories,sites,audit):
            self.pending = self.has_edit

        def weights(self,histories,fresh,paths):
            available = np.asarray([h.weight for h in histories])
            if self.guard:
                result = self.guard.weights(np.asarray([h.mean for h in histories]),available,
                    np.asarray([f["mean"] for f in fresh]),np.asarray([f["second"] for f in fresh]),
                    paths,self.pending)
            else:
                result = np.minimum(available,self.limits) if self.pending else available
            self.pending = False
            return result

    results = []
    for case in tape["cases"]:
        metadata = case["guided"]
        if len(case["sites"]) != len(metadata["baseline_caps"]):
            raise ValueError("missing legacy receiver caps")
        targets = {tuple(math.floor(v) for v in pair["target"])
                   for q in metadata["receivers"] for pair in q["geometry1"]["pairs"]}
        limits,retention = [],[]
        for site,baseline in zip(case["sites"],metadata["baseline_caps"]):
            if not metadata["has_edit"]:
                limits.append(65536)
                retention.append(65536)
                continue
            elif baseline == 16:
                d = 0
            elif len(targets) == metadata["edit_targets"] and targets and metadata["sun_targets"] == 0:
                d = min(math.dist(site["voxel"],p) for p in targets)
            else:
                raise ValueError("legacy tape lacks complete remote geometry/sun evidence")
            limits.append(select_limit(d,None,calibration,budget,prior))
            retention.append(max(limits[-1],QualifiedFloor().limit(d,calibration)))
        for run in case["runs"]:
            for name,values,guarded in (("baseline",metadata["baseline_caps"],False),
                                        ("sun-budget",limits,False),("sun-budget-variance",limits,True),
                                        ("sun-retention",retention,False)):
                policy = Policy(name,values,metadata["has_edit"],guarded)
                result = evaluate(case,run,policy,tape["batch_paths"],tape["warm_paths"],tape["steps"])
                result["bias_noise_conflicts"] = policy.guard.conflicts if policy.guard else 0
                results.append(result)
    return results


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

    def test_variance_guard_preserves_absent_evidence_and_reports_conflict(self):
        guard = VarianceGuard([16])
        old,zero = np.ones((1,5)),np.zeros((1,5))
        self.assertEqual(guard.weights(zero,np.array([4096]),zero,zero,16,True)[0],4096)
        self.assertTrue(guard.pending[0])
        limit = guard.weights(zero,np.array([4096]),old,old*4,16,False)[0]
        self.assertGreater(limit,16)
        self.assertEqual(guard.conflicts,1)
        self.assertFalse(guard.pending[0])
        self.assertEqual(guard.weights(old,np.array([4096]),zero,zero,16,True)[0],16)

    def test_retention_floor_recalibrates_without_changing_spatial_evidence(self):
        quality = SamplingCalibration(3072,"baseline")
        twice = replace(quality,paths_for_quality=1536,sampler_id="hypothetical")
        floor = QualifiedFloor()
        for distance,sun in ((0,None),(48,None),(96,None),(128,None),(512,10),(512,None)):
            old = max(select_limit(distance,sun,quality),floor.limit(distance,quality))
            new = max(select_limit(distance,sun,twice),floor.limit(distance,twice))
            self.assertGreaterEqual(old,16 if distance<=48 else 256)
            self.assertEqual(new,old/2 if old<65536 else old)
        self.assertEqual(max(select_limit(512,10,quality),floor.limit(512,quality)),256)

    def test_legacy_adapter_rejects_uncertified_distances_and_missing_caps(self):
        quality = SamplingCalibration(3072,"test")
        with self.assertRaises(ValueError):
            legacy_controls({},quality,ResponseBudget(),ChangePrior(24))
        with self.assertRaises(ValueError):
            legacy_controls({"cases":[{"sites":[{}],"guided":{"baseline_caps":[]}}]},
                            quality,ResponseBudget(),ChangePrior())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tape",type=Path)
    p.add_argument("metadata",type=Path)
    p.add_argument("--quality-report",type=Path,required=True)
    p.add_argument("--sampler-id",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--legacy-tape",type=Path)
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
    if args.legacy_tape:
        legacy,legacy_hash = load(args.legacy_tape)
        report["legacy_controls"] = legacy_controls(legacy,calibration,budget,prior)
        report["inputs"]["legacy_tape_sha256"] = legacy_hash
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    for c in report["cases"]:
        print(c["name"],c["bounds"],"sun targets",c["sun_targets"])
        for name in (*c["limits"],"sun-budget-variance"):
            for repeated in (False,True):
                runs = [r for r in c["runs"] if r["policy"] == name and r["repeated"] == repeated]
                if runs:
                    print(name,"repeated",repeated,"late p99",max(r["late_rgb"]["p99"] for r in runs),
                          "variation",max(r["late_variation"]["p99"] for r in runs),
                          "settled",[r["all_settled_paths_5pct"] for r in runs])


if __name__ == "__main__":
    main()
