#!/usr/bin/env python3
"""Apply a human-calibrated static quality profile to a qualified sweep report.

This consumes the existing analysis, never redefines its older engineering
gates or treats raw RGB percentages as perceptual distances. The caller must
retain the reference/capture qualification of the generating native study.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import unittest


def complete_rows(rows):
    keys = [(r["window"],r["paths"]) for r in rows]
    counts = sorted({n for _,n in keys})
    if (not counts or any(not isinstance(n,int) or n<=0 for n in counts)
            or len(keys)!=len(set(keys))
            or set(keys)!={(w,n) for w in range(4) for n in counts}):
        raise ValueError("require complete, unique four-window measurements")
    return {(r["window"],r["paths"]):r for r in rows}


def within(row,limits):
    passed = True
    for section,fields in limits.items():
        for key,limit in fields.items():
            value = row[section][key]
            if not all(math.isfinite(v) and v>=0 for v in (value,limit)):
                raise ValueError("invalid quality measurement or limit")
            passed &= value<=limit
    return bool(passed)


def assess(report,profile):
    if profile["schema"]!=1 or report["perception"]["view"]!=profile["view"]:
        raise ValueError("quality profile schema or viewing conditions differ")
    transport = complete_rows(report["measurements"])
    images = complete_rows(report["perception"]["images"])
    if transport.keys()!=images.keys():
        raise ValueError("transport/capture checkpoints differ")
    if set(profile["limits"])!={"centers","corners","flip","flip_32px_tile_means"}:
        raise ValueError("missing or unsupported quality gate")
    rows = []
    for key,row in transport.items():
        raw = within(row,{s:profile["limits"][s] for s in ("centers","corners")})
        image = within(images[key],{s:profile["limits"][s] for s in ("flip","flip_32px_tile_means")})
        rows.append(dict(window=key[0],paths=key[1],transport_pass=raw,image_pass=image,
                         accepted=raw and image))
    counts = sorted({r["paths"] for r in rows})
    minimum = next((n for n in counts if all(r["accepted"] for r in rows if r["paths"]>=n)),None)
    return dict(profile_id=profile["id"],minimum_sustained_paths=minimum,measurements=rows,
                note="Fixed human-calibrated static target, all four windows and larger prefixes. Not temporal stability, interactive FPS, or IR/ionizing acceptance.")


def verify_anchor(report,profile,report_hash):
    """Do not silently retune the accepted target from a future candidate."""
    if report_hash!=profile["baseline"]["analysis_sha256"]:
        raise ValueError("baseline report hash differs")
    n = profile["human_review"]["accepted_paths"]
    for section,fields in profile["limits"].items():
        bank = report["measurements"] if section in ("centers","corners") else report["perception"]["images"]
        rows = [r for r in bank if r["paths"]==n]
        if len(rows)!=4:
            raise ValueError("missing accepted baseline windows")
        for key,limit in fields.items():
            if max(r[section][key] for r in rows)!=limit:
                raise ValueError("quality limit differs from the accepted baseline")
    result = assess(report,profile)
    if result["minimum_sustained_paths"]!=profile["baseline"]["paths_for_quality"]:
        raise ValueError("baseline calibration no longer reproduces")
    for key in ("noticeable_paths","very_noticeable_paths"):
        rows = [r for r in result["measurements"] if r["paths"]==profile["human_review"][key]]
        if len(rows)!=4 or any(r["accepted"] for r in rows):
            raise ValueError("profile does not reject the visible-failure controls")
    return result


class Tests(unittest.TestCase):
    @staticmethod
    def fixture():
        limits = {"centers":{"p99":.05,"max":.1},"corners":{"p99":.025,"max":.1},
                  "flip":{"mean":.03},"flip_32px_tile_means":{"max":.08}}
        profile = dict(schema=1,id="test",view={},limits=limits)
        rows = [dict(window=w,paths=n,**{s:{k:v*factor for k,v in ks.items()} for s,ks in limits.items()})
                for w in range(4) for n,factor in ((1024,2),(2048,1),(4096,.5))]
        return dict(measurements=rows,perception=dict(view={},images=rows.copy())),profile

    def test_fixed_gate_rejects_lower_samples_and_can_reward_better_sampling(self):
        report,profile = self.fixture()
        self.assertEqual(assess(report,profile)["minimum_sustained_paths"],2048)
        for row in report["measurements"]:
            for s,ks in profile["limits"].items():
                for key in ks:
                    row[s][key]*=.5
        self.assertEqual(assess(report,profile)["minimum_sustained_paths"],1024)

    def test_missing_duplicate_nonfinite_and_mismatched_capture_rejected(self):
        for damage in ("missing","duplicate","nonfinite","view"):
            report,profile = self.fixture()
            if damage=="missing": report["perception"]["images"].pop()
            if damage=="duplicate": report["measurements"].append(report["measurements"][0])
            if damage=="nonfinite": report["measurements"][0]["centers"]["p99"]=float("nan")
            if damage=="view": report["perception"]["view"]={"width":1}
            with self.assertRaises(ValueError): assess(report,profile)

    def test_one_bad_later_window_prevents_early_sustained_pass(self):
        report,profile = self.fixture()
        report["measurements"][-1]["flip"]["mean"]=.04
        self.assertIsNone(assess(report,profile)["minimum_sustained_paths"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report",type=Path)
    p.add_argument("--profile",type=Path,required=True)
    p.add_argument("--verify-anchor",action="store_true")
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    raw,profile_raw = args.report.read_bytes(),args.profile.read_bytes()
    report,profile = json.loads(raw),json.loads(profile_raw)
    digest = hashlib.sha256(raw).hexdigest()
    result = verify_anchor(report,profile,digest) if args.verify_anchor else assess(report,profile)
    result["report_sha256"],result["profile_sha256"] = digest,hashlib.sha256(profile_raw).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("minimum sustained paths:",result["minimum_sustained_paths"])


if __name__=="__main__":
    main()
