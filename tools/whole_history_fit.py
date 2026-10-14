#!/usr/bin/env python3
"""Shortlist whole-range history curves from existing native path tapes.

This inexpensive screen measures compressed display-RGB error, NOT FLIP. It
cannot certify a visual gate or gameplay response. Native material replays
remain the final decision. No reference values enter a candidate's cap: every
candidate sees only distance. Fit window zero; report window one separately.
"""
import argparse
import json
from pathlib import Path
import unittest

import numpy as np

from near_history_calibration import (digest, display_step, edit_distances,
    limits, load_tape, relative_error, whole_limits)


def proxy(rgb):
    return rgb / (.25 + rgb)


def evaluate(tape, window, candidates, distances=None, baseline=None, max_sites=512):
    """Vectorized exact moment/display updates; only the error metric is a proxy."""
    distances = edit_distances(tape) if distances is None else distances
    baseline = limits(distances,48,16) if baseline is None else baseline
    order = np.argsort(distances, kind="stable")
    selected = order[np.linspace(0,len(order)-1,min(len(order),max_sites),dtype=int)]
    distance = distances[selected]
    caps = np.stack([baseline[selected]] + [whole_limits(distance,*c) for c in candidates])
    k,n = caps.shape
    bank = [s["data"][window,:,selected,:].transpose(1,0,2).astype(float)
            for s in tape["states"]]
    reference = [np.asarray(s["mean"])[selected,:3] for s in tape["states"]]
    results = []
    for case in ("remove","add","null"):
        old, new = (1,0) if case == "add" else (0,1)
        if case == "null": new = old
        mean = np.broadcast_to(bank[old][0,:,:3],(k,n,3)).copy()
        second = np.broadcast_to(bank[old][0,:,3:],(k,n,3)).copy()
        weight = np.full((k,n),float(tape["warm_paths"]))
        display = mean.copy()
        confidence = np.clip(weight.ravel()/(weight.ravel()+8) /
            (1+relative_error(mean.reshape(-1,3),second.reshape(-1,3),weight.ravel(),.02)),.125,1.)
        initial = np.sqrt(np.mean((proxy(display)-proxy(reference[new]))**2,axis=2))
        early, late = np.zeros(k),np.zeros(k)
        peak = np.zeros(k)
        for step in range(tape["steps"]):
            if step == 0: weight = np.minimum(weight,caps)
            weight = np.minimum(weight,65520) + 16
            alpha = 16/weight[:,:,None]
            incoming = bank[new][step+1]
            mean += (incoming[:,:3]-mean)*alpha
            second += (incoming[:,3:]-second)*alpha
            flat_display,confidence = display_step(display.reshape(-1,3),confidence,
                mean.reshape(-1,3),second.reshape(-1,3),weight.ravel(),.85)
            display = flat_display.reshape(k,n,3)
            error = np.sqrt(np.mean((proxy(display)-proxy(reference[new]))**2,axis=2))
            # Equal mean / p90 sensitivity; not a fitted perceptual pass gate.
            value = .5 * (error.mean(1) + np.quantile(error,.9,axis=1))
            peak = np.maximum(peak,value)
            if step < 32: early += value/32
            else: late += value/(tape["steps"]-32)
        # Every candidate shares the initial image; omit that constant when
        # comparing response integrals. Keep null re-noising separate in output.
        for i in range(k):
            results.append(dict(policy="baseline" if i == 0 else label(candidates[i-1]),
                case=case,window=window,sites=n,early=float(early[i]),late=float(late[i]),
                peak=float(peak[i]),final=float(value[i]),
                early_ratio=float(early[i]/max(early[0],1e-12)),
                late_ratio=float(late[i]/max(late[0],1e-12)),
                initial_mean=float(initial[i].mean())))
    return results


def label(coefficients):
    return "whole-s{}-p{}".format(*(format(v,"g").replace(".","d") for v in coefficients))


def fit(args):
    manifest = json.loads(args.manifest.read_text())
    candidates = [(s,p) for s in args.scales for p in args.exponents]
    rows,inputs = [],[]
    for item in manifest:
        path = Path(item["tape"])
        tape = load_tape(path)
        print("screen",item["name"],"sites",len(tape["sites"]),flush=True)
        for window in args.windows:
            for row in evaluate(tape,window,candidates,max_sites=args.max_sites):
                rows.append(dict(row,fixture=item["name"],family=item["family"]))
        inputs.append(dict(path=str(path),metadata_sha256=digest(path/"near-history.json"),
            moments_sha256={s["moments"]:digest(path/s["moments"]) for s in tape["states"]}))
    rankings = []
    for coefficients in candidates:
        policy = label(coefficients)
        group = [r for r in rows if r["policy"] == policy]
        families = {}
        for family in sorted({r["family"] for r in group}):
            arm = [r for r in group if r["family"] == family]
            families[family] = float(np.exp(np.mean([
                .5*np.log(max(r["early_ratio"],1e-12)) + .5*np.log(max(r["late_ratio"],1e-12))
                for r in arm])))
        rankings.append(dict(policy=policy,scale=coefficients[0],exponent=coefficients[1],
            proxy_score=float(np.exp(np.mean(np.log(list(families.values()))))),families=families,
            worst_early_ratio=max(r["early_ratio"] for r in group),
            worst_late_ratio=max(r["late_ratio"] for r in group)))
    rankings.sort(key=lambda r:r["proxy_score"])
    result = dict(inputs=inputs,manifest_sha256=digest(args.manifest),windows=args.windows,
        max_sites=args.max_sites,rows=rows,rankings=rankings,
        note="Screen only: compressed display-RGB mean/p90 integrals, equal early/late logs, equal cases then scene families. Not FLIP, not a perceptual pass. Deterministic distance-stratified site subset; no corner reconstruction or material rendering. Native image qualification is required before a decision.")
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for row in rankings[:16]: print(row,flush=True)


class Tests(unittest.TestCase):
    def test_vectorized_baseline_matches_scalar_replay(self):
        from near_history_calibration import replay
        rng = np.random.default_rng(21)
        states = []
        for _ in range(2):
            mean = rng.random((1,65,3,3))
            data = np.concatenate([mean,mean**2+.1],axis=3)
            states.append(dict(data=data,mean=np.ones((3,3))*.5))
        tape = dict(states=states,sites=[[[1,0,0],0],[[16,0,0],0],[[96,0,0],0]],
                    edit=[0,0,0],batch_paths=16,warm_paths=4096,steps=64)
        result = evaluate(tape,0,[(16,4)])
        for case in ("remove","add","null"):
            last = list(replay(tape,0,case,48,16,.85,0,[1024]))[-1]
            error = np.sqrt(np.mean((proxy(last[3])-proxy(np.full((3,3),.5)))**2,axis=1))
            row = next(r for r in result if r["policy"] == "baseline" and r["case"] == case)
            self.assertAlmostEqual(row["final"],.5*(error.mean()+np.quantile(error,.9)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("manifest",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--scales",type=float,nargs="+",default=[4,6,8,10,12,14,16,20,24,32,48,64])
    p.add_argument("--exponents",type=float,nargs="+",default=[1,1.5,2,2.5,3,3.5,4,5,6,8])
    p.add_argument("--windows",type=int,nargs="+",default=[0])
    p.add_argument("--max-sites",type=int,default=512)
    fit(p.parse_args())
