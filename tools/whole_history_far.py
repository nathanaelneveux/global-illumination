#!/usr/bin/env python3
"""Qualify free whole-range curves on existing distant-change transport tapes.

These are raw five-band transport controls, not FLIP or gameplay timing. Known
changed-sunlight distances join geometry distances; no future/reference energy
is used to choose a cap. Reuses the original exact native replay implementation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from distance_history_analysis import moments, replay, summary
from near_history_calibration import digest, limits, whole_limits
from sun_retention_study import joined_cases
from whole_history_fit import label


def screen(args):
    """Reject proxy fits with new raw-tail failures before expensive rendering.

    Mean updates are vectorized, not approximated. The existing 5% raw transport
    control is not substituted for FLIP; survivors still need material replay.
    """
    tape = json.loads(args.tape.read_text())
    metadata = json.loads(args.metadata.read_text())
    fitted = json.loads(args.fit.read_text())
    candidates = [(r["scale"],r["exponent"]) for r in fitted["rankings"]]
    policies = ["baseline"] + [label(c) for c in candidates]
    rows = []
    for case,evidence in joined_cases(tape,metadata):
        geometry = np.asarray([s["geometry_distance"] for s in evidence["sites"]])
        sun = np.asarray([s["sun_distance"] if s["sun_distance"] is not None else 1e9
                          for s in evidence["sites"]])
        effective = np.minimum(geometry,sun)
        caps = np.stack([np.minimum(limits(geometry,48,16),limits(sun,0,256))] +
                        [whole_limits(effective,*c) for c in candidates])
        refs = [np.asarray(case["references_"+s]).mean(0)[:,:3] for s in ("before","after")]
        scale = np.maximum(.02,np.linalg.norm(np.maximum(*refs),axis=1))
        for index in args.windows:
            bank = [moments(case[s][index],"batches")[:,:,:3] for s in ("before","after")]
            for reverse in (False,True):
                for repeated in ((False,True) if case["name"] == "cave_far_residency" else (False,)):
                    side = int(reverse)
                    warm = np.asarray([m["mean"][:3] for m in case[("before","after")[side]][index]["warm"]])
                    mean = np.broadcast_to(warm,(len(policies),len(warm),3)).copy()
                    weight = np.full(caps.shape,float(tape["warm_paths"]))
                    last_bad = np.full(len(policies),-1)
                    for step in range(tape["steps"]):
                        if step == 0 or (repeated and step%8 == 0):
                            side=1-side
                            weight=np.minimum(weight,caps)
                        weight=np.minimum(weight,65536-tape["batch_paths"])+tape["batch_paths"]
                        mean+=(bank[side][step]-mean)*(tape["batch_paths"]/weight[:,:,None])
                        error=np.linalg.norm(mean-refs[side],axis=2)/scale
                        last_bad=np.where(np.max(error,axis=1)>.05,step,last_bad)
                    for i,policy in enumerate(policies):
                        rows.append(dict(policy=policy,case=case["name"],bounds=case["changed_bounds"],
                            window=index,reverse=reverse,repeated=repeated,
                            settled=bool(last_bad[i]<tape["steps"]-16),
                            final_p99=float(np.quantile(error[i],.99)),
                            baseline_settled=bool(last_bad[0]<tape["steps"]-16)))
        print("screened",case["name"],case["changed_bounds"],flush=True)
    qualified=[]
    for rank in fitted["rankings"]:
        group=[r for r in rows if r["policy"]==rank["policy"]]
        failures=sum(r["baseline_settled"] and not r["settled"] for r in group)
        qualified.append(dict(rank,new_raw_tail_failures=failures,
            worst_final_raw_p99=max(r["final_p99"] for r in group)))
    args.output.write_text(json.dumps(dict(rankings=qualified,rows=rows,
        fit_sha256=digest(args.fit),tape_sha256=digest(args.tape),metadata_sha256=digest(args.metadata),
        note="Vectorized raw mean transport screen. Native material FLIP still required; no perceptual pass is claimed."),indent=2)+"\n")
    for r in [r for r in qualified if r["new_raw_tail_failures"]==0][:20]: print(r,flush=True)


def run(args):
    tape = json.loads(args.tape.read_text())
    metadata = json.loads(args.metadata.read_text())
    curves = [tuple(map(float,c.split(":"))) for c in args.curves]
    result = []
    for case,evidence in joined_cases(tape,metadata):
        geometry = np.asarray([s["geometry_distance"] for s in evidence["sites"]])
        sun = np.asarray([s["sun_distance"] if s["sun_distance"] is not None else 1e9
                          for s in evidence["sites"]])
        effective = np.minimum(geometry,sun)
        policies = {"baseline":np.minimum(limits(geometry,48,16),limits(sun,0,256))}
        policies.update({label(c):whole_limits(effective,*c) for c in curves})
        runs = []
        print(case["name"],case["changed_bounds"],flush=True)
        for index in args.windows:
            for reverse in (False,True):
                for repeated in ((False,True) if case["name"] == "cave_far_residency" else (False,)):
                    for policy,caps in policies.items():
                        runs.append(replay(case,index,reverse,repeated,policy,
                            tape["batch_paths"],tape["warm_paths"],caps))
        result.append(dict(name=case["name"],bounds=case["changed_bounds"],
            limits={k:summary(v) for k,v in policies.items()},runs=runs))
    args.output.write_text(json.dumps(dict(cases=result,
        inputs={"tape_sha256":digest(args.tape),"metadata_sha256":digest(args.metadata)},
        windows=args.windows,curves=curves,
        note="Raw native transport tails, including five bands and repeated remote events. Not native material FLIP, display or runtime qualification. Same whole-range equation for geometry and known changed-sunlight origins; baseline preserves existing origin-specific floors."),indent=2,allow_nan=False)+"\n")
    for case in result:
        print(case["name"],case["bounds"],flush=True)
        for policy in policies:
            rows=[r for r in case["runs"] if r["policy"]==policy]
            print(policy,"late p99",max(r["late_rgb"]["p99"] for r in rows),
                  "final p99",max(r["final_rgb"]["p99"] for r in rows),
                  "unsettled",sum(r["all_settled_paths_5pct"] is None for r in rows),flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("tape",type=Path)
    p.add_argument("metadata",type=Path)
    p.add_argument("--curves",nargs="+")
    p.add_argument("--fit",type=Path,help="screen every coefficient pair in a near-fit report")
    p.add_argument("--windows",type=int,nargs="+",default=[0,1])
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if bool(args.curves)==bool(args.fit): p.error("choose --curves or --fit")
    (screen if args.fit else run)(args)
