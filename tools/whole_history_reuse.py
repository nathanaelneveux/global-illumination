#!/usr/bin/env python3
"""Reuse native image evidence only when whole-curve frame inputs are identical.

No captures or scores are synthesized. Every candidate RGB/confidence buffer
must be byte-identical to the buffer that produced the original native image.
Keep the original plan, image names, provenance and scores; add an explicit
verified policy alias for aggregation. This avoids rerendering unchanged house
fixtures while rejecting unjustified reuse for different corner donors.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from near_history_calibration import digest, load_tape, replay
from whole_history_fit import label


def reuse(args):
    data=json.loads(args.source.read_text())
    if not data.get("complete"):
        raise ValueError("source assessment must be complete")
    plan=data["plan"]
    tape_path=Path(plan["tape"])
    if digest(tape_path/"near-history.json")!=plan["tape_sha256"]:
        raise ValueError("tape metadata identity changed")
    tape=load_tape(tape_path)
    for name,sha in plan["moments_sha256"].items():
        if digest(tape_path/name)!=sha: raise ValueError("native moments identity changed")
    directory=Path(plan["arguments"]["output"])
    aliases={}
    proofs=[]
    for specification in args.curves:
        old,scale,power=specification.split(":")
        curve=(float(scale),float(power))
        new=label(curve)
        if old in aliases or new in aliases.values(): raise ValueError("duplicate alias")
        frames=[f for f in plan["frames"] if f.get("policy")==old]
        if not frames: raise ValueError("source policy has no recorded frames")
        for case,window in sorted({(f["case"],f["window"]) for f in frames}):
            group=[f for f in frames if (f["case"],f["window"])==(case,window)]
            by_path={f["fresh_paths"]:f for f in group}
            first=group[0]
            rows=replay(tape,window,case,0,0,float(plan["arguments"]["dt"]),
                first["event_interval_paths"],list(by_path),maximum=first["maximum_weight"],whole=curve)
            for paths,after,raw,display,confidence,weight in rows:
                frame=by_path[paths]
                candidate=np.column_stack((display,confidence)).astype("<f4").tobytes()
                original=(directory/(frame["name"]+".bin")).read_bytes()
                if candidate!=original:
                    raise ValueError(f"whole curve changes native frame input: {frame['name']}")
                proofs.append(dict(frame=frame["name"],policy=new,
                    buffer_sha256=hashlib.sha256(candidate).hexdigest()))
        aliases[old]=new
    # Preserve source references and baseline. These are aliases of old native
    # captures, never a claim that another application run occurred.
    for field in ("measurements","summary"):
        kept=[]
        for row in data[field]:
            policy=row.get("policy")
            if policy in aliases:
                row=dict(row,source_policy=policy,policy=aliases[policy])
            elif policy!="baseline" and not row["case"].startswith("warm-"):
                continue
            kept.append(row)
        data[field]=kept
    data["verified_policy_reuse"]=dict(source=str(args.source),source_sha256=digest(args.source),
        aliases=aliases,proofs=proofs,
        note="Original completed native captures and metrics reused. Every candidate RGB/confidence buffer is byte-identical at all captured observations, across every recorded receiver, not merely the visible subset. Original plan/image names/provenance retained; no new native run or reference is claimed.")
    args.output.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    print(args.output,"verified",len(proofs),"native frame buffers",aliases,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    p.add_argument("--curves",nargs="+",required=True,help="source-policy:scale:exponent")
    p.add_argument("--output",type=Path,required=True)
    reuse(p.parse_args())
