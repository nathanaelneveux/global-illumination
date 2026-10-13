#!/usr/bin/env python3
"""Explain accepted history cuts, separately from measured scene-target drift.

No counterfactual policy is applied here. Cause sets can overlap; exclusive
combinations prevent double-counting discarded statistical weight. Six frozen
scene checkpoints measure target motion but cannot classify every intervening
event as harmless. Reference-window disagreement is sensitivity, not a CI.
"""
import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from runtime_retention_analysis import distribution, validate_capture


def groups(events, key):
    result = defaultdict(lambda: dict(cuts=0, weight_removed=0, sites=set()))
    for event in events:
        row = result[key(event)]
        row["cuts"] += 1
        row["weight_removed"] += event["before"] - event["retained"]
        row["sites"].add(event["site"])
    return {k: dict(v, sites=len(v["sites"])) for k, v in sorted(result.items())}


def analyze(folder):
    report = json.loads((folder / "report.json").read_text())
    validate_capture(folder, report)
    audit = json.loads((folder / "retention-audit.json").read_text())
    reference = json.loads((folder / "retention-reference.json").read_text())
    temporal = json.loads((folder / "temporal.json").read_text())
    if (report["profile"] != "cave" or report["spatial_retention"] != "calibrated-cave-user-2048-v1"
            or not report["background_terrain"]["enabled"]
            or temporal["end_s"] < 180 or len(audit["sites"]) != 36):
        raise ValueError("require complete calibrated cave with background terrain")
    if audit["sites"] != reference["sites"]:
        raise ValueError("oracle and attribution receivers differ")
    if len(reference["checkpoints"]) != 6 or reference["paths_per_replicate"] != 65536:
        raise ValueError("incomplete high-sample checkpoints")
    events = audit["events"]
    for event in events:
        decision = event["decision"]
        if decision["cap"] != min(decision["caps"][:4]):
            raise ValueError("diagnostic decomposition does not match actual cap")
        if event["retained"] != min(event["before"], decision["cap"]):
            raise ValueError("invalid retained weight")
        unfolded = decision.get("unfolded")
        if unfolded and unfolded["cap"] != min(unfolded["caps"][:2]):
            raise ValueError("inconsistent shadow journal")
    if any(e["source_changed"] or e["model_restart"] for e in events if e["at_s"] >= 30):
        raise ValueError("source/model changes need separate attribution")
    labels = audit["cap_labels"]

    def mechanism(event):
        d = event["decision"]
        return "+".join(labels[i] for i in range(4) if d["caps"][i] == d["cap"])

    def sunlight(event):
        d = event["decision"]
        matching = [labels[i] for i in range(4, 8) if d["caps"][i] <= d["caps"][1]]
        return "+".join(matching) or "coalescing_only"

    intervals = []
    for start, end in [(0, 30), (30, 60), (60, 120), (120, 180), (30, 180)]:
        selected = [e for e in events if start <= e["at_s"] < end]
        cuts = [e for e in selected if e["retained"] < e["before"]]
        sun = [e for e in cuts if e["decision"]["caps"][1] == e["decision"]["cap"]]
        geo = [e for e in cuts if e["decision"]["caps"][0] == e["decision"]["cap"]]
        folded = [e for e in cuts if min(e["decision"]["caps"][2:4]) == e["decision"]["cap"]]
        explained = [e for e in folded if e["decision"].get("unfolded")]
        if audit["schema_version"] >= 2 and len(explained) != len(folded):
            raise ValueError("shadow journal too short to explain a consumed overflow")
        extra_loss = sum(min(e["before"], e["decision"]["unfolded"]["cap"])-e["retained"] for e in explained)
        unfolded_cuts = [dict(e, decision=e["decision"]["unfolded"],
                             retained=min(e["before"], e["decision"]["unfolded"]["cap"])) for e in explained
                         if e["decision"]["unfolded"]["cap"] < e["before"]]
        unfolded_sun = [e for e in unfolded_cuts if e["decision"]["caps"][1] == e["decision"]["cap"]]
        complete = [e for e in cuts if e["decision"].get("unfolded")]
        single_cause = {}
        for i in [0, 4, 5, 6, 7]:
            losses = [max(0, e["before"]-e["decision"]["unfolded"]["caps"][i]) for e in complete]
            single_cause[labels[i]] = dict(cuts=sum(v > 0 for v in losses), weight_removed=sum(losses))
        intervals.append(dict(
            seconds=[start, end], accepted_jobs=len(selected), cuts=len(cuts),
            weight_removed=sum(e["before"]-e["retained"] for e in cuts),
            mechanisms=groups(cuts, mechanism), sunlight_co_dominant=groups(sun, sunlight),
            geometry_distance=distribution([np.sqrt(e["decision"]["distance_squared"][0]) for e in geo]) if geo else None,
            sunlight_distance=distribution([np.sqrt(e["decision"]["distance_squared"][1]) for e in sun]) if sun else None,
            retained=distribution([e["retained"] for e in cuts]) if cuts else None,
            before=distribution([e["before"] for e in cuts]) if cuts else None,
            superseded_scene_jobs=sum(not e["sky_current"] for e in cuts),
            individual_cause_counterfactual=dict(complete_cuts=len(complete), causes=single_cause,
                note="Each cause alone at observed weights, using unfolded records. Overlapping alternatives; do not sum or call a known distant sun change a measured receiver-lighting change."),
            overflow_counterfactual=dict(available=audit["schema_version"] >= 2, explained=len(explained),
                no_cut_needed=sum(e["decision"]["unfolded"]["cap"] >= e["before"] for e in explained),
                less_severe=sum(e["decision"]["unfolded"]["cap"] > e["retained"] for e in explained),
                extra_weight_removed_by_summary=extra_loss,
                remaining_causes=groups(unfolded_cuts, mechanism),
                remaining_sunlight=groups(unfolded_sun, sunlight),
                largest_cuts=[dict(seconds=e["at_s"], site=e["site"], before=e["before"], retained=e["retained"],
                    unfolded_cap=e["decision"]["unfolded"]["cap"], unfolded_caps=e["decision"]["unfolded"]["caps"],
                    unfolded_distances=[None if v == 9223372036854775807 else float(np.sqrt(v))
                        for v in e["decision"]["unfolded"]["distance_squared"]])
                    for e in sorted(explained, key=lambda e:e["before"]-e["retained"], reverse=True)[:5]],
                note="One-step counterfactual at observed weight, not a no-overflow trajectory replay."),
        ))
    checkpoints = reference["checkpoints"]
    corners = np.asarray([s["corner_references"] for s in checkpoints])[..., :3]
    faces = np.asarray([s["patch_references"] for s in checkpoints])[..., :3][:, :, :36]
    if not np.all(np.isfinite(corners)) or not np.all(np.isfinite(faces)):
        raise ValueError("nonfinite reference")

    def target_motion(values):
        mean = values.mean(axis=1)
        scale = np.maximum(np.linalg.norm(mean.max(axis=0), axis=-1), .02)
        rows = []
        for a, b in combinations(range(len(checkpoints)), 2):
            delta = values[b] - values[a]
            rows.append(dict(
                seconds=[checkpoints[a]["at_s"], checkpoints[b]["at_s"]],
                change=distribution(np.linalg.norm(mean[b]-mean[a], axis=-1) / scale),
                paired_delta_window_disagreement=distribution(np.linalg.norm(delta[1]-delta[0], axis=-1) / scale),
            ))
        return dict(
            checkpoint_window_disagreement=[distribution(np.linalg.norm(v[1]-v[0], axis=-1) / scale) for v in values],
            comparisons=rows,
            note="Fixed brighter-across-checkpoints RGB normalization with .02 norm floor. Common sequence windows; paired-delta disagreement is not a formal confidence bound.",
        )

    return dict(run=str(folder), intervals=intervals, corner_target_motion=target_motion(corners),
                face_target_motion=target_motion(faces), traced_paths=report["traced_paths"],
                max_pages=report["max_loaded_pages"], background=report["background_terrain"],
                note="Instrumentation only, no threshold/policy changes. Mechanism attribution is event-complete at 36 faces; oracle samples six scene checkpoints, not every event. No FPS claim.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run), indent=2))
