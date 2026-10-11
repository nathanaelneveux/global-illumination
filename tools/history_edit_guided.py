#!/usr/bin/env python3
"""Qualify edit/sun-directed history caps using schema-3 native transport tapes.

Only native diagnostic metadata enters cap selection. No references, case labels,
future samples, or null-change annotations are available to the policy. Targeted
queries are diagnostic only and are not inserted into lighting accumulation.
This is a separate experiment from the preserved suspect-halving candidate.
"""
import argparse
import json
import unittest
from pathlib import Path

from history_qualification import (History, error, evaluate, luminance, mean_reference,
                                   percentile, relative_luminance_error, verify_individual_samples)


def significant(pair, margin):
    return pair["known"] and any(
        abs(a - b) > margin * max(abs(a), abs(b), 1e-6)
        for a, b in zip(pair["old"], pair["new"]))


class Guided:
    """One history cap per actual invalidation, never a repeated decay window."""
    def __init__(self, metadata, mode, margin=0.25):
        self.name = mode
        self.has_edit = metadata["has_edit"]
        self.segments_per_event = 0
        if mode == "baseline":
            self.caps = metadata["baseline_caps"]
        elif mode in ("retain", "cap16", "cap256"):
            cap = {"retain":65536, "cap16":16, "cap256":256}[mode]
            self.caps = [cap] * len(metadata["receivers"])
        else:
            queries = [r[mode] for r in metadata["receivers"]]
            self.caps = [16 if any(significant(p, margin) for p in q["pairs"]) else 256
                         for q in queries]
            self.segments_per_event = sum(q["segments"] for q in queries)
        self.events = 0

    def event(self, histories, sites, audit):
        # The cave-null marker has no geometry change. Today's policy does not
        # invalidate it either; do not manufacture a baseline reset to beat.
        self.pending = self.has_edit
        self.events += int(self.has_edit)

    def weights(self, histories, fresh, paths):
        result = [min(h.weight, cap) if self.pending else h.weight
                  for h, cap in zip(histories, self.caps)]
        self.pending = False
        return result


def analyze(tape, margin):
    assert tape["schema"] >= 3, "record edit-directed diagnostic metadata first"
    verify_individual_samples(tape)
    results, uncertainty, detections = [], [], []
    modes = ("baseline", "retain", "cap16", "cap256", "geometry1", "combined1", "combined4")
    for case in tape["cases"]:
        metadata = case["guided"]
        detections.append({"case":case["name"], "has_edit":metadata["has_edit"],
                           "edit_targets":metadata["edit_targets"],
                           "sun_targets":metadata["sun_targets"],
                           "cold_sun_segments":metadata["cold_sun_segments"],
                           "updated_sun_segments":metadata["updated_sun_segments"],
                           "sites":len(case["sites"]),
                           "arms":{mode:{"cap16_faces":Guided(metadata, mode, margin).caps.count(16),
                                          "diagnostic_segments_per_event":Guided(metadata, mode, margin).segments_per_event}
                                   for mode in modes}})
        for which in ("references_before", "references_after"):
            ref = case[which]
            avg = mean_reference(ref)
            errors = [error(a, b, s) for a, b, s in zip(ref[0], ref[1], avg)]
            uncertainty.append({"case":case["name"], "target":which,
                                "rgb_p95":percentile([e[0] for e in errors]),
                                "relative_luminance_p95":percentile([
                                    relative_luminance_error(a, b, luminance(s))
                                    for a, b, s in zip(ref[0], ref[1], avg)]),
                                "all_band_p95":percentile([e[1] for e in errors])})
        for run in case["runs"]:
            for mode in modes:
                policy = Guided(metadata, mode, margin)
                result = evaluate(case, run, policy, tape["batch_paths"], tape["warm_paths"], tape["steps"])
                result["diagnostic_segments_per_site"] = policy.events * policy.segments_per_event / len(case["sites"])
                results.append(result)
                print(json.dumps({k:v for k,v in result.items() if k not in ("curve", "checkpoints")}))
    return {"margin":margin, "detections":detections, "uncertainty":uncertainty, "results":results,
            "note":"Fresh-path accuracy/stability on native frozen-scene tapes. Diagnostic traversal segments are NOT Monte Carlo paths and are reported separately. Sun-cache refresh work is shared with existing direct lighting, not free. Old/new scene residency, indexing and GPU scheduling costs unmeasured. Not full A-SVGF, gameplay FPS, or seconds."}


class Tests(unittest.TestCase):
    def metadata(self, pair, has_edit=True):
        query = {"pairs":[pair], "segments":2}
        return {"has_edit":has_edit, "baseline_caps":[16], "receivers":[{"combined1":query}]}

    def pair(self, old, new, known=True):
        return {"old":old, "new":new, "known":known}

    def test_all_bands_and_unknown_and_reversal(self):
        a, b = [1]*5, [1, 1, 1, 0, 1]
        self.assertTrue(significant(self.pair(a, b), 0.25))
        self.assertTrue(significant(self.pair(b, a), 0.25))
        self.assertFalse(significant(self.pair(a, b, False), 0.25))
        self.assertFalse(significant(self.pair(a, [0.8]*5), 0.25))

    def test_cap_is_applied_once_and_does_not_inflate_history(self):
        history = History({"mean":[1]*5, "second":[1]*5}, 4096)
        policy = Guided(self.metadata(self.pair([1]*5, [0]*5)), "combined1")
        policy.event([history], [], None)
        self.assertEqual(policy.weights([history], [], 16), [16])
        self.assertEqual(policy.weights([history], [], 16), [4096])
        history.weight = 8
        policy.event([history], [], None)
        self.assertEqual(policy.weights([history], [], 16), [8])

    def test_no_edit_means_no_history_cap_or_query_work(self):
        history = History({"mean":[1]*5, "second":[1]*5}, 4096)
        policy = Guided(self.metadata(self.pair([1]*5, [0]*5), False), "combined1")
        policy.event([history], [], None)
        self.assertEqual(policy.weights([history], [], 16), [4096])
        self.assertEqual(policy.events, 0)

    def test_missed_or_unknown_evidence_uses_256_not_retention(self):
        for pair in (self.pair([1]*5, [1]*5), self.pair([1]*5, [0]*5, False)):
            self.assertEqual(Guided(self.metadata(pair), "combined1").caps, [256])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tape", nargs="?")
    parser.add_argument("--output")
    parser.add_argument("--margin", type=float, default=0.25)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if args.test:
        unittest.main(argv=["history_edit_guided.py"])
    else:
        assert 0 < args.margin < 1
        report = analyze(json.loads(Path(args.tape).read_text()), args.margin)
        if args.output:
            Path(args.output).write_text(json.dumps(report))
