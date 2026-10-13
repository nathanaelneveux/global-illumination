"""Small guards for causal accounting; no native renderer needed."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from retention_attribution import analyze, groups
from runtime_retention_analysis import checkpoint_reference


class RetentionAttributionTests(unittest.TestCase):
    def test_overlapping_cause_sets_count_each_cut_once(self):
        events = [dict(before=4000, retained=256, site=0),
                  dict(before=272, retained=256, site=0)]
        result = groups(events, lambda _: "geometry+sunlight")
        self.assertEqual(result, {"geometry+sunlight": dict(cuts=2, weight_removed=3760, sites=1)})

    def documents(self):
        sites = [dict(voxel=[i, 0, 0], face=2) for i in range(36)]
        corners = [[[[1 + replica * .001] * 5] * 4 for _ in sites] for replica in range(2)]
        faces = [[[1 + replica * .001] * 5 for _ in sites] for replica in range(2)]
        maximum = 4294967295
        return {
            "status.json": dict(status="completed"),
            "report.json": dict(status="completed", profile="cave", spatial_retention="calibrated-cave-user-2048-v1",
                capture_mode="offscreen", captures=1, capture_windows=[dict(varied=True, width=2560,
                height=1440, target_valid=True)], background_terrain=dict(enabled=True),
                traced_paths=1000000, max_loaded_pages=100),
            "temporal.json": dict(end_s=182),
            "retention-audit.json": dict(schema_version=1, sites=sites, cap_labels=["geometry", "sunlight", "near", "far",
                "known", "first", "unresolved", "recovered"], events=[dict(at_s=100, site=0, before=1024,
                retained=256, source_changed=False, model_restart=False, sky_current=True,
                decision=dict(cap=256, caps=[maximum, 256, maximum, maximum, maximum, 256, maximum, maximum],
                distance_squared=[1000000, 100, 1000000, 100, 1000000, 1000000]))]),
            "retention-reference.json": dict(sites=sites, paths_per_replicate=65536,
                checkpoints=[dict(at_s=t, corner_references=corners, patch_references=faces)
                             for t in [30, 60, 90, 120, 150, 180]]),
        }

    def run_analysis(self, docs):
        with patch.object(Path, "read_text", lambda path: json.dumps(docs[path.name])):
            return analyze(Path("test"))

    def test_fixed_scene_has_zero_paired_drift_despite_reference_uncertainty(self):
        result = self.run_analysis(self.documents())
        motion = result["corner_target_motion"]
        self.assertEqual(len(motion["comparisons"]), 15)
        self.assertGreater(motion["checkpoint_window_disagreement"][0]["max"], 0)
        self.assertTrue(all(row["change"]["max"] == 0 for row in motion["comparisons"]))
        late = result["intervals"][-1]
        self.assertEqual(late["weight_removed"], 768)
        self.assertEqual(late["sunlight_co_dominant"]["first"]["cuts"], 1)

    def test_rejects_inconsistent_attribution_and_incomplete_oracle(self):
        docs = self.documents()
        docs["retention-audit.json"]["events"][0]["decision"]["cap"] = 16
        with self.assertRaisesRegex(ValueError, "actual cap"):
            self.run_analysis(docs)
        docs = self.documents()
        docs["retention-reference.json"]["checkpoints"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_analysis(docs)

    def test_shadow_journal_accounts_only_for_additional_discarded_weight(self):
        docs = self.documents()
        audit = docs["retention-audit.json"]
        audit["schema_version"] = 2
        event = audit["events"][0]
        maximum = 4294967295
        event["decision"]["caps"] = [maximum, maximum, maximum, 256] + [maximum]*4
        event["decision"]["unfolded"] = dict(cap=768, caps=[768]+[maximum]*7,
            distance_squared=[40000]*6)
        interval = self.run_analysis(docs)["intervals"][-1]
        result = interval["overflow_counterfactual"]
        self.assertEqual(result["extra_weight_removed_by_summary"], 512)
        self.assertEqual(result["remaining_causes"]["geometry"]["weight_removed"], 256)
        self.assertEqual(interval["individual_cause_counterfactual"]["causes"]["geometry"]["weight_removed"], 256)
        event["decision"]["unfolded"] = None
        with self.assertRaisesRegex(ValueError, "shadow journal too short"):
            self.run_analysis(docs)

    def test_distance_summary_identity_and_conservatism_are_checked(self):
        docs = self.documents()
        docs["report.json"]["retention_summary"] = "distance-bands"
        with self.assertRaisesRegex(ValueError, "summary identity"):
            self.run_analysis(docs)
        audit = docs["retention-audit.json"]
        audit["retention_summary"] = "distance-bands"
        self.assertEqual(self.run_analysis(docs)["summary"], "distance-bands")
        audit["events"][0]["decision"]["unfolded"] = dict(cap=16, caps=[16]+[4294967295]*7)
        with self.assertRaisesRegex(ValueError, "exact-history cap"):
            self.run_analysis(docs)

    def test_live_accuracy_uses_final_independent_oracle_and_checks_patch_order(self):
        ref = self.documents()["retention-reference.json"]
        ref["patch_sites"] = ref["sites"] + [dict(voxel=[100, 0, 0], face=2)]
        ref["checkpoints"][-1]["patch_references"][0][0] = [2]*5
        result = checkpoint_reference(ref)
        self.assertEqual(len(result["references"][0]), 36)
        self.assertEqual(result["references"][0][0], [2]*5)
        ref["patch_sites"] = list(reversed(ref["patch_sites"]))
        with self.assertRaisesRegex(ValueError, "patch prefix"):
            checkpoint_reference(ref)
        ref["checkpoints"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete retention reference"):
            checkpoint_reference(ref)


if __name__ == "__main__":
    unittest.main()
