#!/usr/bin/env python3
"""Finite suspect-window experiment: compare, merge, then halve history weight.

This branch is a policy qualification study, not a runtime filter. It consumes
individual native paths during the suspect window, never the 16-path mean.
One shared weight covers all five transport bands. Halving changes only the
weight of future updates, not the just-merged radiance or raw second moment.
"""
import math
import unittest

from history_qualification import History


class SuspectHalving:
    def __init__(self, margin=0.5, metric="bands"):
        if metric not in ("bands", "rgb"):
            raise ValueError(f"unknown disagreement metric: {metric}")
        self.name = f"suspect16-margin{margin:g}" + ("-rgb" if metric == "rgb" else "")
        self.margin = margin
        self.metric = metric
        self.remaining = []

    def event(self, histories, sites, audit):
        # An edit marks evidence suspect, but does not discard anything itself.
        self.remaining = [16] * len(histories)

    def disagrees(self, history, sample):
        # Explicit starting interpretation of "large margin": a relative change
        # in any RGB/IR/ionizing band, with a small numerical floor. No variance
        # gate, replay, brightness clamp, reference target or batch averaging.
        if self.metric == "rgb":
            # Sensitivity control: ignore simulation bands and compare one RGB
            # vector instead. The suspect lifetime and update order are identical.
            a, b = history.mean[:3], sample["mean"][:3]
            norm = lambda values: math.sqrt(sum(v * v for v in values))
            return norm([x - y for x, y in zip(a, b)]) > self.margin * max(norm(a), norm(b), 1e-6)
        return any(abs(a - b) > self.margin * max(abs(a), abs(b), 1e-6)
                   for a, b in zip(history.mean, sample["mean"]))

    def integrate_one(self, history, sample, site):
        suspect = self.remaining[site] > 0
        disagree = suspect and self.disagrees(history, sample)
        history.integrate(sample, 1, history.weight)
        if disagree:
            history.weight *= 0.5
            history.reductions += 1
        if suspect:
            self.remaining[site] -= 1

    def integrate(self, histories, fresh, paths, individual):
        if any(self.remaining):
            if individual is None or len(individual) != paths:
                raise ValueError("suspect history requires individual paths; regenerate the schema-2 tape")
            if any(len(samples) != len(histories) for samples in individual):
                raise ValueError("individual path bank has the wrong receiver count")
            for samples in individual:
                for site, (history, sample) in enumerate(zip(histories, samples)):
                    self.integrate_one(history, sample, site)
        else:
            for history, sample in zip(histories, fresh):
                history.integrate(sample, paths, history.weight)


def policies():
    # The user's window and ordering stay fixed. Only the unspecified margin is
    # swept; 50% is the primary arm, not a post-result selected winner.
    return ([SuspectHalving(margin) for margin in (0.5, 0.25, 0.75, 0.9)]
            + [SuspectHalving(margin, "rgb") for margin in (0.5, 0.9)])


def moment(value):
    return {"mean": [value] * 5, "second": [value * value] * 5}


class Tests(unittest.TestCase):
    def setUp(self):
        self.history = History(moment(1), 4000)
        self.policy = SuspectHalving()
        self.policy.event([self.history], [], None)

    def test_merge_precedes_halving_without_rescaling_the_estimate(self):
        self.policy.integrate_one(self.history, moment(0), 0)
        self.assertEqual(self.history.weight, 2000.5)
        self.assertAlmostEqual(self.history.mean[0], 4000 / 4001)
        self.assertEqual(self.history.mean, self.history.second)
        self.assertEqual(self.history.reductions, 1)
        self.assertEqual(self.policy.remaining, [15])

    def test_comparison_uses_pre_merge_history(self):
        # Pre-merge difference 75% exceeds the chosen threshold of 70%, while
        # the post-merge difference would be only 60%.
        policy = SuspectHalving(0.7)
        h = History(moment(1), 1)
        policy.event([h], [], None)
        policy.integrate_one(h, moment(0.25), 0)
        self.assertEqual(h.weight, 1)
        self.assertEqual(h.mean[0], 0.625)

    def test_sixteen_dark_samples_then_window_closes(self):
        for _ in range(16):
            self.policy.integrate_one(self.history, moment(0), 0)
        expected_mean = 4000 / (3999 + 2**16)
        self.assertAlmostEqual(self.history.mean[0], expected_mean)
        self.assertAlmostEqual(self.history.weight, 1 + 3999 / 2**16)
        self.assertEqual(self.history.reductions, 16)
        self.assertEqual(self.policy.remaining, [0])
        weight = self.history.weight
        self.policy.integrate_one(self.history, moment(0), 0)
        self.assertAlmostEqual(self.history.weight, weight + 1)
        self.assertEqual(self.history.reductions, 16)

    def test_agreement_also_consumes_suspect_samples(self):
        for _ in range(16):
            self.policy.integrate_one(self.history, moment(1), 0)
        self.policy.integrate_one(self.history, moment(0), 0)
        self.assertEqual(self.history.weight, 4017)
        self.assertEqual(self.history.reductions, 0)

    def test_single_outlier_does_not_immediately_replace_history(self):
        self.policy.integrate_one(self.history, moment(0), 0)
        for _ in range(15):
            self.policy.integrate_one(self.history, moment(1), 0)
        self.assertGreater(self.history.mean[0], 0.999)
        self.assertEqual(self.history.reductions, 1)
        self.assertEqual(self.history.weight, 2015.5)

    def test_multiband_disagreement_uses_one_shared_weight(self):
        h = History({"mean": [0, 0, 0, 1, 0], "second": [0, 0, 0, 1, 0]}, 4000)
        self.policy.integrate_one(h, moment(0), 0)
        self.assertEqual(h.weight, 2000.5)
        self.assertAlmostEqual(h.mean[3], 4000 / 4001)
        self.assertEqual(h.mean[:3], [0, 0, 0])

    def test_rgb_sensitivity_arm_does_not_observe_simulation_bands(self):
        policy = SuspectHalving(metric="rgb")
        h = History({"mean": [0, 0, 0, 1, 0], "second": [0, 0, 0, 1, 0]}, 4000)
        policy.event([h], [], None)
        policy.integrate_one(h, moment(0), 0)
        self.assertEqual(h.weight, 4001)
        self.assertEqual(policy.remaining, [15])

    def test_batch_mean_cannot_substitute_for_individual_samples(self):
        with self.assertRaises(ValueError):
            self.policy.integrate([self.history], [moment(0)], 16, None)
        self.assertEqual(self.history.weight, 4000)

    def test_batch_integration_matches_individual_order_then_retains(self):
        self.policy.integrate([self.history], [moment(0)], 16, [[moment(0)] for _ in range(16)])
        expected_weight = self.history.weight
        self.assertAlmostEqual(self.history.mean[0], 4000 / (3999 + 2**16))
        self.policy.integrate([self.history], [moment(0)], 16, None)
        self.assertEqual(self.history.weight, expected_weight + 16)
        self.assertEqual(self.history.reductions, 16)

    def test_new_event_rearms_window_without_cutting_history(self):
        for _ in range(16):
            self.policy.integrate_one(self.history, moment(1), 0)
        self.policy.event([self.history], [], None)
        self.assertEqual(self.policy.remaining, [16])
        self.assertEqual(self.history.weight, 4016)
        self.assertEqual(self.history.reductions, 0)

    def test_unmarked_streaming_keeps_normal_history(self):
        policy = SuspectHalving()
        policy.integrate([self.history], [moment(0)], 16, None)
        self.assertEqual(self.history.weight, 4016)
        self.assertAlmostEqual(self.history.mean[0], 4000 / 4016)
        self.assertEqual(self.history.reductions, 0)


if __name__ == "__main__":
    unittest.main()
