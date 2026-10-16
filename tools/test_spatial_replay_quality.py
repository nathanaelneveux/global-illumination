import unittest

import numpy as np

from spatial_replay_quality import aggregate_errors, equivalent_render


class ReplayTests(unittest.TestCase):
    def test_average_does_not_replace_worst_window_or_local_error(self):
        errors = [dict(flip=dict(mean=v), flip_32px_tile_means=dict(max=2 * v))
                  for v in [.1, .1, .1, .5]]
        result = aggregate_errors(errors)
        self.assertAlmostEqual(result["average_flip"], .2)
        self.assertEqual(result["flip"], .5)
        self.assertAlmostEqual(result["average_worst_tile"], .4)
        self.assertEqual(result["worst_tile"], 1.)
        for invalid in [errors[:3], errors + errors[:1], [dict(flip=dict(mean=float("nan")),
                            flip_32px_tile_means=dict(max=1.))] * 4]:
            with self.assertRaises(ValueError):
                aggregate_errors(invalid)

    def test_identical_and_rare_quantization_pass(self):
        a = np.full((100, 100, 3), .5)
        self.assertEqual(equivalent_render(a, a)["max"], 0)
        b = a.copy()
        b[0, 0, 0] += 1 / 255
        self.assertLess(equivalent_render(a, b)["mean"], 1e-6)

    def test_mismatch_or_nonfinite_is_not_rounding(self):
        a = np.full((100, 100, 3), .5)
        for b in [a + 1 / 255, np.full_like(a, np.nan), a[:10]]:
            with self.assertRaises(ValueError):
                equivalent_render(a, b)


if __name__ == "__main__":
    unittest.main()
