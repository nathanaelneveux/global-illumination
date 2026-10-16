import unittest

import numpy as np

from spatial_replay_quality import equivalent_render


class ReplayTests(unittest.TestCase):
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
