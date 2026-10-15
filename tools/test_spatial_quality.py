import unittest

import numpy as np

from spatial_quality import residual_scales


class ResidualTests(unittest.TestCase):
    def test_smooth_bias_is_not_scored_as_correct(self):
        reference = np.full((128, 128, 3), .2, dtype=np.float32)
        result = residual_scales(reference + .1, reference)
        self.assertEqual(result["1"]["neighbor_difference_p95"], 0)
        self.assertGreater(result["64"]["rms"], .01)

    def test_scales_distinguish_grain_from_coarse_error(self):
        reference = np.full((128, 128, 3), .2, dtype=np.float32)
        noisy = reference.copy()
        noisy[::2, ::2] += .05
        result = residual_scales(noisy, reference)
        self.assertGreater(result["1"]["neighbor_difference_p95"], .01)
        self.assertLess(result["4"]["neighbor_difference_p95"], 1e-7)
        self.assertEqual(residual_scales(reference, reference)["1"]["rms"], 0)


if __name__ == "__main__":
    unittest.main()
