import unittest

import numpy as np

from spatial_quality import residual_scales
from spatial_field_sweep import donor_indices, filtered


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

    def test_field_stencil_stops_at_gaps(self):
        rows = [dict(voxel=p, face=2) for p in ([0, 0, 0], [2, 0, 0], [1, 0, 1])]
        self.assertTrue(np.all(donor_indices(rows)[0] == -1))
        rows.append(dict(voxel=[1, 0, 0], face=2))
        indices = donor_indices(rows)
        self.assertEqual(indices[0, 0], 3)
        self.assertEqual(indices[0, 1], 1)
        self.assertEqual(indices[0, 16], 2)

    def test_field_filter_preserves_constants_and_missing_donors(self):
        mean = np.ones((3, 3))
        second = mean * 10
        donors = np.zeros((3, 20), dtype=int)
        np.testing.assert_allclose(filtered(mean, second, 16, donors, .15), mean)
        mean[0] *= .1
        donors.fill(-1)
        np.testing.assert_allclose(filtered(mean, second, 16, donors, .15), mean)


if __name__ == "__main__":
    unittest.main()
