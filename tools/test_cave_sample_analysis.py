"""Small analytic guards for the cave sample-requirement tool."""
import unittest
import json
import numpy as np
from cave_sample_analysis import (
    corner_members, minimum_sustained, perceptual_difference, reconstruct, srgb_to_oklab,
)


class CaveSampleAnalysisTests(unittest.TestCase):
    def test_shared_corners_and_disconnected_diagonals(self):
        rows = [dict(voxel=p, face=2) for p in [[8, 4, 8], [9, 4, 8], [8, 4, 9], [9, 4, 9]]]
        patch = corner_members(rows, np.zeros(3))
        values = np.array([[v] * 5 for v in [1, 3, 5, 7]])
        corners = reconstruct(values, patch)
        np.testing.assert_allclose(corners[3], 4)
        patch = corner_members([rows[0], rows[3]], np.zeros(3))
        np.testing.assert_allclose(reconstruct(values[[0, 3]], patch)[3], 1)

    def test_sustained_minimum_requires_every_window_and_later_checkpoint(self):
        rows = [dict(paths=n, passed=passed) for n, passed in
                [(16, True), (16, True), (32, True), (32, False), (64, True), (64, True)]]
        self.assertEqual(minimum_sustained(rows, "passed"), 64)
        rows[-1]["passed"] = False
        self.assertIsNone(minimum_sustained(rows, "passed"))

    def test_oklab_black_white_and_red(self):
        values = srgb_to_oklab(np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]], dtype=np.float64))
        np.testing.assert_allclose(values[0], 0, atol=1e-7)
        np.testing.assert_allclose(values[1], [1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(values[2], [.62795536, .22486306, .12584630], atol=1e-7)
        # HSL assigns both 50% lightness; perceptual lightness must not.
        yellow, blue = srgb_to_oklab(np.array([[1, 1, 0], [0, 0, 1]]))
        self.assertGreater(yellow[0] - blue[0], .5)

    def test_official_flip_identity_and_brightness_difference(self):
        image = np.full((32, 32, 3), .4, dtype=np.float32)
        same = perceptual_difference(image, image.copy(), 67)
        self.assertEqual(same["oklab_delta_e"]["max"], 0)
        self.assertLess(same["flip"]["max"], 1e-6)
        json.dumps(same, allow_nan=False)
        different = perceptual_difference(image, image + .05, 67)
        self.assertGreater(different["flip"]["mean"], .01)
        self.assertGreater(different["oklab_delta_e"]["mean"], .01)


if __name__ == "__main__":
    unittest.main()
