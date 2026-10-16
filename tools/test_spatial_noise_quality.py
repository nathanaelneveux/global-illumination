import unittest

import numpy as np

from spatial_noise_quality import band_energy, residual_bands, summarize_windows


class NoiseTests(unittest.TestCase):
    def test_constant_brightness_error_is_not_spatial_noise(self):
        reference = np.full((128, 128, 3), .3)
        bands, bias = residual_bands(reference + .05, reference)
        self.assertLess(sum(band_energy(bands)), 1e-25)
        self.assertGreater(bias, .01)

    def test_matching_reference_has_no_noise_or_bias(self):
        image = np.random.default_rng(1).uniform(.1, .8, (128, 128, 3))
        bands, bias = residual_bands(image, image)
        self.assertEqual(sum(band_energy(bands)), 0)
        self.assertEqual(bias, 0)

    def test_sequence_dispersion_distinguishes_fixed_from_random_error(self):
        reference = np.full((128, 128, 3), .3)
        windows = []
        for seed in range(4):
            noise = np.random.default_rng(seed).normal(0, .02, (8, 8, 1))
            noise = np.repeat(np.repeat(noise, 16, axis=0), 16, axis=1)
            bands, _ = residual_bands(reference + noise, reference)
            windows.append(bands)
        actual = summarize_windows(windows)
        fixed = summarize_windows([windows[0]] * 4)
        self.assertGreater(actual["wall_noise_max"], .001)
        self.assertGreater(actual["sequence_dispersion"], .001)
        self.assertEqual(fixed["sequence_dispersion"], 0)
        self.assertGreater(fixed["wall_noise_max"], 0)

    def test_invalid_captures_fail(self):
        image = np.full((128, 128, 3), .3)
        with self.assertRaises(ValueError):
            residual_bands(image[:32], image[:32])
        image[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            residual_bands(image, image)

    def test_incomplete_window_set_fails(self):
        with self.assertRaises(ValueError):
            summarize_windows([])


if __name__ == "__main__":
    unittest.main()
