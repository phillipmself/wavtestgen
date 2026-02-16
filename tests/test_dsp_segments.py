import unittest

import numpy as np

from wavtestgen.config import normalize_config
from wavtestgen.dsp import dbfs_to_amplitude, render_timeline, sample_count


class DspSegmentTests(unittest.TestCase):
    def test_silence_segment_is_all_zeros(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 16,
            "segments": [{"type": "silence", "duration_s": 0.25}],
        }
        cfg, _ = normalize_config(raw)
        sig = render_timeline(cfg, rng=np.random.default_rng(0))
        self.assertEqual(sig.size, 250)
        self.assertTrue(np.all(sig == 0.0))

    def test_impulses_are_placed_at_expected_indices(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 16,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "times_s": [0.1, 0.5, 0.9],
                    "level_dbfs": -6.0,
                }
            ],
        }
        cfg, _ = normalize_config(raw)
        sig = render_timeline(cfg, rng=np.random.default_rng(0))
        expected = dbfs_to_amplitude(-6.0)
        for idx in (100, 500, 900):
            self.assertAlmostEqual(sig[idx], expected, places=12)
        self.assertEqual(int(np.count_nonzero(sig)), 3)

    def test_impulse_count_mode_evenly_spreads_over_window(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 16,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "count": 5,
                    "window_start_s": 0.1,
                    "window_end_s": 0.5,
                    "level_dbfs": -6.0,
                }
            ],
        }
        cfg, _ = normalize_config(raw)
        sig = render_timeline(cfg, rng=np.random.default_rng(0))
        expected = dbfs_to_amplitude(-6.0)
        for idx in (100, 200, 300, 400, 500):
            self.assertAlmostEqual(sig[idx], expected, places=12)
        self.assertEqual(int(np.count_nonzero(sig)), 5)

    def test_impulse_dbfs_ramp_applies_per_impulse(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 16,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "times_s": [0.1, 0.2, 0.3],
                    "start_dbfs": -20.0,
                    "end_dbfs": -6.0,
                }
            ],
        }
        cfg, _ = normalize_config(raw)
        sig = render_timeline(cfg, rng=np.random.default_rng(0))
        expected_dbfs = [-20.0, -13.0, -6.0]
        for idx, dbfs in zip((100, 200, 300), expected_dbfs):
            self.assertAlmostEqual(sig[idx], dbfs_to_amplitude(dbfs), places=12)

    def test_noise_is_deterministic_with_seed(self):
        raw = {
            "sample_rate": 8000,
            "bit_depth": 16,
            "seed": 123,
            "segments": [{"type": "noise", "duration_s": 0.2, "color": "brown", "level_dbfs": -9.0}],
        }
        cfg, _ = normalize_config(raw)
        a = render_timeline(cfg, rng=np.random.default_rng(cfg["seed"]))
        b = render_timeline(cfg, rng=np.random.default_rng(cfg["seed"]))
        self.assertTrue(np.array_equal(a, b))

    def test_noise_dbfs_ramp_increases_level_over_time(self):
        raw = {
            "sample_rate": 8000,
            "bit_depth": 16,
            "fade_ms": 0.0,
            "seed": 123,
            "segments": [
                {
                    "type": "noise",
                    "duration_s": 1.0,
                    "color": "white",
                    "start_dbfs": -30.0,
                    "end_dbfs": -3.0,
                }
            ],
        }
        cfg, _ = normalize_config(raw)
        sig = render_timeline(cfg, rng=np.random.default_rng(cfg["seed"]))
        first_peak = float(np.max(np.abs(sig[:2000])))
        last_peak = float(np.max(np.abs(sig[-2000:])))
        self.assertGreater(last_peak, first_peak)

    def test_mixed_timeline_length_matches_expected_samples(self):
        raw = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "segments": [
                {"type": "sweep", "duration_s": 0.5, "start_hz": 0, "end_hz": 1000, "level_dbfs": -3.0},
                {"type": "silence", "duration_s": 0.25},
                {"type": "impulses", "duration_s": 0.1, "times_s": [0.01], "level_dbfs": -12.0},
                {"type": "noise", "duration_s": 0.15, "color": "pink", "level_dbfs": -9.0},
            ],
        }
        cfg, _ = normalize_config(raw)
        sig = render_timeline(cfg, rng=np.random.default_rng(1))
        expected = (
            sample_count(0.5, 48000)
            + sample_count(0.25, 48000)
            + sample_count(0.1, 48000)
            + sample_count(0.15, 48000)
        )
        self.assertEqual(sig.size, expected)


if __name__ == "__main__":
    unittest.main()
