import unittest

from wavtestgen.config import ConfigError, normalize_config


def _valid_config():
    return {
        "sample_rate": 48000,
        "bit_depth": 24,
        "segments": [
            {
                "type": "sweep",
                "duration_s": 1.0,
                "start_hz": 20,
                "end_hz": 20000,
                "level_dbfs": -3.0,
            }
        ],
    }


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_are_applied(self):
        cfg, warnings = normalize_config(_valid_config())
        self.assertEqual(cfg["channels"], 1)
        self.assertEqual(cfg["fade_ms"], 1.0)
        self.assertEqual(cfg["min_sweep_freq_hz"], 1.0)
        self.assertEqual(cfg["segments"][0]["curve"], "log")
        self.assertEqual(warnings, [])

    def test_invalid_bit_depth(self):
        raw = _valid_config()
        raw["bit_depth"] = 20
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_sweep_zero_frequency_is_clamped_with_warning(self):
        raw = _valid_config()
        raw["segments"][0]["start_hz"] = 0
        cfg, warnings = normalize_config(raw)
        self.assertEqual(cfg["segments"][0]["start_hz"], 1.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("clamped", warnings[0])

    def test_impulse_times_must_be_strictly_increasing(self):
        raw = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "times_s": [0.5, 0.5],
                    "level_dbfs": -6.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_impulses_can_use_count_and_window(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 24,
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
        times = cfg["segments"][0]["times_s"]
        for observed, expected in zip(times, [0.1, 0.2, 0.3, 0.4, 0.5]):
            self.assertAlmostEqual(observed, expected, places=12)

    def test_impulses_can_use_start_and_end_dbfs(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "times_s": [0.1, 0.3, 0.5],
                    "start_dbfs": -20.0,
                    "end_dbfs": -6.0,
                }
            ],
        }
        cfg, _ = normalize_config(raw)
        seg = cfg["segments"][0]
        self.assertEqual(seg["start_dbfs"], -20.0)
        self.assertEqual(seg["end_dbfs"], -6.0)
        self.assertNotIn("level_dbfs", seg)

    def test_impulses_reject_mixed_timing_modes(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "times_s": [0.1],
                    "count": 2,
                    "level_dbfs": -6.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_impulses_reject_mixed_level_modes(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "times_s": [0.1, 0.2],
                    "level_dbfs": -6.0,
                    "start_dbfs": -20.0,
                    "end_dbfs": -3.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_impulse_count_mode_requires_unique_sample_indices(self):
        raw = {
            "sample_rate": 1000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "impulses",
                    "duration_s": 1.0,
                    "count": 5,
                    "window_start_s": 0.0,
                    "window_end_s": 0.001,
                    "level_dbfs": -6.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_level_must_not_exceed_zero_dbfs(self):
        raw = {
            "sample_rate": 48000,
            "bit_depth": 16,
            "segments": [
                {
                    "type": "noise",
                    "duration_s": 1.0,
                    "color": "white",
                    "level_dbfs": 1.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_noise_can_use_start_and_end_dbfs(self):
        raw = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "noise",
                    "duration_s": 1.0,
                    "color": "pink",
                    "start_dbfs": -30.0,
                    "end_dbfs": -6.0,
                }
            ],
        }
        cfg, _ = normalize_config(raw)
        seg = cfg["segments"][0]
        self.assertEqual(seg["start_dbfs"], -30.0)
        self.assertEqual(seg["end_dbfs"], -6.0)
        self.assertNotIn("level_dbfs", seg)

    def test_noise_rejects_mixed_constant_and_ramp_dbfs(self):
        raw = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "noise",
                    "duration_s": 1.0,
                    "color": "white",
                    "level_dbfs": -6.0,
                    "start_dbfs": -12.0,
                    "end_dbfs": -3.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)

    def test_noise_requires_complete_dbfs_definition(self):
        raw = {
            "sample_rate": 48000,
            "bit_depth": 24,
            "segments": [
                {
                    "type": "noise",
                    "duration_s": 1.0,
                    "color": "brown",
                    "start_dbfs": -12.0,
                }
            ],
        }
        with self.assertRaises(ConfigError):
            normalize_config(raw)


if __name__ == "__main__":
    unittest.main()
