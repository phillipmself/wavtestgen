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


if __name__ == "__main__":
    unittest.main()
