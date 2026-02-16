import json
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path


WORKDIR = Path("/Users/phillipself/dev/audio/sweeper")


class CliRenderTests(unittest.TestCase):
    def test_cli_render_creates_wav(self):
        config = {
            "sample_rate": 16000,
            "bit_depth": 16,
            "channels": 2,
            "seed": 7,
            "segments": [
                {"type": "sweep", "duration_s": 0.2, "start_hz": 0, "end_hz": 2000, "level_dbfs": -3.0},
                {"type": "silence", "duration_s": 0.1},
                {"type": "impulses", "duration_s": 0.2, "times_s": [0.01, 0.1], "level_dbfs": -9.0},
                {"type": "noise", "duration_s": 0.2, "color": "white", "level_dbfs": -12.0},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json"
            output_path = Path(tmp) / "out.wav"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wavtestgen.render",
                    "render",
                    str(config_path),
                    str(output_path),
                    "--seed",
                    "123",
                ],
                cwd=str(WORKDIR),
                text=True,
                capture_output=True,
            )

            self.assertEqual(proc.returncode, 0, msg=proc.stderr)
            self.assertTrue(output_path.exists())
            with wave.open(str(output_path), "rb") as wav_handle:
                self.assertEqual(wav_handle.getframerate(), 16000)
                self.assertEqual(wav_handle.getnchannels(), 2)

    def test_cli_render_invalid_config_fails(self):
        config = {
            "sample_rate": 16000,
            "bit_depth": 16,
            "segments": [
                {"type": "noise", "duration_s": 0.1, "color": "white", "level_dbfs": 1.0}
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bad.json"
            output_path = Path(tmp) / "out.wav"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "wavtestgen.render",
                    "render",
                    str(config_path),
                    str(output_path),
                ],
                cwd=str(WORKDIR),
                text=True,
                capture_output=True,
            )

            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("level_dbfs", proc.stderr)


if __name__ == "__main__":
    unittest.main()

