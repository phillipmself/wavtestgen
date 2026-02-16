import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from wavtestgen.wav_writer import float_to_pcm_bytes, write_wav


class WavEncodingTests(unittest.TestCase):
    def test_16bit_known_values(self):
        signal = np.array([[-1.0], [0.0], [1.0]], dtype=np.float64)
        data = float_to_pcm_bytes(signal, 16)
        decoded = np.frombuffer(data, dtype="<i2")
        self.assertEqual(decoded.tolist(), [-32767, 0, 32767])

    def test_24bit_byte_length(self):
        signal = np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float64)
        data = float_to_pcm_bytes(signal, 24)
        self.assertEqual(len(data), signal.size * 3)

    def test_32bit_byte_length(self):
        signal = np.ones((10, 2), dtype=np.float64) * 0.25
        data = float_to_pcm_bytes(signal, 32)
        self.assertEqual(len(data), signal.size * 4)

    def test_write_wav_metadata(self):
        signal = np.zeros((480, 2), dtype=np.float64)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "test.wav"
            write_wav(out, signal=signal, sample_rate=48000, bit_depth=24)
            with wave.open(str(out), "rb") as wav_handle:
                self.assertEqual(wav_handle.getnchannels(), 2)
                self.assertEqual(wav_handle.getsampwidth(), 3)
                self.assertEqual(wav_handle.getframerate(), 48000)
                self.assertEqual(wav_handle.getnframes(), 480)


if __name__ == "__main__":
    unittest.main()

