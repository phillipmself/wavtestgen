"""WAV file output utilities."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Union

import numpy as np


def float_to_pcm_bytes(signal: np.ndarray, bit_depth: int) -> bytes:
    """Convert float signal matrix (N, C) in [-1, 1] to PCM frame bytes."""
    if signal.ndim != 2:
        raise ValueError("signal must be a 2D array with shape (samples, channels).")
    if bit_depth not in {16, 24, 32}:
        raise ValueError("bit_depth must be one of {16, 24, 32}.")
    if not np.isfinite(signal).all():
        raise ValueError("signal contains non-finite values.")

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1.0 + 1e-9:
        raise ValueError(f"signal exceeds full scale with peak={peak}.")

    max_int = (1 << (bit_depth - 1)) - 1
    clipped = np.clip(signal, -1.0, 1.0)
    pcm = np.rint(clipped * max_int).astype(np.int32)
    interleaved = pcm.reshape(-1)

    if bit_depth == 16:
        return interleaved.astype("<i2").tobytes()
    if bit_depth == 32:
        return interleaved.astype("<i4").tobytes()
    return _pack_24bit(interleaved)


def _pack_24bit(values: np.ndarray) -> bytes:
    uvals = values.astype(np.int64) & 0xFFFFFF
    out = np.empty(uvals.size * 3, dtype=np.uint8)
    out[0::3] = (uvals & 0xFF).astype(np.uint8)
    out[1::3] = ((uvals >> 8) & 0xFF).astype(np.uint8)
    out[2::3] = ((uvals >> 16) & 0xFF).astype(np.uint8)
    return out.tobytes()


def write_wav(
    output_path: Union[str, Path],
    signal: np.ndarray,
    sample_rate: int,
    bit_depth: int,
) -> None:
    """Write PCM WAV to disk from an (N, C) float matrix."""
    path = Path(output_path)
    channels = int(signal.shape[1])
    sampwidth = bit_depth // 8
    frame_bytes = float_to_pcm_bytes(signal, bit_depth)

    with wave.open(str(path), "wb") as wav_handle:
        wav_handle.setnchannels(channels)
        wav_handle.setsampwidth(sampwidth)
        wav_handle.setframerate(sample_rate)
        wav_handle.writeframes(frame_bytes)
