"""DSP primitives for wavtestgen segment rendering."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def dbfs_to_amplitude(level_dbfs: float) -> float:
    """Convert peak dBFS to linear amplitude."""
    return float(10 ** (level_dbfs / 20.0))


def sample_count(duration_s: float, sample_rate: int) -> int:
    """Convert duration to integer sample count using round()."""
    count = int(round(duration_s * sample_rate))
    if count <= 0:
        raise ValueError("Segment duration resolves to zero samples.")
    return count


def apply_fade(signal: np.ndarray, sample_rate: int, fade_ms: float) -> np.ndarray:
    """Apply symmetric fade-in and fade-out to reduce segment boundary clicks."""
    if fade_ms <= 0:
        return signal

    fade_samples = int(round(sample_rate * fade_ms / 1000.0))
    if fade_samples <= 0:
        return signal
    fade_samples = min(fade_samples, signal.size // 2)
    if fade_samples <= 0:
        return signal
    if fade_samples == 1:
        signal[0] = 0.0
        signal[-1] = 0.0
        return signal

    ramp = np.linspace(0.0, 1.0, fade_samples)
    signal[:fade_samples] *= ramp
    signal[-fade_samples:] *= ramp[::-1]
    return signal


def generate_sweep(
    duration_s: float,
    sample_rate: int,
    start_hz: float,
    end_hz: float,
    level_dbfs: float,
    curve: str,
) -> np.ndarray:
    """Generate a sine sweep in linear or logarithmic frequency progression."""
    count = sample_count(duration_s, sample_rate)
    t = np.arange(count, dtype=np.float64) / float(sample_rate)
    amplitude = dbfs_to_amplitude(level_dbfs)

    if curve == "linear":
        k = (end_hz - start_hz) / duration_s
        phase = 2.0 * math.pi * (start_hz * t + 0.5 * k * t * t)
    else:
        if start_hz <= 0 or end_hz <= 0:
            raise ValueError("Log sweep requires strictly positive start/end frequencies.")
        ratio = end_hz / start_hz
        if ratio == 1.0:
            phase = 2.0 * math.pi * start_hz * t
        else:
            phase = (
                2.0
                * math.pi
                * start_hz
                * duration_s
                / math.log(ratio)
                * (np.power(ratio, t / duration_s) - 1.0)
            )

    return amplitude * np.sin(phase)


def generate_noise(
    duration_s: float,
    sample_rate: int,
    color: str,
    start_dbfs: float,
    end_dbfs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate white/pink/brown noise with optional dBFS ramping."""
    count = sample_count(duration_s, sample_rate)
    base = rng.standard_normal(count)
    if color == "white":
        shaped = base
    else:
        shaped = _shape_noise_fft(base, sample_rate, color)
    unit_peak = _normalize_peak(shaped, 1.0)
    if start_dbfs == end_dbfs:
        return unit_peak * dbfs_to_amplitude(start_dbfs)

    envelope_dbfs = np.linspace(start_dbfs, end_dbfs, count, dtype=np.float64)
    envelope_amp = np.power(10.0, envelope_dbfs / 20.0)
    return unit_peak * envelope_amp


def _shape_noise_fft(noise: np.ndarray, sample_rate: int, color: str) -> np.ndarray:
    spectrum = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(noise.size, d=1.0 / float(sample_rate))
    weights = np.ones_like(freqs)

    if color == "pink":
        weights[1:] = 1.0 / np.sqrt(np.maximum(freqs[1:], 1e-12))
    elif color == "brown":
        weights[1:] = 1.0 / np.maximum(freqs[1:], 1e-12)

    weights[0] = 0.0
    shaped = np.fft.irfft(spectrum * weights, n=noise.size)
    return shaped.real


def _normalize_peak(signal: np.ndarray, target_peak: float) -> np.ndarray:
    if target_peak == 0:
        return np.zeros_like(signal)
    peak = float(np.max(np.abs(signal)))
    if peak == 0:
        return np.zeros_like(signal)
    return signal * (target_peak / peak)


def generate_silence(duration_s: float, sample_rate: int) -> np.ndarray:
    """Generate exact silence."""
    return np.zeros(sample_count(duration_s, sample_rate), dtype=np.float64)


def generate_impulses(
    duration_s: float,
    sample_rate: int,
    times_s: list[float],
    levels_dbfs: list[float],
    polarity: str,
) -> np.ndarray:
    """Generate a sparse impulse train with single-sample impulses."""
    count = sample_count(duration_s, sample_rate)
    out = np.zeros(count, dtype=np.float64)
    polarity_sign = -1.0 if polarity == "negative" else 1.0

    for time_s, level_dbfs in zip(times_s, levels_dbfs):
        index = int(round(time_s * sample_rate))
        if index >= count:
            index = count - 1
        amplitude = dbfs_to_amplitude(level_dbfs) * polarity_sign
        out[index] = amplitude
    return out


def render_segment(
    segment: dict[str, Any],
    sample_rate: int,
    fade_ms: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render one validated segment into a mono float signal."""
    seg_type = segment["type"]

    if seg_type == "sweep":
        sweep = generate_sweep(
            duration_s=segment["duration_s"],
            sample_rate=sample_rate,
            start_hz=segment["start_hz"],
            end_hz=segment["end_hz"],
            level_dbfs=segment["level_dbfs"],
            curve=segment["curve"],
        )
        return apply_fade(sweep, sample_rate=sample_rate, fade_ms=fade_ms)

    if seg_type == "noise":
        if "level_dbfs" in segment:
            start_dbfs = segment["level_dbfs"]
            end_dbfs = segment["level_dbfs"]
        else:
            start_dbfs = segment["start_dbfs"]
            end_dbfs = segment["end_dbfs"]
        noise = generate_noise(
            duration_s=segment["duration_s"],
            sample_rate=sample_rate,
            color=segment["color"],
            start_dbfs=start_dbfs,
            end_dbfs=end_dbfs,
            rng=rng,
        )
        return apply_fade(noise, sample_rate=sample_rate, fade_ms=fade_ms)

    if seg_type == "silence":
        return generate_silence(duration_s=segment["duration_s"], sample_rate=sample_rate)

    if seg_type == "impulses":
        times_s = segment["times_s"]
        if "level_dbfs" in segment:
            levels_dbfs = [segment["level_dbfs"]] * len(times_s)
        else:
            start_dbfs = segment["start_dbfs"]
            end_dbfs = segment["end_dbfs"]
            levels_dbfs = np.linspace(
                start_dbfs, end_dbfs, num=len(times_s), dtype=np.float64
            ).tolist()
        return generate_impulses(
            duration_s=segment["duration_s"],
            sample_rate=sample_rate,
            times_s=times_s,
            levels_dbfs=levels_dbfs,
            polarity=segment["polarity"],
        )

    raise ValueError(f"Unsupported segment type: {seg_type}")


def render_timeline(config: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Render all segments into one mono timeline."""
    sample_rate = config["sample_rate"]
    fade_ms = config["fade_ms"]
    segments = config["segments"]

    rendered = [render_segment(seg, sample_rate=sample_rate, fade_ms=fade_ms, rng=rng) for seg in segments]
    signal = np.concatenate(rendered).astype(np.float64, copy=False)

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > 1.0 + 1e-9:
        raise ValueError(f"Rendered signal exceeds full-scale peak: {peak}.")
    return signal


def to_channel_matrix(mono: np.ndarray, channels: int) -> np.ndarray:
    """Expand mono signal to (samples, channels)."""
    if channels == 1:
        return mono.reshape(-1, 1)
    if channels == 2:
        return np.column_stack((mono, mono))
    raise ValueError("channels must be 1 or 2.")
