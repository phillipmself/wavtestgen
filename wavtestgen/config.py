"""Configuration loading and validation for wavtestgen."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from typing import Optional
from typing import Union


SUPPORTED_BIT_DEPTHS = {16, 24, 32}
SUPPORTED_CHANNELS = {1, 2}
SUPPORTED_SWEEP_CURVES = {"log", "linear"}
SUPPORTED_NOISE_COLORS = {"white", "pink", "brown"}
SUPPORTED_IMPULSE_POLARITY = {"positive", "negative"}
SUPPORTED_SEGMENT_TYPES = {"sweep", "noise", "silence", "impulses"}


class ConfigError(ValueError):
    """Raised when configuration input is invalid."""


def load_json_config(config_path: Union[str, Path]) -> dict[str, Any]:
    """Load a JSON configuration file."""
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Top-level JSON config must be an object.")
    return data


def normalize_config(
    raw_config: dict[str, Any], overrides: Optional[dict[str, Any]] = None
) -> tuple[dict[str, Any], list[str]]:
    """Validate and normalize raw config into a strict internal dict."""
    cfg = dict(raw_config)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value

    warnings: list[str] = []

    sample_rate = _require_int(cfg, "sample_rate")
    if sample_rate <= 0:
        raise ConfigError("sample_rate must be greater than 0.")

    bit_depth = _require_int(cfg, "bit_depth")
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ConfigError(f"bit_depth must be one of {sorted(SUPPORTED_BIT_DEPTHS)}.")

    channels = cfg.get("channels", 1)
    channels = _require_int_value(channels, "channels")
    if channels not in SUPPORTED_CHANNELS:
        raise ConfigError(f"channels must be one of {sorted(SUPPORTED_CHANNELS)}.")

    fade_ms = cfg.get("fade_ms", 1.0)
    fade_ms = _require_float_value(fade_ms, "fade_ms")
    if fade_ms < 0:
        raise ConfigError("fade_ms must be >= 0.")

    min_sweep_freq_hz = cfg.get("min_sweep_freq_hz", 1.0)
    min_sweep_freq_hz = _require_float_value(min_sweep_freq_hz, "min_sweep_freq_hz")
    if min_sweep_freq_hz <= 0:
        raise ConfigError("min_sweep_freq_hz must be > 0.")

    seed = cfg.get("seed")
    if seed is not None:
        seed = _require_int_value(seed, "seed")

    segments = cfg.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ConfigError("segments must be a non-empty array.")

    normalized_segments: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ConfigError(f"segments[{index}] must be an object.")
        normalized_segments.append(
            _normalize_segment(
                segment=segment,
                index=index,
                min_sweep_freq_hz=min_sweep_freq_hz,
                sample_rate=sample_rate,
                warnings=warnings,
            )
        )

    normalized = {
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "channels": channels,
        "seed": seed,
        "fade_ms": fade_ms,
        "min_sweep_freq_hz": min_sweep_freq_hz,
        "segments": normalized_segments,
    }
    return normalized, warnings


def _normalize_segment(
    segment: dict[str, Any],
    index: int,
    min_sweep_freq_hz: float,
    sample_rate: int,
    warnings: list[str],
) -> dict[str, Any]:
    seg_type = segment.get("type")
    if seg_type not in SUPPORTED_SEGMENT_TYPES:
        raise ConfigError(
            f"segments[{index}].type must be one of {sorted(SUPPORTED_SEGMENT_TYPES)}."
        )

    duration_s = _require_float(segment, "duration_s", index)
    if duration_s <= 0:
        raise ConfigError(f"segments[{index}].duration_s must be > 0.")

    if seg_type == "silence":
        return {"type": "silence", "duration_s": duration_s}

    if seg_type == "sweep":
        start_hz = _require_float(segment, "start_hz", index)
        end_hz = _require_float(segment, "end_hz", index)
        level_dbfs = _require_float(segment, "level_dbfs", index)
        _validate_dbfs(level_dbfs, index, "level_dbfs")

        if start_hz < 0 or end_hz < 0:
            raise ConfigError(f"segments[{index}] sweep frequencies must be >= 0.")
        if start_hz == 0:
            start_hz = min_sweep_freq_hz
            warnings.append(
                f"segments[{index}].start_hz was 0 and was clamped to {min_sweep_freq_hz} Hz."
            )
        if end_hz == 0:
            end_hz = min_sweep_freq_hz
            warnings.append(
                f"segments[{index}].end_hz was 0 and was clamped to {min_sweep_freq_hz} Hz."
            )

        curve = segment.get("curve", "log")
        if curve not in SUPPORTED_SWEEP_CURVES:
            raise ConfigError(
                f"segments[{index}].curve must be one of {sorted(SUPPORTED_SWEEP_CURVES)}."
            )
        return {
            "type": "sweep",
            "duration_s": duration_s,
            "start_hz": start_hz,
            "end_hz": end_hz,
            "level_dbfs": level_dbfs,
            "curve": curve,
        }

    if seg_type == "noise":
        color = segment.get("color")
        if color not in SUPPORTED_NOISE_COLORS:
            raise ConfigError(
                f"segments[{index}].color must be one of {sorted(SUPPORTED_NOISE_COLORS)}."
            )

        has_level = "level_dbfs" in segment
        has_start = "start_dbfs" in segment
        has_end = "end_dbfs" in segment

        if has_level and (has_start or has_end):
            raise ConfigError(
                f"segments[{index}] noise must use either level_dbfs or "
                "start_dbfs/end_dbfs, not both."
            )

        if has_level:
            level_dbfs = _require_float(segment, "level_dbfs", index)
            _validate_dbfs(level_dbfs, index, "level_dbfs")
            return {
                "type": "noise",
                "duration_s": duration_s,
                "color": color,
                "level_dbfs": level_dbfs,
            }

        if not (has_start and has_end):
            raise ConfigError(
                f"segments[{index}] noise must provide either level_dbfs or "
                "both start_dbfs and end_dbfs."
            )

        start_dbfs = _require_float(segment, "start_dbfs", index)
        end_dbfs = _require_float(segment, "end_dbfs", index)
        _validate_dbfs(start_dbfs, index, "start_dbfs")
        _validate_dbfs(end_dbfs, index, "end_dbfs")
        return {
            "type": "noise",
            "duration_s": duration_s,
            "color": color,
            "start_dbfs": start_dbfs,
            "end_dbfs": end_dbfs,
        }

    polarity = segment.get("polarity", "positive")
    if polarity not in SUPPORTED_IMPULSE_POLARITY:
        raise ConfigError(
            f"segments[{index}].polarity must be one of {sorted(SUPPORTED_IMPULSE_POLARITY)}."
        )

    normalized_times = _normalize_impulse_times(segment, index, duration_s)
    _validate_impulse_sample_indices(
        times_s=normalized_times,
        duration_s=duration_s,
        sample_rate=sample_rate,
        index=index,
    )

    has_level = "level_dbfs" in segment
    has_start = "start_dbfs" in segment
    has_end = "end_dbfs" in segment

    if has_level and (has_start or has_end):
        raise ConfigError(
            f"segments[{index}] impulses must use either level_dbfs or "
            "start_dbfs/end_dbfs, not both."
        )

    if has_level:
        level_dbfs = _require_float(segment, "level_dbfs", index)
        _validate_dbfs(level_dbfs, index, "level_dbfs")
        return {
            "type": "impulses",
            "duration_s": duration_s,
            "times_s": normalized_times,
            "level_dbfs": level_dbfs,
            "polarity": polarity,
        }

    if not (has_start and has_end):
        raise ConfigError(
            f"segments[{index}] impulses must provide either level_dbfs or "
            "both start_dbfs and end_dbfs."
        )

    start_dbfs = _require_float(segment, "start_dbfs", index)
    end_dbfs = _require_float(segment, "end_dbfs", index)
    _validate_dbfs(start_dbfs, index, "start_dbfs")
    _validate_dbfs(end_dbfs, index, "end_dbfs")

    return {
        "type": "impulses",
        "duration_s": duration_s,
        "times_s": normalized_times,
        "start_dbfs": start_dbfs,
        "end_dbfs": end_dbfs,
        "polarity": polarity,
    }


def _normalize_impulse_times(
    segment: dict[str, Any], index: int, duration_s: float
) -> list[float]:
    has_times = "times_s" in segment
    has_count = "count" in segment
    has_window_start = "window_start_s" in segment
    has_window_end = "window_end_s" in segment

    if has_times and (has_count or has_window_start or has_window_end):
        raise ConfigError(
            f"segments[{index}] impulses must use either explicit times_s or "
            "count/window_start_s/window_end_s, not both."
        )

    if has_times:
        times_s = segment.get("times_s")
        if not isinstance(times_s, list) or not times_s:
            raise ConfigError(f"segments[{index}].times_s must be a non-empty array.")
        normalized_times: list[float] = []
        previous_time = -math.inf
        for time_i, time_val in enumerate(times_s):
            time_s = _require_float_value(
                time_val, f"segments[{index}].times_s[{time_i}]"
            )
            if time_s < 0 or time_s >= duration_s:
                raise ConfigError(
                    f"segments[{index}].times_s[{time_i}] must be within [0, duration_s)."
                )
            if time_s <= previous_time:
                raise ConfigError(
                    f"segments[{index}].times_s must be strictly increasing with no duplicates."
                )
            previous_time = time_s
            normalized_times.append(time_s)
        return normalized_times

    if not has_count:
        raise ConfigError(
            f"segments[{index}] impulses must provide either times_s or count."
        )

    count = _require_int_value(segment.get("count"), f"segments[{index}].count")
    if count <= 0:
        raise ConfigError(f"segments[{index}].count must be > 0.")

    window_start_s = _require_float_value(
        segment.get("window_start_s", 0.0), f"segments[{index}].window_start_s"
    )
    window_end_s = _require_float_value(
        segment.get("window_end_s", duration_s), f"segments[{index}].window_end_s"
    )

    if window_start_s < 0 or window_start_s > duration_s:
        raise ConfigError(
            f"segments[{index}].window_start_s must be within [0, duration_s]."
        )
    if window_end_s < 0 or window_end_s > duration_s:
        raise ConfigError(
            f"segments[{index}].window_end_s must be within [0, duration_s]."
        )
    if window_end_s < window_start_s:
        raise ConfigError(
            f"segments[{index}] requires window_end_s >= window_start_s."
        )
    if count > 1 and window_end_s == window_start_s:
        raise ConfigError(
            f"segments[{index}] requires window_end_s > window_start_s when count > 1."
        )

    if count == 1:
        return [window_start_s]

    step = (window_end_s - window_start_s) / float(count - 1)
    return [window_start_s + step * i for i in range(count)]


def _validate_impulse_sample_indices(
    times_s: list[float], duration_s: float, sample_rate: int, index: int
) -> None:
    segment_samples = int(round(duration_s * sample_rate))
    if segment_samples <= 0:
        raise ConfigError(
            f"segments[{index}] duration resolves to zero samples at sample_rate {sample_rate}."
        )

    seen: set[int] = set()
    for time_s in times_s:
        sample_index = int(round(time_s * sample_rate))
        if sample_index >= segment_samples:
            sample_index = segment_samples - 1
        if sample_index in seen:
            raise ConfigError(
                f"segments[{index}] impulse times map to duplicate sample indices; "
                "reduce count or increase time window."
            )
        seen.add(sample_index)


def _validate_dbfs(level_dbfs: float, index: int, field_name: str) -> None:
    if level_dbfs > 0:
        raise ConfigError(f"segments[{index}].{field_name} must be <= 0.")


def _require_int(cfg: dict[str, Any], key: str) -> int:
    if key not in cfg:
        raise ConfigError(f"{key} is required.")
    return _require_int_value(cfg[key], key)


def _require_int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer.")
    return value


def _require_float(cfg: dict[str, Any], key: str, index: int) -> float:
    if key not in cfg:
        raise ConfigError(f"segments[{index}].{key} is required.")
    return _require_float_value(cfg[key], f"segments[{index}].{key}")


def _require_float_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric.")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise ConfigError(f"{name} must be finite.")
    return float_value
