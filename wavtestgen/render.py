"""Top-level render API and CLI for wavtestgen."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any
from typing import Optional

import numpy as np

from .config import ConfigError, load_json_config, normalize_config
from .dsp import render_timeline, to_channel_matrix
from .wav_writer import write_wav


def render_from_dict(
    config: dict[str, Any],
    output_path: str,
    *,
    sample_rate: Optional[int] = None,
    bit_depth: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Render a WAV file from an in-memory config dict."""
    overrides = {
        "sample_rate": sample_rate,
        "bit_depth": bit_depth,
        "seed": seed,
    }
    normalized, config_warnings = normalize_config(config, overrides=overrides)
    _emit_warnings(config_warnings, verbose=verbose)

    rng = np.random.default_rng(normalized["seed"])
    mono = render_timeline(normalized, rng=rng)
    channels = to_channel_matrix(mono, normalized["channels"])
    write_wav(
        output_path=output_path,
        signal=channels,
        sample_rate=normalized["sample_rate"],
        bit_depth=normalized["bit_depth"],
    )


def render_from_config(
    config_path: str,
    output_path: str,
    *,
    sample_rate: Optional[int] = None,
    bit_depth: Optional[int] = None,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> None:
    """Render a WAV file from a JSON config path."""
    raw = load_json_config(config_path)
    render_from_dict(
        raw,
        output_path,
        sample_rate=sample_rate,
        bit_depth=bit_depth,
        seed=seed,
        verbose=verbose,
    )


def _emit_warnings(messages: list[str], verbose: bool) -> None:
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
        if verbose:
            print(f"warning: {message}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wavtestgen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="Render WAV from JSON config.")
    render_parser.add_argument("config_path", help="Path to JSON configuration file.")
    render_parser.add_argument("output_path", help="Path for output WAV file.")
    render_parser.add_argument("--sample-rate", type=int, default=None, dest="sample_rate")
    render_parser.add_argument("--bit-depth", type=int, default=None, dest="bit_depth")
    render_parser.add_argument("--seed", type=int, default=None)
    render_parser.add_argument("--verbose", action="store_true")

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command != "render":
        parser.error(f"Unsupported command: {args.command}")

    try:
        render_from_config(
            config_path=args.config_path,
            output_path=args.output_path,
            sample_rate=args.sample_rate,
            bit_depth=args.bit_depth,
            seed=args.seed,
            verbose=args.verbose,
        )
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover
        print(f"error: unexpected failure: {exc}", file=sys.stderr)
        return 1

    print(f"Rendered WAV: {Path(args.output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
