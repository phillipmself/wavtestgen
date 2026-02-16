# wavtestgen

`wavtestgen` is a small Python library and CLI for generating WAV test files from
JSON configs.

## Features

- Configurable sample rate and bit depth (`16/24/32`-bit integer PCM)
- Segment-based timelines:
  - `sweep` (linear or log)
  - `noise` (white, pink, brown)
  - `silence`
  - `impulses` (single-sample spikes at explicit times)
- Peak-referenced dBFS levels
- Optional deterministic noise with `seed`
- Mono output or dual-mono stereo duplication

## Install

```bash
pip install -e .
```

## CLI Usage

```bash
wavtestgen render /path/to/config.json /path/to/output.wav
```

Optional overrides:

```bash
wavtestgen render config.json out.wav --sample-rate 48000 --bit-depth 24 --seed 123 --verbose
```

## Config Format

Top-level fields:

- `sample_rate` (required, int > 0)
- `bit_depth` (required: `16`, `24`, `32`)
- `segments` (required, non-empty list)
- `channels` (optional, default `1`, valid: `1` or `2`)
- `seed` (optional int)
- `fade_ms` (optional float, default `1.0`)
- `min_sweep_freq_hz` (optional float, default `1.0`)

Segment types:

- `sweep`: `duration_s`, `start_hz`, `end_hz`, `level_dbfs`, optional `curve`
- `noise`: `duration_s`, `color`, `level_dbfs`
- `silence`: `duration_s`
- `impulses`: `duration_s`, `times_s`, `level_dbfs`, optional `polarity`

### Example

See [`examples/example_config.json`](examples/example_config.json).

## Python API

```python
from wavtestgen import render_from_config, render_from_dict

render_from_config("config.json", "out.wav")
render_from_dict({...}, "out.wav")
```
