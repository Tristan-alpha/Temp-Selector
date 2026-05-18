## Why

`parallel_size` reads from `cfg["inference"].get("parallel_size")` in 4 scripts, but no config YAML file contains this key. The value is always `None`. GPU allocation is a runtime concern and belongs in CLI args, not static config.

## What Changes

- Add `--parallel-size` CLI arg to `mil/training.py`, `mil/eval.py`, `scripts/build_dataset.py`
- Remove all `cfg["inference"].get("parallel_size")` config reads
- `ppo/training.py`: remove `inf.get("parallel_size")` fallback, keep existing `--parallel-size` CLI arg

## Capabilities

### Modified Capabilities

- `gpu-allocation`: `parallel_size` is now determined solely by CLI `--parallel-size`, not config file

## Impact

- `mil/training.py`: add CLI arg, remove config read
- `mil/eval.py`: same
- `scripts/build_dataset.py`: same
- `ppo/training.py`: remove config fallback
- `configs/*.yaml`: unchanged (already have no `parallel_size` key)
