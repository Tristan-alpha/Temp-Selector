## Context

After simplifying `_resolve_parallel_size`, `parallel_size` defaults to `None` (auto-detect all GPUs). The config YAML files no longer contain a `parallel_size` key. However, 4 scripts still read `cfg["inference"].get("parallel_size")` which always returns `None`.

## Goals / Non-Goals

**Goals:**
- Remove dead config-read code
- Add `--parallel-size` CLI arg where missing
- `--parallel-size` default: `None` (auto-detect)

**Non-Goals:**
- Changing `run_pipeline.sh`
- Changing `ppo/eval.py` (separate review)

## Decisions

### Decision 1: CLI-only, no config fallback

**Rationale:** GPU allocation is runtime infrastructure, not model/training configuration. Moving it exclusively to CLI makes the separation clear. Configs describe the experiment; CLI describes the execution environment.

## Risks / Trade-offs

- **`run_pipeline.sh` doesn't pass `--parallel-size`**: All scripts will default to `None` (auto-detect all GPUs), which is the correct behavior for the pipeline.
