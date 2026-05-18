## Context

`_resolve_parallel_size` determines the vLLM tensor-parallel size. Currently it:
1. Parses `CUDA_VISIBLE_DEVICES` env var manually
2. Falls back to `torch.cuda.device_count()`
3. Falls back to `1` if torch imports fail
4. Deducts 1 when `engine_preset == "prefill"`

The env-var parsing is brittle and redundant (torch already respects `CUDA_VISIBLE_DEVICES`). The `engine_preset` name obscures its real purpose: reserving a GPU for training. The triple-fallback chain silently masks misconfiguration.

## Goals / Non-Goals

**Goals:**
- Use `torch.cuda.device_count()` as the single source of truth for GPU count
- Replace `engine_preset` with a clear `reserve_training_gpu: bool` parameter
- Raise explicit errors instead of silent fallback

**Non-Goals:**
- Changing `ppo/training.py` or `ppo/eval.py` (separate review later)
- Changing config YAML files (no `parallel_size` keys remain)
- Changing the vLLM `LLM()` constructor call beyond `tensor_parallel_size`

## Decisions

### Decision 1: `torch.cuda.device_count()` only

**Alternative considered:** Keep env-var parsing for explicit control.

**Rationale:** `torch.cuda.device_count()` already respects `CUDA_VISIBLE_DEVICES` set by the process environment. Manual parsing of the env var duplicates framework behavior. Dropping it simplifies the code.

### Decision 2: `RuntimeError` on zero GPUs

**Alternative considered:** Keep `max(1, tp)` fallback.

**Rationale:** A vLLM runner with 0 GPUs cannot function. Silently proceeding with `tp=1` on a CPU-only machine would fail later with a confusing vLLM error. Early failure with a clear message is better.

### Decision 3: `reserve_training_gpu: bool` over `engine_preset: str`

**Alternative considered:** Keeping `engine_preset` with values like `"decode"`, `"prefill"`.

**Rationale:** The two presets only differed in whether they deduct 1 GPU. A boolean parameter directly expresses intent. `engine_preset` suggested vLLM engine configuration differences that don't exist.

## Risks / Trade-offs

- **PPO eval has its own `_resolve_tp`**: Not changed in this pass. The two copies will temporarily diverge in style. → Low risk; PPO eval will be reviewed separately.
- **`torch.cuda.device_count()` may return 0 before CUDA init**: In some environments, `device_count()` returns 0 until `torch.cuda.init()` is called. → Not an issue in practice; the runner is always created in GPU-enabled contexts.
