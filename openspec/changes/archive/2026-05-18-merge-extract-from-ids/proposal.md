## Why

`extract_logprobs_from_ids` and `extract_hidden_from_ids` share the same `llm.generate()` call but are separate methods. When `feature_mode=all`, collate_fn calls both sequentially, running the same prefill twice. Merging them eliminates the duplicate GPU work.

## What Changes

- Merge `extract_logprobs_from_ids` and `extract_hidden_from_ids` into a single `extract_from_ids` method
- Method accepts `return_logprobs: bool` and `return_hidden: bool` flags
- Returns a `dict` with `"logprobs"` and/or `"hidden"` keys
- Update `collate_fn` in `mil/training.py` to use the merged method
- Update `mil/eval.py` collate_fn similarly

## Capabilities

### Modified Capabilities

- `collate-feature-extraction`: unified online extraction method replaces two separate calls

## Impact

- `inference/vllm_runner.py`: delete 2 methods, add 1 merged method
- `mil/training.py`: update collate_fn
- `mil/eval.py`: update collate_fn
