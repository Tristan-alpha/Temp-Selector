## 1. Config changes

- [x] 1.1 All config yamls: `instance_dim: 64` → `4098`, `top_k_logits: 16` → `top_k_logprobs: 4096`

## 2. Global rename topk_logits → topk_logprobs

- [x] 2.1 `features/schema.py`: `TokenFeature.topk_logits` → `topk_logprobs`
- [x] 2.2 `features/vectorizer.py`: all `topk_logits` → `topk_logprobs`
- [x] 2.3 `inference/sglang_runner.py`: field names, variable names
- [x] 2.4 `inference/vllm_runner.py`: field names, variable names
- [x] 2.5 `ppo/training.py`: `top_k_logits` → `top_k_logprobs`, variable names

## 3. extract() returns (hidden, logprobs) tuple

- [x] 3.1 `inference/sglang_runner.py`: `extract()` returns `Tuple[List[torch.Tensor], List[torch.Tensor]]`
- [x] 3.2 `mil/training.py`: BagDataset accepts extract tuple, patches both hidden + logprobs into token features
- [x] 3.3 `mil/eval.py`: same

## 4. build_dataset stops writing logprobs

- [x] 4.1 `scripts/build_dataset.py`: set `TokenFeature.topk_logprobs = None` for all tokens

## 5. Tests

- [x] 5.1 Update test assertions for renamed fields
- [x] 5.2 `python -m pytest tests/ -v` — all pass
