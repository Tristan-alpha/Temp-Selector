## Context

`instance_dim=64` 导致 hidden states 4096-dim 被截断为 ~46 维。上调到 4098 解除瓶颈。同时 term 修正（logits→logprobs）和 logprobs 在线化（不走 JSONL）。

## Decisions

### Decision 1: `instance_dim=4098`, `top_k_logprobs=4096`

base(2) + top_k(4096) = 4098。hidden(4096) 模式下正好放下。logprobs 模式 top-4096 截断取前 4098 维。

### Decision 2: `topk_logits` → `topk_logprobs`

全项目统一重命名。API 始终保持 `return_logprob` 语义。

### Decision 3: logprobs 在线计算，不存 JSONL

与 hidden states 模式统一：`SGLangRunner.extract()` 改为返回 `(hidden_tensors, logprob_tensors)`。MIL 训练/评估时一次 engine prefill 同时获取 hidden + logprobs。

### Decision 4: `extract()` 签名变更

```python
def extract(self, prompts, responses) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    # Returns (hidden_states, logprobs) — both List[torch.Tensor]
```

对于不需要 logprobs 的场景（basic 模式），调用方忽略返回值即可。

## Files

| 操作 | 文件 |
|---|---|
| MODIFY | 12 config yamls: `instance_dim: 4098`, `top_k_logprobs: 4096` |
| MODIFY | `features/schema.py`: `topk_logits` → `topk_logprobs` |
| MODIFY | `features/vectorizer.py`: `topk_logits` → `topk_logprobs` |
| MODIFY | `inference/sglang_runner.py`: `extract()` 返回 tuple; rename |
| MODIFY | `inference/vllm_runner.py`: rename |
| MODIFY | `ppo/training.py`: rename, `top_k_logits` → `top_k_logprobs` |
| MODIFY | `mil/training.py`, `mil/eval.py`: extract 返回 tuple |
| MODIFY | `scripts/build_dataset.py`: stop writing logprobs to JSONL |
