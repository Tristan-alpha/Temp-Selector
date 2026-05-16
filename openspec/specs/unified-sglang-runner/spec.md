### Requirement: vLLM runner rejects hidden_states mode

`VLLMFeatureExporter.__init__` SHALL raise `ValueError` when `feature_mode` is `"hidden_states"` or `"all"`, because vLLM single engine cannot extract hidden states.

#### Scenario: vLLM with hidden_states mode

- **WHEN** `VLLMFeatureExporter(feature_mode="all")` is constructed
- **THEN** a `ValueError` is raised immediately

### Requirement: Config uses parallel_size instead of tensor_parallel_size

All config YAML files SHALL use `parallel_size` (not `tensor_parallel_size`). Runners SHALL map `parallel_size` to backend-specific parameters: SGLang uses `dp_size`, vLLM uses `tp_size`.

#### Scenario: SGLang runner receives parallel_size

- **WHEN** `SGLangRunner(parallel_size=2)` is constructed
- **THEN** the internal engine is created with `dp_size=2, tp_size=1`

## ADDED Requirements

### Requirement: SGLangRunner provides unified generate + extract

`SGLangRunner` SHALL expose `generate(prompts, temperatures, top_k_logits, num_votes) -> List[Dict]` for token-feature generation and `extract(prompts, responses) -> List[torch.Tensor]` for per-token hidden state extraction. Both SHALL use the same internal `sglang.Engine` instance. `generate()` SHALL conditionally include hidden states based on `self.feature_mode`.

#### Scenario: generate with topk_logits mode

- **WHEN** `SGLangRunner(feature_mode="topk_logits")` calls `generate()`
- **THEN** the engine is created without `enable_return_hidden_states`, and token_features contain only logprob/entropy/topk_logits data

#### Scenario: generate with all mode

- **WHEN** `SGLangRunner(feature_mode="all")` calls `generate()`
- **THEN** hidden state tensors are included in the returned payloads for each sample

### Requirement: extract slices hidden states with correct offset

`extract()` SHALL slice hidden states as `hs[prompt_len - 1:]` — one position before the response starts — because `h[i]` is the hidden state that produces the logits for `token[i+1]`.

#### Scenario: Five prompt tokens, two response tokens

- **WHEN** prompt has 5 tokens, response has 2 tokens
- **THEN** `hs[4:]` returns 2 hidden states, each aligned to one response token

### Requirement: All stages use SGLangRunner, not bare Engine

`ppo/training.py`, `mil/training.py`, `mil/eval.py`, and `scripts/build_dataset.py` SHALL use `SGLangRunner` for all SGLang interactions. No stage SHALL directly import `sglang.Engine`.

#### Scenario: PPO training uses runner

- **WHEN** `ppo/training.py --backend sglang` runs
- **THEN** a `SGLangRunner` is created and both generation and hidden extraction go through its methods

## REMOVED Requirements

### Requirement: APIFeatureExporter for Bailian DashScope API

**Reason**: API backend never used in practice

**Migration**: Remove `--backend api` CLI option from build_dataset.py

### Requirement: VLLMHiddenStateExtractor for two-pass hidden extraction

**Reason**: All hidden extraction now goes through SGLang

**Migration**: Use `SGLangRunner.extract()` instead
