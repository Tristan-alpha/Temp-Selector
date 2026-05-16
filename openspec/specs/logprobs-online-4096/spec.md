## ADDED Requirements

### Requirement: SGLangRunner.extract returns hidden and logprobs

`extract(prompts, responses)` SHALL return `Tuple[List[torch.Tensor], List[torch.Tensor]]` — per-sample hidden state tensors and per-sample top-k logprob tensors. A single engine prefill SHALL produce both.

#### Scenario: extract with hidden_states mode

- **WHEN** `extract()` is called with 10 prompt-response pairs
- **THEN** both hidden tensors (10 elements, shape [n_tokens, 4096]) and logprob tensors (10 elements, shape [n_tokens, 4096]) are returned

### Requirement: instance_dim=4098 accommodates full hidden states

All config YAMLs SHALL use `data.instance_dim: 4098`. `token_to_vec()` SHALL pass through full hidden states without truncation.

#### Scenario: hidden state vector passes through un-truncated

- **WHEN** a 4096-dim hidden state vector is passed to `token_to_vec(obs_dim=4098)`
- **THEN the full 4096 values are included, padded with 2 zeros (from logprob+entropy base)

### Requirement: topk_logprobs naming throughout codebase

All code and config SHALL use `topk_logprobs` (not `topk_logits`). Config key SHALL be `inference.top_k_logprobs`.

### Requirement: build_dataset does not store logprobs in JSONL

When `feature_mode` is `"topk_logprobs"`, `scripts/build_dataset.py` SHALL set `TokenFeature.topk_logprobs = None` in JSONL output. Logprobs SHALL be obtained online during MIL training via `SGLangRunner.extract()`.

#### Scenario: build with topk_logprobs mode writes null logprobs

- **WHEN** build_dataset runs with `feature_mode: topk_logprobs`
- **THEN JSONL `token_features[].topk_logprobs` is `null` for every token
