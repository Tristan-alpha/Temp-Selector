## ADDED Requirements

### Requirement: token_to_vec returns a tensor, accepts optional extracted

`token_to_vec` SHALL return a `torch.Tensor` of shape `[obs_dim]` and dtype `float32`. It SHALL accept an optional `extracted: Dict[str, torch.Tensor] | None` parameter for per-token tensors (hidden, topk_logprobs) that are consumed inline and NOT stored in the row dict.

#### Scenario: extracted tensors consumed inline

- **WHEN** `token_to_vec(feat, obs_dim, extracted={"hidden": t[j], "topk_logprobs": t[j]})` is called
- **THEN** the tensor values are concatenated via `torch.cat` and the 1D views go out of scope after the function returns

#### Scenario: no extracted dict (PPO path)

- **WHEN** `token_to_vec(feat, obs_dim)` is called without `extracted`
- **THEN** it reads `topk_logprobs` / `hidden` from `feat` dict directly (backward compat)

### Requirement: token_to_obs returns a tensor

`token_to_obs` SHALL return a `torch.Tensor` of shape `[obs_dim]` and dtype `float32`.

### Requirement: mean_pool_obs accepts list of tensors

`mean_pool_obs` SHALL accept `List[torch.Tensor]` where each tensor has shape `[obs_dim]`, and SHALL return a `torch.Tensor` of shape `[obs_dim]` computed via `torch.stack(obs_list).mean(dim=0)`.

#### Scenario: Empty list

- **WHEN** an empty list is passed
- **THEN** a zero tensor of shape `[obs_dim]` is returned

### Requirement: segment_pooling accepts a tensor

`segment_pooling` SHALL accept `token_tensor: torch.Tensor` of shape `[n_tokens, obs_dim]` and SHALL return a `torch.Tensor` of shape `[n_segments, obs_dim]`. In mean mode, per-segment aggregation SHALL use `chunk.mean(dim=0)`. In concat mode, per-segment aggregation SHALL zero-pad segments shorter than `segment_size` to `segment_size × obs_dim` and SHALL truncate segments exceeding `segment_size` to the first `segment_size` tokens.

#### Scenario: Empty token tensor

- **WHEN** an empty tensor is passed
- **THEN** a `[1, obs_dim]` zero tensor is returned

#### Scenario: Concat mode zero-pads short segment

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has 2 tokens
- **THEN** the output tensor for that segment SHALL have shape `[segment_size × obs_dim]`
- **AND** the first `2 × obs_dim` elements SHALL be real features and the remainder SHALL be zeros

#### Scenario: Concat mode truncates long segment

- **WHEN** `segment_pooling` is called with `mode="concat"`, `segment_size=64`, and a segment has 100 tokens
- **THEN** only the first 64 tokens SHALL be used
- **AND** the output tensor SHALL have shape `[segment_size × obs_dim]`

### Requirement: _patch_features is deleted, collate_fn consumes tensors inline

MIL `collate_fn` SHALL iterate over tokens per row, passing extracted tensor views to `token_to_vec` via the `extracted` parameter. No extracted data SHALL be stored in `token_features` dicts or `BagDataset.rows`. The `_patch_features` helper SHALL be deleted.

#### Scenario: Extracted tensors freed after collate_fn

- **WHEN** collate_fn completes for a batch
- **THEN** `hidden_tensors` and `logprob_tensors` go out of scope with no surviving views into them

### Requirement: PPO uses mean_pool_obs instead of manual loops

The manual mean-pool loops in `_extract_segment_obs` and `_extract_segment_obs_sglang` SHALL be replaced with `mean_pool_obs`. Hidden state mean-pool SHALL use `torch.tensor(hidden_states).mean(dim=0)`.

### Requirement: batch concat fast path output matches segment_pooling concat

The `batch_build_segment_obs_from_lp` concat fast path SHALL produce output tensors of shape `[1, segment_size × obs_dim]` that are element-wise identical to those produced by per-chain `segment_pooling(mode="concat", segment_size=segment_size)` for the same input. The fast path SHALL NOT depend on `max_tok` for its output dimension.

#### Scenario: max_tok less than segment_size

- **WHEN** all chains in the batch have `n_tok < segment_size`
- **THEN** the concat fast path output SHALL have shape `[1, segment_size × obs_dim]`
- **AND** the first `n_tok × obs_dim` elements SHALL contain real features and the remainder SHALL be zeros

#### Scenario: max_tok greater than segment_size

- **WHEN** any chain in the batch has `n_tok > segment_size`
- **THEN** the concat fast path output SHALL have shape `[1, segment_size × obs_dim]`
- **AND** only the first `segment_size` tokens SHALL be used per chain

#### Scenario: Batch concat fast path matches per-chain output

- **WHEN** `batch_build_segment_obs_from_lp` concat fast path and per-chain `build_segment_obs_from_lp` are given the same inputs
- **THEN** their output tensors SHALL be element-wise identical
