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

`segment_pooling` SHALL accept `token_tensor: torch.Tensor` of shape `[n_tokens, obs_dim]` and SHALL return a `torch.Tensor` of shape `[n_segments, obs_dim]`. In mean mode, per-segment aggregation SHALL use `chunk.mean(dim=0)`.

#### Scenario: Empty token tensor

- **WHEN** an empty tensor is passed
- **THEN** a `[1, obs_dim]` zero tensor is returned

### Requirement: _patch_features is deleted, collate_fn consumes tensors inline

MIL `collate_fn` SHALL iterate over tokens per row, passing extracted tensor views to `token_to_vec` via the `extracted` parameter. No extracted data SHALL be stored in `token_features` dicts or `BagDataset.rows`. The `_patch_features` helper SHALL be deleted.

#### Scenario: Extracted tensors freed after collate_fn

- **WHEN** collate_fn completes for a batch
- **THEN** `hidden_tensors` and `logprob_tensors` go out of scope with no surviving views into them

### Requirement: PPO uses mean_pool_obs instead of manual loops

The manual mean-pool loops in `_extract_segment_obs` and `_extract_segment_obs_sglang` SHALL be replaced with `mean_pool_obs`. Hidden state mean-pool SHALL use `torch.tensor(hidden_states).mean(dim=0)`.
