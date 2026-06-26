## MODIFIED Requirements

### Requirement: batch_build_segment_obs_from_lp function

`features/segmenter.py` SHALL provide a `batch_build_segment_obs_from_lp` function that accepts a list of per-chain logprob tensors and computes segment observations in a single GPU batch. It SHALL accept `lp_tensors` (list of `[n_tok_i, top_k+1]` tensors), `segment_size`, `obs_dim`, `device`, and the same `include_topk`, `extra_parts_list`, `segment_mode`, `pooling_mode` parameters as the per-chain function. It SHALL return a list of `[n_segments_i, obs_dim]` CPU tensors. In concat mode, the output dimension SHALL be `segment_size × obs_dim` regardless of the actual token count per chain.

#### Scenario: fixed_window mean pooling on GPU

- **WHEN** `batch_build_segment_obs_from_lp` is called with 1500 logprob tensors of uniform shape [32, 4097], `segment_mode="fixed_window"`, and `pooling_mode="mean"`
- **THEN** all tensors SHALL be stacked and moved to `device` in a single transfer
- **AND** `torch.exp`, `torch.cat`, truncation, and `mean(dim=1)` pooling SHALL execute on GPU
- **AND** the result SHALL be a list of 1500 tensors each of shape [1, obs_dim] on CPU

#### Scenario: fixed_window concat pooling on GPU

- **WHEN** `batch_build_segment_obs_from_lp` is called with `segment_mode="fixed_window"` and `pooling_mode="concat"`
- **THEN** `tok_vecs` SHALL be padded or truncated to `segment_size` tokens along the token dimension before reshaping
- **AND** the output for each chain SHALL have shape `[1, segment_size × obs_dim]`
- **AND** shorter chains SHALL be zero-padded, matching `segment_pooling` concat behavior

#### Scenario: concat fast path handles variable token counts

- **WHEN** chains in the batch have `n_tok < segment_size` or `n_tok > segment_size`
- **THEN** the fast path SHALL produce `segment_size × obs_dim` output for every chain
- **AND** the output SHALL match the per-chain `segment_pooling` concat path element-wise

#### Scenario: step mode pools per-chain after GPU batch

- **WHEN** `batch_build_segment_obs_from_lp` is called with `segment_mode="step"`
- **THEN** token-level math (exp, cat, truncate) SHALL execute on GPU
- **AND** per-chain `segment_pooling` SHALL execute on CPU using each chain's text-derived spans
- **AND** the result SHALL be a list of `[n_segments_i, obs_dim]` CPU tensors

#### Scenario: CPU fallback when no GPU available

- **WHEN** `device` is a CPU device (`torch.device("cpu")`)
- **THEN** the function SHALL delegate to per-chain `build_segment_obs_from_lp` calls
- **AND** produce identical results to the GPU path
