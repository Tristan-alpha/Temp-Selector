## ADDED Requirements

### Requirement: batch_build_segment_obs_from_lp function

`features/segmenter.py` SHALL provide a `batch_build_segment_obs_from_lp` function that accepts a list of per-chain logprob tensors and computes segment observations in a single GPU batch. It SHALL accept `lp_tensors` (list of `[n_tok_i, top_k+1]` tensors), `segment_size`, `obs_dim`, `device`, and the same `include_topk`, `extra_parts_list`, `segment_mode`, `pooling_mode` parameters as the per-chain function. It SHALL return a list of `[n_segments_i, obs_dim]` CPU tensors.

#### Scenario: fixed_window mean pooling on GPU

- **WHEN** `batch_build_segment_obs_from_lp` is called with 1500 logprob tensors of uniform shape [32, 4097], `segment_mode="fixed_window"`, and `pooling_mode="mean"`
- **THEN** all tensors SHALL be stacked and moved to `device` in a single transfer
- **AND** `torch.exp`, `torch.cat`, truncation, and `mean(dim=1)` pooling SHALL execute on GPU
- **AND** the result SHALL be a list of 1500 tensors each of shape [1, obs_dim] on CPU

#### Scenario: step mode pools per-chain after GPU batch

- **WHEN** `batch_build_segment_obs_from_lp` is called with `segment_mode="step"`
- **THEN** token-level math (exp, cat, truncate) SHALL execute on GPU
- **AND** per-chain `segment_pooling` SHALL execute on CPU using each chain's text-derived spans
- **AND** the result SHALL be a list of `[n_segments_i, obs_dim]` CPU tensors

#### Scenario: CPU fallback when no GPU available

- **WHEN** `device` is a CPU device (`torch.device("cpu")`)
- **THEN** the function SHALL delegate to per-chain `build_segment_obs_from_lp` calls
- **AND** produce identical results to the GPU path

### Requirement: Eval resolves GPU device

`OnlineTemperatureEvaluator.__init__` SHALL resolve a `self.device` attribute using the same `cuda:n_gpu-1` pattern as `ppo/training.py`. When `torch.cuda.device_count() == 0`, `self.device` SHALL be `torch.device("cpu")`.

#### Scenario: Multi-GPU system reserves last GPU

- **WHEN** 2 GPUs are available and `reserve_training_gpu=True` was passed to VLLMFeatureExporter
- **THEN** `self.device` SHALL be `torch.device("cuda:1")`

#### Scenario: Single-GPU or CPU-only falls back

- **WHEN** no GPU is available (`torch.cuda.device_count() == 0`)
- **THEN** `self.device` SHALL be `torch.device("cpu")`
- **AND** `batch_build_segment_obs_from_lp` SHALL delegate to per-chain CPU calls

### Requirement: Eval postproc uses batched GPU call

`_evaluate_strategy_batch` SHALL collect logprob tensors from all active chains after `generate_with_features`, call `batch_build_segment_obs_from_lp` once, then distribute results back to `segment_obs[i][v]`. The per-chain `build_segment_obs_from_lp` loop SHALL be replaced.

#### Scenario: Round with 1500 active chains

- **WHEN** `generate_with_features` returns features for 1500 active chains
- **AND** all active chains have `f["logprobs"] is not None`
- **THEN** `batch_build_segment_obs_from_lp` SHALL be called once with all 1500 logprob tensors
- **AND** each resulting obs SHALL be assigned to the correct `segment_obs[i][v]` via `.tolist()`
