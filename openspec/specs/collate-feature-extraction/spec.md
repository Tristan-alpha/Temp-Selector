## ADDED Requirements

### Requirement: Unified extract_from_ids method

`VLLMFeatureExporter` SHALL provide a single `extract_from_ids` method. Online extraction SHALL always be active — `_lazy_init` SHALL always configure speculative decode. `make_collate_fn` SHALL always call `extract_from_ids` when an extractor is available. The method SHALL accept `return_logprobs: bool`, `return_hidden: bool`, and `device: torch.device | None` parameters.

#### Scenario: Always-on extraction

- **WHEN** `make_collate_fn` is called with an extractor
- **THEN** `extract_from_ids` SHALL be called to compute online features

### Requirement: Two feature modes

The system SHALL support exactly two feature modes. Both modes SHALL produce per-token feature vectors of exactly `instance_dim` dims:

- `topk_logprobs`: `[sampled_logprob, entropy, topk_logprob_0..topk_logprob_{top_k-1}]` (2 + top_k dims)
- `hidden_states`: `[sampled_logprob, entropy, hidden_0..hidden_{hidden_dim-1}]` (2 + hidden_dim dims)

Both modes SHALL use `build_segment_obs_from_lp` → `segment_pooling` as the single construction pipeline. `build_segment_obs_from_lp` SHALL accept an `include_topk: bool` parameter to control whether top-k logprobs are appended. The modes `basic` and `all` SHALL NOT exist.

#### Scenario: Valid feature modes

- **WHEN** `feature_mode` is set
- **THEN** it SHALL be either `"topk_logprobs"` or `"hidden_states"`

#### Scenario: topk_logprobs mode includes top-k logprobs directly

- **WHEN** `feature_mode` is `"topk_logprobs"` and `build_segment_obs_from_lp` is called with `include_topk=True`
- **THEN** the per-token vector SHALL include all top-k logprobs values
- **AND** no dimensions SHALL be zero-padded (2 + top_k = instance_dim)

#### Scenario: hidden_states mode includes logprobs and hidden

- **WHEN** `feature_mode` is `"hidden_states"`
- **THEN** `make_collate_fn` SHALL request both logprobs and hidden states from `extract_from_ids`
- **AND** segment vectors SHALL be built via `build_segment_obs_from_lp(lp, extra_parts=[hidden], include_topk=False)` — same composition as PPO

### Requirement: Warning on missing hidden states

The system SHALL log a warning via `logging.getLogger(__name__)` when `hidden_states_path` is `None` in the `llm.generate()` output, instead of silently returning zero tensors.

#### Scenario: hs_path is None

- **WHEN** `extract_from_ids` encounters a sample with `hs_path is None`
- **THEN** `logger.warning` SHALL be called before falling back to a zero tensor

### Requirement: make_collate_fn reads individual_label for bag labels

`make_collate_fn` SHALL construct the `label` tensor in its return dict from the `individual_label` field of each dataset row. The tensor key `"label"` in the return dict is unchanged (internal name), but the source field in the row dict SHALL be `individual_label`.

#### Scenario: Collate function label tensor construction

- **WHEN** `make_collate_fn` builds a batch dict
- **THEN** `batch["label"]` SHALL be a float tensor constructed from `row["individual_label"]` values
- **AND** missing `individual_label` SHALL default to `0.0`

### Requirement: build_segment_obs_from_lp accepts pooling_mode

`build_segment_obs_from_lp` SHALL accept a `pooling_mode: str = "mean"` parameter and SHALL forward it to `segment_pooling(mode=pooling_mode)`.

#### Scenario: concat pooling via config

- **WHEN** `data.segment_pooling` is `"concat"` and `make_collate_fn` calls `build_segment_obs_from_lp`
- **THEN** the pooling mode SHALL be forwarded to `segment_pooling(mode="concat")`
- **AND** the resulting instance tensor SHALL have shape `[K, segment_size * instance_dim]`

#### Scenario: concat drops incomplete last segment

- **WHEN** `pooling_mode == "concat"` and a segment has fewer than `segment_size` tokens
- **THEN** the segment SHALL be skipped (dropped) from the output
- **AND** the output SHALL contain only fully-filled segments

### Requirement: make_cached_collate_fn exists alongside make_collate_fn

`mil/utils.py` SHALL export `make_cached_collate_fn(segment_cache, instance_dim, train_device)`. This factory SHALL return a collate function that reads pre-computed segment tensors, labels, and temp indices from `segment_cache` (a list of dicts). The returned collate_fn SHALL pad instances to max K within the batch and output the same dict format as `make_collate_fn`.

#### Scenario: make_cached_collate_fn returns valid collate_fn

- **WHEN** `make_cached_collate_fn(cache, instance_dim=4098, train_device=device)` is called
- **THEN** it SHALL return a callable that accepts a list of row indices and returns `{instances, mask, label, temp_idx, _batch_tokens}`

## MODIFIED Requirements

### Requirement: VLLMFeatureExporter is the only extraction backend

All stages (MIL training, MIL eval, PPO training, build_dataset) SHALL use `VLLMFeatureExporter` for token feature extraction. No `backend` configuration key SHALL exist. No `if backend == ...` branching SHALL remain.

#### Scenario: MIL training uses vLLM

- **WHEN** `mil/training.py` creates an extraction engine
- **THEN** it creates a `VLLMFeatureExporter` directly without checking a `backend` config key

## REMOVED Requirements

### Requirement: SGLang as alternative backend

**Reason**: vLLM is faster for both generation and extraction; maintaining two backends adds unnecessary branching in every script.

**Migration**: Remove `backend: sglang` from configs. All SGLang-specific CLI args (`--backend`, multiple `parallel_size` mappings, `base_gpu_id`) are removed. `SGLangRunner` class and `_extract_segment_obs_sglang` function deleted.

### Requirement: extract_logprobs_from_ids

**Reason**: Merged into `extract_from_ids`.

**Migration**: Use `extract_from_ids(full_ids, prompt_lens, temperatures=temps, return_logprobs=True)["logprobs"]`.

### Requirement: extract_hidden_from_ids

**Reason**: Merged into `extract_from_ids`.

**Migration**: Use `extract_from_ids(full_ids, prompt_lens, return_hidden=True)["hidden"]`.

### Requirement: feature_mode basic

**Reason**: Fake logprobs (-20.0) in JSONL replaced by always-on online extraction.

**Migration**: Use `feature_mode: topk_logprobs` and `extract_from_ids(return_logprobs=True)`.

### Requirement: feature_mode all

**Reason**: Unused in any config. Both modes can be achieved separately.

**Migration**: Use appropriate mode (`topk_logprobs` or `hidden_states`) depending on which features are needed.

### Requirement: TokenBatchSampler

**Reason**: With pre-computed segment features, per-sample token counts are irrelevant for training batching. `token_batches()` is used for vLLM-aware batching during pre-computation and eval instead.

**Migration**: Replace `TokenBatchSampler(token_counts, max_tokens, shuffle=True)` with `DataLoader(SegmentCacheDataset(N), batch_size=N, shuffle=True, collate_fn=cached_collate)`. Remove the `TokenBatchSampler` class from `mil/utils.py`.
