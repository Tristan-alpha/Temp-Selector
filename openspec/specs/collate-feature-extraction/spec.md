## ADDED Requirements

### Requirement: Unified extract_from_ids method

`VLLMFeatureExporter` SHALL provide a single `extract_from_ids` method. Online extraction SHALL always be active — `_lazy_init` SHALL always configure speculative decode. `make_collate_fn` SHALL always call `extract_from_ids` when an extractor is available. The method SHALL accept `return_logprobs: bool`, `return_hidden: bool`, and `device: torch.device | None` parameters.

#### Scenario: Always-on extraction

- **WHEN** `make_collate_fn` is called with an extractor
- **THEN** `extract_from_ids` SHALL be called to compute online features

### Requirement: Two feature modes

The system SHALL support exactly two feature modes: `topk_logprobs` (extracts logprob + entropy + top-k logprobs per token) and `hidden_states` (extracts hidden states per token). The modes `basic` and `all` SHALL NOT exist.

#### Scenario: Valid feature modes

- **WHEN** `feature_mode` is set
- **THEN** it SHALL be either `"topk_logprobs"` or `"hidden_states"`

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
