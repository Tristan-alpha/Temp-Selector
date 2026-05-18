## ADDED Requirements

### Requirement: Unified extract_from_ids method

`VLLMFeatureExporter` SHALL provide a single `extract_from_ids` method that replaces `extract_logprobs_from_ids` and `extract_hidden_from_ids`. The method SHALL accept `return_logprobs: bool`, `return_hidden: bool`, and `device: torch.device | None` parameters and SHALL make at most one `llm.generate()` call. Logprob computation SHALL be chunked (CHUNK_SIZE=1024) with per-chunk `apply_model` calls, concatenated on `device` (or CPU if `None`). The method SHALL return a dict with optional `"logprobs"` and `"hidden"` keys.

#### Scenario: Both logprobs and hidden requested with training GPU cat

- **WHEN** `extract_from_ids(full_ids, prompt_lens, temperatures=temps, return_logprobs=True, return_hidden=True, device=train_device)` is called
- **THEN** a single `llm.generate()` call SHALL be made, logprob chunks SHALL be cat'd on `train_device`, and the returned dict SHALL contain both `"logprobs"` and `"hidden"` tensors

#### Scenario: Only logprobs requested

- **WHEN** `return_logprobs=True, return_hidden=False`
- **THEN** the returned dict SHALL contain only `"logprobs"` key

#### Scenario: Only hidden requested

- **WHEN** `return_logprobs=False, return_hidden=True`
- **THEN** the returned dict SHALL contain only `"hidden"` key

### Requirement: Warning on missing hidden states

The system SHALL log a warning via `logging.getLogger(__name__)` when `hidden_states_path` is `None` in the `llm.generate()` output, instead of silently returning zero tensors.

#### Scenario: hs_path is None

- **WHEN** `extract_from_ids` encounters a sample with `hs_path is None`
- **THEN** `logger.warning` SHALL be called before falling back to a zero tensor

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
