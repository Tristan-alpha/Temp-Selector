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
