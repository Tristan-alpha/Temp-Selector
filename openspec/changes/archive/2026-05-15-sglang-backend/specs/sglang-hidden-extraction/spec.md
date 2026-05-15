## ADDED Requirements

### Requirement: SGLang engine extracts hidden states during generation

When `feature_mode` is `"hidden_states"` or `"all"`, the SGLang backend SHALL call `engine.generate(prompt, sampling_params, return_hidden_states=True)` and extract per-token hidden states from `output.meta_info["hidden_states"]`.

The extracted hidden states SHALL be returned as `List[torch.Tensor]`, each of shape `[n_response_tokens, hidden_dim]` in the model's native dtype (bf16), matching the existing vLLM extractor's interface.

#### Scenario: Basic generation with hidden states

- **WHEN** `SGLangFeatureExporter.export_token_features_multi_temp()` is called with `feature_mode="hidden_states"`
- **THEN** the returned payloads contain hidden state tensors for each generated token

#### Scenario: Hidden state shape matches response tokens

- **WHEN** a response has `N` tokens
- **THEN** the corresponding hidden state tensor has shape `[N, hidden_dim]`

### Requirement: Single engine for generation and hidden extraction

The SGLang backend SHALL use a single `sglang.Engine` instance for both text generation and hidden state extraction. No second engine instance, speculative config, or multi-process orchestration SHALL be required.

#### Scenario: No second LLM instance needed

- **WHEN** PPO training runs with SGLang backend and `feature_mode="all"`
- **THEN** only one engine instance exists, and hidden states are available directly from generation outputs

### Requirement: Backend selection via CLI and config

The system SHALL support `--backend sglang` (default) and `--backend vllm` (legacy) for both `scripts/build_dataset.py` and `ppo/training.py`. Config SHALL include `inference.backend: sglang`.

#### Scenario: Default backend is SGLang

- **WHEN** `python scripts/build_dataset.py --config configs/dataset.yaml` is run without `--backend`
- **THEN** SGLang is used as the inference backend

#### Scenario: Explicit vLLM backend

- **WHEN** `--backend vllm` is passed
- **THEN** the legacy vLLM backend is used
