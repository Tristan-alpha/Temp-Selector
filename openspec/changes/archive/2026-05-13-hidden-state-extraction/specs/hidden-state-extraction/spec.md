## ADDED Requirements

### Requirement: Two-pass hidden state extraction
`VLLMHiddenStateExtractor` SHALL take a list of (prompt, response) pairs and an existing vLLM `LLM` instance (or create one with `speculative_config`). For each pair, it SHALL:

1. Construct the full text as `prompt + response` (raw concatenation, no chat template)
2. Run `llm.generate(full_text, max_tokens=1)` with `extract_hidden_states` speculative config
3. Read hidden states from the safetensors file written to `kv_transfer_params["hidden_states_path"]`
4. Map the prompt-length offset to extract only the response-token hidden states
5. Return per-token hidden state vectors aligned with the response token positions

#### Scenario: Hidden states match response token count
- **WHEN** a prompt of 5 tokens generates a response of 10 tokens
- **THEN** the extractor returns 10 hidden state vectors (one per response token)

#### Scenario: Hidden states are non-empty
- **WHEN** a valid prompt+response pair is processed
- **THEN** each hidden state vector has length `hidden_size` (4096 for Qwen3-8B)

### Requirement: Four-tier feature_mode dispatch
The `feature_mode` config key SHALL accept `basic`, `topk_logits`, `hidden_states`, or `all`.

| mode | logprob | entropy | topk_logits | hidden_states | use case |
|---|---|---|---|---|---|
| `basic` | ✓ | ✓ | | | PPO training (fast) |
| `topk_logits` | ✓ | ✓ | ✓ | | current dataset gen |
| `hidden_states` | ✓ | ✓ | | ✓ | hidden state experiments |
| `all` | ✓ | ✓ | ✓ | ✓ | full dataset (large) |

#### Scenario: Dispatch table correct
- **WHEN** `feature_mode` is `hidden_states`
- **THEN** the two-pass extractor runs but topk_logits is not exported
- **WHEN** `feature_mode` is `all`
- **THEN** both topk_logits and hidden states are exported

### Requirement: Integration with build_dataset
`scripts/build_dataset.py` SHALL call the hidden state extractor after vLLM generation when config `feature_mode` is `"hidden_states"` or `"all"`. The extracted hidden states SHALL be populated into `TokenFeature.hidden` for each token.

#### Scenario: TokenFeature.hidden populated
- **WHEN** build_dataset runs with `feature_mode: all` or `feature_mode: hidden_states`
- **THEN** `TokenFeature.hidden` contains a list of floats for each generated token

### Requirement: Configurable layer selection
`configs/dataset.yaml` SHALL contain `eagle_aux_hidden_state_layer_ids` in the inference section, selecting which transformer layers' hidden states to extract. For the final hidden state (just before LM head), only the last layer SHALL be selected.

#### Scenario: Last layer only
- **WHEN** `eagle_aux_hidden_state_layer_ids: [28]` for Qwen3-8B (28 layers)
- **THEN** only the final layer's hidden state is extracted per token

### Requirement: Temp directory cleanup
Temporary safetensors files written by the extractor SHALL be cleaned up after hidden states are read.

#### Scenario: No temp files remain
- **WHEN** the hidden state extractor completes
- **THEN** the temporary directory used for safetensors storage no longer exists
