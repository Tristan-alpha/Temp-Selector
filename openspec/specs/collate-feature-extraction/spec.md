## ADDED Requirements

### Requirement: Collate function extracts features per training batch

The MIL training collate function SHALL, for each training batch, invoke the SGLangRunner to extract per-token logprob features (and optionally hidden states when `feature_mode` is `"hidden_states"` or `"all"`), then immediately compute per-segment instance vectors via `token_to_vec` and `segment_pooling`, and pad to a uniform batch tensor.

#### Scenario: Collate with logprob extraction (topk_logprobs mode)

- **WHEN** collate_fn receives a batch of rows with `feature_mode="topk_logprobs"` or `"all"`
- **THEN** it calls `extractor.extract_logprobs(prompts, responses, temperatures=[...])` with per-sample temperatures
- **AND** patches the returned logprob tensors into each row's token features
- **AND** builds segment-level instance vectors via `token_to_vec` → `build_segments` → `segment_pooling`
- **AND** returns a padded batch dict `{"instances": (B, max_K, D), "mask": (B, max_K), "label": (B,), "temp_idx": (B,)}`

#### Scenario: Collate with hidden state extraction (hidden_states mode)

- **WHEN** collate_fn receives a batch with `feature_mode="hidden_states"` or `"all"`
- **THEN** it calls `extractor.extract_hidden(prompts, responses)` for per-token hidden states
- **AND** patches the returned hidden state tensors into each row's token features
- **AND** instance vectors are built as above

#### Scenario: Collate with basic mode (no extraction)

- **WHEN** collate_fn receives a batch with `feature_mode="basic"`
- **THEN** no SGLang call is made; instance vectors are built directly from existing token features (logprob, entropy)

### Requirement: Collate function receives extractor via closure

The collate function SHALL receive the SGLangRunner instance through `make_collate_fn` so it can be shared across train and val DataLoaders without duplicating the engine.

#### Scenario: Shared extractor across loaders

- **WHEN** `train_mil()` creates one SGLangRunner and two DataLoaders (train + val)
- **THEN** both loaders use `make_collate_fn(extractor=runner, ...)` with the same runner instance
- **AND** the runner's engine is kept alive until training completes
