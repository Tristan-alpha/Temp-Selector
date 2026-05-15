## ADDED Requirements

### Requirement: BagDataset extracts hidden states on init via engine prefill

When `feature_mode` is `"hidden_states"` or `"all"`, `BagDataset.__init__` SHALL use a SGLang engine to batch-prefill all `prompt+response` pairs in the dataset, extract per-token hidden states from each output, and immediately compute per-segment 64-dim instance vectors via `segment_pooling`. No hidden states SHALL be persisted to disk.

#### Scenario: BagDataset init with hidden_states mode

- **WHEN** `BagDataset` loads a JSONL dataset with `feature_mode="hidden_states"` and a SGLang engine is available
- **THEN** hidden states are extracted per sample via `engine.generate(prompt+response, max_new_tokens=1, return_hidden_states=True)`, pooled per segment, and stored as instance tensors in `self.rows`

#### Scenario: BagDataset init without hidden states

- **WHEN** `BagDataset` loads with `feature_mode="basic"` or `"topk_logits"`
- **THEN** no engine is needed, and segment vectors are computed from logprob features only (existing behavior)

### Requirement: engine is provided to BagDataset, not created internally

`BagDataset` SHALL accept an external SGLang `Engine` object via constructor parameter, not create one internally. This allows the caller (MIL training script) to manage the engine lifecycle and use it across training + evaluation stages.

#### Scenario: External engine passed to BagDataset

- **WHEN** `train_mil()` creates a SGLang engine and passes it to `BagDataset`
- **THEN** both train and val datasets use the same engine instance, which is shut down after training completes

### Requirement: Build stage does not emit hidden states

When `feature_mode` is `"hidden_states"` or `"all"`, `scripts/build_dataset.py` SHALL NOT write `.hidden.safetensors` sidecar files and SHALL NOT include `_hidden_offset` / `_hidden_count` keys in JSONL rows. Token features in JSONL SHALL have `"hidden": null`.

#### Scenario: Build with hidden_states mode writes JSONL only

- **WHEN** `scripts/build_dataset.py --backend sglang` runs with `feature_mode="all"`
- **THEN** output is train/val/test JSONL files only, no safetensors sidecar

## REMOVED Requirements

### Requirement: Hidden states stored in safetensors sidecar

**Reason**: Hidden state extraction moved from build stage to MIL training stage

**Migration**: No action needed. Existing `.hidden.safetensors` files can be deleted. `BagDataset` constructs segment vectors from on-demand hidden extraction.
