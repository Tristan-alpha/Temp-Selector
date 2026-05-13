## ADDED Requirements

### Requirement: Hidden states stored in safetensors sidecar

When building a dataset with `feature_mode` set to `"hidden_states"` or `"all"`, the system SHALL write hidden state vectors to a companion safetensors file (`<dataset>.jsonl.hidden.safetensors`) as a single tensor named `"hidden_states"` of shape `[total_tokens, hidden_dim]` in the model's native dtype (typically bf16), instead of embedding them inline in JSONL rows.

Each JSONL row SHALL include `_hidden_offset` (starting token index into the tensor) and `_hidden_count` (number of tokens with hidden states for this row).

#### Scenario: Dataset built with hidden_states feature mode

- **WHEN** `scripts/build_dataset.py` runs with `feature_mode: "hidden_states"` or `"all"`
- **THEN** the output dataset consists of a JSONL file where token features have `"hidden": null` and a companion `.hidden.safetensors` file containing all hidden vectors in native dtype (bf16)

#### Scenario: Dataset built without hidden states

- **WHEN** `scripts/build_dataset.py` runs with `feature_mode: "basic"` or `"topk_logits"`
- **THEN** no `.hidden.safetensors` file is created and JSONL rows do not contain `_hidden_offset` or `_hidden_count` keys

### Requirement: BagDataset loads hidden states from sidecar via mmap

When initializing `BagDataset` from a dataset path, the system SHALL check for a `.hidden.safetensors` sidecar file. If present, it SHALL open the file via `with safetensors.safe_open(...) as f:`, use `f.get_slice("hidden_states")` for mmap'd access without loading the full tensor into memory, and patch each row's token feature dicts with hidden vectors using the `_hidden_offset` and `_hidden_count` indices. The processing loop SHALL be inside the `with` block so the mmap is released after all rows are processed. If the sidecar is absent, the system SHALL proceed without patching (existing behavior).

#### Scenario: Dataset has hidden sidecar

- **WHEN** `BagDataset` loads a dataset where `<path>.hidden.safetensors` exists and rows have valid `_hidden_offset`/`_hidden_count`
- **THEN** each token feature dict in the row has its `"hidden"` key populated from the corresponding slice of the safetensors tensor

#### Scenario: Dataset has no hidden sidecar (legacy or basic mode)

- **WHEN** `BagDataset` loads a dataset without a `.hidden.safetensors` file
- **THEN** token feature dicts are processed as-is (with `hidden` already present in JSONL for legacy, or `None` for basic mode)

#### Scenario: Safetensors file is corrupted or unreadable

- **WHEN** `BagDataset` attempts to load a `.hidden.safetensors` file that is malformed
- **THEN** `read_hidden_tensor` returns None and the dataset loads without hidden states (graceful degradation)

#### Scenario: Lazy mmap access — only requested rows are read

- **WHEN** `BagDataset.__init__` opens a `.hidden.safetensors` sidecar via `with safe_open(...) as f:` and uses `f.get_slice("hidden_states")`
- **THEN** only the token ranges accessed via `hs[offset:offset+count, :]` are paged into physical memory, and the mmap is released when the `with` block exits

### Requirement: Build and split merged for hidden_state datasets

When building a dataset with `feature_mode` set to `"hidden_states"` or `"all"`, the system SHALL perform group-aware splitting in-memory after hidden state extraction and write train/val/test JSONL + safetensors sidecars directly, without creating an intermediate "all" dataset safetensors file.

#### Scenario: Build with hidden states writes three splits directly

- **WHEN** `scripts/build_dataset.py` runs with `feature_mode: "hidden_states"` and train/val/test output paths configured
- **THEN** three JSONL files + three `.hidden.safetensors` sidecars are written directly, with no intermediate all_dataset safetensors file

#### Scenario: Build without hidden states retains existing all → split flow

- **WHEN** `scripts/build_dataset.py` runs with `feature_mode: "basic"` or `"topk_logits"`
- **THEN** a single `all_dataset.jsonl` is written and `scripts/split_jsonl.py` is used separately (existing behavior, no sidecar involved)

### Requirement: Split and subsample propagate hidden sidecar

When splitting or subsampling a dataset that has a hidden sidecar, the system SHALL slice the source safetensors tensor and write new sidecar files for each output split, using the `_hidden_offset`/`_hidden_count` of rows assigned to each output.

#### Scenario: Splitting a dataset with hidden states

- **WHEN** `scripts/split_jsonl.py` splits a dataset with a `.hidden.safetensors` sidecar into train/val/test
- **THEN** each output split has its own `.hidden.safetensors` file containing only the hidden vectors for rows in that split, with correct `_hidden_offset` indices

#### Scenario: Subsampling a dataset with hidden states

- **WHEN** `scripts/subsample_jsonl.py` creates a subset of a dataset with a `.hidden.safetensors` sidecar
- **THEN** the output has a `.hidden.safetensors` file containing only the hidden vectors for the sampled rows

#### Scenario: Splitting a dataset without hidden states

- **WHEN** `scripts/split_jsonl.py` splits a dataset with no `.hidden.safetensors` sidecar
- **THEN** no sidecar files are created for the outputs (no-op)

### Requirement: Native dtype precision preserved

Hidden state float values stored in and loaded from the safetensors sidecar SHALL be bitwise-identical to the original tensor values produced by vLLM, without intermediate conversion through Python float64 or JSON decimal strings.

#### Scenario: Round-trip precision (bf16)

- **WHEN** hidden state tensors (bf16) are written to safetensors and then read back
- **THEN** each value is identical to the original — no `.tolist()` (→ float64) or JSON string conversion occurs
