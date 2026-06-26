## ADDED Requirements

### Requirement: generate_with_features exports hidden states in one pass

`VLLMFeatureExporter.generate_with_features` SHALL pass `extra_args={"kv_transfer_params": {"include_output_tokens": True}}` in `SamplingParams` so the connector saves hidden states for `all_token_ids[:-1]` (prompt + all generated tokens except the last) in a single `llm.generate()` call. It SHALL compute per-token logprobs directly from these hidden states without a second `llm.generate()` / `extract_from_ids` call.

#### Scenario: Hidden states cover all response tokens needed for logprobs

- **WHEN** `generate_with_features` is called with 64 `max_tokens` and `include_output_tokens=True`
- **THEN** the returned hidden states tensor SHALL have `hs_seq_len >= prompt_len + resp_len - 1`
- **AND** the slice `hs[p_len-1:][:n_resp]` SHALL yield exactly `n_resp` hidden state vectors, each mapping to one response-token logprob

#### Scenario: Logprobs computed from single-pass hidden states

- **WHEN** `generate_with_features` is called with `return_logprobs=True`
- **THEN** each returned dict SHALL include a `logprobs` tensor of shape `[n_resp, top_k+1]` computed from Pass 1 hidden states
- **AND** no second `llm.generate()` call SHALL be made

### Requirement: Async disk write is synchronized before reading

`generate_with_features` SHALL use `load_hidden_states` from `example_hidden_states_connector` to read hidden state files, blocking on the companion `.lock` file's `flock` until the async writer finishes. It SHALL use `cleanup_hidden_states` to delete both `.safetensors` and `.lock` files after reading.

#### Scenario: Hidden state file not yet written

- **WHEN** the async disk write is still in flight after `llm.generate()` returns
- **THEN** `load_hidden_states` SHALL block on `flock(LOCK_SH)` until the writer releases `flock(LOCK_EX)`
- **AND** no `FileNotFoundError` SHALL be raised

#### Scenario: Hidden state file and lock file cleaned up

- **WHEN** hidden states are successfully read
- **THEN** `cleanup_hidden_states` SHALL remove both the `.safetensors` file and its companion `.lock` file

### Requirement: VLLM_WORKER_MULTIPROC_METHOD set to spawn

`VLLMFeatureExporter._lazy_init` SHALL set `os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")` before initializing the LLM, preventing CUDA fork-safety errors when vLLM is used as a library.

#### Scenario: Engine core starts without CUDA fork error

- **WHEN** `VLLMFeatureExporter` is instantiated from a non-server Python script
- **THEN** the vLLM engine core SHALL start successfully without `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
