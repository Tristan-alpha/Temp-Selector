## 1. Hidden state extractor

- [x] 1.1 Create `inference/vllm_hidden_extractor.py` with `VLLMHiddenStateExtractor` class
- [x] 1.2 Initialize a second LLM instance with `speculative_config` (`extract_hidden_states` method, 1 speculative token, specified layer IDs)
- [x] 1.3 Implement `extract(prompts, responses)` method: concatenate → prefill with max_tokens=1 → read safetensors → slice response positions → return per-token hidden states
- [x] 1.4 Implement temp directory cleanup after extraction

## 2. Config

- [x] 2.1 Add `eagle_aux_hidden_state_layer_ids: [28]` to `configs/dataset.yaml` inference section  (Qwen3-8B last layer)

## 3. Integration with build_dataset

- [x] 3.1 In `scripts/build_dataset.py`, after vLLM generation, call hidden state extractor when `feature_mode` is `"all"`
- [x] 3.2 Map extracted hidden states back to `TokenFeature.hidden` for each token in each generated response
- [x] 3.3 Handle `num_votes > 1` — extract hidden states for each vote independently

## 4. Cleanup feature_mode dispatch

- [x] 4.1 Update `inference/vllm_runner.py`: `_build_feature_payload` dispatch — `basic`/`hidden_states` → no topk_logits; `topk_logits`/`all` → set topk_logits. `hidden` always `None` (populated by extractor in build_dataset).
- [x] 4.2 Update `inference/api_runner.py`: same dispatch

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 5.2 Run `python -m compileall -q` on all modified files
- [x] 5.3 Verify `configs/dataset.yaml` parses

## 6. Documentation

- [x] 6.1 Update PIPELINE.md: feature extraction section, two-pass explanation
- [x] 6.2 Update README.md `dataset.yaml` config description
