## 1. segment_pooling concat zero-padding

- [x] 1.1 Modify `segment_pooling` in `features/segmenter.py`: replace `continue` in concat mode with zero-padding to `segment_size × obs_dim`. Truncate segments exceeding `segment_size` to the first `segment_size` tokens.
- [x] 1.2 Update `test_segment_pooling_concat_padding` in `tests/test_segmenter.py`: the test currently expects an all-zero output for dropped segments. Change to verify real features are preserved in the first `n_tokens × obs_dim` elements and the remainder are zeros.
- [x] 1.3 Add `test_segment_pooling_concat_truncates_long` in `tests/test_segmenter.py`: verify that a segment with more than `segment_size` tokens is truncated to `segment_size` before reshaping.
- [x] 1.4 Update `test_segment_pooling_concat_padding_mixed` in `tests/test_segmenter.py`: verify content (not just shape) of surviving segments after the padding change.

## 2. PPO distributed terminal reward

- [x] 2.1 Modify reward construction in `ppo/training.py`: for each chain, if `mil_model is not None`, compute `attn_weights` via MIL, L1-normalize, and distribute `terminal_reward` across ALL steps (including the final step). If `mil_model is None`, distribute uniformly.
- [x] 2.2 Remove `shaping_coef` from config loading in `ppo/training.py` (lines 162 and 360).
- [x] 2.3 Remove `shaping_coef` key from all training configs: `base.yaml`, `hidden_states.yaml`, `pool_concat.yaml`, `arch_mlp_only.yaml`, `instance_contrastive.yaml`, `instance_soft_pseudo_label.yaml`, `temp_heads.yaml`.
- [x] 2.4 Update PPO reward tests in `tests/test_ppo_model.py` if any test references `shaping_coef` or the old reward scheme. Add a test for the distributed reward formula.

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — all tests must pass.
- [x] 3.2 Run `python -m compileall -q features/segmenter.py ppo/training.py` to catch syntax errors.
- [x] 3.3 Update `ppo/DESIGN.md` — reward section now describes attention-weighted distribution instead of shaping coefficient.
- [x] 3.4 Update `PIPELINE.md` — remove references to `shaping_coef` from config documentation.
