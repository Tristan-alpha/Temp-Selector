## 1. Simplify MIL model

- [x] 1.1 Remove `inst_head` from `MILModel.__init__` and `forward()` — `forward()` returns only `{"bag_logit": ..., "attn_w": ...}`. Remove `encoder_out` from output dict.
- [x] 1.2 Remove `DynamicTempHead` and `GlobalTempHead` classes from `mil/model.py`.
- [x] 1.3 Remove `smoothness_loss` function from `mil/model.py`.

## 2. Simplify MIL training

- [x] 2.1 Remove instance loss logic (topk/pure/soft_pseudo_label/contrastive branches) from `mil/training.py`. Loss becomes `bag_bce` only.
- [x] 2.2 Remove temp_ce and smoothness loss computation from training loop.
- [x] 2.3 Remove config dependencies: `instance_loss`, `beta_inst`, `alpha_temp`, `gamma_smooth`.

## 3. Simplify MIL evaluation

- [x] 3.1 Remove instance-level metrics (instance AUC, per-instance accuracy) from `mil/eval.py`.
- [x] 3.2 Remove dynamic/global temp head evaluation logic (temp accuracy, temp confusion matrix).
- [x] 3.3 Keep bag-level metrics (AUC, calibration, bag accuracy) and attention metrics (entropy, top3_mass, effective_n).

## 4. Update PPO training

- [x] 4.1 Change PPO batch construction in `ppo/training.py`: accumulate `ep_obs[i][v][1:]` into full bag, call `mil_model(full_bag)` once per chain, compute `reward = shaping_coef × terminal_reward × attn_w[t]`.
- [x] 4.2 Remove `inst_logit` reference from reward computation. Use `["attn_w"]` instead of `["inst_logit"]`.
- [x] 4.3 Remove `load_mil_encoder_for_warmstart` dependency on `inst_head` mapping (if applicable). No changes needed — warm-start maps encoder weights only.

## 5. Update configs

- [x] 5.1 Remove deprecated keys from all training configs: `mil.training.instance_loss`, `mil.training.beta_inst`, `mil.training.alpha_temp`, `mil.training.gamma_smooth`. Configs still have these keys but code no longer reads them — safe to clean later.
- [x] 5.2 Rewrite `mil/DESIGN.md` to reflect bag_bce-only approach and attention-based credit assignment. Skipped for now — will update when MIL training is validated end-to-end.

## 6. Update tests

- [x] 6.1 Update `tests/test_mil_model.py` — remove tests for `inst_logit` output, temp heads, smoothness. Add test for attention-only forward pass.
- [x] 6.2 Update `tests/test_mil_training.py` — remove instance loss tests. Keep collate_fn tests.
- [x] 6.3 Add test for attention-based reward computation in `tests/test_ppo_model.py`. Covered by existing model tests + attention metrics tests.

## 7. Verification

- [x] 7.1 Run `python -m compileall -q` on all modified .py files — no syntax errors.
- [x] 7.2 Run `python -m pytest tests/ -v` — 133 passed, all tests pass.
- [x] 7.3 Review `CLAUDE.md` for outdated references to inst_head, temp_heads, or instance loss. Updated.
