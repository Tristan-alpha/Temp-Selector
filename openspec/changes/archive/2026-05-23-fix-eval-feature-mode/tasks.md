## 1. Eval feature_mode support

- [x] 1.1 In `OnlineTemperatureEvaluator.__init__`, compute `self.hs_needed = self.feature_mode == "hidden_states"` after the existing `self.feature_mode` assignment
- [x] 1.2 In `_evaluate_strategy_batch`, pass `return_hidden=self.hs_needed` to `generate_with_features`
- [x] 1.3 In `_evaluate_strategy_batch`, construct `extra = [f["hidden_states"]] if f["hidden_states"] is not None else None` and pass `extra_parts=extra` and `include_topk=(not self.hs_needed)` to `build_segment_obs_from_lp`

## 2. Verification

- [x] 2.1 Run `python -m py_compile ppo/eval.py` to verify syntax
- [x] 2.2 Run `python -m pytest tests/ -v` — all existing tests must pass
- [x] 2.3 Manually verify: run with `base.yaml` (default `topk_logprobs` mode) — behavior must be unchanged
- [x] 2.4 Manually verify: run with `eval_hidden_states.yaml` — log message or dry-run confirms `return_hidden=True` path is taken
