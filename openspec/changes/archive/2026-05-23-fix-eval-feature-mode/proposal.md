## Why

`ppo/eval.py` hardcodes logprob-based feature construction (`include_topk=True`, no `extra_parts`, no `return_hidden`) regardless of the configured `feature_mode`. When training uses `feature_mode: hidden_states`, the PPO policy learns a mapping from hidden-state features to temperature actions, but eval silently feeds logprob-based features instead — producing invalid results with no error.

## What Changes

- `OnlineTemperatureEvaluator` reads `self.feature_mode` (already stored on line 84) and passes `return_hidden` / `extra_parts` / `include_topk` to `generate_with_features` and `build_segment_obs_from_lp` exactly as `ppo/training.py` does
- **No config format changes** — `feature_mode` is already a supported config key

## Capabilities

### New Capabilities
- `eval-feature-mode`: `OnlineTemperatureEvaluator` SHALL respect the `feature_mode` config key, constructing segment observations with the same parameters as training

### Modified Capabilities
<!-- None: this is a bug fix, not a requirement change -->

## Impact

- `ppo/eval.py` — `__init__`, `_evaluate_strategy_batch` (the `generate_with_features` call and `build_segment_obs_from_lp` call)
