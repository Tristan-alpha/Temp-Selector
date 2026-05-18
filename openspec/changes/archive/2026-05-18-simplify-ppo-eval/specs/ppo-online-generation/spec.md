## ADDED Requirements

### Requirement: Shared feature construction helper

`features/segmenter.py` SHALL provide a `build_segment_obs_from_lp` helper that converts `generate_with_features` output (logprob tensor, token_ids, tokens, text) into a segment observation vector via segment pooling. Both `ppo/training.py` and `ppo/eval.py` SHALL use this helper instead of duplicating the feature construction logic.

#### Scenario: Training and eval use the same helper

- **WHEN** constructing segment observations from `generate_with_features` output
- **THEN** both PPO training and PPO eval SHALL call `build_segment_obs_from_lp`

### Requirement: PPO eval uses runner for prompt rendering

`OnlineTemperatureEvaluator` SHALL use `runner.build_math_messages()` and `runner.render_messages()` instead of its own `_render_prompt` method. The `_render_prompt` method SHALL be deleted.

#### Scenario: Prompt rendering via runner

- **WHEN** `_evaluate_strategy_batch` renders a prompt
- **THEN** it SHALL call `self.runner.build_math_messages(question)` followed by `self.runner.render_messages(messages)`
