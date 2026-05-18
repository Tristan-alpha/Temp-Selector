## MODIFIED Requirements

### Requirement: generate_with_features method

`VLLMFeatureExporter` SHALL provide a `generate_with_features` method. The `OnlineTemperatureEvaluator` in `ppo/eval.py` SHALL use `VLLMFeatureExporter` instead of a raw `vllm.LLM` instance. The evaluator SHALL use `generate_with_features` for segment-by-segment generation and feature extraction instead of manual logprob parsing via `_extract_segment_obs`.

#### Scenario: PPO eval uses runner

- **WHEN** `OnlineTemperatureEvaluator` is constructed
- **THEN** it SHALL create a `VLLMFeatureExporter` with `reserve_training_gpu=True` instead of a raw `vllm.LLM`

#### Scenario: PPO eval uses generate_with_features

- **WHEN** `_evaluate_strategy_batch` generates a segment
- **THEN** it SHALL call `runner.generate_with_features` instead of `llm.generate` + `_extract_segment_obs`
