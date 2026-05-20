## ADDED Requirements

### Requirement: PPO supports concat pooling mode

`ppo/training.py` and `ppo/eval.py` SHALL read `data.segment_pooling` from config. When `segment_pooling == "concat"`, `obs_dim` SHALL be `instance_dim * segment_size` and `build_segment_obs_from_lp` SHALL receive `pooling_mode="concat"`.

#### Scenario: PPO training with concat pooling

- **WHEN** `data.segment_pooling` is `"concat"` in config
- **THEN** `train_ppo` SHALL set `obs_dim = instance_dim * segment_size`
- **AND** `build_segment_obs_from_lp` SHALL be called with `pooling_mode="concat"`

#### Scenario: PPO eval with concat pooling

- **WHEN** `data.segment_pooling` is `"concat"` in config
- **THEN** `OnlineTemperatureEvaluator` SHALL set `obs_dim = instance_dim * segment_size`
- **AND** `build_segment_obs_from_lp` SHALL be called with `pooling_mode="concat"`
