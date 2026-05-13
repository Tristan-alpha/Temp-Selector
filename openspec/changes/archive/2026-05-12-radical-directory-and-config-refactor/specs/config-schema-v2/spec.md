## ADDED Requirements

### Requirement: Config has data section with shared parameters
The YAML config SHALL contain a top-level `data` section with `instance_dim` (int) and `temp_bins` (list of floats), shared by MIL and PPO pipelines.

#### Scenario: data section contains instance_dim and temp_bins
- **WHEN** loading the config file
- **THEN** `config["data"]["instance_dim"]` returns an integer and `config["data"]["temp_bins"]` returns a list of 15 floats

### Requirement: Config has mil section with model and training sub-sections
The YAML config SHALL contain `mil.model` (architecture) and `mil.training` (hyperparameters). Neither SHALL contain a `use_ddp` key.

#### Scenario: mil.model contains architecture parameters
- **WHEN** loading the config file
- **THEN** `config["mil"]["model"]["hidden_dim"]` returns 256, `config["mil"]["model"]["aggregator"]` returns "attention"

#### Scenario: mil.training does not contain use_ddp
- **WHEN** loading the config file
- **THEN** `config["mil"]["training"].get("use_ddp")` returns None

### Requirement: Config has ppo section with model and training sub-sections
The YAML config SHALL contain `ppo.model` (architecture, with `hidden_dim` replacing old `policy_hidden_dim`) and `ppo.training`. Neither SHALL contain `use_ddp` or `policy_hidden_dim`.

#### Scenario: ppo.model uses hidden_dim, not policy_hidden_dim
- **WHEN** loading the config file
- **THEN** `config["ppo"]["model"]["hidden_dim"]` returns an integer and `config["ppo"]["model"].get("policy_hidden_dim")` returns None

#### Scenario: ppo section does not contain use_ddp
- **WHEN** loading the config file
- **THEN** `config["ppo"].get("use_ddp")` returns None and `config["ppo"]["training"].get("use_ddp")` returns None

### Requirement: No legacy model or training sections
The YAML config SHALL NOT contain top-level `model` or `training` sections.

#### Scenario: Legacy sections absent
- **WHEN** loading the config file
- **THEN** `config.get("model")` returns None and `config.get("training")` returns None

### Requirement: All 5 config files follow the same schema
All config files (`base.yaml`, `arch_mlp_only.yaml`, `baseline_fixed_window.yaml`, `pool_concat.yaml`, `temp_heads.yaml`) SHALL use the new key structure. Each SHALL differ from base only in the experimental variable it tests.

#### Scenario: All configs are valid YAML with new schema
- **WHEN** loading each of the 5 config files
- **THEN** all contain `data.instance_dim`, `mil.model`, `mil.training`, `ppo.model`, `ppo.training` sections and do NOT contain top-level `model` or `training`
