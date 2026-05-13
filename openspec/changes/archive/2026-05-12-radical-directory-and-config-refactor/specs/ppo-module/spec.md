## ADDED Requirements

### Requirement: PPO module is self-contained under ppo/
The PPO policy/value network, GAE computation, MIL warm-start loader, online training loop, and online evaluation logic SHALL reside under a single `ppo/` top-level directory.

#### Scenario: PPO model is importable from ppo.model
- **WHEN** code executes `from ppo.model import PolicyValueNet, compute_gae, sample_action, load_mil_encoder_for_warmstart`
- **THEN** all four symbols are successfully imported

#### Scenario: PPO training is importable from ppo.training
- **WHEN** code executes `from ppo.training import train_ppo`
- **THEN** the function is successfully imported

#### Scenario: PPO evaluation is importable from ppo.eval
- **WHEN** code executes `from ppo.eval import OnlineTemperatureEvaluator`
- **THEN** the class is successfully imported

#### Scenario: PPO training is runnable as a module
- **WHEN** user executes `python -m ppo.training --config configs/base.yaml`
- **THEN** the PPO training loop starts without import errors

### Requirement: PPO module has no dependency on legacy paths
None of the files under `ppo/` SHALL import from `models.*`, `training.*`, `rl.*`, or `eval.*` paths.

#### Scenario: All ppo/ imports use current paths
- **WHEN** grep for `from models\.` or `from training\.` or `from rl\.` or `from eval\.` in `ppo/`
- **THEN** no matches are found

### Requirement: PPO uses shared vectorizer utilities
`ppo/training.py` and `ppo/eval.py` SHALL import feature utilities from `features.vectorizer` and SHALL NOT contain local copies.

#### Scenario: No duplicate token_to_vec or compute_entropy in ppo/
- **WHEN** searching for `def token_to_vec` or `def _compute_entropy` or `def compute_entropy_from_logprobs` in `ppo/`
- **THEN** no definition is found

### Requirement: PPO uses ep_correct variable name
The PPO training code SHALL use `ep_correct` (1=correct, 0=wrong) instead of `ep_labels` to distinguish from the MIL convention (0=correct, 1=error).

#### Scenario: No ep_labels in ppo/
- **WHEN** searching for `ep_labels` in `ppo/`
- **THEN** no matches are found

### Requirement: PPO eval reads correct config key for hidden_dim
`ppo/eval.py` SHALL read PolicyValueNet hidden_dim from `config["ppo"]["model"]["hidden_dim"]`, not from `config["model"]["hidden_dim"]`.

#### Scenario: Correct config key path
- **WHEN** reading `ppo/eval.py`
- **THEN** hidden_dim is read from `config["ppo"]["model"]["hidden_dim"]`
