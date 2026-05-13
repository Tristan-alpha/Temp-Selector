## ADDED Requirements

### Requirement: PPO training reads prompts from train_dataset
`ppo/training.py` SHALL extract unique (question, answer) pairs from `paths.train_dataset` (a labeled BagSample JSONL) rather than from `paths.raw_input`. This is symmetric with the `ppo/eval.py` fix and ensures PPO online training only samples from the train split.

#### Scenario: Train prompts from train_dataset
- **WHEN** `train_ppo` starts
- **THEN** prompts are loaded from `paths.train_dataset` via `load_train_prompts()`, not from `raw_input`

#### Scenario: No raw_input dependency in PPO training
- **WHEN** reading `ppo/training.py` main()
- **THEN** there is no reference to `paths.raw_input` or `cfg["paths"]["raw_input"]`

### Requirement: load_train_prompts deduplicates by sample_prefix
The helper `load_train_prompts(dataset_path)` SHALL extract unique (question, answer) pairs using `sample_prefix` to deduplicate, identical to `load_test_prompts` in `ppo/eval.py`.

#### Scenario: Dedup works
- **WHEN** `load_train_prompts` processes the labeled train JSONL
- **THEN** each unique prompt appears exactly once in the returned list

### Requirement: raw_input only used by Stage 1
After this change, `paths.raw_input` SHALL only be read by `features/build_dataset.py`. PPO training and evaluation SHALL NOT depend on it.

#### Scenario: raw_input consumers
- **WHEN** grep for `raw_input` in the codebase
- **THEN** matches appear only in `features/build_dataset.py`, config files, and `run_pipeline.sh` env var defaults
