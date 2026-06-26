## ADDED Requirements

### Requirement: MIL training exports per-epoch metrics to JSONL

The MIL training loop SHALL append one JSON object per epoch to a file named
`{run_name}_mil_metrics.jsonl` in the configured log directory.

Each JSON object SHALL contain:
- `epoch` (int): epoch number, 1-indexed
- `loss` (float): mean BCEWithLogitsLoss over training batches
- `train_acc` (float): bag-level accuracy on the training set
- `val_acc` (float): bag-level accuracy on the validation set
- `val_acc_pos` (float): validation accuracy on positive bags (bags containing errors)
- `val_acc_neg` (float): validation accuracy on negative bags (fully correct)
- `grad_norm` (float): mean gradient L2 norm across training steps
- `attn_entropy` (float): mean attention entropy across training batches

#### Scenario: MIL metrics file is created alongside log

- **WHEN** MIL training starts epoch 1
- **THEN** the file `{log_dir}/{run_name}_mil_metrics.jsonl` is created (if it does not exist)
- **AND** one JSON line is appended after each epoch completes

#### Scenario: MIL metrics record train/val gap

- **WHEN** epoch 3 completes with train_acc=0.90 and val_acc=0.72
- **THEN** the JSONL row contains `"train_acc": 0.90, "val_acc": 0.72`
- **AND** the row also contains per-class val accuracy

### Requirement: PPO training exports per-iteration metrics to JSONL

The PPO training loop SHALL append one JSON object per iteration to a file named
`{run_name}_ppo_metrics.jsonl` in the configured log directory.

Each JSON object SHALL contain:
- `iter` (int): iteration number, 1-indexed
- `total_loss` (float): combined PPO loss (policy + value_coef * value - entropy_coef * entropy)
- `policy_loss` (float): clipped policy loss (mean)
- `value_loss` (float): value function MSE loss (mean)
- `entropy` (float): policy entropy (mean across mini-batches)
- `reward_mean` (float): mean terminal reward across training episodes
- `reward_pos_ratio` (float): fraction of episodes receiving +1 reward
- `train_acc` (float): majority-vote accuracy on training rollout
- `val_acc` (float): majority-vote accuracy on fixed validation set
- `temp_dist` (dict[str, int]): count of selections per temperature bin
- `temp_mean` (float): mean selected temperature
- `temp_std` (float): standard deviation of selected temperatures
- `segments_mean` (float): mean number of segments per chain
- `segments_min` (int): minimum segments in any chain
- `segments_max` (int): maximum segments in any chain
- `advantage_mean` (float): mean of GAE advantages
- `advantage_std` (float): standard deviation of GAE advantages
- `clip_fraction` (float): fraction of samples where PPO ratio exceeded [1-eps, 1+eps]
- `total_steps` (int): total PPO training steps in this iteration

#### Scenario: PPO metrics file captures temperature distribution

- **WHEN** iteration 5 completes with 480 total steps
- **THEN** the JSONL row contains `"total_steps": 480`
- **AND** `temp_dist` maps each temperature bin value to its selection count

#### Scenario: PPO metrics file captures advantage statistics

- **WHEN** iteration 3 completes
- **THEN** the JSONL row contains `advantage_mean` and `advantage_std` computed from the GAE advantages

### Requirement: JSONL metrics rows are atomic

Each metrics row SHALL be written as a single `json.dumps()` call followed by a newline.
The training loop SHALL open the file in append mode and flush after each write.

#### Scenario: Incomplete file recovered by plotting script

- **WHEN** a training run is killed mid-write
- **THEN** the plotting script SHALL ignore the last line if it fails JSON parsing
- **AND** render all preceding valid rows
