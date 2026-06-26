## ADDED Requirements

### Requirement: MIL model only outputs bag_logit and attention weights

`MILModel.forward()` SHALL return a dict containing at minimum `bag_logit` (shape `[B]`) and `attn_w` (shape `[B, K]`). It SHALL NOT include `inst_logit` or `encoder_out`. The model SHALL consist of `InstanceEncoder` → optional `PositionEncoding` → optional `BiGRU` → `AttentionAggregator` → `bag_head`.

#### Scenario: Forward pass on a bag of instances

- **WHEN** `MILModel(instances)` is called with `instances` of shape `[B, K, input_dim]`
- **THEN** the returned dict SHALL have key `bag_logit` with shape `[B]`
- **AND** the returned dict SHALL have key `attn_w` with shape `[B, K]`
- **AND** the returned dict SHALL NOT contain `inst_logit` or `encoder_out`

### Requirement: MIL training uses only bag_bce loss

`mil/training.py` SHALL train MIL using only binary cross-entropy loss on `bag_logit` against bag-level labels. The training loop SHALL NOT compute instance loss, temp_ce, or smoothness loss.

#### Scenario: Training step computes only bag_bce

- **WHEN** a training batch `(instances, labels, mask)` is processed
- **THEN** the loss SHALL be `BCE(sigmoid(bag_logit), labels)`
- **AND** no other loss terms SHALL contribute to the optimizer step

### Requirement: MIL evaluation uses bag-level metrics only

`mil/eval.py` SHALL evaluate MIL using bag-level metrics: AUC, calibration, bag accuracy, confusion matrix. Attention metrics (entropy, top3_mass, effective_n) SHALL still be reported for interpretability. Instance-level metrics SHALL NOT be computed. Temperature accuracy metrics SHALL NOT be computed.

#### Scenario: Evaluation outputs bag metrics

- **WHEN** `evaluate_mil` runs on a validation dataset
- **THEN** the results SHALL include `bag_auc`, `bag_accuracy`, `bag_calibration`
- **AND** the results SHALL include `attn_entropy`, `attn_top3_mass`, `attn_effective_n`
- **AND** the results SHALL NOT include instance AUC or temperature accuracy

### Requirement: PPO uses attention-based credit assignment

`ppo/training.py` SHALL compute intermediate-step rewards by accumulating all round segments of a chain into a full bag, calling `mil_model(full_bag)` once per chain, and distributing `terminal_reward × attention_weight` to each step. The shaping computation SHALL occur during PPO batch construction (post-rollout), not during rollout.

#### Scenario: Attention-based reward for a 3-step chain

- **WHEN** a chain has 3 steps and the terminal majority-vote result is correct
- **THEN** `full_bag = torch.stack(ep_obs[i][v][1:])` SHALL produce shape `[3, obs_dim]`
- **AND** `mil_model(full_bag.unsqueeze(0))["attn_w"]` SHALL return attention weights of shape `[1, 3]`
- **AND** step t's reward SHALL be `shaping_coef × 1.0 × attn_w[0, t-1]` for non-terminal steps
- **AND** the terminal step SHALL receive `reward = 1.0`
