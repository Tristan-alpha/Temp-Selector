## ADDED Requirements

### Requirement: mil.training.instance_loss config key
The config key `mil.training.instance_loss` SHALL accept one of `"topk"`, `"pure"`, `"soft_pseudo_label"`, or `"contrastive"`, defaulting to `"pure"`.

#### Scenario: Valid values accepted
- **WHEN** loading config with `mil.training.instance_loss: pure`
- **THEN** training selects the k=1 loss path

#### Scenario: Unknown value raises error
- **WHEN** loading config with `mil.training.instance_loss: unknown`
- **THEN** training raises ValueError

### Requirement: Pure MIL (k=1) instance loss
When `instance_loss` is `"pure"`, positive bags SHALL push only the single highest-scoring instance toward target=1 (k=1), per the standard MIL assumption.

#### Scenario: k=1 for positive bag
- **WHEN** a positive bag has 9 instances
- **THEN** only the instance with the highest `inst_logit` gets target=1; all others get target=0

### Requirement: Soft pseudo-label instance loss
When `instance_loss` is `"soft_pseudo_label"`, positive bag targets SHALL be `sigmoid(inst_logit).detach()` with a SeLa-MIL-inspired constraint: if no soft target exceeds 0.5, the highest-scoring instance's target is clamped to 0.5.

#### Scenario: Soft targets from model output
- **WHEN** a positive bag has instances with inst_logit [5.0, -2.0, -3.0]
- **THEN** targets are approximately [0.99, 0.12, 0.05] (detached)

#### Scenario: Anti-degeneration safety net
- **WHEN** a positive bag has all inst_logit < 0 (all sigmoids < 0.5)
- **THEN** the highest-scoring instance's target is clamped to exactly 0.5

### Requirement: Contrastive instance loss
When `instance_loss` is `"contrastive"`, positive bags SHALL use a contrastive objective: `logsumexp(scores) - max(scores)`, encouraging one instance to stand out above all others. Negative bags SHALL use MSE: `scores.pow(2).mean()`.

#### Scenario: Contrastive loss for positive bag
- **WHEN** a positive bag has inst_logit [5.0, -2.0, -3.0]
- **THEN** loss is computed as `logsumexp(scores) - max(scores)`

#### Scenario: MSE for negative bag
- **WHEN** a negative bag has inst_logit scores
- **THEN** loss is `scores.pow(2).mean()` (all scores pushed toward 0)

### Requirement: Legacy topk preserved
The current `topk` method with `k = max(1, n_valid // 3)` SHALL remain available as a config-selectable option.

#### Scenario: topk still works
- **WHEN** `mil.training.instance_loss: topk`
- **THEN** behavior is identical to the pre-change implementation

### Requirement: Ablation configs
Two new config files SHALL exist: `configs/instance_soft_pseudo_label.yaml` and `configs/instance_contrastive.yaml`, each identical to `base.yaml` except for `mil.training.instance_loss` and dedicated output paths.

#### Scenario: Soft pseudo-label config loads
- **WHEN** loading `configs/instance_soft_pseudo_label.yaml`
- **THEN** `mil.training.instance_loss` is `"soft_pseudo_label"`

#### Scenario: Contrastive config loads
- **WHEN** loading `configs/instance_contrastive.yaml`
- **THEN** `mil.training.instance_loss` is `"contrastive"`
