## ADDED Requirements

### Requirement: Top-k MIL instance loss has explanatory docstring
The top-k instance loss logic in `mil/training.py` (the `for i in range(y.size(0))` loop) SHALL have a comment or docstring explaining: the MIL annotation dilemma (bag-level labels, unknown instance-level ground truth), why all-error labeling fails, what top-k achieves, the choice of k=max(1, n_valid//3), and the different treatment of positive vs negative bags.

#### Scenario: Docstring explains top-k rationale
- **WHEN** reading the instance loss section of mil/training.py
- **THEN** the code includes a comment explaining why top-k is used and how k is chosen

### Requirement: MILModel.forward has explanatory docstring
`MILModel.forward()` in `mil/model.py` SHALL have a docstring tracing the feature flow through encoder, position encoding, BiGRU, aggregator, bag_head, and inst_head, explaining the semantic meaning of each output key.

#### Scenario: Forward docstring present
- **WHEN** reading MILModel.forward in mil/model.py
- **THEN** a docstring explains the tensor shapes and semantics at each stage

### Requirement: First-segment dummy values have explanatory comment
The first-segment branch in `ppo/training.py` (where `segment_obs[i] is None`) SHALL have a comment explaining why there is no observation, why dummy values are recorded, and why they are safely skipped during PPO batch construction.

#### Scenario: Comment present
- **WHEN** reading the first-segment branch in ppo/training.py
- **THEN** a comment already exists (preserved from prior Bug 5 fix)

### Requirement: PPO batch construction has explanatory comment
The PPO batch construction loop in `ppo/training.py` SHALL have a comment explaining the done flag, terminal reward assignment, and the skipping of the first step (range(1, n_steps)).

#### Scenario: Batch construction comment present
- **WHEN** reading the batch construction loop in ppo/training.py
- **THEN** a comment explains the done flag and reward logic

### Requirement: compute_gae has explanatory docstring
`compute_gae` in `ppo/model.py` SHALL have a short docstring explaining the GAE formula, the role of the done mask at episode boundaries, and the advantage standardization step.

#### Scenario: compute_gae docstring present
- **WHEN** reading compute_gae in ppo/model.py
- **THEN** a docstring explains the algorithm

### Requirement: segment_pooling has explanatory docstring
`segment_pooling` in `features/segmenter.py` SHALL expand its existing docstring to explain the mean vs concat choice, the clamping behavior for boundary spans, and the fallback for empty output.

#### Scenario: Docstring expanded
- **WHEN** reading segment_pooling in features/segmenter.py
- **THEN** the docstring covers mode choices and edge case behavior

### Requirement: BagSample docstring explains label semantics
The `BagSample` dataclass in `features/schema.py` SHALL have a comment noting the label semantics (0=correct/negative bag, 1=error/positive bag — flipped from conventional MIL).

#### Scenario: Label semantics documented
- **WHEN** reading the BagSample class in features/schema.py
- **THEN** a comment notes the flipped label convention

### Requirement: token_to_vec docstring explains feature merging
`token_to_vec` in `features/vectorizer.py` SHALL have a docstring explaining the merge order (logprob, entropy, topk_logits, hidden), default values, and the padding/truncation strategy.

#### Scenario: Docstring present
- **WHEN** reading token_to_vec in features/vectorizer.py
- **THEN** a docstring explains the merge strategy and defaults
