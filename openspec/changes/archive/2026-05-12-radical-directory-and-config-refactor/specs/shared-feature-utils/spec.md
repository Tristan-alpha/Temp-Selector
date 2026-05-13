## ADDED Requirements

### Requirement: Feature vector utilities live in features.vectorizer
The functions `token_to_vec`, `token_to_obs`, `mean_pool_obs`, and `compute_entropy` SHALL be defined exactly once, at `features/vectorizer.py`. All consumers SHALL import from `features.vectorizer`.

#### Scenario: All four functions importable from features.vectorizer
- **WHEN** executing `from features.vectorizer import token_to_vec, token_to_obs, mean_pool_obs, compute_entropy`
- **THEN** all four imports succeed

#### Scenario: No duplicate definitions in mil/
- **WHEN** searching for `def token_to_vec` or `def _compute_entropy` in `mil/`
- **THEN** no definition is found

#### Scenario: No duplicate definitions in ppo/
- **WHEN** searching for `def token_to_vec` or `def _compute_entropy` or `def compute_entropy_from_logprobs` in `ppo/`
- **THEN** no definition is found

#### Scenario: segmenter.py contains no vector utilities
- **WHEN** searching for `def token_to_vec` or `def compute_entropy` in `features/segmenter.py`
- **THEN** no definition is found

### Requirement: token_to_vec behavior unchanged
Same defaults (-20.0 logprob, 0.0 entropy), field merging order (logprob, entropy, topk_logits, hidden), padding/truncation to `obs_dim`.

#### Scenario: Missing fields get defaults
- **WHEN** `token_to_vec({}, 64)` is called
- **THEN** first element is -20.0, second is 0.0

#### Scenario: Excess fields truncated
- **WHEN** `token_to_vec({"logprob": -1.0, "entropy": 0.5, "topk_logits": list(range(100))}, 10)` is called
- **THEN** result has length 10

### Requirement: compute_entropy behavior unchanged
Computes entropy of a categorical distribution from log-probabilities, identical to the original implementations.

#### Scenario: Uniform distribution
- **WHEN** `compute_entropy([-1.0986, -1.0986, -1.0986])` is called
- **THEN** result is approximately 1.0986

#### Scenario: Deterministic distribution
- **WHEN** `compute_entropy([0.0, -100.0, -100.0])` is called
- **THEN** result is less than 0.01
