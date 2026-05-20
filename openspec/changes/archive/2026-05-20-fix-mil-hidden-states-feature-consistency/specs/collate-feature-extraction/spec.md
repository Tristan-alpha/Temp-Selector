## MODIFIED Requirements

### Requirement: Two feature modes

The system SHALL support exactly two feature modes. Both modes SHALL produce per-token feature vectors of exactly `instance_dim` dims:

- `topk_logprobs`: `[sampled_logprob, entropy, topk_logprob_0..topk_logprob_{top_k-1}]` (2 + top_k dims)
- `hidden_states`: `[sampled_logprob, entropy, hidden_0..hidden_{hidden_dim-1}]` (2 + hidden_dim dims)

Both mode SHALL use `build_segment_obs_from_lp` → `segment_pooling` as the single construction pipeline. `build_segment_obs_from_lp` SHALL accept an `include_topk: bool` parameter to control whether top-k logprobs are appended.

#### Scenario: Valid feature modes

- **WHEN** `feature_mode` is set
- **THEN** it SHALL be either `"topk_logprobs"` or `"hidden_states"`

#### Scenario: topk_logprobs mode includes top-k logprobs directly

- **WHEN** `feature_mode` is `"topk_logprobs"` and `build_segment_obs_from_lp` is called with `include_topk=True`
- **THEN** the per-token vector SHALL include all top-k logprobs values
- **AND** no dimensions SHALL be zero-padded (2 + top_k = instance_dim)

#### Scenario: hidden_states mode includes logprobs and hidden

- **WHEN** `feature_mode` is `"hidden_states"`
- **THEN** `make_collate_fn` SHALL request both logprobs and hidden states from `extract_from_ids`
- **AND** segment vectors SHALL be built via `build_segment_obs_from_lp(lp, extra_parts=[hidden], include_topk=False)` — same composition as PPO
