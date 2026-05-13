## ADDED Requirements

### Requirement: hidden_states.yaml config exists
`configs/hidden_states.yaml` SHALL exist with `feature_mode: hidden_states`, `instance_dim: 4096` (Qwen3-8B hidden_size), and shared dataset paths.

#### Scenario: Config loads
- **WHEN** loading `configs/hidden_states.yaml`
- **THEN** `data.instance_dim` is 4096 and `inference.feature_mode` is `hidden_states`

### Requirement: feature_mode values are cleanly named
All configs and runner code SHALL use these three feature_mode values:

| value | includes |
|---|---|
| `basic` | logprob + entropy |
| `topk_logits` | + top-16 logprob values |
| `hidden_states` | + per-token LLM hidden state (4096-dim) |

The old `combined` and un-named empty string SHALL NOT appear in any config or code.

#### Scenario: base.yaml uses topk_logits
- **WHEN** loading `configs/base.yaml`
- **THEN** `inference.feature_mode` is `topk_logits`

#### Scenario: No combined alias in code
- **WHEN** grep for `"combined"` in `inference/vllm_runner.py` and `inference/api_runner.py`
- **THEN** no match is found (only `"topk_logits"` in the dispatch)

#### Scenario: Runner dispatch matches new names
- **WHEN** feature_mode is `basic`
- **THEN** neither topk_logits nor hidden are set on TokenFeature
- **WHEN** feature_mode is `topk_logits`
- **THEN** topk_logits field is set, hidden is None
- **WHEN** feature_mode is `hidden_states`
- **THEN** hidden field is set, topk_logits is None
