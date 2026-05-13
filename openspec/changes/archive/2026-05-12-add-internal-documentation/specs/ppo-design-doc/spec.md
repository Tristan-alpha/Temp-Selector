## ADDED Requirements

### Requirement: ppo/DESIGN.md exists
A `ppo/DESIGN.md` file SHALL exist explaining the PPO module's design rationale.

#### Scenario: File present in ppo/
- **WHEN** checking `ppo/DESIGN.md`
- **THEN** the file exists and is non-empty

### Requirement: ppo/DESIGN.md covers key design decisions
The file SHALL explain: why online PPO is necessary vs offline (causal action-reward chain), the per-segment generation loop with vLLM APC, GAE + PPO clip mechanics, terminal reward vs shaping reward (MIL inst_logit as intermediate signal), overfitting diagnostic (value vs val_value), ep_correct semantics vs MIL label convention, and the first-segment dummy value rationale.

#### Scenario: Design rationale sections present
- **WHEN** reading ppo/DESIGN.md
- **THEN** it covers online vs offline rationale, generation loop, reward design, and diagnostic strategy
