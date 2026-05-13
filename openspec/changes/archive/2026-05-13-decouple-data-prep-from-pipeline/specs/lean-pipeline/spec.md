## ADDED Requirements

### Requirement: Default STAGES is training-only
`run_pipeline.sh` default STAGES SHALL be `mil,eval,ppo,eval_ol`. The `build`, `split`, `eval_ds` stages SHALL NOT appear in the default.

#### Scenario: Default run skips data prep
- **WHEN** `bash scripts/run_pipeline.sh` is executed without STAGES override
- **THEN** only mil/eval/ppo/eval_ol stages execute

### Requirement: Legacy stage names still work if explicitly requested
If a user explicitly sets `STAGES=build,split,eval_ds,mil,...`, the pipeline SHALL still execute those stages.

#### Scenario: Full run with override
- **WHEN** `STAGES=build,split,eval_ds,mil,eval,ppo,eval_ol bash scripts/run_pipeline.sh` is executed
- **THEN** all stages run including data preparation
