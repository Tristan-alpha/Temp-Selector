## Context

`OnlineTemperatureEvaluator` is the last place in the codebase that directly instantiates `vllm.LLM()` outside of `VLLMFeatureExporter`. It was written before `generate_with_features` existed. The eval loop runs 3 strategies (ppo/best-fixed/random) per prompt set, each generating segment-by-segment.

## Goals / Non-Goals

**Goals:**
- Replace raw `LLM` with `VLLMFeatureExporter`
- Replace manual logprob parsing with `generate_with_features`
- Delete duplicated `_resolve_tp`

**Non-Goals:**
- Changing the 3-strategy evaluation structure
- Changing `run_pipeline.sh`

## Decisions

### Decision 1: Single VLLMFeatureExporter instance

**Rationale:** The 3 strategies are run sequentially, not concurrently. A single runner instance serves all three.

### Decision 2: Segment pooling via `build_segments` + `segment_pooling`

**Rationale:** Consistent with PPO training. The old `mean_pool_obs` just averages per-token vectors without respecting segment boundaries — `segment_pooling` is more correct for this use case.
