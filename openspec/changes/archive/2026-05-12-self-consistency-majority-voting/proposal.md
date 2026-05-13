## Why

The intended majority voting mechanism is self-consistency: extract the answer from each response, find the modal answer (plurality), and compare the mode to gold. However, a previous coding agent misinterpreted this and implemented a stricter "count per-vote correctness" check — `n_correct >= (num_votes+1)//2` — which requires gold to appear in at least half of all votes. This threshold condition is more restrictive than true self-consistency: gold could be the plurality winner with, say, 3/8 votes, but still fail the `>=4` threshold and be labeled "wrong." Replacing the threshold-based check with true self-consistency restores the original design intent.

## What Changes

- Add `extract_answer()` and `verify_answer_by_value()` to `utils/answer_verifier.py` — extraction-only and extraction+verification utilities
- Change Stage 1 label computation in `features/build_dataset.py` (both vLLM and API paths): extract answer from each response, find modal answer via `Counter.most_common(1)`, compare mode to gold
- Change Stage 3 terminal reward in `ppo/training.py`: same self-consistency logic for determining if majority vote was correct
- Change Stage 4 online evaluation in `ppo/eval.py`: same self-consistency logic
- **Not changed**: `BagSample.metadata["individual_correct"]` — remains per-vote `verify_answer()` against gold (auxiliary statistic)
- **Not changed**: `features/dataset_eval.py` statistics — `individual_correct` and `individual_accuracy` remain as-is (reference metric, not label signal)

## Capabilities

### New Capabilities

- `answer-extraction`: Dedicated `extract_answer()` function using `math_verify.parse()` to extract math expressions from free-text responses
- `self-consistency-labeling`: Self-consistency majority voting: extract answers, find mode, compare mode to gold — used for Stage 1 labels, Stage 3 terminal reward, and Stage 4 evaluation accuracy

### Modified Capabilities

<!-- None -->

## Impact

- **Code**: `utils/answer_verifier.py` (+2 functions), `features/build_dataset.py` (2 sites), `ppo/training.py` (2 sites), `ppo/eval.py` (1 site)
- **Behavior**: Label distribution may shift slightly for questions where wrong answers converge on one incorrect value
- **Data**: `BagSample.label` semantics change — re-running Stage 1 on the same prompts may produce different labels
- **Existing checkpoints**: MIL/PPO checkpoints trained with old labels may need re-training for self-consistency evaluation
- **Docs**: PIPELINE.md label semantics section, ppo/DESIGN.md terminal reward section, features/DESIGN.md majority voting section
