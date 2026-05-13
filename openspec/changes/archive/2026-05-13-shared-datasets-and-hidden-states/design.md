## Context

Two root causes prevent dataset sharing: (1) `segment_mode` and `segment_size` control segmentation in Stage 1, creating config-specific JSONL; (2) `feature_mode` naming is confusing. Moving segmentation to Stage 2 and cleaning up names solves both.

## Goals / Non-Goals

**Goals:**
- All configs share a single Stage 1 output (`datasets/all.jsonl`)
- Segmentation (`build_segments`) runs in `BagDataset` at load time, configurable per-experiment
- `feature_mode` renamed to `basic` / `topk_logits` / `hidden_states`; dead `combined` alias removed
- New `hidden_states.yaml` and `ppo_control.yaml` configs
- PPO training reads prompts from `train_dataset` (not `raw_input`), symmetric with `ppo/eval.py` fix

**Non-Goals:**
- Changing how hidden states are extracted from vLLM
- Adding a mode that merges hidden_states + topk_logits simultaneously

## Decisions

### D1: Segmentation in BagDataset

```python
# BagDataset.__init__ — new signature
def __init__(self, data_path, temp_bins, instance_dim,
             pooling_mode="mean", segment_size=32,
             segment_mode="step"):          # ← NEW param
    ...
    token_texts = [tf.text for tf in token_features]
    spans = build_segments(                 # ← was in build_dataset.py
        tokens=token_texts, mode=segment_mode,
        segment_size=segment_size,
        response=row.get("response", ""),
    )
    inst_vecs = segment_pooling(token_vecs, spans, instance_dim, ...)
```

Stage 1 (`build_dataset.py`) still writes `segment_spans` to JSONL using step segmentation — this serves as the canonical/default. `BagDataset` can override based on config params.

### D2: feature_mode dispatch

```python
# vllm_runner.py / api_runner.py — simplified
if feature_mode == "topk_logits":
    topk_logits = dist
elif feature_mode == "hidden_states":
    hidden = gen.hidden_states[i]
# "basic" → neither → both remain None
```

### D3: Config path unification

All 9 configs share:
```yaml
paths:
  raw_input: data/math-small.jsonl
  all_dataset: datasets/all.jsonl
  train_dataset: datasets/train.jsonl
  val_dataset: datasets/val.jsonl
  test_dataset: datasets/test.jsonl
```

Only `mil_ckpt` and `ppo_ckpt` are config-specific (e.g., `checkpoints/mil_mlp_ckpt.pt`).

### D4: PPO training from train_dataset

`ppo/training.py` shares helper code with `ppo/eval.py`: extract unique (question, answer) pairs from a labeled BagSample JSONL via `sample_prefix` dedup. PPO online training uses `paths.train_dataset` (not `raw_input`), ensuring prompts come only from the train split.

After this change, `paths.raw_input` is only read by `features/build_dataset.py` (Stage 1 full data generation).

## Risks

- **[Hidden state file size]** 4096 floats per token → impractical for full dataset. Use `scripts/subsample_jsonl.py` first.
- **[Backward compat]** Old JSONL files have pre-computed `segment_spans` with the wrong mode. New BagDataset ignores them and recomputes from `token_features[].text` + `response`.
