## Context

`build`, `split`, `eval_ds` produce artifacts that all configs now share. Re-running them on every experiment wastes GPU time (build) and is redundant (split, eval_ds). They should be one-time operations outside the iterative training loop.

## Goals / Non-Goals

**Goals:**
- `build_dataset.py` moved to `scripts/` and run independently
- Pipeline defaults to `mil,eval,ppo,eval_ol`
- `dataset_eval` writes results to a JSON file
- Legacy stage names preserved for explicit overrides

**Non-Goals:**
- Changing how any stage works internally
- Removing any stage capability (still runnable via STAGES override)

## Decisions

### D1: build_dataset.py location

Moved from `features/` to `scripts/` because it's not a reusable library — it's a CLI-driven operation script. Import references in `run_pipeline.sh` change accordingly.

### D2: Pipeline stages

```bash
# Default: training + evaluation only
STAGES=${STAGES:-mil,eval,ppo,eval_ol}

# Data prep: run once manually before experiments
python scripts/build_dataset.py --config configs/base.yaml
python scripts/split_jsonl.py --config configs/base.yaml
python -m features.dataset_eval --config configs/base.yaml
```

### D3: eval output file

```python
# features/dataset_eval.py main()
result = evaluate_dataset(data_path)
out_path = args.output or "datasets/eval_stats.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)
```

## Risks

- **[build stage]** Moved to scripts but still imports from `features/` — `sys.path` adjustment needed in the script header
