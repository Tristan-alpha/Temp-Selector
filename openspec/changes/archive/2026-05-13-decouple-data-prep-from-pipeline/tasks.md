## 1. Move build_dataset to scripts/

- [x] 1.1 Move `features/build_dataset.py` → `scripts/build_dataset.py`; add `sys.path.insert(0, ...)` for imports
- [x] 1.2 Update `run_pipeline.sh` `build` stage: `python -m features.build_dataset` → `python scripts/build_dataset.py`
- [x] 1.3 Update `README.md`, `PIPELINE.md`, `CLAUDE.md` with new script location

## 2. Slim down pipeline

- [x] 2.1 Change default `STAGES` in `run_pipeline.sh` to `mil,eval,ppo,eval_ol`
- [x] 2.2 Keep `build`, `split`, `eval_ds` stage blocks intact (still runnable via STAGES override)
- [x] 2.3 Add data prep commands as one-time setup in README and PIPELINE.md

## 3. dataset_eval JSON output

- [x] 3.1 Add `--output` argument to `features/dataset_eval.py` main(), default to `datasets/eval_stats.json`
- [x] 3.2 Write result dict as JSON to output file in addition to logging
- [x] 3.3 Update `run_pipeline.sh` `eval_ds` stage to pass `--output "$EVAL_STATS_FILE"`

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 4.2 Run `python scripts/build_dataset.py --help` — CLI works from new location
- [x] 4.3 Run `python -m features.dataset_eval --config configs/base.yaml` — JSON output file created

## 5. Documentation

- [x] 5.1 Update README.md: pipeline section, data prep commands
- [x] 5.2 Update PIPELINE.md: pipeline flow diagram, env vars table
- [x] 5.3 Update CLAUDE.md: directory structure
