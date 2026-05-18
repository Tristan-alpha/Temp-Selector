## 1. Add shared helper to features/segmenter.py

- [x] 1.1 Add `build_segment_obs_from_lp` to `features/segmenter.py`
- [x] 1.2 Update `ppo/training.py` to use shared helper
- [x] 1.3 Update `ppo/eval.py` to use shared helper

## 2. Clean up ppo/eval.py

- [x] 2.1 Replace `_render_prompt` with `runner.build_math_messages` + `runner.render_messages`
- [x] 2.2 Delete `load_prompts` (dead code) and inline `_get_question` + `load_config`
- [x] 2.3 Remove `errors` field from `OnlineResult`
- [x] 2.4 Fix `--parallel-size` CLI arg type to `int`

## 3. Verification

- [x] 3.1 Run `python -m pytest tests/ -v` — all tests must pass
- [x] 3.2 Run `python -m compileall -q` on all modified files
