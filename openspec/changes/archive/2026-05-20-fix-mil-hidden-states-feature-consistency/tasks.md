## 1. Modify build_segment_obs_from_lp in features/segmenter.py

- [x] 1.1 Add `include_topk: bool = False` parameter
- [x] 1.2 When `include_topk=True`, append `lp_tensor[:, 1:]` (top-k logprobs) to parts; when False, keep current behavior (only logprob + entropy as base)

## 2. Fix make_collate_fn in mil/utils.py

- [x] 2.1 Change `need_logprobs` to be True for both feature modes
- [x] 2.2 Remove the `elif hidden_tensors is not None` branch; both modes go through `build_segment_obs_from_lp`
- [x] 2.3 Pass `include_topk=(feature_mode == "topk_logprobs")` to `build_segment_obs_from_lp`

## 3. Update PPO training in ppo/training.py

- [x] 3.1 Pass `include_topk=True` when `hs_needed=False` (topk_logprobs mode) in `build_segment_obs_from_lp` call

## 4. Documentation

- [x] 4.1 Update `mil/DESIGN.md`: document new feature vector layout for both modes

## 5. Verification

- [x] 5.1 Run `python -m pytest tests/ -v` — all tests must pass; update tests if needed
- [x] 5.2 Run `python -m compileall -q features/segmenter.py mil/utils.py ppo/training.py`
