## 1. CLAUDE.md

- [x] 1.1 Create `CLAUDE.md` at project root with sections: project overview, directory structure (per-file one-liners), key data flow, coding conventions, common pitfalls, common tasks

## 2. Module DESIGN.md files

- [x] 2.1 Create `mil/DESIGN.md` covering: problem formulation and label flip rationale, model architecture and component roles, loss function design (four terms and weights), top-k instance loss motivation, pos_weight formula, DynamicTempHead integration and Bug 1 lesson
- [x] 2.2 Create `ppo/DESIGN.md` covering: online vs offline PPO rationale, per-segment generation loop with vLLM APC, GAE + PPO clip mechanics, reward design (terminal + shaping), overfitting diagnostic, ep_correct vs MIL label semantics, first-segment dummy values
- [x] 2.3 Create `features/DESIGN.md` covering: full data pipeline (JSONL to BagSample to instance tensor), segmentation mode tradeoffs, vectorizer feature merging strategy, majority voting computation, inter-stage data format contract

## 3. Critical docstrings and comments

- [x] 3.1 Add comment block above top-k instance loss logic in `mil/training.py` explaining MIL annotation dilemma, what top-k achieves, why k=n_valid//3, and positive vs negative bag treatment
- [x] 3.2 Add docstring to `MILModel.forward()` in `mil/model.py` tracing feature flow through encoder, position encoding, BiGRU, aggregator, and two heads, with tensor shapes and output key semantics
- [x] 3.3 Verify first-segment comment in `ppo/training.py` exists (already added in prior Bug 5 fix) — no action needed
- [x] 3.4 Add comment above PPO batch construction loop in `ppo/training.py` explaining done flag, terminal vs intermediate reward, and range(1, n_steps) skip
- [x] 3.5 Add docstring to `compute_gae()` in `ppo/model.py` explaining GAE formula, done mask at episode boundaries, and advantage standardization
- [x] 3.6 Expand docstring on `segment_pooling()` in `features/segmenter.py` covering mean vs concat choice, boundary clamping, and empty output fallback
- [x] 3.7 Add comment on `BagSample.label` field in `features/schema.py` documenting the flipped label convention (0=correct, 1=error)
- [x] 3.8 Expand docstring on `token_to_vec()` in `features/vectorizer.py` explaining merge order, default values, and padding/truncation

## 4. Verification

- [x] 4.1 Verify `CLAUDE.md` and all 3 `DESIGN.md` files exist and are non-empty
- [x] 4.2 Verify all 8 docstring/comment targets are present by reading each file
- [x] 4.3 Run `python -m pytest tests/ -v` to confirm no behavior change (all 80 tests pass)
