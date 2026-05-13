## 1. Drop feature_mode from configs

- [x] 1.1 Remove `feature_mode` key from `configs/dataset.yaml`

## 2. Simplify runners

- [x] 2.1 In `inference/vllm_runner.py`: always set `topk_logits=dist` on TokenFeature; remove `feature_mode` param from `_build_feature_payload` and `export_token_features_multi_temp`
- [x] 2.2 In `inference/api_runner.py`: same simplification

## 3. Update build_dataset.py

- [x] 3.1 Remove `feature_mode` arguments from exporter calls in `scripts/build_dataset.py`

## 4. Verification

- [x] 4.1 Run `python -m pytest tests/ -v` — all tests pass
- [x] 4.2 Run `python -m compileall -q` on modified files
- [x] 4.3 Verify `configs/dataset.yaml` parses

## 5. Documentation

- [x] 5.1 Update README.md if feature_mode is referenced
- [x] 5.2 Update PIPELINE.md if feature_mode is referenced
