## Why

BagDataset currently extracts all logprob/hidden features upfront in `__init__`, converting SGLang tensors to Python lists and accumulating them in memory before training starts. For a 9600-row dataset with 500-token mean responses, this creates 600+ GB of Python float objects — `hidden_batch_size` batches the SGLang calls but does not control memory peak. Moving extraction to the DataLoader's `collate_fn` makes it truly lazy: only one training batch is materialized at a time.

## What Changes

- **BagDataset** stores only row metadata (no extraction, no tensor building in `__init__`)
- `__getitem__` returns raw row dicts instead of pre-built `RowTensor` instances
- New `collate_fn` performs per-batch SGLang extraction → token vectorization → segment pooling → padding
- `hidden_batch_size` replaced by training `batch_size` itself as the memory control
- DataLoader `num_workers` set to 0 (SGLangRunner cannot be serialized to worker processes)
- MIL training and eval scripts pass extractor to collate_fn instead of BagDataset constructor

## Capabilities

### New Capabilities

- `collate-feature-extraction`: Collate function orchestrates SGLang logprob/hidden extraction per training batch, then builds segment-level instance tensors via segment pooling

### Modified Capabilities

- `mil-online-hidden-extract`: BagDataset no longer extracts hidden states in `__init__`; extraction is deferred to `collate_fn` during DataLoader iteration

## Impact

- `mil/training.py` — BagDataset, collate_fn, train()
- `mil/eval.py` — BagDataset usage, collate_fn, evaluation loop
- `inference/sglang_runner.py` — no changes (already provides per-batch `extract_hidden`/`extract_logprobs`)
- `configs/*.yaml` — remove `hidden_batch_size`, optionally bump MIL `batch_size` to 128-256
