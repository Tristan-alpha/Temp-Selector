## Design

### TokenBatchSampler

A custom sampler that receives pre-computed token counts (from `len(r["_full_ids"])`) and groups samples so each batch stays under `max_tokens_per_batch`.

```python
class TokenBatchSampler(Sampler):
    def __init__(self, token_counts, max_tokens, shuffle=True):
        self.token_counts = token_counts
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.token_counts)))
        if self.shuffle:
            random.shuffle(indices)
        batch = []
        batch_tokens = 0
        for idx in indices:
            n = self.token_counts[idx]
            if batch and batch_tokens + n > self.max_tokens:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += n
        if batch:
            yield batch

    def __len__(self):
        return max(1, sum(self.token_counts) // self.max_tokens)
```

### Integration

```python
# train():
dataset = BagDataset(data_path)
# ... pre-tokenize ...
token_counts = [len(r["_full_ids"]) for r in dataset.rows]
sampler = TokenBatchSampler(token_counts, max_tokens=cfg["mil"]["training"]["max_tokens_per_batch"])
loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, num_workers=0)
```

Collate_fn is unchanged — it receives a list of rows and processes them. The batch size is now variable but bounded by token count.

### Config

```yaml
mil:
  training:
    max_tokens_per_batch: 100000  # ~78% of SGLang's max_total_num_tokens (128K)
    batch_size: 128               # REMOVED
```

### Per-batch log

Add token count to batch log: `epoch=1 batch=5/18 tokens=97821 bags=121 loss=0.5234 ...`

### Valuation

Validation uses the same `TokenBatchSampler` (no shuffle), ensuring deterministic eval batches.

### Edge cases

- A single sample exceeds `max_tokens_per_batch`: yield it alone (the sampler naturally handles this — first sample alone in batch)
- Empty dataset: `__len__` returns 1, sampler yields empty batches
