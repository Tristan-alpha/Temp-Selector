## Design

### Core principle

Every function that currently returns `List[float]` or `List[List[float]]` returns `torch.Tensor` instead. Extraction results are consumed directly per-token, never written back to dataset dicts — so tensors are freed when collate_fn returns.

### Why NOT store extraction results in dicts

Storing `t[j]` (a view into the extracted `[n_tokens, dim]` tensor) in `token_features[j][field]` keeps the large parent tensor alive even after the extract method returns. Collate_fn exits but the row dicts (in `BagDataset.rows`) hold the view → the 1+ GB extraction tensor leaks across epochs.

**Fix**: `token_to_vec` accepts an optional `extracted` dict with per-token tensor values. These come directly from the extraction loop, are consumed to build the vector, and go out of scope — no leak.

### token_to_vec: optional extracted parameter

```python
def token_to_vec(
    feat: Dict[str, Any],
    obs_dim: int,
    extracted: Dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """[obs_dim] float32.

    ``feat`` provides the mandatory scalar fields (logprob, entropy).
    ``extracted`` provides optional per-token tensor fields (hidden,
    topk_logprobs) that are consumed inline and NOT stored in the dict.
    """
    parts = [torch.tensor([float(feat["logprob"]), float(feat["entropy"])])]
    if extracted:
        for key in ("topk_logprobs", "hidden"):
            v = extracted.get(key)
            if v is not None:
                parts.append(v.float())  # already a 1D tensor
    else:
        for key in ("topk_logprobs", "hidden"):
            v = feat.get(key)
            if v is not None:
                parts.append(v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
    merged = torch.cat(parts)
    if merged.shape[0] >= obs_dim:
        return merged[:obs_dim]
    return torch.cat([merged, torch.zeros(obs_dim - merged.shape[0])])

def token_to_obs(logprob, entropy_val, topk_logprobs, obs_dim) -> torch.Tensor:
    """[obs_dim] float32 — same logic, no dict."""
    base = torch.tensor([float(logprob), float(entropy_val)])
    topk = torch.tensor([float(x) for x in topk_logprobs]) if topk_logprobs else torch.zeros(0)
    merged = torch.cat([base, topk])
    if merged.shape[0] >= obs_dim:
        return merged[:obs_dim]
    return torch.cat([merged, torch.zeros(obs_dim - merged.shape[0])])

def mean_pool_obs(obs_list: List[torch.Tensor], obs_dim: int) -> torch.Tensor:
    """[obs_dim] float32"""
    if not obs_list:
        return torch.zeros(obs_dim)
    return torch.stack(obs_list).mean(dim=0)

def segment_pooling(token_tensor, spans, obs_dim, mode, segment_size) -> torch.Tensor:
    """token_tensor: [n_tokens, obs_dim]; returns [n_segments, obs_dim]"""
    out = []
    for s in spans:
        chunk = token_tensor[s.start:s.end]    # tensor slice
        if mode == "concat":
            flat = chunk.reshape(-1)
            target = segment_size * obs_dim
            if flat.shape[0] < target:
                flat = torch.cat([flat, torch.zeros(target - flat.shape[0])])
            out.append(flat[:target])
        else:
            out.append(chunk.mean(dim=0))
    if not out:
        return torch.zeros(1, obs_dim)
    return torch.stack(out)
```

### collate_fn: consume tensors inline, delete _patch_features

```python
def collate_fn(batch_rows):
    if need_hidden or need_logprobs:
        # ... extract hidden_tensors, logprob_tensors ...

    for i, row in enumerate(batch_rows):
        token_features = row.get("token_features", [])
        h_t = hidden_tensors[i] if need_hidden else None
        l_t = logprob_tensors[i] if need_logprobs else None

        token_vecs = []
        for j, tf in enumerate(token_features):
            extra = {}
            if h_t is not None and j < h_t.shape[0]:
                extra["hidden"] = h_t[j]
            if l_t is not None and j < l_t.shape[0]:
                extra["topk_logprobs"] = l_t[j]
            token_vecs.append(token_to_vec(tf, instance_dim, extra if extra else None))
        # extra dicts and 1D views go out of scope immediately

        t = torch.stack(token_vecs)            # [n_tokens, dim]
        spans = build_segments(...)
        inst = segment_pooling(t, spans, ...)  # [n_segments, dim]
        instances_list.append(inst)
    # hidden_tensors, logprob_tensors freed here — no views leak
```

### Unify PPO mean-pool

Both `_extract_segment_obs` (vLLM) and `_extract_segment_obs_sglang` (SGLang) have identical manual mean-pool blocks. Replace with `mean_pool_obs`. Hidden state mean-pool becomes `torch.tensor(hidden_states).mean(dim=0)`.

### Risks

- `torch.cat` per token adds overhead over the old list concatenation — dominated by SGLang prefill (seconds vs microseconds)
- PPO `_extract_segment_obs` return changes to `Optional[torch.Tensor]` — call sites already use `.unsqueeze(0)` which works on tensors unchanged
