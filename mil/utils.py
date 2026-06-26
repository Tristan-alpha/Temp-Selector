from __future__ import annotations

import json
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset

from features.segmenter import build_segment_obs_from_lp


def token_batches(rows: List[Dict[str, Any]], max_tokens: int) -> List[List[int]]:
    """Yield lists of row indices where sum of ``_full_ids`` token counts ≤ max_tokens.

    Used for pre-computation and eval where vLLM ``extract_from_ids`` is called —
    the GPU memory constraint is total tokens per batch, not number of rows.
    """
    batches: List[List[int]] = []
    batch: List[int] = []
    batch_tokens = 0
    for idx, row in enumerate(rows):
        n = len(row.get("_full_ids", row.get("token_ids", [])))
        if batch and batch_tokens + n > max_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(idx)
        batch_tokens += n
    if batch:
        batches.append(batch)
    return batches


class BagDataset(Dataset):
    """Lazy dataset: stores row metadata only.  Feature extraction happens in collate_fn."""

    def __init__(self, data_path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def make_collate_fn(
    extractor=None,
    feature_mode: str = "basic",
    instance_dim: int = 4098,
    segment_mode: str = "step",
    segment_size: int = 256,
    pooling_mode: str = "mean",
    temp_bins: List[float] | None = None,
    train_device: torch.device | None = None,
):
    """Factory that returns a collate function for MIL training.

    The returned function receives a list of row dicts from BagDataset,
    optionally extracts logprob/hidden features via VLLMFeatureExporter,
    builds per-segment instance vectors, and returns a padded batch dict.
    """
    if temp_bins is None:
        temp_bins = [0.0]
    bin_map = {float(v): i for i, v in enumerate(temp_bins)}
    need_hidden = feature_mode == "hidden_states" and extractor is not None
    need_logprobs = feature_mode in ("topk_logprobs", "hidden_states") and extractor is not None
    has_extractor = extractor is not None

    def collate_fn(batch_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        hidden_tensors = None
        logprob_tensors = None
        if has_extractor:
            result = extractor.extract_from_ids(
                [r["_full_ids"] for r in batch_rows],
                [r["_prompt_len"] for r in batch_rows],
                temperatures=[float(r.get("temperature", 0.0)) for r in batch_rows],
                return_logprobs=need_logprobs,
                return_hidden=need_hidden,
                device=train_device,
            )
            logprob_tensors = result.get("logprobs")
            hidden_tensors = result.get("hidden")

        instances_list: List[torch.Tensor] = []
        labels: List[float] = []
        temp_indices: List[int] = []

        for i, row in enumerate(batch_rows):
            token_ids = row["token_ids"]
            n = len(token_ids)
            if n == 0:
                obs_dim = instance_dim * segment_size if pooling_mode == "concat" else instance_dim
                instances_list.append(torch.zeros(1, obs_dim, device=train_device))
                labels.append(float(row.get("individual_label", 0)))
                temp_indices.append(bin_map.get(float(row.get("temperature", temp_bins[0])), 0))
                continue

            if logprob_tensors is not None:
                extra = [hidden_tensors[i][:n]] if hidden_tensors is not None else None
                inst = build_segment_obs_from_lp(
                    logprob_tensors[i][:n], row["tokens"], row["response"],
                    segment_size, instance_dim, device=train_device,
                    extra_parts=extra,
                    segment_mode=segment_mode,
                    include_topk=(feature_mode == "topk_logprobs"),
                    pooling_mode=pooling_mode,
                )
            else:
                obs_dim = instance_dim * segment_size if pooling_mode == "concat" else instance_dim
                inst = torch.zeros(1, obs_dim, device=train_device)
            instances_list.append(inst)
            labels.append(float(row.get("individual_label", 0)))
            t_val = float(row.get("temperature", temp_bins[0]))
            temp_indices.append(bin_map.get(t_val, 0))

        if not instances_list:
            return {
                "instances": torch.empty(0, 0, instance_dim, device=train_device),
                "mask": torch.empty(0, 0, device=train_device),
                "label": torch.empty(0, device=train_device),
                "temp_idx": torch.empty(0, dtype=torch.long, device=train_device),
            }

        max_k = max(inst.shape[0] for inst in instances_list)
        d = instances_list[0].shape[1]
        # Shape contract: every instance must have the same last dimension.
        # Mixed dims indicate a bug (e.g. empty-row zeros with wrong obs_dim).
        expected_d = instance_dim * segment_size if pooling_mode == "concat" else instance_dim
        for idx, inst in enumerate(instances_list):
            assert inst.dim() == 2, \
                f"collate_fn: inst {idx} must be 2D [K,D], got {inst.shape}"
            assert inst.shape[1] == d, \
                f"collate_fn: inst {idx} dim={inst.shape[1]} != batch dim={d}"
        assert d == expected_d, \
            f"collate_fn: batch dim={d} != expected={expected_d} for pooling={pooling_mode}"
        b = len(instances_list)
        x = torch.zeros((b, max_k, d), dtype=torch.float32, device=train_device)
        mask = torch.zeros((b, max_k), dtype=torch.float32, device=train_device)
        y = torch.tensor(labels, dtype=torch.float32, device=train_device)
        t = torch.tensor(temp_indices, dtype=torch.long, device=train_device)

        for i, inst in enumerate(instances_list):
            k = inst.shape[0]
            x[i, :k] = inst
            mask[i, :k] = 1.0

        batch_tokens = sum(len(r["_full_ids"]) for r in batch_rows)

        return {"instances": x, "mask": mask, "label": y, "temp_idx": t,
                "_batch_tokens": batch_tokens}

    return collate_fn


class SegmentCacheDataset(Dataset):
    """Minimal dataset that yields row indices for use with make_cached_collate_fn."""

    def __init__(self, n_samples: int):
        self.n = n_samples

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> int:
        return idx


def make_cached_collate_fn(
    segment_cache: List[Dict[str, Any]],
    instance_dim: int = 4098,
    train_device: torch.device | None = None,
):
    """Factory that returns a collate function reading from a pre-computed segment cache.

    ``segment_cache`` is a list of dicts with keys ``instances`` ([K_i, instance_dim]),
    ``label`` (float), and ``temp_idx`` (int).  The returned collate_fn receives a list of
    integer indices from ``SegmentCacheDataset`` and pads instances to max K within the batch.
    """
    def collate_fn(batch_indices: List[int]) -> Dict[str, torch.Tensor]:
        instances_list: List[torch.Tensor] = []
        labels: List[float] = []
        temp_indices: List[int] = []

        for idx in batch_indices:
            entry = segment_cache[idx]
            inst = entry["instances"]
            if train_device is not None:
                inst = inst.to(train_device)
            instances_list.append(inst)
            labels.append(entry["label"])
            temp_indices.append(entry["temp_idx"])

        if not instances_list:
            return {
                "instances": torch.empty(0, 0, instance_dim, device=train_device or torch.device("cpu")),
                "mask": torch.empty(0, 0, device=train_device or torch.device("cpu")),
                "label": torch.empty(0, device=train_device or torch.device("cpu")),
                "temp_idx": torch.empty(0, dtype=torch.long, device=train_device or torch.device("cpu")),
            }

        max_k = max(inst.shape[0] for inst in instances_list)
        d = instances_list[0].shape[1]
        # Shape contract: all pre-computed instances must share the same last dim.
        for idx, inst in enumerate(instances_list):
            assert inst.dim() == 2, \
                f"cached_collate: inst {idx} must be 2D [K,D], got {inst.shape}"
            assert inst.shape[1] == d, \
                f"cached_collate: inst {idx} dim={inst.shape[1]} != batch dim={d}"
        b = len(instances_list)
        x = torch.zeros((b, max_k, d), dtype=torch.float32, device=train_device)
        mask = torch.zeros((b, max_k), dtype=torch.float32, device=train_device)
        y = torch.tensor(labels, dtype=torch.float32, device=train_device)
        t = torch.tensor(temp_indices, dtype=torch.long, device=train_device)

        for i, inst in enumerate(instances_list):
            k = inst.shape[0]
            x[i, :k] = inst
            mask[i, :k] = 1.0

        return {"instances": x, "mask": mask, "label": y, "temp_idx": t,
                "_batch_tokens": 0}

    return collate_fn


# ═══════════════════════════════════════════════════════════════════
# Segment feature cache
# ═══════════════════════════════════════════════════════════════════

def _build_cache_path(
    cache_dir: str,
    split: str,
    segment_mode: str,
    pooling_mode: str,
    feature_mode: str,
    instance_dim: int,
    segment_size: int,
) -> str:
    """Build a deterministic cache path from extraction parameters.

    Dashes separate components; underscores within values are preserved.
    Returns the .safetensors path; a legacy .pt file at the same stem is
    also checked during loading.
    """
    import os
    return os.path.join(
        cache_dir,
        f"{split}-{segment_mode}-{pooling_mode}-{feature_mode}-{instance_dim}-{segment_size}.safetensors",
    )


def _pack_segment_cache(cache: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Flatten segment cache into a safetensors-compatible dict.

    Variable-length instances are concatenated; ``_csr_splits`` enables
    O(1) per-bag indexing.
    """
    all_inst: List[torch.Tensor] = []
    splits = [0]
    labels: List[float] = []
    temp_indices: List[int] = []
    for entry in cache:
        inst = entry["instances"]
        all_inst.append(inst)
        splits.append(splits[-1] + inst.shape[0])
        labels.append(entry["label"])
        temp_indices.append(entry["temp_idx"])
    return {
        "instances": torch.cat(all_inst, dim=0),
        "_csr_splits": torch.tensor(splits, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.float32),
        "temp_indices": torch.tensor(temp_indices, dtype=torch.int64),
    }


def _unpack_segment_cache(packed: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
    """Reconstruct list-of-dicts format from packed safetensors dict."""
    instances = packed["instances"]
    splits = packed["_csr_splits"].tolist()
    labels = packed["labels"].tolist()
    temp_indices = packed["temp_indices"].tolist()
    cache: List[Dict[str, Any]] = []
    for i in range(len(splits) - 1):
        s, e = int(splits[i]), int(splits[i + 1])
        cache.append({
            "instances": instances[s:e],
            "label": float(labels[i]),
            "temp_idx": int(temp_indices[i]),
        })
    return cache


def _load_or_build_segment_cache(
    dataset_rows: List[Dict[str, Any]],
    collate_fn,
    cache_path: str,
    max_tokens_per_batch: int,
    logger,
) -> List[Dict[str, Any]]:
    """Return segment feature cache, loading from disk or building via vLLM.

    Checks *cache_path* (``.safetensors``) first, then a legacy ``.pt``
    file at the same stem.  New caches are always written as
    ``.safetensors``.
    """
    import os

    # Check safetensors path first, then legacy .pt
    cache_pt = os.path.splitext(cache_path)[0] + ".pt"
    if os.path.exists(cache_path):
        logger.info("segment_cache_load path=%s", cache_path)
        packed = {}
        from safetensors.torch import load_file as _sf_load
        packed = _sf_load(cache_path)
        return _unpack_segment_cache(packed)
    elif os.path.exists(cache_pt):
        logger.info("segment_cache_load (legacy .pt) path=%s", cache_pt)
        return torch.load(cache_pt, weights_only=False)

    logger.info("segment_cache_build path=%s n_rows=%d", cache_path, len(dataset_rows))
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)

    batches = token_batches(dataset_rows, max_tokens_per_batch)
    segment_cache: List[Dict[str, Any]] = []
    for indices in batches:
        batch_rows = [dataset_rows[i] for i in indices]
        batch = collate_fn(batch_rows)
        x = batch["instances"].cpu()
        y = batch["label"].cpu()
        t = batch["temp_idx"].cpu()
        mask = batch["mask"].cpu()
        for i in range(x.shape[0]):
            n_valid = int(mask[i].sum().item())
            segment_cache.append({
                "instances": x[i, :n_valid].clone(),
                "label": float(y[i].item()),
                "temp_idx": int(t[i].item()),
            })

    from safetensors.torch import save_file as _sf_save
    packed = _pack_segment_cache(segment_cache)
    _sf_save(packed, cache_path)
    logger.info("segment_cache_saved path=%s entries=%d", cache_path, len(segment_cache))
    return segment_cache


def _fit_apply_scaler(
    train_cache: List[Dict[str, Any]],
    val_cache: List[Dict[str, Any]],
    logger,
    device: torch.device | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Fit a per-feature StandardScaler on training instances, apply to both splits.

    Returns ``(train_cache_scaled, val_cache_scaled, scaler_params)``.
    The scaler is fitted on pooled bag-level mean vectors to avoid being
    dominated by bags with many segments.  GPU-accelerated when *device*
    is provided.
    """
    dev = device or torch.device("cpu")

    # Fit on per-bag mean vectors (one per bag, avoids K imbalance)
    train_means = torch.stack([
        e["instances"].to(dev).float().mean(dim=0) for e in train_cache
    ])  # [N_train, D]
    mean = train_means.mean(dim=0, keepdim=True)  # [1, D]
    std = train_means.std(dim=0, keepdim=True).clamp(min=1e-8)
    logger.info("scaler_fit mean_range=[%.2e, %.2e] std_range=[%.2e, %.2e]",
                 float(mean.min()), float(mean.max()),
                 float(std.min()), float(std.max()))

    def _apply(cache: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for entry in cache:
            inst = (entry["instances"].to(dev).float() - mean) / std
            out.append({
                "instances": inst.cpu(),
                "label": entry["label"],
                "temp_idx": entry["temp_idx"],
            })
        return out

    t0 = __import__("time").time()
    out_train = _apply(train_cache)
    t1 = __import__("time").time()
    if logger:
        logger.info("scaler_apply_train n=%d dt=%.1fs", len(out_train), t1 - t0)
    out_val = _apply(val_cache)
    scaler_params = {"mean": mean.cpu().tolist(), "std": std.cpu().tolist()}
    return out_train, out_val, scaler_params
