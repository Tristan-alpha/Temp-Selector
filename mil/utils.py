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
                instances_list.append(torch.zeros(1, instance_dim, device=train_device))
                labels.append(float(row.get("individual_label", 0)))
                temp_indices.append(bin_map.get(float(row.get("temperature", temp_bins[0]))))
                continue

            if logprob_tensors is not None:
                extra = [hidden_tensors[i][:n]] if hidden_tensors is not None else None
                inst = build_segment_obs_from_lp(
                    logprob_tensors[i][:n], row["tokens"], row["response"],
                    segment_size, instance_dim, device=train_device,
                    extra_parts=extra,
                    include_topk=(feature_mode == "topk_logprobs"),
                    pooling_mode=pooling_mode,
                )
            else:
                inst = torch.zeros(1, instance_dim, device=train_device)
            instances_list.append(inst)
            labels.append(float(row.get("individual_label", 0)))
            t_val = float(row.get("temperature", temp_bins[0]))
            temp_indices.append(bin_map.get(t_val, 0))

        max_k = max(inst.shape[0] for inst in instances_list)
        d = instances_list[0].shape[1]
        b = len(instances_list)
        x = torch.zeros((b, max_k, d), dtype=torch.float32, device=train_device)
        mask = torch.zeros((b, max_k), dtype=torch.float32, device=train_device)
        y = torch.tensor(labels, dtype=torch.float32, device=train_device)
        t = torch.tensor(temp_indices, dtype=torch.long, device=train_device)

        for i, inst in enumerate(instances_list):
            k = inst.shape[0]
            x[i, :k] = inst
            mask[i, :k] = 1.0

        batch_tokens = sum(len(r.get("_full_ids", r.get("token_features", []))) for r in batch_rows)

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

        max_k = max(inst.shape[0] for inst in instances_list)
        d = instances_list[0].shape[1]
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
