"""Data preparation for continuation-supervised prefix value learning."""

from __future__ import annotations

import math
import random
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
from scipy.stats import beta as beta_distribution
from torch.utils.data import Dataset
from tqdm import tqdm

from features.segmenter import build_masked_concat_segment_obs_from_lp
from utils.jsonl import sample_prefix


def problem_id(row: Dict[str, Any]) -> str:
    return sample_prefix(str(row.get("sample_id", "")))


ANCHOR_QUANTILES: List[Tuple[str, float]] = [
    ("anchor_25", 0.25),
    ("anchor_50", 0.50),
    ("anchor_75", 0.75),
]

RANDOM_QUANTILE_BANDS: List[Tuple[str, float, float]] = [
    ("random_early", 0.05, 0.30),
    ("random_middle", 0.30, 0.65),
    ("random_late", 0.65, 0.95),
]


def _stable_trajectory_seed(sampling_seed: int, sample_id: str) -> int:
    material = f"{int(sampling_seed)}:{sample_id}".encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**31)


def _prefix_stage(prefix_segments: int, n_segments: int) -> str:
    q = prefix_segments / max(1, n_segments)
    if q < 0.30:
        return "early"
    if q < 0.65:
        return "middle"
    return "late"


def _clipped_prefix_count(q: float, n_segments: int) -> int:
    return min(n_segments - 1, max(1, math.ceil(float(q) * n_segments)))


def prefix_segment_specs(n_tokens: int, segment_size: int, sampling_seed: int = 42,
                         source_sample_id: str = "") -> List[Dict[str, Any]]:
    """Select anchor and stratified-random prefixes, excluding full responses."""
    n_segments = max(1, math.ceil(n_tokens / max(1, segment_size)))
    if n_segments < 2:
        return []
    trajectory_seed = _stable_trajectory_seed(sampling_seed, source_sample_id)
    rng = random.Random(trajectory_seed)
    candidates: Dict[int, Dict[str, Any]] = {}

    def add_candidate(source: str, quantile: float) -> None:
        count = _clipped_prefix_count(quantile, n_segments)
        entry = candidates.setdefault(
            count,
            {
                "prefix_segments": count,
                "prefix_sources": [],
                "prefix_quantiles": [],
            },
        )
        entry["prefix_sources"].append(source)
        entry["prefix_quantiles"].append(float(quantile))

    for source, quantile in ANCHOR_QUANTILES:
        add_candidate(source, quantile)
    add_candidate("anchor_penultimate", (n_segments - 1) / n_segments)

    for source, lo, hi in RANDOM_QUANTILE_BANDS:
        add_candidate(source, rng.uniform(lo, hi))

    specs = []
    for count in sorted(candidates):
        entry = candidates[count]
        specs.append({
            **entry,
            "prefix_stage": _prefix_stage(count, n_segments),
            "n_segments": n_segments,
            "prefix_sampling_seed": int(sampling_seed),
            "trajectory_sampling_seed": int(trajectory_seed),
        })
    return specs


def prefix_segment_counts(n_tokens: int, segment_size: int, sampling_seed: int = 42,
                          source_sample_id: str = "") -> List[int]:
    """Return selected prefix segment counts for a trajectory."""
    return [
        int(spec["prefix_segments"])
        for spec in prefix_segment_specs(
            n_tokens, segment_size,
            sampling_seed=sampling_seed,
            source_sample_id=source_sample_id,
        )
    ]


def select_continuation_prefixes(rows: Sequence[Dict[str, Any]],
                                 segment_size: int,
                                 sampling_seed: int = 42) -> List[Dict[str, Any]]:
    """Deterministically choose one correct and one incorrect trajectory per problem."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[problem_id(row)].append(row)

    selected: List[Dict[str, Any]] = []
    for pid in sorted(grouped):
        bucket = sorted(grouped[pid], key=lambda r: str(r.get("sample_id", "")))
        correct = [r for r in bucket if int(r.get("individual_label", 1)) == 0]
        wrong = [r for r in bucket if int(r.get("individual_label", 0)) == 1]
        chosen: List[Dict[str, Any]] = []
        if correct:
            chosen.append(correct[0])
        if wrong:
            chosen.append(wrong[0])
        if len(chosen) < 2:
            for row in bucket:
                if row not in chosen:
                    chosen.append(row)
                if len(chosen) == 2:
                    break

        for row in chosen:
            token_ids = list(row.get("token_ids", []))
            source_sample_id = str(row.get("sample_id", ""))
            for prefix_spec in prefix_segment_specs(
                len(token_ids), segment_size,
                sampling_seed=sampling_seed,
                source_sample_id=source_sample_id,
            ):
                count = int(prefix_spec["prefix_segments"])
                selected.append({
                    "problem_id": pid,
                    "source_sample_id": source_sample_id,
                    "prefix_segments": count,
                    "prefix_token_end": min(len(token_ids), count * segment_size),
                    "source_individual_label": int(row.get("individual_label", 0)),
                    **prefix_spec,
                })
    return selected


def jeffreys_interval(n_correct: int, n_total: int,
                      credibility: float = 0.80) -> Tuple[float, float]:
    alpha = float(n_correct) + 0.5
    beta = float(n_total - n_correct) + 0.5
    tail = (1.0 - credibility) / 2.0
    return (
        float(beta_distribution.ppf(tail, alpha, beta)),
        float(beta_distribution.ppf(1.0 - tail, alpha, beta)),
    )


def posterior_mean(record: Dict[str, Any]) -> float:
    correct = float(record["n_correct"])
    total = float(record["n_total"])
    return (correct + 0.5) / (total + 1.0)


def build_ranking_pairs(records: Sequence[Dict[str, Any]], seed: int = 42,
                        max_pairs_per_problem: int = 64) -> List[Tuple[int, int]]:
    """Create balanced same-trajectory and cross-trajectory pairs per problem."""
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, record in enumerate(records):
        grouped[str(record["problem_id"])].append(idx)

    rng = random.Random(seed)
    all_pairs: List[Tuple[int, int]] = []
    for pid in sorted(grouped):
        same: List[Tuple[int, int]] = []
        cross: List[Tuple[int, int]] = []
        indices = grouped[pid]
        intervals = {
            idx: jeffreys_interval(int(records[idx]["n_correct"]), int(records[idx]["n_total"]))
            for idx in indices
        }
        for pos, a in enumerate(indices):
            for b in indices[pos + 1:]:
                lo_a, hi_a = intervals[a]
                lo_b, hi_b = intervals[b]
                if not (hi_a < lo_b or hi_b < lo_a):
                    continue
                pair = (a, b)
                if records[a]["source_sample_id"] == records[b]["source_sample_id"]:
                    same.append(pair)
                else:
                    cross.append(pair)
        rng.shuffle(same)
        rng.shuffle(cross)
        half = max_pairs_per_problem // 2
        chosen = same[:half] + cross[:half]
        if len(chosen) < max_pairs_per_problem:
            remaining = same[half:] + cross[half:]
            rng.shuffle(remaining)
            chosen.extend(remaining[:max_pairs_per_problem - len(chosen)])
        all_pairs.extend(chosen)
    return all_pairs


def pretokenize_rows(rows: Sequence[Dict[str, Any]], tokenizer) -> None:
    prompts = [r.get("metadata", {}).get("rendered_prompt") or r.get("prompt", "") for r in rows]
    if not prompts:
        return
    encoded = tokenizer(prompts, add_special_tokens=False)
    for row, prompt_ids in zip(rows, encoded.input_ids):
        response_ids = list(row.get("token_ids", []))
        row["_prompt_len"] = len(prompt_ids)
        row["_full_ids"] = list(prompt_ids) + response_ids


def _token_batches(rows: Sequence[Dict[str, Any]], max_tokens: int) -> List[List[int]]:
    batches: List[List[int]] = []
    current: List[int] = []
    total = 0
    for idx, row in enumerate(rows):
        size = len(row.get("_full_ids", row.get("token_ids", [])))
        if current and total + size > max_tokens:
            batches.append(current)
            current = []
            total = 0
        current.append(idx)
        total += size
    if current:
        batches.append(current)
    return batches


def precompute_feature_cache(
    rows: Sequence[Dict[str, Any]],
    extractor,
    segment_size: int,
    token_dim: int,
    top_k: int,
    max_tokens_per_batch: int,
    device: torch.device,
    description: str,
) -> List[Dict[str, Any]]:
    """Extract all response features once and retain them in CPU RAM as float16."""
    pretokenize_rows(rows, extractor.tokenizer)
    cache: List[Dict[str, Any]] = []
    for indices in tqdm(
        _token_batches(rows, max_tokens_per_batch), desc=description,
    ):
        batch_rows = [rows[i] for i in indices]
        result = extractor.extract_from_ids(
            [r["_full_ids"] for r in batch_rows],
            [r["_prompt_len"] for r in batch_rows],
            temperatures=[float(r.get("temperature", 1.0)) for r in batch_rows],
            top_k=top_k,
            return_logprobs=True,
            return_hidden=False,
            device=device,
        )
        for row, lp in zip(batch_rows, result["logprobs"]):
            n = len(row.get("token_ids", []))
            masked = build_masked_concat_segment_obs_from_lp(
                lp[:n], list(row.get("tokens", [])), str(row.get("response", "")),
                segment_size=segment_size, token_dim=token_dim, device=device,
                segment_mode="fixed_window",
            )
            cache.append({
                "sample_id": str(row.get("sample_id", "")),
                "problem_id": problem_id(row),
                "features": masked.features.detach().cpu().to(torch.float16),
                "token_mask": masked.token_mask.detach().cpu().to(torch.uint8),
                "terminal_target": 1.0 - float(row.get("individual_label", 0)),
            })
    return cache


def pad_prefix_entries(entries: Sequence[Tuple[Dict[str, Any], int | None]]) -> Dict[str, torch.Tensor]:
    if not entries:
        raise ValueError("cannot collate an empty prefix batch")
    lengths = [
        min(int(prefix_len), int(entry["features"].shape[0])) if prefix_len is not None
        else int(entry["features"].shape[0])
        for entry, prefix_len in entries
    ]
    max_k = max(lengths)
    feature_dim = int(entries[0][0]["features"].shape[1])
    segment_size = int(entries[0][0]["token_mask"].shape[1])
    features = torch.zeros(len(entries), max_k, feature_dim, dtype=torch.float32)
    token_mask = torch.zeros(len(entries), max_k, segment_size, dtype=torch.float32)
    segment_mask = torch.zeros(len(entries), max_k, dtype=torch.float32)
    for i, ((entry, _), length) in enumerate(zip(entries, lengths)):
        features[i, :length] = entry["features"][:length].float()
        token_mask[i, :length] = entry["token_mask"][:length].float()
        segment_mask[i, :length] = 1.0
    return {
        "features": features,
        "token_mask": token_mask,
        "segment_mask": segment_mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }


class IndexDataset(Dataset):
    def __init__(self, size: int):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> int:
        return index


def terminal_collate(cache: Sequence[Dict[str, Any]], indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    batch = pad_prefix_entries([(cache[i], None) for i in indices])
    batch["target"] = torch.tensor(
        [float(cache[i]["terminal_target"]) for i in indices], dtype=torch.float32,
    )
    return batch


def continuation_collate(cache_by_id: Dict[str, Dict[str, Any]],
                         records: Sequence[Dict[str, Any]],
                         indices: Sequence[int]) -> Dict[str, torch.Tensor]:
    chosen = [records[i] for i in indices]
    batch = pad_prefix_entries([
        (cache_by_id[str(r["source_sample_id"])], int(r["prefix_segments"]))
        for r in chosen
    ])
    batch["n_correct"] = torch.tensor([float(r["n_correct"]) for r in chosen])
    batch["n_total"] = torch.tensor([float(r["n_total"]) for r in chosen])
    batch["target"] = torch.tensor([posterior_mean(r) for r in chosen])
    return batch


def ranking_collate(cache_by_id: Dict[str, Dict[str, Any]],
                    records: Sequence[Dict[str, Any]],
                    pairs: Sequence[Tuple[int, int]],
                    indices: Sequence[int]) -> Dict[str, Any]:
    selected = [pairs[i] for i in indices]
    rec_a = [records[a] for a, _ in selected]
    rec_b = [records[b] for _, b in selected]
    return {
        "a": continuation_collate(cache_by_id, rec_a, list(range(len(rec_a)))),
        "b": continuation_collate(cache_by_id, rec_b, list(range(len(rec_b)))),
    }
