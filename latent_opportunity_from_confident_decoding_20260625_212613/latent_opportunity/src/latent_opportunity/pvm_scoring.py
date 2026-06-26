"""Prefix scoring helpers backed by the frozen tf-mil PVM teacher."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def ensure_local_imports(repo_root: str | Path) -> None:
    root = Path(repo_root).resolve()
    for path in (root / "pvm_value" / "src", root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def score_prefixes_with_teacher(
    prefix_records: list[dict[str, Any]],
    *,
    repo_root: str | Path,
    tf_mil_root: str | Path,
    pvm_checkpoint: str | Path,
    model_name_or_path: str | None,
    teacher_top_k: int = 64,
    batch_size: int = 32,
    parallel_size: int | None = 1,
    gpu_memory_utilization: float = 0.70,
    enable_prefix_caching: bool = False,
) -> list[dict[str, Any]]:
    ensure_local_imports(repo_root)
    from pvm_value.pvm.pvm_wrapper import ChildPrefix, TfMilPVMTeacher

    teacher = TfMilPVMTeacher(
        tf_mil_root=tf_mil_root,
        checkpoint_path=pvm_checkpoint,
        model_name_or_path=model_name_or_path,
        top_k=teacher_top_k,
        parallel_size=parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=enable_prefix_caching,
    )
    children = [
        ChildPrefix(
            prompt=str(row.get("prompt", "")),
            response_token_ids=[int(x) for x in row.get("prefix_token_ids", [])],
        )
        for row in prefix_records
    ]
    scores = teacher.score_children(children, batch_size=batch_size)
    scored = []
    for row, score in zip(prefix_records, scores):
        item = dict(row)
        item["prefix_pvm_score"] = float(score)
        scored.append(item)
    return scored


def assign_pvm_groups(prefix_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored = [row for row in prefix_records if row.get("prefix_pvm_score") is not None]
    if not scored:
        return [dict(row, pvm_group="") for row in prefix_records]
    ordered = sorted(float(row["prefix_pvm_score"]) for row in scored)
    n = len(ordered)
    low_cut = ordered[max(0, min(n - 1, n // 3 - 1))]
    high_cut = ordered[max(0, min(n - 1, (2 * n) // 3 - 1))]
    out: list[dict[str, Any]] = []
    for row in prefix_records:
        item = dict(row)
        score = item.get("prefix_pvm_score")
        if score is None:
            item["pvm_group"] = ""
        elif float(score) <= low_cut:
            item["pvm_group"] = "low"
        elif float(score) <= high_cut:
            item["pvm_group"] = "mid"
        else:
            item["pvm_group"] = "high"
        out.append(item)
    return out
