"""Standard records and metrics for segment-level latent path analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


PVM_GROUP_ORDER = {"low": 0, "mid": 1, "high": 2}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(dict(row)), ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_safe(dict(data)), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(str(key))
                seen.add(str(key))
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_safe(row.get(key)) for key in fields})


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def csv_safe(value: Any) -> Any:
    value = json_safe(value)
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def delta_label(delta: float) -> str:
    return f"{float(delta):.2f}"


def temperature_key(temperature: float) -> str:
    return str(float(temperature))


def _mean_or_none(values: Iterable[float | int | None]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return sum(vals) / len(vals) if vals else None


def _segment_key(row: Mapping[str, Any]) -> str:
    token_ids = row.get("segment_token_ids")
    if isinstance(token_ids, list):
        material = json.dumps([int(item) for item in token_ids], separators=(",", ":"))
    else:
        material = str(row.get("segment_text", ""))
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


def assign_pvm_groups(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Assign low/mid/high tertiles from prefix PVM scores."""
    scores_by_prefix: dict[str, float] = {}
    for row in rows:
        if row.get("prefix_pvm_score") is None:
            continue
        scores_by_prefix.setdefault(str(row["prefix_id"]), float(row["prefix_pvm_score"]))
    ordered = sorted(scores_by_prefix.items(), key=lambda item: (item[1], item[0]))
    n = len(ordered)
    groups: dict[str, str] = {}
    if n == 0:
        return groups
    for idx, (prefix_id, _score) in enumerate(ordered):
        if idx < n / 3:
            groups[prefix_id] = "low"
        elif idx < 2 * n / 3:
            groups[prefix_id] = "mid"
        else:
            groups[prefix_id] = "high"
    return groups


def standardize_segment_records(raw_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw generated candidates into the standard segment record shape.

    Reward is empirical success rate for duplicate child segments within a prefix.
    If correctness is absent for a candidate, the reward is left as None.
    """
    by_prefix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        by_prefix[str(row["prefix_id"])].append(row)

    pvm_groups = assign_pvm_groups(raw_rows)
    records: list[dict[str, Any]] = []
    for prefix_id, rows in sorted(by_prefix.items()):
        greedy_rows = [row for row in rows if str(row.get("candidate_role", "")) == "greedy"]
        sample_rows = [row for row in rows if str(row.get("candidate_role", "")) != "greedy"]
        if not greedy_rows:
            raise ValueError(f"prefix {prefix_id} has no greedy candidate")

        reward_by_segment: dict[str, float | None] = {}
        count_by_segment: dict[str, int] = {}
        correct_by_segment: dict[str, int] = {}
        for row in rows:
            key = _segment_key(row)
            if row.get("correct") is None:
                continue
            count_by_segment[key] = count_by_segment.get(key, 0) + 1
            correct_by_segment[key] = correct_by_segment.get(key, 0) + int(bool(row.get("correct")))
        for key, n_total in count_by_segment.items():
            reward_by_segment[key] = correct_by_segment[key] / max(1, n_total)

        first = greedy_rows[0]
        greedy_key = _segment_key(first)
        greedy_reward = reward_by_segment.get(greedy_key)
        greedy = {
            "segment": str(first.get("segment_text", "")),
            "segment_token_ids": list(first.get("segment_token_ids", [])),
            "child_pvm": float(first["child_pvm_score"]) if first.get("child_pvm_score") is not None else None,
            "reward": greedy_reward,
            "correct": bool(first.get("correct")) if first.get("correct") is not None else None,
            "n_success": correct_by_segment.get(greedy_key),
            "n_total": count_by_segment.get(greedy_key),
        }

        samples: list[dict[str, Any]] = []
        for row in sorted(sample_rows, key=lambda item: (
            float(item.get("temperature", 0.0)),
            int(item.get("seed_index", 0)),
            str(item.get("candidate_id", "")),
        )):
            key = _segment_key(row)
            samples.append({
                "temperature": float(row["temperature"]),
                "seed_index": int(row.get("seed_index", 0)),
                "generation_seed": int(row.get("generation_seed", 0)),
                "segment": str(row.get("segment_text", "")),
                "segment_token_ids": list(row.get("segment_token_ids", [])),
                "child_pvm": (
                    float(row["child_pvm_score"])
                    if row.get("child_pvm_score") is not None else None
                ),
                "reward": reward_by_segment.get(key),
                "correct": bool(row.get("correct")) if row.get("correct") is not None else None,
                "n_success": correct_by_segment.get(key),
                "n_total": count_by_segment.get(key),
                "candidate_id": row.get("candidate_id", ""),
            })

        records.append({
            "problem_id": str(first.get("problem_id", "")),
            "prefix_id": prefix_id,
            "source_sample_id": str(first.get("source_sample_id", "")),
            "prefix_text": str(first.get("prefix_text", "")),
            "prefix_token_end": int(first.get("prefix_token_end", 0)),
            "prefix_segments": int(first.get("prefix_segments", 0)),
            "prefix_pvm_score": (
                float(first["prefix_pvm_score"]) if first.get("prefix_pvm_score") is not None else None
            ),
            "pvm_group": str(first.get("pvm_group") or pvm_groups.get(prefix_id, "")),
            "greedy": greedy,
            "samples": samples,
        })
    return records


def opportunity_by_pvm_group(
    records: Sequence[Mapping[str, Any]],
    *,
    deltas: Sequence[float],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("pvm_group", ""))].append(record)
    rows: list[dict[str, Any]] = []
    for group, items in sorted(grouped.items(), key=lambda item: PVM_GROUP_ORDER.get(item[0], 99)):
        if not group:
            continue
        out: dict[str, Any] = {
            "pvm_group": group,
            "num_prefixes": len(items),
            "mean_prefix_pvm_score": _mean_or_none(row.get("prefix_pvm_score") for row in items),
            "mean_greedy_reward": _mean_or_none(row.get("greedy", {}).get("reward") for row in items),
            "mean_best_sample_reward": _mean_or_none(
                max(
                    [float(sample["reward"]) for sample in row.get("samples", []) if sample.get("reward") is not None],
                    default=None,
                )
                for row in items
            ),
        }
        for delta in deltas:
            label = delta_label(delta)
            values: list[float] = []
            for record in items:
                greedy_reward = record.get("greedy", {}).get("reward")
                sample_rewards = [
                    float(sample["reward"])
                    for sample in record.get("samples", [])
                    if sample.get("reward") is not None
                ]
                if greedy_reward is None or not sample_rewards:
                    continue
                values.append(1.0 if max(sample_rewards) > float(greedy_reward) + float(delta) else 0.0)
            out[f"opportunity_rate_delta_{label}"] = _mean_or_none(values)
            out[f"num_evaluable_delta_{label}"] = len(values)
        rows.append(out)
    return rows


def proposal_yield_by_temperature(
    records: Sequence[Mapping[str, Any]],
    *,
    delta: float,
) -> list[dict[str, Any]]:
    by_temp: dict[float, list[tuple[Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for record in records:
        for sample in record.get("samples", []):
            by_temp[float(sample["temperature"])].append((record, sample))
    rows: list[dict[str, Any]] = []
    for temp, items in sorted(by_temp.items()):
        row_success: list[float] = []
        rewards: list[float] = []
        by_prefix: dict[str, list[float]] = defaultdict(list)
        by_prefix_greedy: dict[str, float] = {}
        for record, sample in items:
            greedy_reward = record.get("greedy", {}).get("reward")
            reward = sample.get("reward")
            if greedy_reward is None or reward is None:
                continue
            reward_f = float(reward)
            greedy_f = float(greedy_reward)
            row_success.append(1.0 if reward_f > greedy_f + float(delta) else 0.0)
            rewards.append(reward_f)
            prefix_id = str(record["prefix_id"])
            by_prefix[prefix_id].append(reward_f)
            by_prefix_greedy[prefix_id] = greedy_f
        best_values = [
            1.0 if max(values) > by_prefix_greedy[prefix_id] + float(delta) else 0.0
            for prefix_id, values in by_prefix.items()
            if values
        ]
        rows.append({
            "temperature": temp,
            "delta": float(delta),
            "n_samples": len(row_success),
            "n_prefixes": len(by_prefix),
            "yield": _mean_or_none(row_success),
            "best_of_n_yield": _mean_or_none(best_values),
            "mean_sample_reward": _mean_or_none(rewards),
            "mean_best_reward": _mean_or_none(max(values) for values in by_prefix.values() if values),
        })
    return rows


def selection_gain(
    records: Sequence[Mapping[str, Any]],
    *,
    scopes: Sequence[str] = ("all", "low_pvm"),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        if scope == "all":
            use_records = list(records)
        elif scope == "low_pvm":
            use_records = [row for row in records if row.get("pvm_group") == "low"]
        else:
            raise ValueError(f"unknown selection scope: {scope}")

        greedy_values: list[float] = []
        random_values: list[float] = []
        pvm_best_values: list[float] = []
        for record in use_records:
            greedy_reward = record.get("greedy", {}).get("reward")
            samples = [
                sample for sample in record.get("samples", [])
                if sample.get("reward") is not None and sample.get("child_pvm") is not None
            ]
            if greedy_reward is None or not samples:
                continue
            greedy_values.append(float(greedy_reward))
            random_values.append(mean(float(sample["reward"]) for sample in samples))
            best = max(
                samples,
                key=lambda sample: (
                    float(sample["child_pvm"]),
                    -float(sample["temperature"]),
                    -int(sample.get("seed_index", 0)),
                ),
            )
            pvm_best_values.append(float(best["reward"]))
        greedy_acc = _mean_or_none(greedy_values)
        random_acc = _mean_or_none(random_values)
        pvm_best_acc = _mean_or_none(pvm_best_values)
        rows.append({
            "scope": scope,
            "n_prefixes": len(greedy_values),
            "Acc_greedy": greedy_acc,
            "Acc_random_sampled": random_acc,
            "Acc_PVM_best": pvm_best_acc,
            "PVM_best_minus_random": (
                pvm_best_acc - random_acc
                if pvm_best_acc is not None and random_acc is not None else None
            ),
            "PVM_best_minus_greedy": (
                pvm_best_acc - greedy_acc
                if pvm_best_acc is not None and greedy_acc is not None else None
            ),
        })
    return rows


def prefix_summary_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    default_delta: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        greedy_reward = record.get("greedy", {}).get("reward")
        sample_rewards = [
            float(sample["reward"])
            for sample in record.get("samples", [])
            if sample.get("reward") is not None
        ]
        child_scores = [
            float(sample["child_pvm"])
            for sample in record.get("samples", [])
            if sample.get("child_pvm") is not None
        ]
        best_reward = max(sample_rewards, default=None)
        rows.append({
            "prefix_id": record.get("prefix_id", ""),
            "problem_id": record.get("problem_id", ""),
            "source_sample_id": record.get("source_sample_id", ""),
            "pvm_group": record.get("pvm_group", ""),
            "prefix_pvm_score": record.get("prefix_pvm_score"),
            "greedy_reward": greedy_reward,
            "best_sample_reward": best_reward,
            "best_sample_minus_greedy": (
                best_reward - float(greedy_reward)
                if best_reward is not None and greedy_reward is not None else None
            ),
            "max_sample_child_pvm": max(child_scores, default=None),
            "n_samples": len(record.get("samples", [])),
            "has_opportunity_default_delta": (
                best_reward > float(greedy_reward) + float(default_delta)
                if best_reward is not None and greedy_reward is not None else None
            ),
        })
    return rows


def top_level_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    opportunity_rows: Sequence[Mapping[str, Any]],
    temperature_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    default_delta: float,
) -> dict[str, Any]:
    all_selection = next((row for row in selection_rows if row.get("scope") == "all"), {})
    low_selection = next((row for row in selection_rows if row.get("scope") == "low_pvm"), {})
    return {
        "n_prefixes": len(records),
        "n_problem_ids": len({str(row.get("problem_id", "")) for row in records}),
        "n_samples": sum(len(row.get("samples", [])) for row in records),
        "default_delta": float(default_delta),
        "opportunity_by_group": list(opportunity_rows),
        "temperature_with_best_yield": max(
            temperature_rows,
            key=lambda row: float(row.get("best_of_n_yield") or -1.0),
            default=None,
        ),
        "selection_gain_all": dict(all_selection),
        "selection_gain_low_pvm": dict(low_selection),
    }

