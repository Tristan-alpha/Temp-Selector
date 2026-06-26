"""Opportunity, temperature, and negative-control metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def delta_label(delta: float) -> str:
    return f"{float(delta):.2f}"


def mean(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def median_or_none(values: Iterable[float]) -> float | None:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return None
    return float(median(vals))


def bootstrap_ci(values: Sequence[float], *, n_bootstrap: int, seed: int) -> tuple[float | None, float | None]:
    vals = np.asarray([float(v) for v in values], dtype=float)
    if vals.size == 0:
        return None, None
    if vals.size == 1 or n_bootstrap <= 0:
        value = float(vals.mean())
        return value, value
    rng = np.random.default_rng(seed)
    means = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(vals, size=vals.size, replace=True)
        means[i] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def group_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def build_prefix_summaries(
    prefix_records: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    deltas: Sequence[float],
    default_delta: float,
) -> list[dict[str, Any]]:
    candidates_by_prefix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in candidate_records:
        candidates_by_prefix[str(row["prefix_id"])].append(row)

    summaries: list[dict[str, Any]] = []
    for prefix in prefix_records:
        prefix_id = str(prefix["prefix_id"])
        candidates = candidates_by_prefix.get(prefix_id, [])
        final_greedy = [row for row in candidates if bool(row.get("is_final_greedy"))]
        if len(final_greedy) != 1:
            raise ValueError(f"expected exactly one final greedy candidate for {prefix_id}, got {len(final_greedy)}")
        greedy = final_greedy[0]
        v_g = float(greedy["child_pvm_score"])
        alternatives = [row for row in candidates if not bool(row.get("is_final_greedy"))]
        final_alts = [row for row in alternatives if bool(row.get("appears_final_topk_alt"))]
        near_alts = [row for row in alternatives if bool(row.get("appears_near_final_topk_alt"))]

        best_final = max(final_alts, key=lambda row: float(row["child_pvm_score"]), default=None)
        best_near = max(near_alts, key=lambda row: float(row["child_pvm_score"]), default=None)
        best_all = max(alternatives, key=lambda row: float(row["child_pvm_score"]), default=None)

        final_delta = float(best_final["child_pvm_score"]) - v_g if best_final else None
        near_delta = float(best_near["child_pvm_score"]) - v_g if best_near else None
        all_delta = float(best_all["child_pvm_score"]) - v_g if best_all else None
        near_advantage = (
            float(best_near["child_pvm_score"]) - float(best_final["child_pvm_score"])
            if best_near and best_final
            else None
        )
        if best_all is None:
            best_source_type = ""
        elif bool(best_all.get("appears_final_topk_alt")):
            best_source_type = "final_topk_alt"
        else:
            best_source_type = "near_final_topk_alt"

        out = {
            "prefix_id": prefix_id,
            "problem_id": prefix.get("problem_id", ""),
            "sample_id": prefix.get("sample_id", ""),
            "pvm_group": prefix.get("pvm_group", ""),
            "prefix_pvm_score": prefix.get("prefix_pvm_score"),
            "relative_position": prefix.get("relative_position"),
            "relative_position_decile": prefix.get("relative_position_decile"),
            "final_correct": prefix.get("final_correct"),
            "V_g": v_g,
            "V_best_final_alt": float(best_final["child_pvm_score"]) if best_final else None,
            "V_best_near_final": float(best_near["child_pvm_score"]) if best_near else None,
            "V_best_all": float(best_all["child_pvm_score"]) if best_all else None,
            "best_final_alt_delta": final_delta,
            "best_near_final_delta": near_delta,
            "best_all_delta": all_delta,
            "near_final_advantage": near_advantage,
            "best_alt_token_id": best_all.get("candidate_token_id") if best_all else None,
            "best_alt_token_text": best_all.get("candidate_token_text") if best_all else None,
            "best_source_type": best_source_type,
            "default_delta": float(default_delta),
        }
        for delta in deltas:
            label = delta_label(delta)
            out[f"overall_opportunity_{label}"] = bool(all_delta is not None and all_delta > delta)
            out[f"final_opportunity_{label}"] = bool(final_delta is not None and final_delta > delta)
            out[f"near_final_opportunity_{label}"] = bool(near_delta is not None and near_delta > delta)
            out[f"near_final_only_opportunity_{label}"] = bool(
                near_delta is not None
                and near_delta > delta
                and not (final_delta is not None and final_delta > delta)
            )
        summaries.append(out)
    return summaries


def opportunity_table_by_group(
    prefix_summaries: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    deltas: Sequence[float],
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group, items in sorted(group_by(prefix_summaries, group_key).items()):
        if group == "":
            continue
        out: dict[str, Any] = {
            group_key: group,
            "num_prefixes": len(items),
            "mean_prefix_pvm": mean(row.get("prefix_pvm_score") for row in items),
            "mean_best_all_delta": mean(row.get("best_all_delta") for row in items),
            "mean_best_final_alt_delta": mean(row.get("best_final_alt_delta") for row in items),
            "mean_best_near_final_delta": mean(row.get("best_near_final_delta") for row in items),
            "mean_near_final_advantage": mean(row.get("near_final_advantage") for row in items),
        }
        for delta in deltas:
            label = delta_label(delta)
            for name in ("overall", "final", "near_final", "near_final_only"):
                values = [1.0 if row.get(f"{name}_opportunity_{label}") else 0.0 for row in items]
                ci_low, ci_high = bootstrap_ci(values, n_bootstrap=n_bootstrap, seed=seed)
                out[f"{name}_opportunity_rate_delta_{label}"] = mean(values)
                out[f"{name}_opportunity_ci_low_delta_{label}"] = ci_low
                out[f"{name}_opportunity_ci_high_delta_{label}"] = ci_high
        rows.append(out)
    return rows


def best_candidate_source_table(
    prefix_summaries: Sequence[Mapping[str, Any]],
    candidate_records: Sequence[Mapping[str, Any]],
    *,
    default_delta: float,
) -> list[dict[str, Any]]:
    label = delta_label(default_delta)
    opportunity_prefixes = {
        str(row["prefix_id"])
        for row in prefix_summaries
        if bool(row.get(f"overall_opportunity_{label}"))
    }
    all_prefix_count = len(prefix_summaries)
    opportunity_count = len(opportunity_prefixes)
    by_key = {
        (str(row["prefix_id"]), int(row["candidate_token_id"])): row
        for row in candidate_records
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for summary in prefix_summaries:
        token_id = summary.get("best_alt_token_id")
        if token_id is None:
            continue
        source_type = str(summary.get("best_source_type", ""))
        candidate = by_key.get((str(summary["prefix_id"]), int(token_id)))
        if candidate is not None:
            grouped[source_type].append(candidate)

    rows: list[dict[str, Any]] = []
    for source_type, items in sorted(grouped.items()):
        opp_items = [row for row in items if str(row["prefix_id"]) in opportunity_prefixes]
        use_items = opp_items or items
        rows.append({
            "source_type": source_type,
            "count": len(items),
            "percentage_all_prefixes": len(items) / max(1, all_prefix_count),
            "percentage_opportunity_prefixes": len(opp_items) / max(1, opportunity_count),
            "mean_child_pvm": mean(row.get("child_pvm_score") for row in use_items),
            "mean_delta_vs_greedy": mean(row.get("delta_vs_final_greedy") for row in use_items),
            "median_rank": median_or_none(row.get("best_source_rank") for row in use_items),
            "mean_source_probability": mean(row.get("max_source_prob") for row in use_items),
        })
    return rows


def high_value_rank_table(
    source_records: Sequence[Mapping[str, Any]],
    *,
    default_delta: float,
) -> list[dict[str, Any]]:
    high_value = [
        row for row in source_records
        if row.get("source_type") in {"final_topk_alt", "near_final_topk_alt"}
        and float(row.get("delta_vs_final_greedy", 0.0)) > float(default_delta)
    ]
    rows: list[dict[str, Any]] = []
    for source_type, items in sorted(group_by(high_value, "source_type").items()):
        ranks = [int(row["candidate_rank"]) for row in items]
        probs = [float(row["candidate_prob"]) for row in items]
        rows.append({
            "source_type": source_type,
            "n_high_value_source_records": len(items),
            "mean_rank": mean(ranks),
            "median_rank": median_or_none(ranks),
            "p_rank_le_3": mean(1.0 if rank <= 3 else 0.0 for rank in ranks),
            "p_rank_le_5": mean(1.0 if rank <= 5 else 0.0 for rank in ranks),
            "p_rank_le_10": mean(1.0 if rank <= 10 else 0.0 for rank in ranks),
            "p_rank_le_20": mean(1.0 if rank <= 20 else 0.0 for rank in ranks),
            "mean_prob": mean(probs),
            "median_prob": median_or_none(probs),
        })
    return rows


def temperature_weights(probs: Sequence[float], temperature: float) -> list[float]:
    if not probs:
        return []
    if float(temperature) == 0.0:
        weights = [0.0] * len(probs)
        weights[0] = 1.0
        return weights
    arr = np.asarray([max(float(p), 1e-300) for p in probs], dtype=float)
    arr = np.power(arr, 1.0 / float(temperature))
    total = float(arr.sum())
    if total <= 0.0:
        return [1.0 / len(probs)] * len(probs)
    return [float(x / total) for x in arr]


def _best_near_layer(prefix: Mapping[str, Any], source_rows: list[Mapping[str, Any]]) -> int | None:
    final_layer = int(prefix["final_layer"])
    final_greedy = int(prefix["final_greedy_token_id"])
    best_layer = None
    best_score = -float("inf")
    for layer, items in group_by(source_rows, "source_layer").items():
        layer_id = int(layer)
        if layer_id == final_layer:
            continue
        alt_scores = [
            float(row["child_pvm_score"])
            for row in items
            if int(row["candidate_token_id"]) != final_greedy
        ]
        if not alt_scores:
            continue
        score = max(alt_scores)
        if score > best_score:
            best_score = score
            best_layer = layer_id
    return best_layer


def per_prefix_temperature_rows(
    prefix_records: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    *,
    temperatures: Sequence[float],
    default_delta: float,
) -> list[dict[str, Any]]:
    sources_by_prefix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_records:
        sources_by_prefix[str(row["prefix_id"])].append(row)
    rows: list[dict[str, Any]] = []
    for prefix in prefix_records:
        prefix_id = str(prefix["prefix_id"])
        source_rows = sources_by_prefix.get(prefix_id, [])
        layer_choices = [("final_layer", int(prefix["final_layer"]))]
        near_layer = _best_near_layer(prefix, source_rows)
        if near_layer is not None:
            layer_choices.append(("near_final_oracle_layer", near_layer))
        v_g = float(prefix["final_greedy_child_pvm_score"])
        for source_family, layer_id in layer_choices:
            items = sorted(
                [row for row in source_rows if int(row["source_layer"]) == int(layer_id)],
                key=lambda row: int(row["candidate_rank"]),
            )
            if not items:
                continue
            probs = [float(row["candidate_prob"]) for row in items]
            scores = [float(row["child_pvm_score"]) for row in items]
            plus = [score > v_g + float(default_delta) for score in scores]
            minus = [score < v_g - float(default_delta) for score in scores]
            p0_plus = None
            p0_minus = None
            for temp in temperatures:
                weights = temperature_weights(probs, float(temp))
                p_plus = sum(weight for weight, is_plus in zip(weights, plus) if is_plus)
                p_minus = sum(weight for weight, is_minus in zip(weights, minus) if is_minus)
                expected = sum(weight * score for weight, score in zip(weights, scores))
                if float(temp) == 0.0:
                    p0_plus = p_plus
                    p0_minus = p_minus
                denom = (p_minus - (p0_minus or 0.0)) + 1e-12
                rows.append({
                    "prefix_id": prefix_id,
                    "pvm_group": prefix.get("pvm_group", ""),
                    "source_family": source_family,
                    "source_layer": layer_id,
                    "temperature": float(temp),
                    "P_T_A_plus": float(p_plus),
                    "P_T_A_minus": float(p_minus),
                    "expected_child_pvm": float(expected),
                    "elicitation_ratio": float((p_plus - (p0_plus or 0.0)) / denom),
                    "delta": float(default_delta),
                    "conditioned_on_saved_topk": True,
                })
    return rows


def temperature_aggregate_table(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("pvm_group", "")), str(row["source_family"]), float(row["temperature"]))].append(row)
    out: list[dict[str, Any]] = []
    for (pvm_group, source_family, temperature), items in sorted(grouped.items()):
        if not pvm_group:
            continue
        out.append({
            "pvm_group": pvm_group,
            "source_family": source_family,
            "temperature": temperature,
            "num_prefixes": len({str(row["prefix_id"]) for row in items}),
            "P_T_A_plus": mean(row["P_T_A_plus"] for row in items),
            "P_T_A_minus": mean(row["P_T_A_minus"] for row in items),
            "elicitation_ratio": mean(row["elicitation_ratio"] for row in items),
            "expected_child_pvm": mean(row["expected_child_pvm"] for row in items),
        })
    return out


def negative_control_table(
    prefix_summaries: Sequence[Mapping[str, Any]],
    source_records: Sequence[Mapping[str, Any]],
    temperature_rows: Sequence[Mapping[str, Any]],
    *,
    default_delta: float,
    n_bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    label = delta_label(default_delta)
    sources_by_prefix: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in source_records:
        sources_by_prefix[str(row["prefix_id"])].append(row)
    rng = random.Random(seed)
    diffs: list[float] = []
    for summary in prefix_summaries:
        if not summary.get(f"overall_opportunity_{label}"):
            continue
        prefix_id = str(summary["prefix_id"])
        best_token = summary.get("best_alt_token_id")
        if best_token is None:
            continue
        source_type = str(summary.get("best_source_type", ""))
        same_source = [
            row for row in sources_by_prefix.get(prefix_id, [])
            if row.get("source_type") == source_type
            and int(row["candidate_token_id"]) == int(best_token)
        ]
        if not same_source:
            continue
        best_source = min(same_source, key=lambda row: int(row["candidate_rank"]))
        best_rank = int(best_source["candidate_rank"])
        candidates = [
            row for row in sources_by_prefix[prefix_id]
            if row.get("source_type") == source_type
            and int(row["candidate_token_id"]) != int(best_token)
            and int(row["candidate_rank"]) == best_rank
        ]
        if not candidates:
            if best_rank <= 3:
                rank_cap = 3
            elif best_rank <= 5:
                rank_cap = 5
            else:
                rank_cap = 20
            candidates = [
                row for row in sources_by_prefix[prefix_id]
                if row.get("source_type") == source_type
                and int(row["candidate_token_id"]) != int(best_token)
                and int(row["candidate_rank"]) <= rank_cap
            ]
        if not candidates:
            continue
        control = rng.choice(candidates)
        diffs.append(float(best_source["child_pvm_score"]) - float(control["child_pvm_score"]))
    ci_low, ci_high = bootstrap_ci(diffs, n_bootstrap=n_bootstrap, seed=seed)
    significance = None
    if diffs:
        arr = np.asarray(diffs, dtype=float)
        boot = []
        rng_np = np.random.default_rng(seed)
        for _ in range(max(1, int(n_bootstrap))):
            boot.append(float(rng_np.choice(arr, size=arr.size, replace=True).mean()))
        significance = sum(1 for item in boot if item > 0.0) / len(boot)

    rows = [{
        "comparison": "best_alt_vs_random_same_rank",
        "n": len(diffs),
        "mean_delta": mean(diffs),
        "median_delta": median_or_none(diffs),
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "p_value_or_bootstrap_significance": significance,
    }]

    opportunity_prefixes = {
        str(row["prefix_id"])
        for row in prefix_summaries
        if bool(row.get(f"overall_opportunity_{label}"))
    }
    final_high = [
        row for row in temperature_rows
        if row.get("source_family") == "final_layer" and float(row["temperature"]) == 1.0
    ]
    for comparison, items in (
        ("fixed_high_temperature_final_all", final_high),
        (
            "fixed_high_temperature_final_opportunity_prefixes",
            [row for row in final_high if str(row["prefix_id"]) in opportunity_prefixes],
        ),
    ):
        rows.append({
            "comparison": comparison,
            "n": len(items),
            "mean_delta": None,
            "median_delta": None,
            "bootstrap_ci_low": None,
            "bootstrap_ci_high": None,
            "p_value_or_bootstrap_significance": None,
            "P_T_A_plus": mean(row["P_T_A_plus"] for row in items),
            "P_T_A_minus": mean(row["P_T_A_minus"] for row in items),
            "expected_child_pvm": mean(row["expected_child_pvm"] for row in items),
            "elicitation_ratio": mean(row["elicitation_ratio"] for row in items),
        })
    return rows


def top_level_summary(
    prefix_summaries: Sequence[Mapping[str, Any]],
    *,
    deltas: Sequence[float],
    default_delta: float,
    candidate_top_k: int | None,
) -> dict[str, Any]:
    label = delta_label(default_delta)
    return {
        "n_prefixes": len(prefix_summaries),
        "n_problem_ids": len({str(row.get("problem_id", "")) for row in prefix_summaries}),
        "candidate_top_k": candidate_top_k,
        "topk_scope_note": (
            "This first run reuses saved top-k candidates; top-10/top-20 claims require a top_k=20 rerun."
            if candidate_top_k is not None and candidate_top_k < 20
            else "Candidate top-k supports top-20 rank diagnostics."
        ),
        "default_delta": float(default_delta),
        "delta_thresholds": [float(x) for x in deltas],
        "mean_V_g": mean(row["V_g"] for row in prefix_summaries),
        "mean_V_best_all": mean(row.get("V_best_all") for row in prefix_summaries),
        "mean_best_all_delta": mean(row.get("best_all_delta") for row in prefix_summaries),
        "overall_opportunity_rate_default_delta": mean(
            1.0 if row.get(f"overall_opportunity_{label}") else 0.0
            for row in prefix_summaries
        ),
        "final_opportunity_rate_default_delta": mean(
            1.0 if row.get(f"final_opportunity_{label}") else 0.0
            for row in prefix_summaries
        ),
        "near_final_opportunity_rate_default_delta": mean(
            1.0 if row.get(f"near_final_opportunity_{label}") else 0.0
            for row in prefix_summaries
        ),
        "near_final_only_opportunity_rate_default_delta": mean(
            1.0 if row.get(f"near_final_only_opportunity_{label}") else 0.0
            for row in prefix_summaries
        ),
        "interpretation_questions": {
            "high_value_alternative_branch_exists": "overall_opportunity_rate_default_delta > 0",
            "final_temperature_can_sample_it": "final_opportunity_rate_default_delta > 0 plus rank/prob diagnostics",
            "near_final_layers_add_visibility": "near_final_only_opportunity_rate_default_delta > 0",
        },
    }
