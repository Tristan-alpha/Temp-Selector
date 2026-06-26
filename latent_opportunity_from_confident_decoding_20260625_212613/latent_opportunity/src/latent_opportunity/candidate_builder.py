"""Build prefix, candidate, and source-level records from layer traces."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class TokenSource:
    source_type: str
    layer_id: int
    rank: int
    prob: float


def infer_final_correct(row: Mapping[str, Any]) -> bool | None:
    value = row.get("final_correct")
    if value is not None:
        return bool(value)
    if "individual_label" in row:
        try:
            return int(row["individual_label"]) == 0
        except Exception:
            return None
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping) and metadata.get("individual_correct") is not None:
        return bool(metadata["individual_correct"])
    return None


def relative_decile(value: float | None) -> int | None:
    if value is None or math.isnan(float(value)):
        return None
    return min(9, max(0, int(float(value) * 10.0)))


def trace_prefix_record(row: Mapping[str, Any]) -> dict[str, Any]:
    prefix_id = str(row.get("trace_id") or f"{row.get('problem_id')}::tok{row.get('token_index')}")
    raw_prefix_score = row.get("pvm_score_prefix", row.get("pvm_phi"))
    try:
        prefix_pvm_score = float(raw_prefix_score) if raw_prefix_score is not None else None
    except Exception:
        prefix_pvm_score = None
    return {
        "prefix_id": prefix_id,
        "problem_id": str(row.get("problem_id", "")),
        "sample_id": str(row.get("source_sample_id", "")),
        "dataset_name": str(row.get("dataset_name", "")),
        "split": str(row.get("split", "")),
        "prompt": str(row.get("prompt", "")),
        "full_generated_text": str(row.get("generated_text", "")),
        "prefix_text": str(row.get("prefix_text", "")),
        "prefix_token_ids": [int(x) for x in row.get("prefix_token_ids", [])],
        "token_position": int(row.get("token_index", row.get("position", 0))),
        "relative_position": float(row.get("relative_position", 0.0)),
        "relative_position_decile": relative_decile(float(row.get("relative_position", 0.0))),
        "prefix_pvm_score": prefix_pvm_score,
        "pvm_group": "",
        "final_answer": row.get("final_answer"),
        "final_correct": infer_final_correct(row),
    }


def build_target_lookup(target_rows: list[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in target_rows:
        key = (str(row["trace_id"]), int(row["layer_id"]))
        if key in lookup:
            raise ValueError(f"duplicate target row for {key}")
        lookup[key] = row
    return lookup


def token_text_map(
    token_ids: set[int],
    model_name_or_path: str | None = None,
) -> dict[int, str]:
    if not model_name_or_path:
        return {token_id: f"<tok:{token_id}>" for token_id in token_ids}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        return {
            token_id: tokenizer.decode([token_id], skip_special_tokens=False)
            for token_id in token_ids
        }
    except Exception:
        return {token_id: f"<tok:{token_id}>" for token_id in token_ids}


def build_records_from_trace_targets(
    trace_rows: list[Mapping[str, Any]],
    target_rows: list[Mapping[str, Any]],
    *,
    model_name_or_path: str | None = None,
    score_tolerance: float = 1e-6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return prefix records, unique candidate records, and source records.

    Candidate rows are unique per ``(prefix_id, token_id)``. Source rows keep
    every layer/rank appearance so rank and visibility analysis can remain
    faithful to the original top-k distributions.
    """

    target_lookup = build_target_lookup(target_rows)
    all_token_ids: set[int] = set()
    for row in trace_rows:
        for layer_tokens in row.get("topk_token_ids_by_layer", []):
            all_token_ids.update(int(x) for x in layer_tokens)
    texts = token_text_map(all_token_ids, model_name_or_path=model_name_or_path)

    prefix_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []

    for trace in trace_rows:
        prefix = trace_prefix_record(trace)
        prefix_id = str(prefix["prefix_id"])
        problem_id = str(prefix["problem_id"])
        sample_id = str(prefix["sample_id"])
        layer_ids = [int(x) for x in trace["layer_ids"]]
        if not layer_ids:
            continue
        final_layer = max(layer_ids)
        layer_offsets = {int(layer_id): i for i, layer_id in enumerate(layer_ids)}
        final_offset = layer_offsets[final_layer]
        final_tokens = [int(x) for x in trace["topk_token_ids_by_layer"][final_offset]]
        final_probs = [float(x) for x in trace["topk_probs_by_layer"][final_offset]]
        if not final_tokens:
            raise ValueError(f"trace {prefix_id} has empty final-layer top-k")
        final_greedy_token_id = int(final_tokens[0])

        candidates: dict[int, dict[str, Any]] = {}
        source_by_token: dict[int, list[TokenSource]] = {}
        source_rows_local: list[dict[str, Any]] = []

        for layer_id in layer_ids:
            offset = layer_offsets[layer_id]
            target = target_lookup.get((prefix_id, layer_id))
            if target is not None:
                tokens = [int(x) for x in target["topk_token_ids"]]
                probs = [float(x) for x in target["topk_probs"]]
                scores = [float(x) for x in target["child_pvm_scores"]]
            else:
                tokens = [int(x) for x in trace["topk_token_ids_by_layer"][offset]]
                probs = [float(x) for x in trace["topk_probs_by_layer"][offset]]
                raw_scores = trace.get("child_pvm_scores_by_layer")
                if raw_scores is None:
                    raise ValueError(
                        f"trace {prefix_id} lacks target scores for layer {layer_id}; "
                        "provide value_targets or child_pvm_scores_by_layer"
                    )
                scores = [float(x) for x in raw_scores[offset]]
            if not (len(tokens) == len(probs) == len(scores)):
                raise ValueError(f"top-k token/prob/score length mismatch for {prefix_id} layer {layer_id}")

            for zero_rank, (token_id, prob, score) in enumerate(zip(tokens, probs, scores)):
                rank = zero_rank + 1
                if layer_id == final_layer and rank == 1:
                    source_type = "final_greedy"
                elif layer_id == final_layer:
                    source_type = "final_topk_alt"
                else:
                    source_type = "near_final_topk_alt"
                token_id = int(token_id)
                prior = candidates.get(token_id)
                if prior is None:
                    candidates[token_id] = {
                        "prefix_id": prefix_id,
                        "problem_id": problem_id,
                        "sample_id": sample_id,
                        "candidate_token_id": token_id,
                        "candidate_token_text": texts.get(token_id, f"<tok:{token_id}>"),
                        "child_pvm_score": float(score),
                        "is_final_greedy": token_id == final_greedy_token_id,
                        "appears_final_topk_alt": False,
                        "appears_near_final_topk_alt": False,
                        "source_types": [],
                    }
                elif abs(float(prior["child_pvm_score"]) - float(score)) > float(score_tolerance):
                    raise ValueError(
                        "same child prefix received inconsistent PVM scores: "
                        f"prefix_id={prefix_id} token_id={token_id} "
                        f"{prior['child_pvm_score']} vs {score}"
                    )

                if source_type == "final_topk_alt":
                    candidates[token_id]["appears_final_topk_alt"] = True
                if source_type == "near_final_topk_alt":
                    candidates[token_id]["appears_near_final_topk_alt"] = True
                if source_type not in candidates[token_id]["source_types"]:
                    candidates[token_id]["source_types"].append(source_type)
                source_by_token.setdefault(token_id, []).append(
                    TokenSource(source_type=source_type, layer_id=layer_id, rank=rank, prob=float(prob))
                )
                source_rows_local.append({
                    "prefix_id": prefix_id,
                    "problem_id": problem_id,
                    "sample_id": sample_id,
                    "source_type": source_type,
                    "source_layer": int(layer_id),
                    "candidate_rank": int(rank),
                    "candidate_token_id": token_id,
                    "candidate_token_text": texts.get(token_id, f"<tok:{token_id}>"),
                    "candidate_prob": float(prob),
                    "candidate_logprob": math.log(max(float(prob), 1e-300)),
                    "child_pvm_score": float(score),
                })

        greedy = candidates.get(final_greedy_token_id)
        if greedy is None:
            raise ValueError(f"final greedy token missing from candidates for {prefix_id}")
        greedy_score = float(greedy["child_pvm_score"])
        final_rank = {int(token_id): i + 1 for i, token_id in enumerate(final_tokens)}
        final_prob = {int(token_id): float(prob) for token_id, prob in zip(final_tokens, final_probs)}

        for token_id, candidate in sorted(candidates.items()):
            srcs = source_by_token.get(token_id, [])
            ranks = [src.rank for src in srcs]
            probs = [src.prob for src in srcs]
            source_types = sorted({src.source_type for src in srcs})
            candidate["source_types"] = source_types
            candidate["best_source_rank"] = min(ranks) if ranks else None
            candidate["max_source_prob"] = max(probs) if probs else None
            candidate["rank_in_final_layer"] = final_rank.get(token_id)
            candidate["prob_in_final_layer"] = final_prob.get(token_id)
            candidate["final_greedy_token_id"] = final_greedy_token_id
            candidate["final_greedy_child_pvm_score"] = greedy_score
            candidate["delta_vs_final_greedy"] = float(candidate["child_pvm_score"]) - greedy_score
            candidate["is_duplicate_candidate"] = len(srcs) > 1
            candidate_records.append(dict(candidate))

        for source in source_rows_local:
            token_id = int(source["candidate_token_id"])
            source["rank_in_final_layer"] = final_rank.get(token_id)
            source["prob_in_final_layer"] = final_prob.get(token_id)
            source["final_greedy_token_id"] = final_greedy_token_id
            source["final_greedy_child_pvm_score"] = greedy_score
            source["delta_vs_final_greedy"] = float(source["child_pvm_score"]) - greedy_score
            source_records.append(source)

        prefix["final_layer"] = final_layer
        prefix["final_greedy_token_id"] = final_greedy_token_id
        prefix["final_greedy_token_text"] = texts.get(final_greedy_token_id, f"<tok:{final_greedy_token_id}>")
        prefix["final_greedy_child_pvm_score"] = greedy_score
        prefix["candidate_top_k"] = max(len(x) for x in trace.get("topk_token_ids_by_layer", [[]]))
        prefix_records.append(prefix)

    return prefix_records, candidate_records, source_records
