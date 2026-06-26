#!/usr/bin/env python3
"""Train a frozen-backbone segment advantage head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TF_MIL_ROOT = EXPERIMENT_ROOT.parent
SRC = EXPERIMENT_ROOT / "src"
for path in (SRC, TF_MIL_ROOT, EXPERIMENT_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.segment_advantage import (  # noqa: E402
    build_advantage_examples,
    divergence_diagnostics_rows,
    generate_pairwise_indices,
    group_kfold_indices,
    prefix_rank_summary,
    roc_auc,
    selection_gain_rows,
    train_val_split_by_group,
)
from data.segment_records import (  # noqa: E402
    read_jsonl,
    standardize_segment_records,
    write_csv,
    write_json,
    write_jsonl,
)
from features.segmenter import build_masked_concat_segment_obs_from_lp  # noqa: E402
from generate_segment_candidates import (  # noqa: E402
    load_config,
    load_value_model,
    render_prompt,
    resolve_path,
)
from inference.vllm_runner import VLLMFeatureExporter  # noqa: E402
from mil.prefix_data import pad_prefix_entries  # noqa: E402
from mil.prefix_value import calibrated_probability  # noqa: E402


DEFAULT_CONFIG = TF_MIL_ROOT / "configs" / "training" / "min_pvm_ppo_500_seed42.yaml"
DEFAULT_INPUT = (
    EXPERIMENT_ROOT / "outputs" / "segment_latent_paths" / "segment_candidate_records.jsonl"
)
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "segment_advantage_head"


class SegmentAdvantageHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(32, hidden_dim // 4)),
            nn.ReLU(),
            nn.Linear(max(32, hidden_dim // 4), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "extract-hidden", "train"], default="all")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hidden-cache", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--pvm-checkpoint", type=Path, default=None)
    parser.add_argument("--max-prefixes", type=int, default=0)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--bce-loss-weight", type=float, default=0.25)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--train-device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--top-k-logprobs", type=int, default=64)
    parser.add_argument("--feature-temperature", type=float, default=1.0)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-micro-batch-size", type=int, default=None)
    parser.add_argument("--overwrite-hidden", action="store_true")
    return parser.parse_args()


def _select_prefixes(raw_rows: Sequence[Mapping[str, Any]], max_prefixes: int) -> list[dict[str, Any]]:
    if int(max_prefixes) <= 0:
        return [dict(row) for row in raw_rows]
    keep: set[str] = set()
    ordered: list[str] = []
    for row in raw_rows:
        prefix_id = str(row["prefix_id"])
        if prefix_id not in keep:
            keep.add(prefix_id)
            ordered.append(prefix_id)
            if len(ordered) >= int(max_prefixes):
                break
    keep = set(ordered)
    return [dict(row) for row in raw_rows if str(row["prefix_id"]) in keep]


def _strip_example_for_json(example: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(example)
    out.pop("segment_token_ids", None)
    out.pop("greedy_segment_token_ids", None)
    return out


def _tokenizer_decode(tokenizer, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


@torch.no_grad()
def extract_prefix_hidden(
    *,
    records: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    feature_cache_path: Path,
    pvm_checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
    cache_by_id = {str(entry["sample_id"]): entry for entry in cache}
    model, _ = load_value_model(cfg, pvm_checkpoint_path, device)
    result: dict[str, torch.Tensor] = {}
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start:start + int(batch_size)]
        batch = pad_prefix_entries([
            (dict(cache_by_id[str(row["source_sample_id"])]), int(row["prefix_segments"]))
            for row in batch_records
        ])
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        output = model(
            tensor_batch["features"],
            tensor_batch["token_mask"],
            tensor_batch["segment_mask"],
            prompt_hidden=tensor_batch.get("prompt_hidden"),
        )
        hidden = output["terminal_hidden"].detach().cpu().float()
        for row, vector in zip(batch_records, hidden):
            result[str(row["prefix_id"])] = vector
    return result


@torch.no_grad()
def extract_child_hidden(
    *,
    examples: Sequence[Mapping[str, Any]],
    raw_by_candidate: Mapping[str, Mapping[str, Any]],
    cfg: Mapping[str, Any],
    source_dataset_path: Path,
    pvm_checkpoint_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, list[float]]:
    source_rows = read_jsonl(source_dataset_path)
    source_by_id = {str(row["sample_id"]): row for row in source_rows}
    inference_cfg = cfg["inference"]
    runner = VLLMFeatureExporter(
        model_name_or_path=str(inference_cfg["model_name_or_path"]),
        max_new_tokens=int(inference_cfg.get("max_new_tokens", 8192)),
        parallel_size=(
            int(args.parallel_size)
            if args.parallel_size is not None else max(1, torch.cuda.device_count())
        ),
        gpu_memory_utilization=(
            float(args.gpu_memory_utilization)
            if args.gpu_memory_utilization is not None
            else float(inference_cfg.get("gpu_memory_utilization", 0.80))
        ),
        max_batch_size=(
            int(args.vllm_micro_batch_size)
            if args.vllm_micro_batch_size is not None
            else int(inference_cfg.get("vllm_micro_batch_size", 64))
        ),
        enforce_eager=bool(inference_cfg.get("vllm_enforce_eager", False)),
        enable_prefix_caching=False,
    )
    model, calibration_temperature = load_value_model(cfg, pvm_checkpoint_path, device)
    tokenizer = runner.tokenizer
    segment_size = int(cfg["data"]["segment_size"])
    token_dim = int(cfg["data"]["instance_dim"])
    child_hidden: list[torch.Tensor] = []
    child_scores: list[float] = []
    prompt_id_cache: dict[str, list[int]] = {}

    for start in range(0, len(examples), int(args.score_batch_size)):
        end = min(start + int(args.score_batch_size), len(examples))
        batch_examples = examples[start:end]
        full_ids: list[list[int]] = []
        prompt_lens: list[int] = []
        response_ids_list: list[list[int]] = []
        response_texts: list[str] = []
        response_tokens: list[list[str]] = []
        for example in batch_examples:
            raw = raw_by_candidate[str(example["candidate_id"])]
            source_id = str(raw["source_sample_id"])
            if source_id not in source_by_id:
                raise RuntimeError(f"missing source row for sample_id={source_id}")
            if source_id not in prompt_id_cache:
                rendered = render_prompt(tokenizer, source_by_id[source_id], cfg)
                prompt_id_cache[source_id] = [
                    int(item)
                    for item in tokenizer(rendered, add_special_tokens=False).input_ids
                ]
            prompt_ids = prompt_id_cache[source_id]
            response_ids = [int(item) for item in raw.get("prefix_token_ids", [])] + [
                int(item) for item in raw.get("segment_token_ids", [])
            ]
            full_ids.append(prompt_ids + response_ids)
            prompt_lens.append(len(prompt_ids))
            response_ids_list.append(response_ids)
            response_texts.append(_tokenizer_decode(tokenizer, response_ids))
            response_tokens.append([
                str(item) for item in tokenizer.convert_ids_to_tokens(response_ids)
            ])

        extracted = runner.extract_from_ids(
            full_ids,
            prompt_lens,
            temperatures=[float(args.feature_temperature)] * len(full_ids),
            top_k=int(args.top_k_logprobs),
            return_logprobs=True,
            return_hidden=False,
            device=device,
        )
        entries: list[dict[str, torch.Tensor]] = []
        for logprobs, response_ids, tokens, text in zip(
            extracted["logprobs"],
            response_ids_list,
            response_tokens,
            response_texts,
        ):
            masked = build_masked_concat_segment_obs_from_lp(
                logprobs[: len(response_ids)],
                tokens,
                text,
                segment_size=segment_size,
                token_dim=token_dim,
                device=device,
                segment_mode="fixed_window",
            )
            entries.append({
                "features": masked.features.detach().cpu().to(torch.float16),
                "token_mask": masked.token_mask.detach().cpu().to(torch.uint8),
            })

        for entry_start in range(0, len(entries), int(args.score_batch_size)):
            batch_entries = entries[entry_start:entry_start + int(args.score_batch_size)]
            batch = pad_prefix_entries([(entry, None) for entry in batch_entries])
            tensor_batch = {
                key: value.to(device)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
            }
            output = model(
                tensor_batch["features"],
                tensor_batch["token_mask"],
                tensor_batch["segment_mask"],
                prompt_hidden=tensor_batch.get("prompt_hidden"),
            )
            probs = calibrated_probability(output["terminal_logits"], calibration_temperature)
            child_hidden.extend(
                vector.detach().cpu().float()
                for vector in output["terminal_hidden"]
            )
            child_scores.extend(float(value) for value in probs.detach().cpu().tolist())
        print(f"extracted child hidden {end}/{len(examples)}", flush=True)

    return torch.stack(child_hidden, dim=0), child_scores


def build_hidden_cache(args: argparse.Namespace, cfg: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    raw_rows = _select_prefixes(read_jsonl(args.candidates), int(args.max_prefixes))
    records = standardize_segment_records(raw_rows)
    examples = build_advantage_examples(records, delta=float(args.delta))
    raw_by_candidate = {
        str(row.get("candidate_id", "")): row
        for row in raw_rows
        if str(row.get("candidate_role", "")) != "greedy"
    }
    missing = [
        str(example["candidate_id"])
        for example in examples
        if str(example["candidate_id"]) not in raw_by_candidate
    ]
    if missing:
        raise RuntimeError(f"missing raw candidate rows for {len(missing)} examples")

    paths = cfg["paths"]
    feature_cache_path = resolve_path(args.feature_cache or paths["val_feature_cache"], TF_MIL_ROOT)
    pvm_checkpoint_path = resolve_path(args.pvm_checkpoint or paths["prefix_value_ckpt"], TF_MIL_ROOT)
    source_dataset_path = resolve_path(args.source_dataset or paths["val_dataset"], TF_MIL_ROOT)
    assert feature_cache_path is not None
    assert pvm_checkpoint_path is not None
    assert source_dataset_path is not None

    if not torch.cuda.is_available():
        raise RuntimeError("hidden extraction requires CUDA/vLLM")
    device = torch.device("cuda:0")
    print(f"extracting prefix hidden for prefixes={len(records)}", flush=True)
    prefix_hidden_by_id = extract_prefix_hidden(
        records=records,
        cfg=cfg,
        feature_cache_path=feature_cache_path,
        pvm_checkpoint_path=pvm_checkpoint_path,
        device=device,
        batch_size=int(args.score_batch_size),
    )
    print(f"extracting child hidden for samples={len(examples)}", flush=True)
    h_child, child_scores = extract_child_hidden(
        examples=examples,
        raw_by_candidate=raw_by_candidate,
        cfg=cfg,
        source_dataset_path=source_dataset_path,
        pvm_checkpoint_path=pvm_checkpoint_path,
        device=device,
        args=args,
    )
    h_prefix = torch.stack([
        prefix_hidden_by_id[str(example["prefix_id"])]
        for example in examples
    ], dim=0)
    for example, score in zip(examples, child_scores):
        example["recomputed_child_pvm"] = float(score)

    cache = {
        "examples": examples,
        "h_prefix": h_prefix.to(torch.float16).cpu(),
        "h_child": h_child.to(torch.float16).cpu(),
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "candidates": str(args.candidates.resolve()),
            "max_prefixes": int(args.max_prefixes),
            "n_prefixes": len(records),
            "n_examples": len(examples),
            "delta": float(args.delta),
            "config": str(args.config.resolve()),
            "feature_cache": str(feature_cache_path.resolve()),
            "pvm_checkpoint": str(pvm_checkpoint_path.resolve()),
            "source_dataset": str(source_dataset_path.resolve()),
            "top_k_logprobs": int(args.top_k_logprobs),
            "feature_temperature": float(args.feature_temperature),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    hidden_cache = args.hidden_cache or (output_dir / "advantage_hidden_cache.pt")
    torch.save(cache, hidden_cache)
    write_jsonl(output_dir / "advantage_examples.jsonl", [
        _strip_example_for_json(example) for example in examples
    ])
    print(f"wrote hidden cache: {hidden_cache}", flush=True)
    return cache


def _train_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--train-device=cuda requested but CUDA is unavailable")
        return torch.device("cuda:0")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _temperature_values(examples: Sequence[Mapping[str, Any]]) -> list[float]:
    return sorted({float(example["temperature"]) for example in examples})


def _feature_batch(
    h_prefix: torch.Tensor,
    h_child: torch.Tensor,
    temperatures: torch.Tensor,
    temperature_values: Sequence[float],
    indices: Sequence[int] | torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    idx = torch.as_tensor(indices, dtype=torch.long)
    prefix = h_prefix[idx].to(device=device, dtype=torch.float32)
    child = h_child[idx].to(device=device, dtype=torch.float32)
    temps = temperatures[idx].to(device=device, dtype=torch.float32)
    temp_features = torch.zeros((idx.numel(), len(temperature_values)), device=device)
    for col, value in enumerate(temperature_values):
        temp_features[:, col] = (temps == float(value)).to(torch.float32)
    return torch.cat([prefix, child, child - prefix, child * prefix, temp_features], dim=1)


def _precompute_feature_matrix(
    h_prefix: torch.Tensor,
    h_child: torch.Tensor,
    temperatures: torch.Tensor,
    temperature_values: Sequence[float],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    all_indices = list(range(int(h_prefix.shape[0])))
    for start in range(0, len(all_indices), int(batch_size)):
        batch_idx = all_indices[start:start + int(batch_size)]
        chunks.append(_feature_batch(
            h_prefix,
            h_child,
            temperatures,
            temperature_values,
            batch_idx,
            device,
        ).detach())
    return torch.cat(chunks, dim=0)


def _pairwise_loss(
    model: nn.Module,
    features: torch.Tensor,
    pairs: Sequence[tuple[int, int, float]],
    device: torch.device,
) -> torch.Tensor:
    if not pairs:
        return torch.zeros((), device=device)
    good = [item[0] for item in pairs]
    bad = [item[1] for item in pairs]
    weights = torch.tensor([item[2] for item in pairs], dtype=torch.float32, device=device)
    good_x = features[torch.as_tensor(good, dtype=torch.long, device=device)]
    bad_x = features[torch.as_tensor(bad, dtype=torch.long, device=device)]
    return (weights * F.softplus(-(model(good_x) - model(bad_x)))).mean()


def _bce_loss(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    if not indices:
        return torch.zeros((), device=device)
    index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
    x = features[index_tensor]
    y = labels[torch.as_tensor(indices, dtype=torch.long)].to(device=device, dtype=torch.float32)
    if y.sum() > 0 and y.sum() < y.numel():
        pos_weight = ((y.numel() - y.sum()) / y.sum()).clamp(1.0, 20.0)
        return F.binary_cross_entropy_with_logits(model(x), y, pos_weight=pos_weight)
    return F.binary_cross_entropy_with_logits(model(x), y)


@torch.no_grad()
def _predict_scores(
    model: nn.Module,
    features: torch.Tensor,
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> dict[int, float]:
    model.eval()
    scores: dict[int, float] = {}
    for start in range(0, len(indices), int(batch_size)):
        batch_idx = list(indices[start:start + int(batch_size)])
        x = features[torch.as_tensor(batch_idx, dtype=torch.long, device=device)]
        pred = model(x).detach().cpu().tolist()
        for idx, score in zip(batch_idx, pred):
            scores[int(idx)] = float(score)
    return scores


def _evaluate_loss(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    pairs: Sequence[tuple[int, int, float]],
    sample_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    with torch.no_grad():
        pair_loss = _pairwise_loss(
            model,
            features,
            pairs[: min(len(pairs), 8192)],
            device,
        )
        bce = _bce_loss(
            model,
            features,
            labels,
            list(sample_indices)[:8192],
            device,
        )
        return float((pair_loss + float(args.bce_loss_weight) * bce).detach().cpu())


def train_one_fold(
    *,
    fold: int,
    examples: Sequence[Mapping[str, Any]],
    features: torch.Tensor,
    temperature_values: Sequence[float],
    labels: torch.Tensor,
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[int, float]]:
    inner_train, val_indices = train_val_split_by_group(
        examples,
        train_indices,
        group_key="problem_id",
        val_fraction=float(args.val_fraction),
        seed=int(args.seed) + int(fold),
    )
    train_pairs = generate_pairwise_indices(examples, inner_train)
    val_pairs = generate_pairwise_indices(examples, val_indices) if val_indices else []
    if not train_pairs and not inner_train:
        raise RuntimeError(f"fold {fold} has no train data")

    input_dim = int(features.shape[1])
    model = SegmentAdvantageHead(
        input_dim=input_dim,
        hidden_dim=int(args.head_hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    rng = random.Random(int(args.seed) + int(fold) * 1009)
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    patience_left = int(args.patience)
    best_epoch = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        rng.shuffle(train_pairs)
        step_losses: list[float] = []
        n_steps = max(1, math.ceil(len(train_pairs) / max(1, int(args.batch_size))))
        for step in range(n_steps):
            pair_batch = train_pairs[
                step * int(args.batch_size):(step + 1) * int(args.batch_size)
            ]
            if inner_train:
                sample_batch = [
                    inner_train[rng.randrange(len(inner_train))]
                    for _ in range(min(int(args.batch_size), len(inner_train)))
                ]
            else:
                sample_batch = []
            optimizer.zero_grad(set_to_none=True)
            pair_loss = _pairwise_loss(
                model,
                features,
                pair_batch,
                device,
            )
            bce = _bce_loss(
                model,
                features,
                labels,
                sample_batch,
                device,
            )
            loss = pair_loss + float(args.bce_loss_weight) * bce
            loss.backward()
            optimizer.step()
            step_losses.append(float(loss.detach().cpu()))

        eval_indices = val_indices if val_indices else inner_train
        eval_pairs = val_pairs if val_indices else train_pairs
        val_loss = _evaluate_loss(
            model,
            features,
            labels,
            eval_pairs,
            eval_indices,
            args,
            device,
        )
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = int(args.patience)
        else:
            patience_left -= 1
        if epoch == 1 or epoch % 10 == 0 or patience_left <= 0:
            print(
                f"fold={fold} epoch={epoch} train_loss={mean(step_losses):.4f} "
                f"val_loss={val_loss:.4f} best={best_val:.4f}",
                flush=True,
            )
        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "fold": int(fold),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "temperature_values": list(temperature_values),
        "input_dim": int(input_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "dropout": float(args.dropout),
    }, output_dir / f"advantage_head_fold{fold}.pt")

    scores = _predict_scores(
        model,
        features,
        test_indices,
        device,
        batch_size=int(args.batch_size),
    )
    rank_summary = prefix_rank_summary(examples, scores, test_indices)
    labels_test = [float(examples[idx]["better_than_greedy_delta"]) for idx in test_indices]
    scores_test = [scores[idx] for idx in test_indices]
    auc = roc_auc(labels_test, scores_test)
    selection = selection_gain_rows(
        examples,
        scores=scores,
        indices=test_indices,
        delta=float(args.delta),
        scopes=("all",),
    )[0]
    row = {
        "fold": int(fold),
        "n_train_examples": len(inner_train),
        "n_val_examples": len(val_indices),
        "n_test_examples": len(test_indices),
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "better_than_greedy_auc": auc,
        **rank_summary,
        **{f"test_{key}": value for key, value in selection.items() if key != "scope"},
    }
    return row, scores


def plot_outputs(
    *,
    fold_rows: Sequence[Mapping[str, Any]],
    selection_rows: Sequence[Mapping[str, Any]],
    divergence_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    all_row = next((row for row in selection_rows if row.get("scope") == "all"), None)
    if all_row:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        labels = ["greedy", "random", "child-PVM", "advantage", "oracle"]
        fields = [
            "Acc_greedy",
            "Acc_random_sampled",
            "Acc_child_PVM_best",
            "Acc_advantage_best",
            "Acc_oracle_sample_best",
        ]
        values = [float(all_row.get(field) or 0.0) for field in fields]
        ax.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f58518", "#54a24b", "#b279a2"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Held-out reward / correctness")
        ax.set_title("Segment Selector Gain")
        fig.tight_layout()
        fig.savefig(output_dir / "fig_advantage_selection_gain.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    folds = [int(row["fold"]) for row in fold_rows]
    spearman = [
        float(row.get("mean_within_prefix_spearman") or 0.0)
        for row in fold_rows
    ]
    auc = [float(row.get("better_than_greedy_auc") or 0.0) for row in fold_rows]
    xs = list(range(len(folds)))
    width = 0.35
    ax.bar([x - width / 2 for x in xs], spearman, width=width, label="within-prefix Spearman")
    ax.bar([x + width / 2 for x in xs], auc, width=width, label="better-than-greedy AUC")
    ax.set_xticks(xs, [str(fold) for fold in folds])
    ax.set_xlabel("Fold")
    ax.set_title("Held-out Ranking Diagnostics")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fig_within_prefix_rank.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bins = [str(row["divergence_bin"]) for row in divergence_rows]
    rates = [float(row.get("better_than_greedy_rate") or 0.0) for row in divergence_rows]
    ax.bar(bins, rates, color="#4c78a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Better-than-greedy rate")
    ax.set_title("First-Divergence Position Diagnostic")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_divergence_position.png", dpi=150)
    plt.close(fig)


def train_from_cache(args: argparse.Namespace, cache: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    examples = [dict(item) for item in cache["examples"]]
    h_prefix = cache["h_prefix"].float()
    h_child = cache["h_child"].float()
    temperatures = torch.tensor(
        [float(example["temperature"]) for example in examples],
        dtype=torch.float32,
    )
    labels = torch.tensor(
        [float(example["better_than_greedy_delta"]) for example in examples],
        dtype=torch.float32,
    )
    temperature_values = _temperature_values(examples)
    split_group_key = "problem_id"
    if len({str(example.get("problem_id", "")) for example in examples}) < 2:
        split_group_key = "prefix_id"
    splits = group_kfold_indices(
        examples,
        group_key=split_group_key,
        folds=int(args.folds),
        seed=int(args.seed),
    )
    device = _train_device(str(args.train_device))
    print(f"training advantage head on device={device} folds={len(splits)}", flush=True)
    features = _precompute_feature_matrix(
        h_prefix,
        h_child,
        temperatures,
        temperature_values,
        device,
        batch_size=max(1024, int(args.batch_size)),
    )
    print(
        f"precomputed feature matrix shape={tuple(features.shape)} device={features.device}",
        flush=True,
    )

    all_scores: dict[int, float] = {}
    fold_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(splits):
        row, scores = train_one_fold(
            fold=fold,
            examples=examples,
            features=features,
            temperature_values=temperature_values,
            labels=labels,
            train_indices=train_idx,
            test_indices=test_idx,
            args=args,
            device=device,
            output_dir=output_dir,
        )
        fold_rows.append(row)
        all_scores.update(scores)

    all_indices = sorted(all_scores)
    selection_rows = selection_gain_rows(
        examples,
        scores=all_scores,
        indices=all_indices,
        delta=float(args.delta),
    )
    divergence_rows = divergence_diagnostics_rows(
        examples,
        scores=all_scores,
        indices=all_indices,
    )
    rank_summary = prefix_rank_summary(examples, all_scores, all_indices)
    auc = roc_auc(
        [float(examples[idx]["better_than_greedy_delta"]) for idx in all_indices],
        [float(all_scores[idx]) for idx in all_indices],
    )

    examples_out = []
    for idx, example in enumerate(examples):
        item = _strip_example_for_json(example)
        if idx in all_scores:
            item["advantage_score"] = float(all_scores[idx])
        examples_out.append(item)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stage": "train",
        "n_examples": len(examples),
        "n_prefixes": len({str(example["prefix_id"]) for example in examples}),
        "n_problem_ids": len({str(example["problem_id"]) for example in examples}),
        "folds": len(splits),
        "split_group_key": split_group_key,
        "delta": float(args.delta),
        "seed": int(args.seed),
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "patience": int(args.patience),
            "bce_loss_weight": float(args.bce_loss_weight),
            "head_hidden_dim": int(args.head_hidden_dim),
            "dropout": float(args.dropout),
            "train_device": str(device),
        },
        "hidden_cache_metadata": cache.get("metadata", {}),
        "overall_better_than_greedy_auc": auc,
        **rank_summary,
        "outputs": {
            "advantage_examples": str(output_dir / "advantage_examples.jsonl"),
            "fold_metrics": str(output_dir / "fold_metrics.csv"),
            "selection_gain": str(output_dir / "selection_gain.csv"),
            "divergence_position_diagnostics": str(output_dir / "divergence_position_diagnostics.csv"),
            "summary": str(output_dir / "summary.json"),
            "fig_advantage_selection_gain": str(output_dir / "fig_advantage_selection_gain.png"),
            "fig_within_prefix_rank": str(output_dir / "fig_within_prefix_rank.png"),
            "fig_divergence_position": str(output_dir / "fig_divergence_position.png"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "advantage_examples.jsonl", examples_out)
    write_csv(output_dir / "fold_metrics.csv", fold_rows)
    write_csv(output_dir / "selection_gain.csv", selection_rows)
    write_csv(output_dir / "divergence_position_diagnostics.csv", divergence_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "run_manifest.json", {
        "summary": summary,
        "fold_metrics": fold_rows,
        "selection_gain": selection_rows,
        "divergence_position_diagnostics": divergence_rows,
    })
    plot_outputs(
        fold_rows=fold_rows,
        selection_rows=selection_rows,
        divergence_rows=divergence_rows,
        output_dir=output_dir,
    )
    print(f"wrote advantage-head outputs to {output_dir}", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    hidden_cache = args.hidden_cache or (args.output_dir / "advantage_hidden_cache.pt")
    args.hidden_cache = hidden_cache.resolve()
    cfg = load_config(args.config)

    cache: Mapping[str, Any] | None = None
    if args.stage in {"all", "extract-hidden"}:
        if args.hidden_cache.exists() and not args.overwrite_hidden:
            print(f"reusing hidden cache: {args.hidden_cache}", flush=True)
            cache = torch.load(args.hidden_cache, map_location="cpu", weights_only=False)
        else:
            cache = build_hidden_cache(args, cfg)
    if args.stage in {"all", "train"}:
        if cache is None:
            if not args.hidden_cache.exists():
                raise RuntimeError(f"hidden cache does not exist: {args.hidden_cache}")
            cache = torch.load(args.hidden_cache, map_location="cpu", weights_only=False)
        summary = train_from_cache(args, cache)
        print(json.dumps({
            "n_examples": summary["n_examples"],
            "overall_auc": summary.get("overall_better_than_greedy_auc"),
            "mean_within_prefix_spearman": summary.get("mean_within_prefix_spearman"),
            "output_dir": str(args.output_dir),
        }, indent=2), flush=True)


if __name__ == "__main__":
    main()
