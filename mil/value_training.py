"""Train the continuation-supervised causal Prefix Value Model."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from inference.vllm_runner import VLLMFeatureExporter
from mil.prefix_data import (
    IndexDataset,
    build_ranking_pairs,
    continuation_collate,
    oracle_temperature_stats,
    precompute_feature_cache,
    ranking_collate,
    terminal_collate,
)
from mil.prefix_value import (
    PrefixValueModel,
    binomial_nll,
    calibrated_probability,
    masked_binomial_nll,
    paired_ranking_loss,
)
from utils.exp_logger import log_exception, setup_experiment_logger
from utils.jsonl import load_jsonl


def _to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _repeat(loader: DataLoader):
    while True:
        yield from loader


def _loader_kwargs(training_cfg: Dict[str, Any]) -> Dict[str, Any]:
    num_workers = int(training_cfg.get("num_workers", 0))
    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": bool(training_cfg.get("pin_memory", num_workers > 0 and torch.cuda.is_available())),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(training_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(training_cfg.get("prefetch_factor", 2))
    return kwargs


def _prompt_dim(cfg: Dict[str, Any]) -> int:
    return int(cfg.get("prefix_value", {}).get("model", {}).get("prompt_dim", 0) or 0)


def _prompt_integration(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("prefix_value", {}).get("model", {}).get("prompt_integration", "none"))


def _ece(probabilities: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    edges = torch.linspace(0.0, 1.0, n_bins + 1, device=probabilities.device)
    result = probabilities.new_zeros(())
    for i in range(n_bins):
        mask = (probabilities > edges[i]) & (probabilities <= edges[i + 1])
        if torch.any(mask):
            result += mask.float().mean() * torch.abs(
                probabilities[mask].mean() - targets[mask].mean()
            )
    return float(result.item())


def _binomial_nll_per_record(logits: torch.Tensor, n_correct: torch.Tensor,
                             n_total: torch.Tensor,
                             calibration_temperature: float = 1.0) -> torch.Tensor:
    temperature = torch.as_tensor(
        calibration_temperature, dtype=logits.dtype, device=logits.device,
    ).clamp_min(1e-4)
    calibrated_logits = logits / temperature
    n_total = n_total.to(logits.dtype).clamp_min(1.0)
    n_correct = n_correct.to(logits.dtype)
    return -(n_correct * F.logsigmoid(calibrated_logits) +
             (n_total - n_correct) * F.logsigmoid(-calibrated_logits)) / n_total


def _binomial_nll_for_constant(probability: torch.Tensor, n_correct: torch.Tensor,
                               n_total: torch.Tensor) -> torch.Tensor:
    p = probability.to(n_correct.dtype).clamp(1e-6, 1.0 - 1e-6)
    n_total = n_total.to(n_correct.dtype).clamp_min(1.0)
    n_correct = n_correct.to(n_correct.dtype)
    return -(n_correct * torch.log(p) +
             (n_total - n_correct) * torch.log1p(-p)) / n_total


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    values = values.detach().cpu().to(torch.float64)
    order = torch.argsort(values)
    ranks = torch.empty_like(values)
    sorted_values = values[order]
    start = 0
    n = int(values.numel())
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def _spearman(values: torch.Tensor, targets: torch.Tensor) -> float:
    if values.numel() < 2:
        return 0.0
    rank_values = _average_ranks(values)
    rank_targets = _average_ranks(targets)
    vx = rank_values - rank_values.mean()
    vy = rank_targets - rank_targets.mean()
    denom = torch.sqrt(torch.sum(vx * vx) * torch.sum(vy * vy))
    if float(denom.item()) <= 0.0:
        return 0.0
    return float(torch.sum(vx * vy).item() / denom.item())


def _record_stage(record: Dict[str, Any]) -> str:
    stage = record.get("prefix_stage")
    if stage:
        return str(stage)
    n_segments = int(record.get("n_segments", 0))
    prefix_segments = int(record.get("prefix_segments", 0))
    if n_segments <= 0 or prefix_segments <= 0:
        return "unknown"
    q = prefix_segments / n_segments
    if q < 0.30:
        return "early"
    if q < 0.65:
        return "middle"
    return "late"


def _load_or_build_cache(path: str, rows: List[Dict[str, Any]], extractor,
                         cfg: Dict[str, Any], device: torch.device,
                         description: str) -> List[Dict[str, Any]]:
    cache_path = Path(path)
    prompt_dim = _prompt_dim(cfg)
    if cache_path.exists():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        expected_dim = int(cfg["data"]["segment_size"]) * int(cfg["data"]["instance_dim"])
        row_ids = [str(row.get("sample_id", "")) for row in rows]
        cache_ids = [str(entry.get("sample_id", "")) for entry in cache]
        prompt_ok = (
            prompt_dim <= 0 or
            (not cache or (
                "prompt_hidden" in cache[0] and
                int(cache[0]["prompt_hidden"].shape[-1]) == prompt_dim
            ))
        )
        if (len(cache) == len(rows) and cache_ids == row_ids and
                (not cache or int(cache[0]["features"].shape[1]) == expected_dim) and
                prompt_ok):
            return cache
    cache = precompute_feature_cache(
        rows=rows,
        extractor=extractor,
        segment_size=int(cfg["data"]["segment_size"]),
        token_dim=int(cfg["data"]["instance_dim"]),
        top_k=int(cfg["inference"]["top_k_logprobs"]),
        max_tokens_per_batch=int(cfg["prefix_value"]["training"].get(
            "max_tokens_per_batch", 131072,
        )),
        device=device,
        description=description,
        prompt_dim=prompt_dim,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _forward_model(model: PrefixValueModel, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return model(
        batch["features"],
        batch["token_mask"],
        batch["segment_mask"],
        prompt_hidden=batch.get("prompt_hidden"),
    )


def _forward_terminal(model: PrefixValueModel, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return _forward_model(model, batch)["terminal_logits"]


def _q_metrics(q_logits: torch.Tensor, q_n_correct: torch.Tensor,
               q_n_total: torch.Tensor, q_mask: torch.Tensor,
               temp_bins: List[float] | None,
               records: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    valid = q_mask > 0
    if not torch.any(valid):
        return {}
    q_probs = torch.sigmoid(q_logits)
    q_target = (q_n_correct + 0.5) / (q_n_total + 1.0)
    q_observed = q_n_correct / q_n_total.clamp_min(1.0)
    q_loss = masked_binomial_nll(q_logits, q_n_correct, q_n_total, q_mask)
    result: Dict[str, Any] = {
        "q_brier": float(torch.mean((q_probs[valid] - q_target[valid]) ** 2).item()),
        "q_ece": _ece(q_probs[valid], q_observed[valid]),
        "q_binomial_nll": float(q_loss.item()),
        "q_valid_labels": int(valid.sum().item()),
    }
    if temp_bins is None or q_logits.shape[1] != len(temp_bins):
        return result

    regrets: List[float] = []
    top1_matches: List[float] = []
    top3_coverages: List[float] = []
    selected_temps: Dict[str, int] = defaultdict(int)
    for row_idx in range(q_logits.shape[0]):
        valid_idx = torch.nonzero(q_mask[row_idx] > 0, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            continue
        row_probs = q_probs[row_idx, valid_idx]
        row_observed = q_observed[row_idx, valid_idx]
        local_selected = int(torch.argmax(row_probs).item())
        selected_idx = int(valid_idx[local_selected].item())
        selected_rate = float(q_observed[row_idx, selected_idx].item())
        oracle_rate = float(row_observed.max().item())
        oracle_indices = valid_idx[row_observed == row_observed.max()]
        regrets.append(oracle_rate - selected_rate)
        top1_matches.append(float(selected_idx in {int(i.item()) for i in oracle_indices}))
        k = min(3, int(valid_idx.numel()))
        topk_local = torch.topk(row_probs, k=k).indices
        topk_indices = {int(valid_idx[int(i.item())].item()) for i in topk_local}
        top3_coverages.append(float(any(int(i.item()) in topk_indices for i in oracle_indices)))
        selected_temps[str(float(temp_bins[selected_idx]))] += 1

    if regrets:
        result.update({
            "q_oracle_regret": float(sum(regrets) / len(regrets)),
            "q_top1_oracle_match": float(sum(top1_matches) / len(top1_matches)),
            "q_top3_oracle_coverage": float(sum(top3_coverages) / len(top3_coverages)),
            "q_selected_temperature_distribution": dict(sorted(selected_temps.items())),
        })
    if records is not None and len(records) == q_logits.shape[0]:
        oracle_summary = [
            oracle_temperature_stats(record)
            for record in records
        ]
        result["q_record_oracle_available"] = sum(
            1 for item in oracle_summary if item["oracle_temperature"] is not None
        )
    return result


@torch.no_grad()
def evaluate_value_model(model: PrefixValueModel,
                         continuation_loader: DataLoader,
                         ranking_loader: DataLoader | None,
                         device: torch.device,
                         calibration_temperature: float = 1.0,
                         records: List[Dict[str, Any]] | None = None,
                         temp_bins: List[float] | None = None) -> Dict[str, Any]:
    model.eval()
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    correct_all: List[torch.Tensor] = []
    total_all: List[torch.Tensor] = []
    q_logits_all: List[torch.Tensor] = []
    q_correct_all: List[torch.Tensor] = []
    q_total_all: List[torch.Tensor] = []
    q_mask_all: List[torch.Tensor] = []
    for batch in continuation_loader:
        batch = _to_device(batch, device)
        output = _forward_model(model, batch)
        logits = output["terminal_logits"]
        logits_all.append(logits.cpu())
        targets_all.append(batch["target"].cpu())
        correct_all.append(batch["n_correct"].cpu())
        total_all.append(batch["n_total"].cpu())
        q_logits = output.get("terminal_q_logits")
        if q_logits is not None and "q_n_correct" in batch:
            q_logits_all.append(q_logits.cpu())
            q_correct_all.append(batch["q_n_correct"].cpu())
            q_total_all.append(batch["q_n_total"].cpu())
            q_mask_all.append(batch["q_mask"].cpu())
    if not logits_all:
        return {"brier": 1.0, "ece": 1.0, "binomial_nll": float("inf"), "pair_accuracy": 0.0}
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)
    n_correct = torch.cat(correct_all)
    n_total = torch.cat(total_all)
    probs = calibrated_probability(logits, calibration_temperature)
    per_record_nll = _binomial_nll_per_record(
        logits, n_correct, n_total, calibration_temperature,
    )
    observed_rate = n_correct / n_total.clamp_min(1.0)
    constant_probability = (n_correct.sum() / n_total.sum().clamp_min(1.0)).clamp(1e-6, 1.0 - 1e-6)
    constant_nll = _binomial_nll_for_constant(constant_probability, n_correct, n_total)
    constant_brier = torch.mean((constant_probability - targets) ** 2)

    pair_correct = pair_total = 0
    if ranking_loader is not None:
        for pair_batch in ranking_loader:
            a = _to_device(pair_batch["a"], device)
            b = _to_device(pair_batch["b"], device)
            logits_a = _forward_terminal(model, a)
            logits_b = _forward_terminal(model, b)
            target_a = a["target"]
            target_b = b["target"]
            pair_correct += int(((logits_a > logits_b) == (target_a > target_b)).sum().item())
            pair_total += int(logits_a.numel())
    result = {
        "brier": float(torch.mean((probs - targets) ** 2).item()),
        "ece": _ece(probs, targets),
        "binomial_nll": float(per_record_nll.mean().item()),
        "constant_brier": float(constant_brier.item()),
        "constant_binomial_nll": float(constant_nll.mean().item()),
        "constant_mean_probability": float(constant_probability.item()),
        "spearman": _spearman(probs, observed_rate),
        "pair_accuracy": pair_correct / max(1, pair_total),
        "n_prefixes": int(targets.numel()),
        "n_pairs": pair_total,
        "n_total_distribution": {
            str(int(value.item())): int((n_total == value).sum().item())
            for value in torch.unique(n_total)
        },
        **_record_level_metrics(records, probs, targets, per_record_nll, observed_rate),
    }
    if q_logits_all:
        result.update(_q_metrics(
            torch.cat(q_logits_all),
            torch.cat(q_correct_all),
            torch.cat(q_total_all),
            torch.cat(q_mask_all),
            temp_bins=temp_bins,
            records=records,
        ))
    return result


def _record_level_metrics(records: List[Dict[str, Any]] | None,
                          probs: torch.Tensor,
                          targets: torch.Tensor,
                          per_record_nll: torch.Tensor,
                          observed_rate: torch.Tensor) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    if records is not None and len(records) == int(probs.numel()):
        by_stage: Dict[str, List[int]] = defaultdict(list)
        for idx, record in enumerate(records):
            by_stage[_record_stage(record)].append(idx)
        stage_metrics: Dict[str, Any] = {}
        for stage, indices in sorted(by_stage.items()):
            idx = torch.tensor(indices, dtype=torch.long)
            stage_metrics[stage] = {
                "n_prefixes": len(indices),
                "brier": float(torch.mean((probs[idx] - targets[idx]) ** 2).item()),
                "binomial_nll": float(per_record_nll[idx].mean().item()),
                "mean_phi": float(probs[idx].mean().item()),
                "observed_rate": float(observed_rate[idx].mean().item()),
            }
        result["stage_metrics"] = stage_metrics

    n = int(probs.numel())
    if n > 0:
        k = max(1, n // 4)
        order = torch.argsort(probs)
        bottom = order[:k]
        top = order[-k:]
        result["phi_quartiles"] = {
            "bottom_n": int(bottom.numel()),
            "top_n": int(top.numel()),
            "bottom_mean_phi": float(probs[bottom].mean().item()),
            "top_mean_phi": float(probs[top].mean().item()),
            "bottom_observed_rate": float(observed_rate[bottom].mean().item()),
            "top_observed_rate": float(observed_rate[top].mean().item()),
            "observed_rate_delta": float(
                observed_rate[top].mean().item() - observed_rate[bottom].mean().item()
            ),
        }
    return result


def fit_temperature(model: PrefixValueModel, loader: DataLoader,
                    device: torch.device) -> float:
    model.eval()
    logits_all: List[torch.Tensor] = []
    targets_all: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            logits_all.append(_forward_terminal(model, batch))
            targets_all.append(batch["target"])
    logits = torch.cat(logits_all).detach()
    targets = torch.cat(targets_all).detach()
    log_temperature = torch.zeros((), device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def train(config_path: str, parallel_size: int | None = None,
          run_name: str | None = None, log_dir: str = "logs") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger, log_path, final_run_name = setup_experiment_logger(
        component="prefix_value_training", run_name=run_name, log_dir=log_dir, config=cfg,
    )
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_rows = load_jsonl(cfg["paths"]["train_dataset"])
    val_rows = load_jsonl(cfg["paths"]["val_dataset"])
    train_records = load_jsonl(cfg["paths"]["train_continuations"])
    val_records = load_jsonl(cfg["paths"]["val_continuations"])
    logger.info("train_rows=%d val_rows=%d train_prefixes=%d val_prefixes=%d",
                len(train_rows), len(val_rows), len(train_records), len(val_records))
    if not train_records or not val_records:
        raise RuntimeError("continuation label files are empty; run build_prefix_continuations first")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{max(0, n_gpu - 1)}") if n_gpu else torch.device("cpu")
    extractor = VLLMFeatureExporter(
        model_name_or_path=cfg["inference"]["model_name_or_path"],
        max_new_tokens=int(cfg["inference"].get("max_new_tokens", 8192)),
        parallel_size=parallel_size,
        gpu_memory_utilization=float(cfg["inference"].get("gpu_memory_utilization", 0.90)),
        reserve_training_gpu=True,
        enable_prefix_caching=cfg["inference"].get("enable_prefix_caching", False),
    )
    train_cache = _load_or_build_cache(
        cfg["paths"]["train_feature_cache"], train_rows, extractor, cfg, device,
        "Prefix train features",
    )
    val_cache = _load_or_build_cache(
        cfg["paths"]["val_feature_cache"], val_rows, extractor, cfg, device,
        "Prefix val features",
    )
    train_by_id = {entry["sample_id"]: entry for entry in train_cache}
    val_by_id = {entry["sample_id"]: entry for entry in val_cache}

    train_pairs = build_ranking_pairs(train_records, seed=seed, max_pairs_per_problem=64)
    val_pairs = build_ranking_pairs(val_records, seed=seed, max_pairs_per_problem=64)
    logger.info("train_pairs=%d val_pairs=%d", len(train_pairs), len(val_pairs))

    training_cfg = cfg["prefix_value"]["training"]
    batch_size = int(training_cfg.get("batch_size", 32))
    temp_bins = [float(x) for x in cfg["data"].get("temp_bins", [])]
    n_temps = int(cfg["prefix_value"]["model"].get("n_temps", 0))
    q_temp_bins = temp_bins if n_temps > 0 else None
    generator = torch.Generator().manual_seed(seed)
    loader_kwargs = _loader_kwargs(training_cfg)
    logger.info("dataloader_kwargs=%s", json.dumps(loader_kwargs, sort_keys=True))
    train_terminal_loader = DataLoader(
        IndexDataset(len(train_cache)), batch_size=batch_size, shuffle=True,
        generator=generator, **loader_kwargs,
        collate_fn=partial(terminal_collate, train_cache),
    )
    train_cont_loader = DataLoader(
        IndexDataset(len(train_records)), batch_size=batch_size, shuffle=True,
        generator=generator, **loader_kwargs,
        collate_fn=partial(continuation_collate, train_by_id, train_records, temp_bins=q_temp_bins),
    )
    train_rank_loader = DataLoader(
        IndexDataset(len(train_pairs)), batch_size=batch_size, shuffle=True,
        generator=generator, **loader_kwargs,
        collate_fn=partial(ranking_collate, train_by_id, train_records, train_pairs, temp_bins=q_temp_bins),
    ) if train_pairs else None
    val_cont_loader = DataLoader(
        IndexDataset(len(val_records)), batch_size=batch_size, shuffle=False,
        **loader_kwargs,
        collate_fn=partial(continuation_collate, val_by_id, val_records, temp_bins=q_temp_bins),
    )
    val_rank_loader = DataLoader(
        IndexDataset(len(val_pairs)), batch_size=batch_size, shuffle=False,
        **loader_kwargs,
        collate_fn=partial(ranking_collate, val_by_id, val_records, val_pairs, temp_bins=q_temp_bins),
    ) if val_pairs else None

    model = PrefixValueModel(
        token_dim=int(cfg["data"]["instance_dim"]),
        segment_size=int(cfg["data"]["segment_size"]),
        hidden_dim=int(cfg["prefix_value"]["model"]["hidden_dim"]),
        max_segments=int(cfg["prefix_value"]["model"].get("max_segments", 8192)),
        n_temps=n_temps,
        prompt_dim=_prompt_dim(cfg),
        prompt_integration=_prompt_integration(cfg),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training_cfg.get("lr", 2e-4)))
    rank_weight = float(training_cfg.get("ranking_weight", 0.5))
    q_weight = float(training_cfg.get("q_weight", 0.0))
    max_epochs = int(training_cfg.get("max_epochs", 50))
    patience_limit = int(training_cfg.get("early_stop_patience", 5))

    best_nll = float("inf")
    best_state: Dict[str, torch.Tensor] | None = None
    patience = 0
    for epoch in range(max_epochs):
        model.train()
        cont_iter = _repeat(train_cont_loader)
        rank_iter = _repeat(train_rank_loader) if train_rank_loader is not None else None
        sums = {"total": 0.0, "terminal": 0.0, "continuation": 0.0, "ranking": 0.0, "q": 0.0}
        steps = 0
        for terminal_batch in train_terminal_loader:
            terminal_batch = _to_device(terminal_batch, device)
            continuation_batch = _to_device(next(cont_iter), device)
            terminal_output = _forward_model(model, terminal_batch)
            continuation_output = _forward_model(model, continuation_batch)
            terminal_logits = terminal_output["terminal_logits"]
            continuation_logits = continuation_output["terminal_logits"]
            loss_terminal = F.binary_cross_entropy_with_logits(
                terminal_logits, terminal_batch["target"],
            )
            loss_continuation = binomial_nll(
                continuation_logits,
                continuation_batch["n_correct"], continuation_batch["n_total"],
            )
            loss_q = terminal_logits.new_zeros(())
            if q_weight > 0.0:
                q_logits = continuation_output.get("terminal_q_logits")
                if q_logits is None:
                    raise RuntimeError("prefix_value.training.q_weight > 0 requires prefix_value.model.n_temps > 0")
                loss_q = masked_binomial_nll(
                    q_logits,
                    continuation_batch["q_n_correct"],
                    continuation_batch["q_n_total"],
                    continuation_batch["q_mask"],
                )
            loss_ranking = terminal_logits.new_zeros(())
            if rank_iter is not None:
                pair_batch = next(rank_iter)
                a = _to_device(pair_batch["a"], device)
                b = _to_device(pair_batch["b"], device)
                loss_ranking = paired_ranking_loss(
                    _forward_terminal(model, a), _forward_terminal(model, b),
                    a["target"], b["target"],
                )
            loss = loss_continuation + loss_terminal + q_weight * loss_q + rank_weight * loss_ranking
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sums["total"] += float(loss.item())
            sums["terminal"] += float(loss_terminal.item())
            sums["continuation"] += float(loss_continuation.item())
            sums["ranking"] += float(loss_ranking.item())
            sums["q"] += float(loss_q.item())
            steps += 1

        metrics = evaluate_value_model(
            model, val_cont_loader, val_rank_loader, device, temp_bins=q_temp_bins,
        )
        logger.info(
            "epoch=%d loss=%.6f terminal=%.6f continuation=%.6f q=%.6f ranking=%.6f val=%s",
            epoch + 1, sums["total"] / max(1, steps),
            sums["terminal"] / max(1, steps), sums["continuation"] / max(1, steps),
            sums["q"] / max(1, steps), sums["ranking"] / max(1, steps),
            json.dumps(metrics, sort_keys=True),
        )
        if metrics["binomial_nll"] < best_nll:
            best_nll = metrics["binomial_nll"]
            patience = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_limit:
                break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    calibration_temperature = fit_temperature(model, val_cont_loader, device)
    calibrated_metrics = evaluate_value_model(
        model, val_cont_loader, val_rank_loader, device, calibration_temperature,
        records=val_records, temp_bins=q_temp_bins,
    )
    checkpoint = {
        "prefix_value": best_state,
        "calibration_temperature": calibration_temperature,
        "config": cfg,
        "validation_metrics": calibrated_metrics,
        "train_prefixes": len(train_records),
        "train_pairs": len(train_pairs),
    }
    checkpoint_path = Path(cfg["paths"]["prefix_value_ckpt"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info("saved=%s calibration_temperature=%.6f metrics=%s run_name=%s log=%s",
                checkpoint_path, calibration_temperature,
                json.dumps(calibrated_metrics, sort_keys=True), final_run_name, log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    try:
        train(args.config, args.parallel_size, args.run_name, args.log_dir)
    except Exception as exc:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger, _, _ = setup_experiment_logger(
            component="prefix_value_training", run_name=args.run_name,
            log_dir=args.log_dir, config=cfg,
        )
        log_exception(logger, exc)
        raise


if __name__ == "__main__":
    main()
