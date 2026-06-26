"""MIL model evaluation and metric computation functions."""

from __future__ import annotations

import argparse
import json

from typing import Any, Dict, List

import torch
import yaml
from utils.exp_logger import setup_experiment_logger

from mil.model import MILModel
from mil.utils import (BagDataset, make_collate_fn, token_batches,
                       _build_cache_path, _load_or_build_segment_cache)
from utils.math import safe_div


# ═══════════════════════════  metric functions  ═══════════════════════════

def _to_np(t: torch.Tensor):
    import numpy as np
    return t.detach().cpu().numpy()


def compute_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    """ROC-AUC via trapezoidal rule (no sklearn dependency)."""
    import numpy as np
    y = _to_np(labels).astype(np.int32)
    s = _to_np(scores)
    order = np.argsort(s)[::-1]
    y = y[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.concatenate([[0.0], tp / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fp / n_neg, [1.0]])
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(tpr, fpr))


def compute_bag_metrics(labels: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    """Binary classification metrics for bag-level correctness prediction."""
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    tp = float(((preds == 1) & (labels == 1)).sum())
    tn = float(((preds == 0) & (labels == 0)).sum())
    fp = float(((preds == 1) & (labels == 0)).sum())
    fn = float(((preds == 0) & (labels == 1)).sum())

    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    auc = compute_auc(labels, logits)

    return {
        "bag_accuracy": accuracy,
        "bag_precision": precision,
        "bag_recall": recall,
        "bag_f1": f1,
        "bag_auc": auc,
        "bag_tp": tp,
        "bag_tn": tn,
        "bag_fp": fp,
        "bag_fn": fn,
    }


def compute_calibration(labels: torch.Tensor, logits: torch.Tensor, n_bins: int = 10) -> Dict[str, float]:
    """ECE and Brier score."""
    import numpy as np
    probs = torch.sigmoid(logits)
    p = _to_np(probs)
    y = _to_np(labels)

    brier = float(np.mean((p - y) ** 2))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (p > bin_edges[i]) & (p <= bin_edges[i + 1])
        n_b = int(mask.sum())
        if n_b == 0:
            continue
        bin_acc = float(y[mask].mean())
        bin_conf = float(p[mask].mean())
        ece += (n_b / len(y)) * abs(bin_acc - bin_conf)

    return {"brier_score": brier, "ece": ece}


def compute_attention_metrics(attn_weights: torch.Tensor) -> Dict[str, float]:
    """Entropy and sparsity of attention weight distributions."""
    entropy = float(torch.special.entr(attn_weights).sum(dim=-1).mean())
    top3, _ = torch.topk(attn_weights, k=min(3, attn_weights.size(-1)), dim=-1)
    sparsity = float(top3.sum(dim=-1).mean())
    eff_n = float(1.0 / (attn_weights ** 2).sum(dim=-1).mean())
    return {"attn_entropy": entropy, "attn_top3_mass": sparsity, "attn_effective_n": eff_n}


# ═══════════════════════════  evaluate_mil  ═══════════════════════════

def evaluate_mil(
    mil_ckpt: str,
    data_path: str,
    config: Dict[str, Any],
    device: torch.device,
    parallel_size: int | None = None,
) -> Dict[str, Any]:
    """Evaluate MIL model with bag-level metrics and attention interpretability."""
    ckpt = torch.load(mil_ckpt, map_location=device, weights_only=False)
    mil_state = ckpt["mil"]

    instance_dim = int(config["data"]["instance_dim"])
    segment_size = int(config["data"].get("segment_size", 32))
    pooling_mode = config["data"].get("segment_pooling", "mean")
    if pooling_mode == "concat":
        model_input_dim = instance_dim * segment_size
    else:
        model_input_dim = instance_dim
    hidden_dim = int(config["mil"]["model"]["hidden_dim"])
    temp_bins = [float(x) for x in config["data"]["temp_bins"]]
    max_tokens_per_batch = int(config["mil"]["training"].get("max_tokens_per_batch", 131072))

    mil = MILModel(
        input_dim=model_input_dim, hidden_dim=hidden_dim,
        aggregator=config["mil"]["model"].get("aggregator", "attention"),
        use_position=config["mil"]["model"].get("use_position", True),
        use_gru=config["mil"]["model"].get("use_gru", True),
        gated_attention=config["mil"]["model"].get("gated_attention", False),
        num_heads=int(config["mil"]["model"].get("num_heads", 1)),
    ).to(device)
    mil.load_state_dict(mil_state)
    mil.eval()

    feature_mode = config["inference"].get("feature_mode", "topk_logprobs")

    from inference.vllm_runner import VLLMFeatureExporter
    runner = VLLMFeatureExporter(
        model_name_or_path=config["inference"]["model_name_or_path"],
        max_new_tokens=int(config["inference"].get("max_new_tokens", 8192)),
        parallel_size=parallel_size,
        gpu_memory_utilization=float(config["inference"].get("gpu_memory_utilization", 0.90)),
        reserve_training_gpu=True,
        enable_prefix_caching=False,
    )

    dataset = BagDataset(data_path=data_path)

    if runner is not None:
        prompts = [
            r.get("metadata", {}).get("rendered_prompt") or r.get("prompt", "")
            for r in dataset.rows
        ]
        if prompts:
            encoded = runner.tokenizer(prompts, add_special_tokens=False)
            for row, pids in zip(dataset.rows, encoded.input_ids):
                resp_ids = row["token_ids"]
                row["_full_ids"] = pids + resp_ids
                row["_prompt_len"] = len(pids)

    collate_fn = make_collate_fn(
        extractor=runner,
        feature_mode=feature_mode,
        instance_dim=instance_dim,
        segment_mode=config["data"].get("segment_mode", "fixed_window"),
        segment_size=int(config["data"].get("segment_size", 32)),
        pooling_mode=config["data"].get("segment_pooling", "mean"),
        temp_bins=temp_bins,
        train_device=device,
    )

    # ---- Pre-compute segment features (cached) ----
    segment_mode = config["data"].get("segment_mode", "fixed_window")
    pooling_mode = config["data"].get("segment_pooling", "mean")
    segment_size_for_cache = int(config["data"].get("segment_size", 32))
    # Derive split name from data path (e.g. "test" from "datasets/test.jsonl")
    import os as _os
    split = _os.path.splitext(_os.path.basename(data_path))[0]
    cache_path = _build_cache_path(
        "datasets/cache", split, segment_mode, pooling_mode,
        feature_mode, instance_dim, segment_size_for_cache,
    )
    # Create a minimal logger for cache messages if not already available
    import logging as _logging
    _cache_logger = _logging.getLogger("mil.eval.cache")
    if not _cache_logger.handlers:
        _cache_logger.addHandler(_logging.NullHandler())
    segment_cache = _load_or_build_segment_cache(
        dataset.rows, collate_fn, cache_path,
        max_tokens_per_batch, _cache_logger,
    )

    all_bag_logits: List[torch.Tensor] = []
    all_bag_labels: List[torch.Tensor] = []
    all_attn_weights: List[torch.Tensor] = []

    train_batch_size = int(config["mil"]["training"].get("batch_size", 32))

    with torch.no_grad():
        for start in range(0, len(segment_cache), train_batch_size):
            end = min(start + train_batch_size, len(segment_cache))
            chunk = segment_cache[start:end]

            max_k = max(entry["instances"].shape[0] for entry in chunk)
            d = chunk[0]["instances"].shape[1]
            b = len(chunk)
            x = torch.zeros((b, max_k, d), dtype=torch.float32, device=device)
            mask = torch.zeros((b, max_k), dtype=torch.float32, device=device)
            y = torch.tensor([entry["label"] for entry in chunk],
                             dtype=torch.float32, device=device)

            for j, entry in enumerate(chunk):
                inst = entry["instances"].to(device)
                k = inst.shape[0]
                x[j, :k] = inst
                mask[j, :k] = 1.0

            out = mil(x)

            all_bag_logits.append(out["bag_logit"].cpu())
            all_bag_labels.append(y.cpu())

            for j in range(b):
                n_valid = int(mask[j].sum().item())
                if n_valid == 0:
                    continue
                all_attn_weights.append(out["attn_w"][j, :n_valid].cpu())

    bag_labels = torch.cat(all_bag_labels)
    bag_logits = torch.cat(all_bag_logits)

    bag_metrics = compute_bag_metrics(bag_labels, bag_logits)
    calibration = compute_calibration(bag_labels, bag_logits)

    if all_attn_weights:
        entropies, top3s, eff_ns = [], [], []
        for w in all_attn_weights:
            m = compute_attention_metrics(w.unsqueeze(0))  # [1, K]
            entropies.append(m["attn_entropy"])
            top3s.append(m["attn_top3_mass"])
            eff_ns.append(m["attn_effective_n"])
        attention_metrics = {
            "attn_entropy": sum(entropies) / len(entropies),
            "attn_top3_mass": sum(top3s) / len(top3s),
            "attn_effective_n": sum(eff_ns) / len(eff_ns),
        }
    else:
        attention_metrics = {
            "attn_entropy": 0.0, "attn_top3_mass": 0.0, "attn_effective_n": 0.0,
        }

    return {
        **bag_metrics,
        **calibration,
        **attention_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIL model with bag-level metrics.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", default=None, help="Override paths.test_dataset from config")
    parser.add_argument("--mil-ckpt", default=None, help="Override paths.mil_ckpt from config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--parallel-size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    data_path = args.data or config["paths"]["test_dataset"]
    mil_ckpt = args.mil_ckpt or config["paths"]["mil_ckpt"]

    logger, _log_path, final_run_name = setup_experiment_logger(
        component="mil_eval",
        run_name=args.run_name,
        log_dir=args.log_dir,
        config={"data": data_path, "mil_ckpt": mil_ckpt},
    )

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{max(0, n_gpu - 1)}") if n_gpu > 0 else torch.device("cpu")
    logger.info("device=%s n_gpu=%d", device, n_gpu)
    metrics = evaluate_mil(mil_ckpt, data_path, config, device, parallel_size=args.parallel_size)
    logger.info("mil_metrics=%s", json.dumps(metrics, indent=2, default=str))

    print("\n" + "=" * 60)
    print("MIL EVALUATION")
    print("=" * 60)
    print(f"  Bag accuracy:  {metrics.get('bag_accuracy', 0):.4f}")
    print(f"  Bag F1:        {metrics.get('bag_f1', 0):.4f}")
    print(f"  Bag AUC:       {metrics.get('bag_auc', 0):.4f}")
    print(f"  ECE:           {metrics.get('ece', 0):.4f}")
    print(f"  Attn entropy:  {metrics.get('attn_entropy', 0):.4f}")
    print(f"  Attn top3:     {metrics.get('attn_top3_mass', 0):.4f}")
    print(f"  Attn eff_n:    {metrics.get('attn_effective_n', 0):.4f}")
    print("=" * 60 + "\n")

    logger.info("mil_eval_complete run_name=%s", final_run_name)


if __name__ == "__main__":
    main()
