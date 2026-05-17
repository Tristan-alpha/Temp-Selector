"""MIL model evaluation and metric computation functions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List

import torch
import yaml
from torch.utils.data import DataLoader

from utils.exp_logger import setup_experiment_logger

from mil.model import MILModel, DynamicTempHead, GlobalTempHead, smoothness_loss
from mil.training import BagDataset, make_collate_fn, TokenBatchSampler
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
    return float(np.trapezoid(tpr, fpr))


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


def compute_confusion_matrix(labels: torch.Tensor, preds: torch.Tensor, n_classes: int) -> List[List[int]]:
    cm = [[0] * n_classes for _ in range(n_classes)]
    for t, p in zip(_to_np(labels).astype(int), _to_np(preds).astype(int)):
        cm[t][p] += 1
    return cm


def compute_multiclass_metrics(
    labels: torch.Tensor, logits: torch.Tensor, n_classes: int
) -> Dict[str, Any]:
    """Per-class and macro-averaged precision/recall/f1 + accuracy + confusion matrix."""
    import numpy as np
    preds = logits.argmax(dim=-1)
    acc = float((preds == labels).float().mean())

    per_class: Dict[str, Dict[str, float]] = {}
    tp_all = fp_all = fn_all = 0
    for c in range(n_classes):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[str(c)] = {"precision": precision, "recall": recall, "f1": f1, "support": int((labels == c).sum())}
        tp_all += tp
        fp_all += fp
        fn_all += fn

    macro_p = np.mean([v["precision"] for v in per_class.values()])
    macro_r = np.mean([v["recall"] for v in per_class.values()])
    macro_f1 = safe_div(2 * macro_p * macro_r, macro_p + macro_r)

    cm = compute_confusion_matrix(labels, preds, n_classes)

    return {
        "temp_accuracy": acc,
        "temp_macro_precision": float(macro_p),
        "temp_macro_recall": float(macro_r),
        "temp_macro_f1": float(macro_f1),
        "temp_per_class": per_class,
        "temp_confusion_matrix": cm,
    }


def compute_attention_metrics(attn_weights: torch.Tensor) -> Dict[str, float]:
    """Entropy and sparsity of attention weight distributions."""
    eps = 1e-8
    w = attn_weights + eps
    entropy = float(-(w * torch.log(w)).sum(dim=-1).mean())
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
    eval_temp: bool = False,
) -> Dict[str, Any]:
    """Comprehensive MIL model evaluation."""
    ckpt = torch.load(mil_ckpt, map_location=device, weights_only=False)
    mil_state = ckpt["mil"]
    global_state = ckpt.get("global_head", {}) if eval_temp else {}
    dynamic_state = ckpt.get("dynamic_head", {}) if eval_temp else {}

    instance_dim = int(config["data"]["instance_dim"])
    hidden_dim = int(config["mil"]["model"]["hidden_dim"])
    temp_bins = [float(x) for x in config["data"]["temp_bins"]]
    n_temps = len(temp_bins)
    max_tokens_per_batch = int(config["mil"]["training"].get("max_tokens_per_batch", 100000))

    mil = MILModel(
        input_dim=instance_dim, hidden_dim=hidden_dim,
        aggregator=config["mil"]["model"].get("aggregator", "attention"),
        use_position=config["mil"]["model"].get("use_position", True),
        use_gru=config["mil"]["model"].get("use_gru", True),
    ).to(device)
    mil.load_state_dict(mil_state)
    mil.eval()

    global_head = GlobalTempHead(hidden_dim=hidden_dim, n_bins=n_temps).to(device)
    dynamic_head = DynamicTempHead(hidden_dim=hidden_dim, n_bins=n_temps).to(device)

    has_global = bool(global_state)
    has_dynamic = bool(dynamic_state)
    if has_global:
        global_head.load_state_dict(global_state)
        global_head.eval()
    if has_dynamic:
        dynamic_head.load_state_dict(dynamic_state)
        dynamic_head.eval()

    feature_mode = config["inference"].get("feature_mode", "basic")

    extraction_logprobs = feature_mode in {"topk_logprobs", "all"}
    runner = None
    if extraction_logprobs or feature_mode in {"hidden_states", "all"}:
        from inference.vllm_runner import VLLMFeatureExporter
        runner = VLLMFeatureExporter(
            model_name_or_path=config["inference"]["model_name_or_path"],
            max_new_tokens=int(config["inference"].get("max_new_tokens", 8192)),
            parallel_size=config["inference"].get("parallel_size", "auto"),
            gpu_memory_utilization=float(config["inference"].get("gpu_memory_utilization", 0.90)),
            feature_mode=feature_mode,
            engine_preset="prefill",
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
                resp_ids = [tf["token_id"] for tf in row.get("token_features", [])]
                row["_full_ids"] = pids + resp_ids
                row["_prompt_len"] = len(pids)

    collate_fn = make_collate_fn(
        extractor=runner,
        feature_mode=feature_mode,
        instance_dim=instance_dim,
        segment_mode=config["data"].get("segment_mode", "step"),
        segment_size=int(config["data"].get("segment_size", 32)),
        pooling_mode=config["data"].get("segment_pooling", "mean"),
        temp_bins=temp_bins,
    )
    token_counts = [len(r.get("_full_ids", r.get("token_features", []))) for r in dataset.rows]
    eval_sampler = TokenBatchSampler(token_counts, max_tokens_per_batch, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=eval_sampler, collate_fn=collate_fn, num_workers=0)

    all_bag_logits: List[torch.Tensor] = []
    all_bag_labels: List[torch.Tensor] = []
    all_temp_idx: List[torch.Tensor] = []
    all_global_logits: List[torch.Tensor] = []
    all_dynamic_logits: List[torch.Tensor] = []
    all_dyn_inst_logits: List[torch.Tensor] = []
    all_dyn_inst_labels: List[torch.Tensor] = []
    all_dyn_smoothness: List[float] = []
    per_temp_bag_correct: Dict[int, List[float]] = defaultdict(list)

    inst_pos_vals: List[float] = []
    inst_neg_vals: List[float] = []
    all_attn_weights: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)
            t = batch["temp_idx"].to(device)

            out = mil(x)

            all_bag_logits.append(out["bag_logit"].cpu())
            all_bag_labels.append(y.cpu())
            all_temp_idx.append(t.cpu())

            bag_probs = torch.sigmoid(out["bag_logit"])
            bag_preds = (bag_probs > 0.5).float()
            for i in range(y.size(0)):
                ti = int(t[i].item())
                correct = float(bag_preds[i].item() == y[i].item())
                per_temp_bag_correct[ti].append(correct)

            for i in range(y.size(0)):
                n_valid = int(mask[i].sum().item())
                if n_valid == 0:
                    continue
                inst_i = out["inst_logit"][i, :n_valid].cpu()
                attn_i = out["attn_w"][i, :n_valid].cpu()

                if y[i].item() > 0.5:
                    inst_pos_vals.extend(inst_i.tolist())
                else:
                    inst_neg_vals.extend(inst_i.tolist())

                all_attn_weights.append(attn_i)

            if has_global:
                all_global_logits.append(global_head(out["bag_repr"]).cpu())
            if has_dynamic:
                inst_repr = out["encoder_out"]
                dyn_logits = dynamic_head(inst_repr)
                dyn_avg = []
                for i in range(y.size(0)):
                    n_valid = int(mask[i].sum().item())
                    if n_valid > 0:
                        dyn_avg.append(dyn_logits[i, :n_valid].mean(dim=0))
                    else:
                        dyn_avg.append(dyn_logits[i].mean(dim=0))
                all_dynamic_logits.append(torch.stack(dyn_avg).cpu())
                for i in range(y.size(0)):
                    n_valid = int(mask[i].sum().item())
                    if n_valid > 0:
                        all_dyn_inst_logits.append(dyn_logits[i, :n_valid].cpu())
                        all_dyn_inst_labels.append(t[i].cpu().repeat(n_valid))
                    if n_valid >= 2:
                        all_dyn_smoothness.append(float(smoothness_loss(dyn_logits[i, :n_valid].unsqueeze(0)).cpu()))

    bag_labels = torch.cat(all_bag_labels)
    bag_logits = torch.cat(all_bag_logits)
    temp_idx = torch.cat(all_temp_idx)

    bag_metrics = compute_bag_metrics(bag_labels, bag_logits)
    calibration = compute_calibration(bag_labels, bag_logits)

    per_temp_bag_acc = {}
    for ti in sorted(per_temp_bag_correct.keys()):
        vals = per_temp_bag_correct[ti]
        t_val = temp_bins[ti]
        per_temp_bag_acc[f"t={t_val:.1f}"] = {
            "accuracy": safe_div(sum(vals), len(vals)),
            "n_samples": len(vals),
        }

    temp_cls_metrics: Dict[str, Any] = {}
    if has_global and all_global_logits:
        global_logits = torch.cat(all_global_logits)
        temp_cls_metrics["global_head"] = compute_multiclass_metrics(temp_idx, global_logits, n_temps)

    if has_dynamic and all_dynamic_logits:
        dynamic_logits = torch.cat(all_dynamic_logits)
        temp_cls_metrics["dynamic_head"] = compute_multiclass_metrics(temp_idx, dynamic_logits, n_temps)

        if all_dyn_inst_logits:
            inst_logits_cat = torch.cat(all_dyn_inst_logits)
            inst_labels_cat = torch.cat(all_dyn_inst_labels)
            temp_cls_metrics["dynamic_head_per_instance"] = compute_multiclass_metrics(
                inst_labels_cat, inst_logits_cat, n_temps
            )
            if all_dyn_smoothness:
                temp_cls_metrics["dynamic_head_smoothness"] = {
                    "mean": float(sum(all_dyn_smoothness) / len(all_dyn_smoothness)),
                    "min": float(min(all_dyn_smoothness)),
                    "max": float(max(all_dyn_smoothness)),
                }
            n_inst_per_sample = [t.size(0) for t in all_dyn_inst_logits]
            assert sum(n_inst_per_sample) == inst_labels_cat.size(0)
            n_dyn_samples = len(all_dyn_inst_logits)
            dyn_bag_labels = bag_labels[:n_dyn_samples]
            error_mask = (dyn_bag_labels > 0.5).repeat_interleave(
                torch.tensor(n_inst_per_sample)
            )
            inst_preds_cat = inst_logits_cat.argmax(dim=-1)
            corr_temp_dist: Dict[int, Dict[str, float]] = {}
            for label_val, label_name in [(1.0, "error_bags"), (0.0, "correct_bags")]:
                mask = (error_mask == label_val)
                if mask.any():
                    preds_masked = inst_preds_cat[mask]
                    total = float(mask.sum())
                    counts = torch.bincount(preds_masked, minlength=n_temps).float()
                    corr_temp_dist[label_name] = {
                        f"t={temp_bins[i]:.1f}": float(counts[i].item() / total)
                        for i in range(n_temps)
                    }
            temp_cls_metrics["dynamic_head_correctness_temp_distribution"] = corr_temp_dist

    import numpy as np

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

    instance_metrics = {
        "inst_logit_mean_error_bags": float(np.mean(inst_pos_vals)) if inst_pos_vals else 0.0,
        "inst_logit_mean_correct_bags": float(np.mean(inst_neg_vals)) if inst_neg_vals else 0.0,
        "inst_logit_separation": (
            float(np.mean(inst_pos_vals) - np.mean(inst_neg_vals))
            if inst_pos_vals and inst_neg_vals else 0.0
        ),
        **attention_metrics,
    }

    return {
        **bag_metrics,
        **calibration,
        "per_temperature_bag_accuracy": per_temp_bag_acc,
        "temp_classification": temp_cls_metrics,
        "instance_analysis": instance_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MIL model with comprehensive metrics.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--data", default=None, help="Override paths.test_dataset from config")
    parser.add_argument("--mil-ckpt", default=None, help="Override paths.mil_ckpt from config")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--eval-temp", action="store_true", default=False)
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

    device = torch.device(args.device)
    logger.info("data=%s mil_ckpt=%s device=%s", data_path, mil_ckpt, device)
    metrics = evaluate_mil(mil_ckpt, data_path, config, device, eval_temp=args.eval_temp)
    logger.info("mil_metrics=%s", json.dumps(metrics, indent=2, default=str))

    print("\n" + "=" * 60)
    print("MIL EVALUATION")
    print("=" * 60)
    print(f"  Bag accuracy: {metrics.get('bag_accuracy', 0):.4f}")
    print(f"  Bag F1:       {metrics.get('bag_f1', 0):.4f}")
    print(f"  Bag AUC:      {metrics.get('bag_auc', 0):.4f}")
    print(f"  ECE:          {metrics.get('ece', 0):.4f}")
    inst = metrics.get("instance_analysis", {})
    print(f"  Inst separation: {inst.get('inst_logit_separation', 0):.4f}")
    print("=" * 60 + "\n")

    logger.info("mil_eval_complete run_name=%s", final_run_name)


if __name__ == "__main__":
    main()
