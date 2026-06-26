from __future__ import annotations

import argparse
import json
import random
import yaml
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mil.model import MILModel
from mil.utils import (BagDataset, make_collate_fn, SegmentCacheDataset,
                       make_cached_collate_fn, token_batches,
                       _build_cache_path, _load_or_build_segment_cache,
                       _fit_apply_scaler)
from utils.exp_logger import log_exception, setup_experiment_logger


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config_path: str, data_path: str, run_name: str | None = None, log_dir: str = "logs",
          parallel_size: int | None = None) -> None:
    cfg = load_config(config_path)
    logger, log_path, final_run_name = setup_experiment_logger(
        component="mil_training",
        run_name=run_name,
        log_dir=log_dir,
        config=cfg,
    )

    seed_everything(int(cfg.get("seed", 42)))
    logger.info("data_path=%s", data_path)

    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    instance_dim = int(cfg["data"]["instance_dim"])
    hidden_dim = int(cfg["mil"]["model"]["hidden_dim"])
    feature_mode = cfg["inference"].get("feature_mode", "topk_logprobs")

    # Create extraction engine (shared across train + val)
    runner = None
    from inference.vllm_runner import VLLMFeatureExporter
    runner = VLLMFeatureExporter(
        model_name_or_path=cfg["inference"]["model_name_or_path"],
        max_new_tokens=int(cfg["inference"].get("max_new_tokens", 8192)),
        parallel_size=parallel_size,
        gpu_memory_utilization=float(cfg["inference"].get("gpu_memory_utilization", 0.90)),
        reserve_training_gpu=True,
        enable_prefix_caching=False,
    )
    logger.info("VLLMFeatureExporter ready for online feature extraction")

    dataset = BagDataset(data_path=data_path)
    logger.info("dataset_size=%d", len(dataset))

    # Pre-tokenize prompts once; response IDs come from token_features
    # (preserves original generation tokenization across boundaries).
    # SGLang receives pre-built input_ids, skipping internal tokenization.
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
            logger.info("pre_tokenized prompts batch_encoded=%d", len(prompts))

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{max(0, n_gpu - 1)}") if n_gpu > 0 else torch.device("cpu")
    logger.info("device=%s n_gpu=%d", device, n_gpu)

    collate_fn = make_collate_fn(
        extractor=runner,
        feature_mode=feature_mode,
        instance_dim=instance_dim,
        segment_mode=cfg["data"].get("segment_mode", "fixed_window"),
        segment_size=int(cfg["data"].get("segment_size", 32)),
        pooling_mode=cfg["data"].get("segment_pooling", "mean"),
        temp_bins=temp_bins,
        train_device=device,
    )

    # ---- Pre-compute segment features for training set (cached) ----
    max_tokens_per_batch = int(cfg["mil"]["training"].get("max_tokens_per_batch", 131072))
    segment_mode = cfg["data"].get("segment_mode", "fixed_window")
    pooling_mode = cfg["data"].get("segment_pooling", "mean")
    segment_size_for_cache = int(cfg["data"].get("segment_size", 32))
    train_cache_path = _build_cache_path(
        "datasets/cache", "train", segment_mode, pooling_mode,
        feature_mode, instance_dim, segment_size_for_cache,
    )
    segment_cache = _load_or_build_segment_cache(
        dataset.rows, collate_fn, train_cache_path,
        max_tokens_per_batch, logger,
    )

    train_batch_size = int(cfg["mil"]["training"].get("batch_size", 32))
    cached_collate = make_cached_collate_fn(segment_cache, instance_dim, device)
    loader = DataLoader(
        SegmentCacheDataset(len(segment_cache)),
        batch_size=train_batch_size, shuffle=True,
        collate_fn=cached_collate, num_workers=0,
    )

    pooling_mode = cfg["data"].get("segment_pooling", "mean")
    segment_size_for_model = int(cfg["data"].get("segment_size", 32))
    if pooling_mode == "concat":
        model_input_dim = instance_dim * segment_size_for_model
    else:
        model_input_dim = instance_dim
    logger.info("pooling_mode=%s model_input_dim=%d", pooling_mode, model_input_dim)

    mil = MILModel(
        input_dim=model_input_dim, hidden_dim=hidden_dim,
        aggregator=cfg["mil"]["model"].get("aggregator", "attention"),
        use_position=cfg["mil"]["model"].get("use_position", True),
        use_gru=cfg["mil"]["model"].get("use_gru", True),
        gated_attention=cfg["mil"]["model"].get("gated_attention", False),
        num_heads=int(cfg["mil"]["model"].get("num_heads", 1)),
    ).to(device)

    optimizer = optim.Adam(mil.parameters(), lr=float(cfg["mil"]["training"]["lr"]))

    n_pos = sum(1 for r in dataset.rows if float(r.get("individual_label", 0)) > 0.5)
    n_neg = len(dataset.rows) - n_pos
    logger.info("class_balance n_wrong=%d n_correct=%d", n_pos, n_neg)
    bce = nn.BCEWithLogitsLoss()

    max_epochs = int(cfg["mil"]["training"]["max_epochs"])
    early_stop_patience = int(cfg["mil"]["training"]["early_stop_patience"])

    # ---- validation DataLoader for early stopping ----
    val_path = cfg["paths"]["val_dataset"]
    val_dataset = BagDataset(data_path=val_path)

    if runner is not None:
        prompts = [
            r.get("metadata", {}).get("rendered_prompt") or r.get("prompt", "")
            for r in val_dataset.rows
        ]
        if prompts:
            encoded = runner.tokenizer(prompts, add_special_tokens=False)
            for row, pids in zip(val_dataset.rows, encoded.input_ids):
                resp_ids = row["token_ids"]
                row["_full_ids"] = pids + resp_ids
                row["_prompt_len"] = len(pids)

    # ---- Pre-compute segment features for validation set (cached) ----
    val_cache_path = _build_cache_path(
        "datasets/cache", "val", segment_mode, pooling_mode,
        feature_mode, instance_dim, segment_size_for_cache,
    )
    val_segment_cache = _load_or_build_segment_cache(
        val_dataset.rows, collate_fn, val_cache_path,
        max_tokens_per_batch, logger,
    )

    # ---- Optional: StandardScaler on input features ----
    use_scaler = cfg["mil"]["training"].get("use_input_scaler", False)
    if use_scaler:
        segment_cache, val_segment_cache, scaler_params = _fit_apply_scaler(
            segment_cache, val_segment_cache, logger, device=device)
        logger.info("scaler_applied n_features=%d",
                     len(scaler_params["mean"]))

    val_cached_collate = make_cached_collate_fn(val_segment_cache, instance_dim, device)
    val_loader = DataLoader(
        SegmentCacheDataset(len(val_segment_cache)),
        batch_size=train_batch_size, shuffle=False,
        collate_fn=val_cached_collate, num_workers=0,
    )

    def _validate() -> Tuple[float, float, float]:
        """Compute bag-level accuracy on validation set for early stopping.
        Returns (overall_acc, pos_acc, neg_acc)."""
        mil.eval()
        correct = 0
        total = 0
        pos_correct = 0; pos_total = 0
        neg_correct = 0; neg_total = 0
        with torch.no_grad():
            for batch_v in val_loader:
                x_v = batch_v["instances"]
                y_v = batch_v["label"]
                out_v = mil(x_v)
                pred = (torch.sigmoid(out_v["bag_logit"]) > 0.5).float()
                correct += (pred == y_v).sum().item()
                total += y_v.size(0)
                pos_mask = (y_v > 0.5)
                neg_mask = ~pos_mask
                pos_correct += (pred[pos_mask] == y_v[pos_mask]).sum().item()
                pos_total += pos_mask.sum().item()
                neg_correct += (pred[neg_mask] == y_v[neg_mask]).sum().item()
                neg_total += neg_mask.sum().item()
        mil.train()
        return (correct / max(1, total),
                pos_correct / max(1, pos_total),
                neg_correct / max(1, neg_total))
    patience_counter = 0
    best_val_acc = float("-inf")
    best_ckpt: Dict[str, Any] | None = None

    # ---- Metrics JSONL ----
    metrics_path = f"{log_dir}/{final_run_name}_mil_metrics.jsonl"
    metrics_fh = open(metrics_path, "a", encoding="utf-8")
    logger.info("metrics_jsonl=%s", metrics_path)

    for epoch in range(max_epochs):
        mil.train()

        sum_loss = 0.0
        sum_grad_norm = 0.0
        sum_attn_entropy = 0.0
        train_correct = 0
        train_total = 0
        n_train_batches = 0
        pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{max_epochs}",
                     unit="batch", dynamic_ncols=True)
        for batch in loader:
            x = batch["instances"]
            y = batch["label"]

            out = mil(x)
            bag_logit = out["bag_logit"]

            loss = bce(bag_logit, y)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(mil.parameters(), max_norm=1.0)
            optimizer.step()

            _batch_loss = float(loss.item())
            sum_loss += _batch_loss
            sum_grad_norm += float(grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm))
            if "attn_w" in out:
                attn = out["attn_w"]
                attn_ent = -(attn * (attn + 1e-12).log()).sum(dim=-1).mean().item()
                sum_attn_entropy += attn_ent

            with torch.no_grad():
                pred_train = (torch.sigmoid(bag_logit) > 0.5).float()
                train_correct += (pred_train == y).sum().item()
                train_total += y.size(0)
            n_train_batches += 1

            del x, y, out, loss, batch

            pbar.update(1)
            pbar.set_postfix(loss=f"{_batch_loss:.4f}")

        pbar.close()
        avg = sum_loss / max(1, len(loader))
        train_acc_val = train_correct / max(1, train_total)
        avg_grad_norm = sum_grad_norm / max(1, n_train_batches)
        avg_attn_entropy = sum_attn_entropy / max(1, n_train_batches)

        # ---- validation + early stopping ----
        val_acc, val_acc_pos, val_acc_neg = _validate()
        logger.info("epoch=%d done avg_loss=%.6f val_acc=%.4f best=%.4f patience=%d/%d",
                     epoch + 1, avg, val_acc, best_val_acc,
                     patience_counter, early_stop_patience)

        metrics_fh.write(json.dumps({
            "epoch": epoch + 1,
            "loss": avg,
            "train_acc": round(train_acc_val, 4),
            "val_acc": round(val_acc, 4),
            "val_acc_pos": round(val_acc_pos, 4),
            "val_acc_neg": round(val_acc_neg, 4),
            "grad_norm": round(avg_grad_norm, 4),
            "attn_entropy": round(avg_attn_entropy, 4),
        }) + "\n")
        metrics_fh.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # clone to CPU — state_dict() returns references to live parameters,
            # which are mutated in-place by optimizer.step().
            ckpt: Dict[str, Any] = {
                "mil": {k: v.detach().cpu().clone() for k, v in mil.state_dict().items()},
                "config": cfg,
            }
            best_ckpt = ckpt
            logger.info("new_best separation=%.4f", best_val_acc)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("early_stop val_acc=%.4f best=%.4f", val_acc, best_val_acc)
                break

    metrics_fh.close()

    if best_ckpt is None:
        best_ckpt = {
            "mil": {k: v.detach().cpu().clone() for k, v in mil.state_dict().items()},
            "config": cfg,
        }
    ckpt_path = cfg["paths"]["mil_ckpt"]
    torch.save(best_ckpt, ckpt_path)
    logger.info("saved_checkpoint=%s best_val_acc=%.4f run_name=%s log_path=%s",
                 ckpt_path, best_val_acc, final_run_name, log_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", default=None, help="Override paths.train_dataset from config")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--parallel-size", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_path = args.data or cfg["paths"]["train_dataset"]
    try:
        train(config_path=args.config, data_path=data_path, run_name=args.run_name,
              log_dir=args.log_dir, parallel_size=args.parallel_size)
    except Exception as exc:
        cfg = load_config(args.config)
        logger, _, _ = setup_experiment_logger(
            component="mil_training",
            run_name=args.run_name,
            log_dir=args.log_dir,
            config=cfg,
        )
        log_exception(logger, exc)
        raise


if __name__ == "__main__":
    main()
