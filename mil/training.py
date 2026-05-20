from __future__ import annotations

import argparse
import random
import yaml
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mil.model import MILModel, DynamicTempHead, GlobalTempHead, smoothness_loss
from mil.utils import BagDataset, make_collate_fn, SegmentCacheDataset, make_cached_collate_fn, token_batches
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
        segment_mode=cfg["data"].get("segment_mode", "step"),
        segment_size=int(cfg["data"].get("segment_size", 32)),
        pooling_mode=cfg["data"].get("segment_pooling", "mean"),
        temp_bins=temp_bins,
        train_device=device,
    )

    # ---- Pre-compute segment features for training set ----
    max_tokens_per_batch = int(cfg["mil"]["training"].get("max_tokens_per_batch", 131072))
    segment_cache: List[Dict[str, Any]] = []
    train_batches = token_batches(dataset.rows, max_tokens_per_batch)
    logger.info("precomputing train features n_rows=%d n_batches=%d max_tokens_per_batch=%d",
                 len(dataset), len(train_batches), max_tokens_per_batch)
    for indices in tqdm(train_batches, desc="Precompute train features"):
        batch_rows = [dataset[i] for i in indices]
        batch = collate_fn(batch_rows)
        x = batch["instances"].cpu()
        y = batch["label"].cpu()
        t = batch["temp_idx"].cpu()
        mask = batch["mask"].cpu()
        for i in range(x.shape[0]):
            n_valid = int(mask[i].sum().item())
            segment_cache.append({
                "instances": x[i, :n_valid].clone(),
                "label": float(y[i].item()),
                "temp_idx": int(t[i].item()),
            })
    logger.info("precomputed train segment_cache_size=%d", len(segment_cache))

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
    ).to(device)

    alpha = float(cfg["mil"]["training"]["alpha_temp"])
    use_temp = (alpha > 0)
    global_head = None
    dynamic_head = None
    if use_temp:
        global_head = GlobalTempHead(hidden_dim=hidden_dim, n_bins=len(temp_bins)).to(device)
        dynamic_head = DynamicTempHead(hidden_dim=hidden_dim, n_bins=len(temp_bins)).to(device)

    params = list(mil.parameters())
    if use_temp:
        params += list(global_head.parameters()) + list(dynamic_head.parameters())
    optimizer = optim.Adam(params, lr=float(cfg["mil"]["training"]["lr"]))

    n_pos = sum(1 for r in dataset.rows if float(r.get("individual_label", 0)) > 0.5)
    n_neg = len(dataset.rows) - n_pos
    pos_weight = torch.tensor([(n_neg / max(1, n_pos)) ** 0.5], device=device)
    logger.info("bce_pos_weight=%.4f (n_wrong=%d n_correct=%d)", pos_weight.item(), n_pos, n_neg)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ce = nn.CrossEntropyLoss()

    instance_loss_method = cfg["mil"]["training"].get("instance_loss", "topk")
    valid_methods = {"topk", "pure", "soft_pseudo_label", "contrastive"}
    if instance_loss_method not in valid_methods:
        raise ValueError(f"Unknown instance_loss method: {instance_loss_method}. "
                         f"Must be one of {valid_methods}")
    logger.info("instance_loss_method=%s", instance_loss_method)

    alpha = float(cfg["mil"]["training"]["alpha_temp"])
    beta = float(cfg["mil"]["training"]["beta_inst_aux"])
    gamma = float(cfg["mil"]["training"]["gamma_smooth"])
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

    # ---- Pre-compute segment features for validation set ----
    val_segment_cache: List[Dict[str, Any]] = []
    val_batches = token_batches(val_dataset.rows, max_tokens_per_batch)
    logger.info("precomputing val features n_rows=%d n_batches=%d", len(val_dataset), len(val_batches))
    for indices in tqdm(val_batches, desc="Precompute val features"):
        batch_rows = [val_dataset[i] for i in indices]
        batch = collate_fn(batch_rows)
        x = batch["instances"].cpu()
        y = batch["label"].cpu()
        t = batch["temp_idx"].cpu()
        mask = batch["mask"].cpu()
        for i in range(x.shape[0]):
            n_valid = int(mask[i].sum().item())
            val_segment_cache.append({
                "instances": x[i, :n_valid].clone(),
                "label": float(y[i].item()),
                "temp_idx": int(t[i].item()),
            })
    logger.info("precomputed val segment_cache_size=%d", len(val_segment_cache))

    val_cached_collate = make_cached_collate_fn(val_segment_cache, instance_dim, device)
    val_loader = DataLoader(
        SegmentCacheDataset(len(val_segment_cache)),
        batch_size=train_batch_size, shuffle=False,
        collate_fn=val_cached_collate, num_workers=0,
    )

    def _validate() -> float:
        """Compute inst_logit_separation on the validation set.

        Mean inst_logit on error bags minus mean on correct bags.
        Higher = model discriminates error segments from correct segments better.
        """
        mil.eval()
        pos_means: List[float] = []
        neg_means: List[float] = []
        with torch.no_grad():
            for batch_v in val_loader:
                x_v = batch_v["instances"]
                mask_v = batch_v["mask"]
                y_v = batch_v["label"]
                inst = mil(x_v)["inst_logit"]
                for i in range(y_v.size(0)):
                    n_valid = int(mask_v[i].sum().item())
                    if n_valid == 0:
                        continue
                    bag_mean = inst[i, :n_valid].mean().item()
                    if y_v[i].item() > 0.5:  # label=1: positive bag (contains errors)
                        pos_means.append(bag_mean)
                    else:  # label=0: negative bag (no errors)
                        neg_means.append(bag_mean)
        mil.train()
        if pos_means and neg_means:
            import numpy as np
            return float(np.mean(pos_means) - np.mean(neg_means))
        return 0.0

    best_separation = -float("inf")
    patience_counter = 0
    best_ckpt: Dict[str, Any] | None = None

    for epoch in range(max_epochs):
        mil.train()
        if use_temp:
            global_head.train()
            dynamic_head.train()

        sum_loss = 0.0
        pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{max_epochs}",
                     unit="batch", dynamic_ncols=True)
        for batch in loader:
            x = batch["instances"]
            mask = batch["mask"]
            y = batch["label"]
            t = batch["temp_idx"]
            bags = x.shape[0]

            out = mil(x)
            bag_logit = out["bag_logit"]
            inst_logit = out["inst_logit"]
            bag_repr = out["bag_repr"]
            inst_repr = out["encoder_out"]

            loss_bag = bce(bag_logit, y)

            # Instance-level auxiliary loss.  Method is configurable because
            # the optimal strategy for assigning instance-level targets from
            # bag-level labels is an open research question.  See mil/DESIGN.md.
            inst_loss_total = 0.0
            inst_count = 0
            for i in range(y.size(0)):
                n_valid = int(mask[i].sum().item())
                if n_valid == 0:
                    continue
                scores = inst_logit[i, :n_valid]  # [n_valid]

                if y[i].item() > 0.5:  # label=1: positive bag (contains errors)
                    # ---- positive bag (wrong answer) ----
                    if instance_loss_method == "topk":
                        k = max(1, n_valid // 3)
                        topk_logprobs, topk_idx = torch.topk(scores, k)
                        loss_pos = bce(topk_logprobs, torch.ones(k, device=device))
                        all_idx = set(range(n_valid))
                        rest_idx = torch.tensor(sorted(all_idx - set(topk_idx.tolist())), device=device)
                        if len(rest_idx) > 0:
                            rest_logits = scores[rest_idx]
                            loss_rest = bce(rest_logits, torch.zeros(len(rest_idx), device=device))
                            inst_loss_total += loss_pos.sum() + loss_rest.sum()
                            inst_count += k + len(rest_idx)
                        else:
                            inst_loss_total += loss_pos.sum()
                            inst_count += k

                    elif instance_loss_method == "pure":
                        k = 1
                        topk_logprobs, topk_idx = torch.topk(scores, k)
                        loss_pos = bce(topk_logprobs, torch.ones(k, device=device))
                        all_idx = set(range(n_valid))
                        rest_idx = torch.tensor(sorted(all_idx - set(topk_idx.tolist())), device=device)
                        if len(rest_idx) > 0:
                            rest_logits = scores[rest_idx]
                            loss_rest = bce(rest_logits, torch.zeros(len(rest_idx), device=device))
                            inst_loss_total += loss_pos.sum() + loss_rest.sum()
                            inst_count += k + len(rest_idx)
                        else:
                            inst_loss_total += loss_pos.sum()
                            inst_count += k

                    elif instance_loss_method == "soft_pseudo_label":
                        probs = torch.sigmoid(scores).detach()
                        if probs.max() < 0.5:
                            probs[probs.argmax()] = 0.5
                        loss = bce(scores, probs)
                        inst_loss_total += loss.sum()
                        inst_count += n_valid

                    elif instance_loss_method == "contrastive":
                        # Relative: encourage one score to stand out above others.
                        # Absolute: push the max score into the positive range
                        # (otherwise all scores drift negative while negative-bag
                        # scores are pushed to 0 → negative separation).
                        loss_val = (torch.logsumexp(scores, dim=0) - scores.max()
                                    + nn.functional.softplus(-scores.max()))
                        inst_loss_total += loss_val
                        inst_count += 1

                else:  # label=0: negative bag (no errors)
                    # ---- negative bag (correct answer) ----
                    if instance_loss_method == "contrastive":
                        loss_neg = scores.pow(2).mean()
                        inst_loss_total += loss_neg * n_valid
                    else:
                        loss_neg = bce(scores, torch.zeros(n_valid, device=device))
                        inst_loss_total += loss_neg.sum()
                    inst_count += n_valid

            loss_inst = inst_loss_total / max(1, inst_count)

            loss_temp = torch.tensor(0.0, device=device)
            loss_smo = torch.tensor(0.0, device=device)
            if use_temp:
                temp_logits_global = global_head(bag_repr)
                loss_temp_global = ce(temp_logits_global, t)
                temp_logits_dyn = dynamic_head(inst_repr)
                temp_logits_dyn_avg = temp_logits_dyn.mean(dim=1)
                loss_temp_dyn = ce(temp_logits_dyn_avg, t)
                loss_temp = alpha * (loss_temp_global + loss_temp_dyn) * 0.5
                loss_smo = gamma * smoothness_loss(temp_logits_dyn)

            loss = loss_bag + loss_temp + beta * loss_inst + loss_smo

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            _batch_loss = float(loss.item())
            _batch_bag = float(loss_bag.item())
            _batch_inst = float(loss_inst.item())
            sum_loss += _batch_loss

            del x, mask, y, t, out, loss, loss_bag, loss_inst, batch

            pbar.update(1)
            pbar.set_postfix(loss=f"{_batch_loss:.4f}",
                             bag=f"{_batch_bag:.4f}",
                             inst=f"{_batch_inst:.4f}",
                             bags=bags)

        pbar.close()
        avg = sum_loss / max(1, len(loader))

        # ---- validation + early stopping ----
        separation = _validate()
        logger.info("epoch=%d done avg_loss=%.6f separation=%.4f best=%.4f patience=%d/%d",
                     epoch + 1, avg, separation, best_separation,
                     patience_counter, early_stop_patience)

        if separation > best_separation:
            best_separation = separation
            patience_counter = 0
            # clone to CPU — state_dict() returns references to live parameters,
            # which are mutated in-place by optimizer.step().
            ckpt: Dict[str, Any] = {
                "mil": {k: v.detach().cpu().clone() for k, v in mil.state_dict().items()},
                "config": cfg,
            }
            if use_temp:
                ckpt["global_head"] = {k: v.detach().cpu().clone() for k, v in global_head.state_dict().items()}
                ckpt["dynamic_head"] = {k: v.detach().cpu().clone() for k, v in dynamic_head.state_dict().items()}
            best_ckpt = ckpt
            logger.info("new_best separation=%.4f", best_separation)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("early_stop separation=%.4f best=%.4f", separation, best_separation)
                break

    if best_ckpt is None:
        best_ckpt = {
            "mil": {k: v.detach().cpu().clone() for k, v in mil.state_dict().items()},
            "config": cfg,
        }
        if use_temp:
            best_ckpt["global_head"] = {k: v.detach().cpu().clone() for k, v in global_head.state_dict().items()}
            best_ckpt["dynamic_head"] = {k: v.detach().cpu().clone() for k, v in dynamic_head.state_dict().items()}
    ckpt_path = cfg["paths"]["mil_ckpt"]
    torch.save(best_ckpt, ckpt_path)
    logger.info("saved_checkpoint=%s best_separation=%.4f run_name=%s log_path=%s",
                 ckpt_path, best_separation, final_run_name, log_path)


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
