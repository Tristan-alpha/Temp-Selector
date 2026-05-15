from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

from features.segmenter import build_segments, segment_pooling
from features.vectorizer import token_to_vec
from mil.model import MILModel, DynamicTempHead, GlobalTempHead, smoothness_loss
from utils.exp_logger import log_exception, setup_experiment_logger


@dataclass
class RowTensor:
    instances: torch.Tensor
    label: torch.Tensor
    temp_idx: torch.Tensor


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class BagDataset(Dataset):
    def __init__(self, data_path: str, temp_bins: List[float], instance_dim: int,
                 pooling_mode: str = "mean", segment_size: int = 32,
                 segment_mode: str = "step", feature_mode: str = "basic",
                 extractor=None, hidden_batch_size: int = 256):
        self.rows: List[Tuple[torch.Tensor, int, int]] = []
        bin_map = {float(v): i for i, v in enumerate(temp_bins)}

        need_hidden = feature_mode in {"hidden_states", "all"} and extractor is not None

        # Load all rows from JSONL
        all_rows = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                all_rows.append(json.loads(line))

        if need_hidden:
            # Batch extract hidden states, pool, and store
            for batch_start in range(0, len(all_rows), hidden_batch_size):
                batch_rows = all_rows[batch_start:batch_start + hidden_batch_size]
                prompts = [r["metadata"]["rendered_prompt"] if "metadata" in r and "rendered_prompt" in r.get("metadata", {})
                           else r["prompt"] for r in batch_rows]
                responses = [r["response"] for r in batch_rows]
                hs_tensors = extractor.extract(prompts, responses)

                for row, hs in zip(batch_rows, hs_tensors):
                    token_features = row.get("token_features", [])
                    hs_list = hs.tolist() if hasattr(hs, "tolist") else hs
                    for j in range(min(len(hs_list), len(token_features))):
                        token_features[j]["hidden"] = hs_list[j]

                    token_texts = [tf.get("text", "") if isinstance(tf, dict) else getattr(tf, "text", "")
                                   for tf in token_features]
                    response = row.get("response", "")
                    spans = build_segments(
                        tokens=token_texts, mode=segment_mode,
                        segment_size=segment_size, response=response,
                    )
                    token_vecs = [token_to_vec(tf, instance_dim) for tf in token_features]
                    inst_vecs = segment_pooling(token_vecs, spans, instance_dim,
                                               mode=pooling_mode, segment_size=segment_size)
                    instances = torch.tensor(inst_vecs, dtype=torch.float32)
                    label = int(row.get("label", 0))
                    t = float(row.get("temperature", temp_bins[0]))
                    temp_idx = bin_map.get(t, 0)
                    self.rows.append((instances, label, temp_idx))
        else:
            # Logprob-only features (no hidden states)
            for row in all_rows:
                token_features = row.get("token_features", [])
                token_texts = [tf.get("text", "") if isinstance(tf, dict) else getattr(tf, "text", "")
                               for tf in token_features]
                response = row.get("response", "")
                spans = build_segments(
                    tokens=token_texts, mode=segment_mode,
                    segment_size=segment_size, response=response,
                )
                token_vecs = [token_to_vec(tf, instance_dim) for tf in token_features]
                inst_vecs = segment_pooling(token_vecs, spans, instance_dim,
                                           mode=pooling_mode, segment_size=segment_size)
                instances = torch.tensor(inst_vecs, dtype=torch.float32)
                label = int(row.get("label", 0))
                t = float(row.get("temperature", temp_bins[0]))
                temp_idx = bin_map.get(t, 0)
                self.rows.append((instances, label, temp_idx))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> RowTensor:
        ins, y, tidx = self.rows[idx]
        return RowTensor(
            instances=ins,
            label=torch.tensor(y, dtype=torch.float32),
            temp_idx=torch.tensor(tidx, dtype=torch.long),
        )


def collate_rows(rows: List[RowTensor]) -> Dict[str, torch.Tensor]:
    max_k = max(r.instances.shape[0] for r in rows)
    d = rows[0].instances.shape[1]
    b = len(rows)
    x = torch.zeros((b, max_k, d), dtype=torch.float32)
    mask = torch.zeros((b, max_k), dtype=torch.float32)
    y = torch.stack([r.label for r in rows], dim=0)
    t = torch.stack([r.temp_idx for r in rows], dim=0)

    for i, r in enumerate(rows):
        k = r.instances.shape[0]
        x[i, :k] = r.instances
        mask[i, :k] = 1.0

    return {"instances": x, "mask": mask, "label": y, "temp_idx": t}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(config_path: str, data_path: str, run_name: str | None = None, log_dir: str = "logs") -> None:
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
    feature_mode = cfg["inference"].get("feature_mode", "basic")
    hidden_batch_size = int(cfg["mil"]["training"].get("hidden_batch_size", 256))
    backend = cfg["inference"].get("backend", "sglang")

    # Create SGLang engine for online hidden state extraction
    engine = None
    extractor = None
    if feature_mode in {"hidden_states", "all"} and backend == "sglang":
        from sglang import Engine
        gpu_mem = float(cfg["inference"].get("gpu_memory_utilization", 0.90))
        tp_size = 1
        tp_str = cfg["inference"].get("tensor_parallel_size", "auto")
        if isinstance(tp_str, str) and tp_str == "auto":
            import os as _os
            visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            tp_size = max(1, len([d for d in visible.split(",") if d.strip() and d.strip() != "-1"])) if visible else 1
        else:
            tp_size = max(1, int(tp_str))
        max_tokens = int(cfg["inference"].get("max_new_tokens", 8192))
        engine = Engine(
            model_path=cfg["inference"]["model_name_or_path"],
            tp_size=tp_size,
            mem_fraction_static=gpu_mem,
            context_length=max_tokens + 2048,
            random_seed=int(cfg.get("seed", 42)),
            log_level="error",
            enable_return_hidden_states=True,
        )
        from inference.sglang_hidden_extractor import SGLangHiddenStateExtractor
        extractor = SGLangHiddenStateExtractor(engine)
        logger.info("SGLang engine ready for online hidden extraction")

    dataset = BagDataset(
        data_path=data_path, temp_bins=temp_bins, instance_dim=instance_dim,
        pooling_mode=cfg["data"].get("segment_pooling", "mean"),
        segment_size=int(cfg["data"].get("segment_size", 32)),
        segment_mode=cfg["data"].get("segment_mode", "step"),
        feature_mode=feature_mode,
        extractor=extractor,
        hidden_batch_size=hidden_batch_size,
    )
    logger.info("dataset_size=%d", len(dataset))

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    n_gpu = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    logger.info("device=%s n_gpu=%d", device, n_gpu)

    loader = DataLoader(
        dataset,
        batch_size=int(cfg["mil"]["training"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_rows,
        pin_memory=True,
        num_workers=2,
    )

    mil = MILModel(
        input_dim=instance_dim, hidden_dim=hidden_dim,
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

    n_pos = sum(1 for _, label, _ in dataset.rows if label > 0.5)
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
    val_dataset = BagDataset(
        data_path=val_path, temp_bins=temp_bins, instance_dim=instance_dim,
        pooling_mode=cfg["data"].get("segment_pooling", "mean"),
        segment_size=int(cfg["data"].get("segment_size", 32)),
        segment_mode=cfg["data"].get("segment_mode", "step"),
        feature_mode=feature_mode,
        extractor=extractor,
        hidden_batch_size=hidden_batch_size,
    )

    # Engine no longer needed after all datasets constructed
    if engine is not None:
        engine.shutdown()

    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["mil"]["training"]["batch_size"]),
        shuffle=False,
        collate_fn=collate_rows,
        pin_memory=True,
        num_workers=2,
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
                x_v = batch_v["instances"].to(device)
                mask_v = batch_v["mask"].to(device)
                y_v = batch_v["label"].to(device)
                inst = mil(x_v)["inst_logit"]
                for i in range(y_v.size(0)):
                    n_valid = int(mask_v[i].sum().item())
                    if n_valid == 0:
                        continue
                    bag_mean = inst[i, :n_valid].mean().item()
                    if y_v[i].item() > 0.5:
                        pos_means.append(bag_mean)
                    else:
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
        for batch in loader:
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)
            t = batch["temp_idx"].to(device)

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

                if y[i].item() > 0.5:
                    # ---- positive bag (wrong answer) ----
                    if instance_loss_method == "topk":
                        k = max(1, n_valid // 3)
                        topk_logits, topk_idx = torch.topk(scores, k)
                        loss_pos = bce(topk_logits, torch.ones(k, device=device))
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
                        topk_logits, topk_idx = torch.topk(scores, k)
                        loss_pos = bce(topk_logits, torch.ones(k, device=device))
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

                else:
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

            sum_loss += float(loss.item())

        avg = sum_loss / max(1, len(loader))

        # ---- validation + early stopping ----
        separation = _validate()
        logger.info("epoch=%d loss=%.6f separation=%.4f best=%.4f patience=%d/%d",
                     epoch + 1, avg, separation, best_separation,
                     patience_counter, early_stop_patience)

        if separation > best_separation:
            best_separation = separation
            patience_counter = 0
            ckpt: Dict[str, Any] = {
                "mil": mil.state_dict(),
                "config": cfg,
            }
            if use_temp:
                ckpt["global_head"] = global_head.state_dict()
                ckpt["dynamic_head"] = dynamic_head.state_dict()
            best_ckpt = ckpt
            logger.info("new_best separation=%.4f", best_separation)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("early_stop separation=%.4f best=%.4f", separation, best_separation)
                break

    if best_ckpt is None:
        best_ckpt = {
            "mil": mil.state_dict(),
            "config": cfg,
        }
        if use_temp:
            best_ckpt["global_head"] = global_head.state_dict()
            best_ckpt["dynamic_head"] = dynamic_head.state_dict()
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
    args = parser.parse_args()
    cfg = load_config(args.config)
    data_path = args.data or cfg["paths"]["train_dataset"]
    try:
        train(config_path=args.config, data_path=data_path, run_name=args.run_name, log_dir=args.log_dir)
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
