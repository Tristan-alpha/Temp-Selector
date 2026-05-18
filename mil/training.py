from __future__ import annotations

import argparse
import json
import random
import time
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset, Sampler

from features.segmenter import build_segments, segment_pooling
from features.vectorizer import token_to_vec
from mil.model import MILModel, DynamicTempHead, GlobalTempHead, smoothness_loss
from utils.exp_logger import log_exception, setup_experiment_logger


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TokenBatchSampler(Sampler):
    """Group samples so each batch stays under ``max_tokens`` total (prompt + response)."""

    def __init__(self, token_counts: List[int], max_tokens: int, shuffle: bool = True):
        self.token_counts = token_counts
        self.max_tokens = max_tokens
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.token_counts)))
        if self.shuffle:
            random.shuffle(indices)
        batch: List[int] = []
        batch_tokens = 0
        for idx in indices:
            n = self.token_counts[idx]
            if batch and batch_tokens + n > self.max_tokens:
                yield batch
                batch = []
                batch_tokens = 0
            batch.append(idx)
            batch_tokens += n
        if batch:
            yield batch

    def __len__(self) -> int:
        return max(1, sum(self.token_counts) // max(1, self.max_tokens))


class BagDataset(Dataset):
    """Lazy dataset: stores row metadata only.  Feature extraction happens in collate_fn."""

    def __init__(self, data_path: str):
        self.rows: List[Dict[str, Any]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                self.rows.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.rows[idx]


def make_collate_fn(
    extractor=None,
    feature_mode: str = "basic",
    instance_dim: int = 4098,
    segment_mode: str = "step",
    segment_size: int = 256,
    pooling_mode: str = "mean",
    temp_bins: List[float] | None = None,
    train_device: torch.device | None = None,
):
    """Factory that returns a collate function with per-batch SGLang extraction.

    The returned function receives a list of row dicts from BagDataset,
    optionally extracts logprob/hidden features via SGLang, builds per-segment
    instance vectors, and returns a padded batch dict for MIL training.
    """
    if temp_bins is None:
        temp_bins = [0.0]
    bin_map = {float(v): i for i, v in enumerate(temp_bins)}
    need_hidden = feature_mode in {"hidden_states", "all"} and extractor is not None
    need_logprobs = feature_mode in {"topk_logprobs", "all"} and extractor is not None

    def collate_fn(batch_rows: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        _t = [time.perf_counter()]
        hidden_tensors = None
        logprob_tensors = None
        if need_hidden or need_logprobs:
            full_ids = [r["_full_ids"] for r in batch_rows]
            prompt_lens = [r["_prompt_len"] for r in batch_rows]

            if need_hidden:
                hidden_tensors = extractor.extract_hidden_from_ids(full_ids, prompt_lens)
            if need_logprobs:
                temps = [float(r.get("temperature", 0.0)) for r in batch_rows]
                logprob_tensors = extractor.extract_logprobs_from_ids(full_ids, prompt_lens, temperatures=temps)
        _t.append(time.perf_counter())  # _t[1] = after SGLand extract

        instances_list: List[torch.Tensor] = []
        labels: List[float] = []
        temp_indices: List[int] = []

        _sub_t = [0.0, 0.0, 0.0]  # build_tensor, build_segments, segment_pooling
        for i, row in enumerate(batch_rows):
            token_features = row.get("token_features", [])
            if not token_features:
                instances_list.append(torch.zeros(1, instance_dim))
                labels.append(float(row.get("label", 0)))
                t_val = float(row.get("temperature", temp_bins[0]))
                temp_indices.append(bin_map.get(t_val, 0))
                continue

            h_t = hidden_tensors[i] if hidden_tensors is not None else None
            l_t = logprob_tensors[i] if logprob_tensors is not None else None
            n = len(token_features)

            _t0 = time.perf_counter()
            parts = [torch.tensor([[float(tf.get("logprob", -20.0)), float(tf.get("entropy", 0.0))]
                                   for tf in token_features], dtype=torch.float32)]
            if l_t is not None:
                parts.append(l_t[:n])
            if h_t is not None:
                parts.append(h_t[:n])
            t = torch.cat(parts, dim=1)
            if t.shape[1] < instance_dim:
                t = torch.cat([t, torch.zeros(n, instance_dim - t.shape[1])], dim=1)
            else:
                t = t[:, :instance_dim]
            _sub_t[0] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            token_texts = [tf.get("text", "") if isinstance(tf, dict) else getattr(tf, "text", "")
                           for tf in token_features]
            response = row.get("response", "")
            spans = build_segments(
                tokens=token_texts, mode=segment_mode,
                segment_size=segment_size, response=response,
            )
            _sub_t[1] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            if train_device is not None and train_device.type == "cuda":
                inst = segment_pooling(t.to(train_device), spans, instance_dim,
                                       mode=pooling_mode, segment_size=segment_size).cpu()
            else:
                inst = segment_pooling(t, spans, instance_dim,
                                       mode=pooling_mode, segment_size=segment_size)
            _sub_t[2] += time.perf_counter() - _t0
            instances_list.append(inst)
            labels.append(float(row.get("label", 0)))
            t_val = float(row.get("temperature", temp_bins[0]))
            temp_indices.append(bin_map.get(t_val, 0))

        _t.append(time.perf_counter())  # _t[2] = after token→segment loop

        max_k = max(inst.shape[0] for inst in instances_list)
        d = instances_list[0].shape[1]
        b = len(instances_list)
        x = torch.zeros((b, max_k, d), dtype=torch.float32)
        mask = torch.zeros((b, max_k), dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)
        t = torch.tensor(temp_indices, dtype=torch.long)

        for i, inst in enumerate(instances_list):
            k = inst.shape[0]
            x[i, :k] = inst
            mask[i, :k] = 1.0

        _t.append(time.perf_counter())  # _t[3] = after pad

        batch_tokens = sum(len(r.get("_full_ids", r.get("token_features", []))) for r in batch_rows)
        _coll_timings = {
            "extract_s": _t[1] - _t[0],
            "token2seg_s": _t[2] - _t[1],
            "pad_s": _t[3] - _t[2],
            "total_s": _t[3] - _t[0],
            "tok2seg_build_s": _sub_t[0],
            "tok2seg_seg_s": _sub_t[1],
            "tok2seg_pool_s": _sub_t[2],
        }

        return {"instances": x, "mask": mask, "label": y, "temp_idx": t,
                "_batch_tokens": batch_tokens, "_coll_timings": _coll_timings}

    return collate_fn


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

    # Create extraction engine (shared across train + val)
    extraction_logprobs = feature_mode in {"topk_logprobs", "all"}
    runner = None
    if extraction_logprobs or feature_mode in {"hidden_states", "all"}:
        from inference.vllm_runner import VLLMFeatureExporter
        runner = VLLMFeatureExporter(
            model_name_or_path=cfg["inference"]["model_name_or_path"],
            max_new_tokens=int(cfg["inference"].get("max_new_tokens", 8192)),
            parallel_size=cfg["inference"].get("parallel_size", "auto"),
            gpu_memory_utilization=float(cfg["inference"].get("gpu_memory_utilization", 0.90)),
            feature_mode=feature_mode,
            engine_preset="prefill",
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
                resp_ids = [tf["token_id"] for tf in row.get("token_features", [])]
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

    max_tokens_per_batch = int(cfg["mil"]["training"].get("max_tokens_per_batch", 100000))
    token_counts = [len(r.get("_full_ids", r.get("token_features", []))) for r in dataset.rows]
    logger.info("max_tokens_per_batch=%d total_tokens=%d", max_tokens_per_batch, sum(token_counts))

    train_sampler = TokenBatchSampler(token_counts, max_tokens_per_batch, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=0,
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

    n_pos = sum(1 for r in dataset.rows if float(r.get("label", 0)) > 0.5)
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
                resp_ids = [tf["token_id"] for tf in row.get("token_features", [])]
                row["_full_ids"] = pids + resp_ids
                row["_prompt_len"] = len(pids)

    val_token_counts = [len(r.get("_full_ids", r.get("token_features", []))) for r in val_dataset.rows]
    val_sampler = TokenBatchSampler(val_token_counts, max_tokens_per_batch, shuffle=False)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=0,
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
        cumul_tokens = 0
        total_tokens = sum(token_counts)
        for batch_idx, batch in enumerate(loader):
            t0 = time.perf_counter()
            x = batch["instances"].to(device)
            mask = batch["mask"].to(device)
            y = batch["label"].to(device)
            t = batch["temp_idx"].to(device)
            bags = x.shape[0]

            out = mil(x)
            bag_logit = out["bag_logit"]
            inst_logit = out["inst_logit"]
            bag_repr = out["bag_repr"]
            inst_repr = out["encoder_out"]
            t1 = time.perf_counter()

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
            t2 = time.perf_counter()

            _batch_loss = float(loss.item())
            _batch_bag = float(loss_bag.item())
            _batch_inst = float(loss_inst.item())
            _batch_tokens = batch.get("_batch_tokens", 0)
            _ct = batch.get("_coll_timings", {})
            sum_loss += _batch_loss

            del x, mask, y, t, out, loss, loss_bag, loss_inst, batch

            cumul_tokens += _batch_tokens
            logger.info("epoch=%d batch=%d tokens=%d/%d bags=%d loss=%.4f bag=%.4f inst=%.4f | "
                         "coll[extract=%.2fs tok2seg=%.2fs(build=%.2f seg=%.2f pool=%.2f) "
                         "pad=%.3fs] fwd=%.2fs bwd=%.2fs",
                         epoch + 1, batch_idx + 1,
                         cumul_tokens, total_tokens,
                         bags,
                         _batch_loss, _batch_bag, _batch_inst,
                         _ct.get("extract_s", 0), _ct.get("token2seg_s", 0),
                         _ct.get("tok2seg_build_s", 0), _ct.get("tok2seg_seg_s", 0),
                         _ct.get("tok2seg_pool_s", 0), _ct.get("pad_s", 0),
                         t1 - t0, t2 - t1)

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
