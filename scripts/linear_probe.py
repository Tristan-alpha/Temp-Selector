"""Linear probe: logistic regression on cached MIL segment features.

Tests whether concat-pooled 4096-dim features contain linearly separable
error signal.  If LR cannot beat the class-prior baseline, no neural
network will.

Usage:
    conda activate edu
    python scripts/linear_probe.py --cache datasets/cache/train-fixed_window-concat-topk_logprobs-64-64.pt
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear probe on MIL segment features")
    parser.add_argument("--cache", required=True, help="Path to .pt or .safetensors cache file")
    parser.add_argument("--max-bags", type=int, default=0,
                        help="Subsample N bags (0 = all). Use for quick diagnosis.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading cache ...")
    t0 = time.time()
    if args.cache.endswith(".safetensors"):
        from safetensors.torch import load_file as _sf_load
        packed = _sf_load(args.cache)
        # Reconstruct list-of-dicts using the same logic as mil.utils
        instances = packed["instances"]
        splits = packed["_csr_splits"].tolist()
        labels = packed["labels"].tolist()
        tmp_indices = packed["temp_indices"].tolist()
        cache = []
        for i in range(len(splits) - 1):
            s, e = int(splits[i]), int(splits[i + 1])
            cache.append({
                "instances": instances[s:e],
                "label": float(labels[i]),
                "temp_idx": int(tmp_indices[i]),
            })
    else:
        cache = torch.load(args.cache, weights_only=False)
    rng = np.random.RandomState(args.seed)
    if args.max_bags > 0 and args.max_bags < len(cache):
        idx = rng.choice(len(cache), args.max_bags, replace=False)
        cache = [cache[i] for i in idx]
        print(f"Subsampled to {len(cache)} bags")
    y = np.array([e["label"] for e in cache])
    n_bags = len(cache)
    n_pos = int(y.sum())
    n_neg = n_bags - n_pos
    prior = max(n_pos, n_neg) / n_bags
    print(f"Loaded {n_bags} bags in {time.time() - t0:.0f}s")
    print(f"  positive (error): {n_pos} ({100 * n_pos / n_bags:.1f}%)")
    print(f"  negative (clean): {n_neg}")
    print(f"  class-prior baseline: {prior:.4f}\n")

    # ── Probe 1: bag-level mean pooling ──
    print("--- Bag mean-pool ---")
    X_mean = np.stack([e["instances"].float().mean(dim=0).numpy() for e in cache])
    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf.fit(X_mean, y)
    acc_mean = accuracy_score(y, clf.predict(X_mean))
    print(f"  Train acc: {acc_mean:.4f}  (baseline: {prior:.4f})")

    # ── Probe 2: bag-level max pooling ──
    print("--- Bag max-pool ---")
    X_max = np.stack([e["instances"].float().max(dim=0).values.numpy() for e in cache])
    clf2 = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf2.fit(X_max, y)
    acc_max = accuracy_score(y, clf2.predict(X_max))
    print(f"  Train acc: {acc_max:.4f}  (baseline: {prior:.4f})")

    # ── Probe 3: segment-level (subsample to limit memory) ──
    print(f"\n--- Segment-level ---")
    n_max_seg = min(100000, sum(e["instances"].shape[0] for e in cache))
    # Memory estimate: 4096 dims × 100k segs × 4 bytes = 1.6 GB
    X_seg_chunks = []
    y_seg_chunks = []
    seg_total = 0
    for e in cache:
        inst = e["instances"]  # [K, 4096]
        k = inst.shape[0]
        if seg_total + k > n_max_seg:
            break
        X_seg_chunks.append(inst.float().numpy())
        y_seg_chunks.append(np.full(k, e["label"]))
        seg_total += k
    X_seg = np.concatenate(X_seg_chunks, axis=0)
    y_seg = np.concatenate(y_seg_chunks)
    print(f"  Loaded {seg_total} segments ({X_seg.shape})")

    # Shuffle and subsample to 50k
    n_sample = min(50000, seg_total)
    idx = rng.choice(seg_total, n_sample, replace=False)
    X_sub, y_sub = X_seg[idx], y_seg[idx]
    print(f"  Training on {n_sample} segments ...")
    t0 = time.time()
    clf3 = LogisticRegression(max_iter=2000, C=1.0, random_state=args.seed)
    clf3.fit(X_sub, y_sub)
    acc_seg = accuracy_score(y_sub, clf3.predict(X_sub))
    print(f"  LR fit: {time.time() - t0:.0f}s")
    print(f"  Train acc: {acc_seg:.4f}  (baseline: {max(y_sub.mean(), 1 - y_sub.mean()):.4f})")

    # ── Summary ──
    print(f"\n{'=' * 55}")
    print(f"Summary")
    print(f"{'=' * 55}")
    print(f"  Class-prior baseline:        {prior:.4f}")
    print(f"  Bag mean-pool LR acc:        {acc_mean:.4f}")
    print(f"  Bag max-pool LR acc:         {acc_max:.4f}")
    print(f"  Segment-level LR acc:        {acc_seg:.4f}")
    print()
    if max(acc_mean, acc_max, acc_seg) < prior + 0.02:
        print("=> Features contain almost NO linearly separable error signal.")
        print("   The problem is not the model architecture — it is the features.")
    elif max(acc_mean, acc_max, acc_seg) < prior + 0.05:
        print("=> Features contain WEAK signal (marginally better than prior).")
        print("   A neural network might help, but gains will be limited.")
    else:
        print("=> Features contain USEFUL signal. Consider architectural improvements.")
    print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
