"""Render a multi-subfigure PNG from a training metrics JSONL file.

Usage:
    python scripts/plot_training.py --metrics logs/exp_mil_metrics.jsonl
    python scripts/plot_training.py --metrics logs/exp_ppo_metrics.jsonl --output plots/exp_ppo.png

Auto-detects MIL vs PPO from the first JSONL row (``epoch`` vs ``iter`` key).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


def load_metrics(path: str) -> List[Dict[str, Any]]:
    """Load a metrics JSONL file, skipping the last line if it is incomplete."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    if not lines:
        return rows
    for line in lines[:-1]:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # silently skip corrupted lines
            continue
    # Try last line — skip if truncated
    last = lines[-1].strip()
    if last:
        try:
            rows.append(json.loads(last))
        except json.JSONDecodeError:
            pass
    return rows


def _safe_get(rows: List[Dict[str, Any]], key: str, default: Any = None) -> List[Any]:
    """Extract a list of values for *key* from all rows, returning *default* if absent."""
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            v = default
        vals.append(v)
    return vals


def _has_key(rows: List[Dict[str, Any]], key: str) -> bool:
    return any(key in r for r in rows)


# ──────────────────────── matplotlib import guard ────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")                      # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
# ─────────────────────────────────────────────────────────────────────────


def _plot_or_no_data(ax, rows, keys, labels, ylabel="", title="",
                     x_values=None, ylim=None):
    """Plot one or more lines; if keys are all absent render 'No data'."""
    if not HAS_MPL:
        return
    present_keys = [k for k in keys if _has_key(rows, k)]
    if not present_keys:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=12)
        ax.set_title(title)
        return

    xs = x_values if x_values is not None else list(range(1, len(rows) + 1))
    for key in present_keys:
        label = labels[keys.index(key)] if len(keys) > 1 else (labels[0] if labels else key)
        vals = _safe_get(rows, key)
        ax.plot(xs, vals, label=label, linewidth=1.0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if len(present_keys) > 1:
        ax.legend(fontsize=7)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)


def plot_mil(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Render MIL 3×2 grid PNG."""
    if not HAS_MPL:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    epochs = list(range(1, len(rows) + 1))

    _plot_or_no_data(axes[0, 0], rows, ["loss"], ["BCE Loss"],
                     "Loss", "Training Loss")
    axes[0, 0].set_xlabel("Epoch")

    _plot_or_no_data(axes[0, 1], rows, ["train_acc", "val_acc"],
                     ["Train", "Val"], "Accuracy", "Accuracy", ylim=(0, 1))
    axes[0, 1].set_xlabel("Epoch")

    _plot_or_no_data(axes[1, 0], rows, ["grad_norm"], ["Grad Norm"],
                     "L2 Norm", "Gradient Norm")
    axes[1, 0].set_xlabel("Epoch")

    _plot_or_no_data(axes[1, 1], rows, ["attn_entropy"], ["Entropy"],
                     "Entropy (nats)", "Attention Entropy")
    axes[1, 1].set_xlabel("Epoch")

    _plot_or_no_data(axes[2, 0], rows, ["val_acc_pos", "val_acc_neg"],
                     ["Pos (errors)", "Neg (clean)"], "Accuracy",
                     "Validation Per-Class Accuracy", ylim=(0, 1))
    axes[2, 0].set_xlabel("Epoch")

    # (2, 1) left empty
    axes[2, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"MIL plot saved to {output_path}")


def plot_ppo(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Render PPO 4×3 grid PNG."""
    if not HAS_MPL:
        print("matplotlib is required for plotting. Install with: pip install matplotlib")
        sys.exit(1)

    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    iters = list(range(1, len(rows) + 1))

    _plot_or_no_data(axes[0, 0], rows, ["train_acc", "val_acc"],
                     ["Train", "Val"], "Accuracy", "Accuracy", ylim=(0, 1))
    axes[0, 0].set_xlabel("Iteration")

    _plot_or_no_data(axes[0, 1], rows, ["policy_loss"], ["Policy Loss"],
                     "Loss", "Policy Loss")
    axes[0, 1].set_xlabel("Iteration")

    _plot_or_no_data(axes[0, 2], rows, ["value_loss"], ["Value Loss"],
                     "Loss", "Value Loss")
    axes[0, 2].set_xlabel("Iteration")

    _plot_or_no_data(axes[1, 0], rows, ["entropy"], ["Entropy"],
                     "Entropy (nats)", "Policy Entropy")
    axes[1, 0].set_xlabel("Iteration")

    _plot_or_no_data(axes[1, 1], rows, ["reward_pos_ratio"], ["Pos Ratio"],
                     "Ratio", "Reward Positive Ratio", ylim=(0, 1))
    axes[1, 1].set_xlabel("Iteration")

    # Temperature distribution heatmap
    if _has_key(rows, "temp_dist") and HAS_MPL:
        temp_keys: Optional[List[str]] = None
        for r in rows:
            if "temp_dist" in r and r["temp_dist"]:
                temp_keys = sorted(r["temp_dist"].keys(), key=lambda x: float(x))
                break
        if temp_keys:
            heatmap_data: List[List[int]] = []
            for r in rows:
                td = r.get("temp_dist", {})
                heatmap_data.append([td.get(k, 0) for k in temp_keys])
            im = axes[1, 2].imshow(list(zip(*heatmap_data))[::-1], aspect="auto", cmap="YlOrRd")
            axes[1, 2].set_xticks(range(0, len(iters), max(1, len(iters) // 5)))
            axes[1, 2].set_xticklabels([str(iters[i]) for i in range(0, len(iters), max(1, len(iters) // 5))])
            yticks_pos = range(len(temp_keys))
            axes[1, 2].set_yticks(yticks_pos)
            axes[1, 2].set_yticklabels(temp_keys[::-1] if len(temp_keys) <= 10
                                       else [temp_keys[::-1][i] for i in range(0, len(temp_keys), max(1, len(temp_keys)//6))])
            axes[1, 2].set_title("Temperature Distribution")
            axes[1, 2].set_xlabel("Iteration")
            plt.colorbar(im, ax=axes[1, 2], label="Count")
        else:
            axes[1, 2].text(0.5, 0.5, "No data", ha="center", va="center",
                            transform=axes[1, 2].transAxes, color="gray", fontsize=12)
            axes[1, 2].set_title("Temperature Distribution")
    else:
        axes[1, 2].text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=axes[1, 2].transAxes, color="gray", fontsize=12)
        axes[1, 2].set_title("Temperature Distribution")

    _plot_or_no_data(axes[2, 0], rows, ["temp_mean", "temp_std"],
                     ["Mean", "Std"], "Temperature",
                     "Temperature (Mean ± Std)")
    axes[2, 0].set_xlabel("Iteration")

    _plot_or_no_data(axes[2, 1], rows, ["segments_mean", "segments_min", "segments_max"],
                     ["Mean", "Min", "Max"], "Segments",
                     "Segments per Chain")
    axes[2, 1].set_xlabel("Iteration")

    _plot_or_no_data(axes[2, 2], rows, ["advantage_mean", "advantage_std"],
                     ["Mean", "Std"], "Advantage",
                     "Advantage (Mean ± Std)")
    axes[2, 2].set_xlabel("Iteration")

    _plot_or_no_data(axes[3, 0], rows, ["clip_fraction"], ["Clip Fraction"],
                     "Fraction", "PPO Clip Fraction")
    axes[3, 0].set_xlabel("Iteration")

    _plot_or_no_data(axes[3, 1], rows, ["total_steps"], ["Steps"],
                     "Steps", "Total Training Steps")
    axes[3, 1].set_xlabel("Iteration")

    axes[3, 2].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PPO plot saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render training metrics JSONL as a multi-subfigure PNG."
    )
    parser.add_argument("--metrics", required=True,
                        help="Path to a {run_name}_{stage}_metrics.jsonl file")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: <metrics_basename>_training.png)")
    args = parser.parse_args()

    metrics_path = args.metrics
    if not os.path.exists(metrics_path):
        print(f"File not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_metrics(metrics_path)
    if not rows:
        print(f"No valid rows in {metrics_path}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect stage
    if "epoch" in rows[0]:
        stage = "mil"
    elif "iter" in rows[0]:
        stage = "ppo"
    else:
        print("Cannot detect stage: first row has neither 'epoch' nor 'iter'.",
              file=sys.stderr)
        sys.exit(1)

    # Derive output path
    if args.output:
        output_path = args.output
    else:
        base = os.path.splitext(os.path.basename(metrics_path))[0]
        output_path = os.path.join(os.path.dirname(metrics_path) or ".",
                                    f"{base}_training.png")

    if stage == "mil":
        plot_mil(rows, output_path)
    else:
        plot_ppo(rows, output_path)


if __name__ == "__main__":
    main()
