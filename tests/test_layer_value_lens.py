from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.extract_layer_value_lens_cache import PrefixRecord, gather_prefix_endpoints
from scripts.train_layer_value_lens import (
    evaluate_probabilities,
    expected_calibration_error,
    split_problem_indices,
)


def test_gather_prefix_endpoints_uses_prefix_end_minus_one():
    records = [
        PrefixRecord(0, "train", "p0", "s0", 1, "early", 1, 0, 4, 1),
        PrefixRecord(1, "train", "p0", "s0", 2, "middle", 3, 4, 4, 0),
        PrefixRecord(2, "train", "p1", "s1", 1, "late", 2, 2, 4, 0),
    ]
    source_hidden = {
        "s0": torch.arange(5 * 2 * 3, dtype=torch.float32).view(5, 2, 3),
        "s1": 100 + torch.arange(4 * 2 * 3, dtype=torch.float32).view(4, 2, 3),
    }
    source_logits = {
        "s0": torch.arange(5 * 2 * 4, dtype=torch.float32).view(5, 2, 4),
        "s1": 200 + torch.arange(4 * 2 * 4, dtype=torch.float32).view(4, 2, 4),
    }

    hidden, logits = gather_prefix_endpoints(records, source_hidden, source_logits)

    assert hidden.shape == (3, 2, 3)
    assert logits.shape == (3, 2, 4)
    assert torch.equal(hidden[0], source_hidden["s0"][0])
    assert torch.equal(hidden[1], source_hidden["s0"][2])
    assert torch.equal(hidden[2], source_hidden["s1"][1])


def test_split_problem_indices_keeps_problem_groups_disjoint():
    rows = []
    for problem in range(6):
        for prefix in range(3):
            rows.append({"problem_id": f"p{problem}", "prefix": prefix})

    train_idx, dev_idx = split_problem_indices(rows, dev_fraction=0.33, seed=7)

    train_problems = {rows[i]["problem_id"] for i in train_idx}
    dev_problems = {rows[i]["problem_id"] for i in dev_idx}
    assert train_problems
    assert dev_problems
    assert train_problems.isdisjoint(dev_problems)
    assert len(train_idx) + len(dev_idx) == len(rows)


def test_metrics_match_simple_hand_computation():
    rows = [
        {
            "problem_id": "p0",
            "source_sample_id": "a",
            "n_correct": 0,
            "n_total": 10,
            "target": 0.5 / 11.0,
            "observed_success_rate": 0.0,
            "prefix_stage": "early",
        },
        {
            "problem_id": "p0",
            "source_sample_id": "b",
            "n_correct": 10,
            "n_total": 10,
            "target": 10.5 / 11.0,
            "observed_success_rate": 1.0,
            "prefix_stage": "late",
        },
    ]
    probs = np.asarray([0.1, 0.9])

    metrics = evaluate_probabilities(probs, rows, seed=42, label="test")

    expected_brier = np.mean((probs - np.asarray([0.5 / 11.0, 10.5 / 11.0])) ** 2)
    assert metrics["brier"] == pytest.approx(expected_brier)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["pair_accuracy"] == pytest.approx(1.0)
    assert metrics["n_pairs"] == 1
    assert expected_calibration_error([0.1, 0.9], [0.0, 1.0], n_bins=10) == pytest.approx(0.1)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_tiny_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True)
    layer_ids = [1, 2]
    hidden_dim = 4

    def rows(split: str, n_problems: int) -> list[dict]:
        out = []
        for problem in range(n_problems):
            for prefix in range(2):
                correct = 4 if (problem + prefix) % 2 else 0
                out.append({
                    "record_index": len(out),
                    "split": split,
                    "problem_id": f"{split}_p{problem}",
                    "source_sample_id": f"{split}_s{problem}_{prefix}",
                    "prefix_segments": prefix + 1,
                    "prefix_stage": "early" if prefix == 0 else "late",
                    "prefix_token_end": prefix + 1,
                    "n_correct": correct,
                    "n_total": 4,
                    "observed_success_rate": correct / 4,
                    "target": (correct + 0.5) / 5,
                    "source_individual_label": 0 if correct else 1,
                })
        return out

    train_rows = rows("train", 4)
    val_rows = rows("val", 3)
    _write_jsonl(cache_dir / "train_metadata.jsonl", train_rows)
    _write_jsonl(cache_dir / "val_metadata.jsonl", val_rows)

    for split, split_rows in (("train", train_rows), ("val", val_rows)):
        n = len(split_rows)
        hidden = torch.zeros(n, len(layer_ids), hidden_dim, dtype=torch.float16)
        logits = torch.zeros(n, len(layer_ids), 4, dtype=torch.float32)
        for i, row in enumerate(split_rows):
            sign = 1.0 if row["observed_success_rate"] > 0.5 else -1.0
            hidden[i, :, 0] = sign
            hidden[i, :, 1] = float(row["prefix_segments"])
            logits[i, :, 0] = -sign
            logits[i, :, 1] = sign
        torch.save({
            "split": split,
            "layer_ids": layer_ids,
            "prefix_hidden": hidden,
            "logit_features": logits,
            "logit_feature_names": ["entropy", "sampled_logprob", "top1_logprob", "top1_margin"],
        }, cache_dir / f"{split}_layers_0001_0002.pt")

    manifest = {
        "config": "",
        "model_path": "dummy",
        "num_hidden_layers": 2,
        "hidden_size": hidden_dim,
        "layer_ids": layer_ids,
        "splits": {
            "train": {"n_prefixes": len(train_rows), "n_problem_ids": 4, "metadata": "train_metadata.jsonl"},
            "val": {"n_prefixes": len(val_rows), "n_problem_ids": 3, "metadata": "val_metadata.jsonl"},
        },
        "overlap": {"train_val_problem_overlap": 0},
        "chunks": [
            {"split": "train", "cache_file": "train_layers_0001_0002.pt", "layer_ids": layer_ids},
            {"split": "val", "cache_file": "val_layers_0001_0002.pt", "layer_ids": layer_ids},
        ],
    }
    (cache_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_train_layer_value_lens_cli_smoke_on_tiny_cache(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    output_dir = tmp_path / "out"
    _make_tiny_cache(cache_dir)

    cmd = [
        sys.executable,
        "scripts/train_layer_value_lens.py",
        "--cache-dir",
        str(cache_dir),
        "--output-dir",
        str(output_dir),
        "--skip-pvm-baseline",
        "--probe-families",
        "hidden,logit",
        "--max-layers",
        "2",
        "--epochs",
        "3",
        "--patience",
        "2",
        "--batch-size",
        "2",
        "--eval-batch-size",
        "4",
        "--dev-fraction",
        "0.5",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["problem_split"]["train_val_problem_overlap"] == 0
    assert (output_dir / "metrics_by_layer.csv").exists()
    assert (output_dir / "fig_layer_metrics.png").exists()
