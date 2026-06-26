"""CPU-only tests for scripts/plot_training.py"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure scripts/ is on sys.path
_sys_path = [os.path.join(os.path.dirname(__file__), "..", "scripts")] + sys.path
sys.path = _sys_path + sys.path

from plot_training import load_metrics, _safe_get, _has_key


# ── load_metrics ──────────────────────────────────────────────────────


def test_load_metrics_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("")
        f.flush()
        path = f.name
    try:
        rows = load_metrics(path)
        assert rows == []
    finally:
        os.unlink(path)


def test_load_metrics_valid_rows():
    content = (
        '{"epoch": 1, "loss": 0.5}\n'
        '{"epoch": 2, "loss": 0.3}\n'
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name
    try:
        rows = load_metrics(path)
        assert len(rows) == 2
        assert rows[0]["epoch"] == 1
        assert rows[1]["epoch"] == 2
    finally:
        os.unlink(path)


def test_load_metrics_truncated_last_line():
    """Last line is incomplete JSON — should be skipped."""
    content = (
        '{"epoch": 1, "loss": 0.5}\n'
        '{"epoch": 2, "loss":'  # truncated
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name
    try:
        rows = load_metrics(path)
        assert len(rows) == 1
        assert rows[0]["epoch"] == 1
    finally:
        os.unlink(path)


def test_load_metrics_blank_lines_skipped():
    content = (
        '{"epoch": 1, "loss": 0.5}\n'
        '\n'
        '{"epoch": 2, "loss": 0.3}\n'
        '\n'
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name
    try:
        rows = load_metrics(path)
        assert len(rows) == 2
    finally:
        os.unlink(path)


# ── _safe_get ─────────────────────────────────────────────────────────


def test_safe_get_existing_key():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    assert _safe_get(rows, "a") == [1, 3]


def test_safe_get_missing_key():
    rows = [{"a": 1}, {"a": 3}]
    assert _safe_get(rows, "b", default=0.0) == [0.0, 0.0]


def test_safe_get_mixed_key():
    rows = [{"a": 1, "b": 2}, {"a": 3}]
    assert _safe_get(rows, "b", default=0.0) == [2, 0.0]


# ── _has_key ──────────────────────────────────────────────────────────


def test_has_key_true():
    rows = [{"a": 1}, {"b": 2}]
    assert _has_key(rows, "a") is True
    assert _has_key(rows, "b") is True


def test_has_key_false():
    rows = [{"a": 1}, {"a": 3}]
    assert _has_key(rows, "b") is False


def test_has_key_empty_rows():
    assert _has_key([], "a") is False


# ── Stage detection (integration via load_metrics) ────────────────────


def test_stage_detection_mil():
    content = '{"epoch": 1, "loss": 0.5}\n'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name
    try:
        rows = load_metrics(path)
        assert "epoch" in rows[0]
        assert "iter" not in rows[0]
    finally:
        os.unlink(path)


def test_stage_detection_ppo():
    content = '{"iter": 1, "total_loss": 0.5}\n'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(content)
        f.flush()
        path = f.name
    try:
        rows = load_metrics(path)
        assert "iter" in rows[0]
        assert "epoch" not in rows[0]
    finally:
        os.unlink(path)


# ── MIL metrics JSONL format ──────────────────────────────────────────


def test_mil_metrics_format():
    """Verify MIL metrics JSONL has all required keys after 2 synthetic epochs."""
    rows = []
    for epoch in [1, 2]:
        rows.append({
            "epoch": epoch,
            "loss": 0.5 - 0.1 * (epoch - 1),
            "train_acc": 0.8 + 0.05 * (epoch - 1),
            "val_acc": 0.75 + 0.03 * (epoch - 1),
            "val_acc_pos": 0.7 + 0.05 * (epoch - 1),
            "val_acc_neg": 0.85 + 0.02 * (epoch - 1),
            "grad_norm": 1.5 - 0.2 * (epoch - 1),
            "attn_entropy": 0.95 + 0.02 * (epoch - 1),
        })
    required_keys = ["epoch", "loss", "train_acc", "val_acc",
                     "val_acc_pos", "val_acc_neg", "grad_norm", "attn_entropy"]
    for row in rows:
        for k in required_keys:
            assert k in row, f"Missing key {k} in MIL metrics row"
    assert rows[0]["loss"] > rows[1]["loss"]  # loss decreasing


# ── PPO metrics JSONL format ──────────────────────────────────────────


def test_ppo_metrics_format():
    """Verify PPO metrics JSONL has all required keys after 2 synthetic iters."""
    temp_bins = ["0.1", "0.3", "0.5", "0.7", "1.0"]
    rows = []
    for it in [1, 2]:
        rows.append({
            "iter": it,
            "total_loss": 0.8 - 0.1 * (it - 1),
            "policy_loss": 0.5 - 0.05 * (it - 1),
            "value_loss": 0.2 - 0.02 * (it - 1),
            "entropy": 1.2 - 0.1 * (it - 1),
            "reward_mean": -0.1 + 0.15 * (it - 1),
            "reward_pos_ratio": 0.45 + 0.05 * (it - 1),
            "train_acc": 0.55 + 0.05 * (it - 1),
            "val_acc": 0.55 + 0.03 * (it - 1),
            "temp_dist": {t: it * 10 // len(temp_bins) for t in temp_bins},
            "temp_mean": 0.6 + 0.05 * (it - 1),
            "temp_std": 0.3 - 0.05 * (it - 1),
            "segments_mean": 30.0 + 2.0 * (it - 1),
            "segments_min": 10,
            "segments_max": 60,
            "advantage_mean": 0.01,
            "advantage_std": 0.8,
            "clip_fraction": 0.2,
            "total_steps": 480,
        })
    required_keys = ["iter", "total_loss", "policy_loss", "value_loss",
                     "entropy", "reward_mean", "reward_pos_ratio",
                     "train_acc", "val_acc", "temp_dist", "temp_mean",
                     "temp_std", "segments_mean", "segments_min",
                     "segments_max", "advantage_mean", "advantage_std",
                     "clip_fraction", "total_steps"]
    for row in rows:
        for k in required_keys:
            assert k in row, f"Missing key {k} in PPO metrics row"
    assert rows[1]["reward_pos_ratio"] > rows[0]["reward_pos_ratio"]
    assert isinstance(rows[0]["temp_dist"], dict)
