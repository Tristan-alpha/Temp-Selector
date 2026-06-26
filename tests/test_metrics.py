"""Tests for metric-computation functions.  All CPU-only, no models loaded."""

from __future__ import annotations

import torch

from mil.eval import (
    compute_auc,
    compute_bag_metrics,
    compute_calibration,
    compute_attention_metrics,
)
from utils.math import safe_div


# ═══════════════════════════════════════════════════════════════
# safe_div
# ═══════════════════════════════════════════════════════════════

def test_safe_div_normal():
    assert safe_div(3.0, 6.0) == 0.5


def test_safe_div_zero_denom():
    assert safe_div(5.0, 0.0) == 0.0


def test_safe_div_zero_num():
    assert safe_div(0.0, 5.0) == 0.0


# ═══════════════════════════════════════════════════════════════
# compute_auc
# ═══════════════════════════════════════════════════════════════

def test_auc_perfect():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    scores = torch.tensor([-2.0, -1.0, 1.0, 2.0])
    assert compute_auc(labels, scores) == 1.0


def test_auc_random():
    # Random scores uncorrelated with labels → AUC ≈ 0.5
    torch.manual_seed(42)
    labels = torch.cat([torch.zeros(50), torch.ones(50)])
    scores = torch.randn(100)
    auc = compute_auc(labels, scores)
    assert abs(auc - 0.5) < 0.1


def test_auc_inverted():
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.float32)
    scores = torch.tensor([2.0, 1.0, -1.0, -2.0])
    assert compute_auc(labels, scores) == 0.0


def test_auc_all_positive():
    labels = torch.tensor([1, 1, 1, 1], dtype=torch.float32)
    scores = torch.randn(4)
    assert compute_auc(labels, scores) == 0.5  # undefined, returns 0.5


def test_auc_all_negative():
    labels = torch.tensor([0, 0, 0, 0], dtype=torch.float32)
    scores = torch.randn(4)
    assert compute_auc(labels, scores) == 0.5  # undefined, returns 0.5


# ═══════════════════════════════════════════════════════════════
# compute_bag_metrics
# ═══════════════════════════════════════════════════════════════

def test_bag_metrics_perfect():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([-5.0, -5.0, 5.0, 5.0])
    m = compute_bag_metrics(labels, logits)
    assert m["bag_accuracy"] == 1.0
    assert m["bag_precision"] == 1.0
    assert m["bag_recall"] == 1.0
    assert m["bag_f1"] == 1.0
    assert m["bag_auc"] == 1.0
    assert m["bag_tp"] == 2.0
    assert m["bag_tn"] == 2.0
    assert m["bag_fp"] == 0.0
    assert m["bag_fn"] == 0.0


def test_bag_metrics_all_wrong():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
    m = compute_bag_metrics(labels, logits)
    assert m["bag_accuracy"] == 0.0
    assert m["bag_precision"] == 0.0  # tp=0 → precision=0
    assert m["bag_recall"] == 0.0     # tp=0 → recall=0


def test_bag_metrics_all_positive():
    labels = torch.tensor([1.0, 1.0, 1.0])
    logits = torch.tensor([5.0, 5.0, 5.0])
    m = compute_bag_metrics(labels, logits)
    assert m["bag_tp"] == 3.0
    assert m["bag_tn"] == 0.0
    assert m["bag_fp"] == 0.0
    assert m["bag_fn"] == 0.0


# ═══════════════════════════════════════════════════════════════
# compute_calibration
# ═══════════════════════════════════════════════════════════════

def test_calibration_perfect():
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    logits = torch.tensor([-5.0, 5.0, -5.0, 5.0])
    c = compute_calibration(labels, logits)
    assert c["ece"] < 0.1
    assert c["brier_score"] < 0.01


def test_calibration_overconfident():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([10.0, 10.0, 10.0, 10.0])  # all predicted as 1 with high confidence
    c = compute_calibration(labels, logits)
    # 2 of 4 are wrong → expected ECE ~ 0.5, Brier ~ 0.5
    assert c["ece"] > 0.3
    assert c["brier_score"] > 0.3


# ═══════════════════════════════════════════════════════════════
# compute_attention_metrics
# ═══════════════════════════════════════════════════════════════

def test_attention_uniform():
    w = torch.ones(3, 8) / 8.0  # uniform attention over 8 instances
    m = compute_attention_metrics(w)
    assert abs(m["attn_entropy"] - 2.079) < 0.01  # ln(8) ≈ 2.079
    assert m["attn_effective_n"] > 7.0


def test_attention_sparse():
    w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # all mass on first instance
    m = compute_attention_metrics(w)
    assert m["attn_entropy"] < 0.01
    assert abs(m["attn_top3_mass"] - 1.0) < 0.01
    assert abs(m["attn_effective_n"] - 1.0) < 0.01


def test_attention_single_instance():
    w = torch.tensor([[1.0]])  # only 1 instance
    m = compute_attention_metrics(w)
    # entropy of single-element distribution = 0
    assert m["attn_entropy"] < 0.01
    assert m["attn_top3_mass"] == 1.0


def test_attention_varying_lengths():
    """Varying segment counts — the real eval usage pattern (Bug 6 regression)."""
    weights = [
        torch.tensor([0.5, 0.3, 0.2]),        # 3 segments
        torch.tensor([0.7, 0.3]),              # 2 segments
        torch.tensor([0.25, 0.25, 0.25, 0.25]),  # 4 segments
    ]
    entropies, top3s, eff_ns = [], [], []
    for w in weights:
        m = compute_attention_metrics(w.unsqueeze(0))
        entropies.append(m["attn_entropy"])
        top3s.append(m["attn_top3_mass"])
        eff_ns.append(m["attn_effective_n"])
    avg_entropy = sum(entropies) / len(entropies)
    avg_top3 = sum(top3s) / len(top3s)
    avg_eff_n = sum(eff_ns) / len(eff_ns)
    # 3-segment: entropy ≈ 1.03; 2-segment: ≈ 0.61; 4-segment: ≈ 1.39 → avg ≈ 1.01
    assert 0.5 < avg_entropy < 2.0
    assert 0.0 < avg_top3 <= 1.0
    assert avg_eff_n > 1.0


# ═══════════════════════════════════════════════════════════════
# Edge case tests
# ═══════════════════════════════════════════════════════════════

def test_auc_all_positive():
    labels = torch.ones(10)
    scores = torch.randn(10)
    auc = compute_auc(labels, scores)
    assert auc == 0.5  # no negative samples → baseline


def test_auc_all_negative():
    labels = torch.zeros(10)
    scores = torch.randn(10)
    auc = compute_auc(labels, scores)
    assert auc == 0.5  # no positive samples → baseline


def test_bag_metrics_extreme_logits():
    """Very large/small logits should not produce NaN."""
    labels = torch.tensor([0.0, 1.0])
    logits = torch.tensor([-100.0, 100.0])
    m = compute_bag_metrics(labels, logits)
    assert m["bag_accuracy"] == 1.0
    assert m["bag_auc"] == 1.0


def test_calibration_perfect_separation():
    """Extreme logits → low ECE, low Brier."""
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    logits = torch.tensor([-10.0, -10.0, 10.0, 10.0])
    c = compute_calibration(labels, logits)
    assert c["brier_score"] < 0.01
    assert c["ece"] < 0.01


# ═══════════════════════════════════════════════════════════════
# AttentionAggregator direct tests
# ═══════════════════════════════════════════════════════════════

from mil.model import AttentionAggregator


def test_attention_aggregator_shapes():
    agg = AttentionAggregator(hidden_dim=16)
    h = torch.randn(3, 5, 16)
    bag, w = agg(h)
    assert bag.shape == (3, 16)
    assert w.shape == (3, 5)


def test_attention_aggregator_weights_sum_to_one():
    agg = AttentionAggregator(hidden_dim=8)
    h = torch.randn(2, 10, 8)
    _, w = agg(h)
    assert torch.allclose(w.sum(dim=-1), torch.ones(2), atol=1e-6)


def test_attention_aggregator_equal_scores():
    agg = AttentionAggregator(hidden_dim=4)
    # Zero out weights to get equal scores
    agg.attn.weight.data.zero_()
    agg.attn.bias.data.zero_()
    h = torch.randn(1, 6, 4)
    _, w = agg(h)
    # All attention scores = 0 → softmax → uniform
    assert torch.allclose(w, torch.full((1, 6), 1.0 / 6), atol=1e-6)


def test_attention_aggregator_no_nan():
    """Extreme input values should not produce NaN in weights."""
    agg = AttentionAggregator(hidden_dim=4)
    h = torch.tensor([[[1e3, -1e3, 0.0, 0.0], [-1e3, 1e3, 0.0, 0.0]]])
    _, w = agg(h)
    assert not torch.isnan(w).any()
    assert torch.allclose(w.sum(dim=-1), torch.tensor([1.0]), atol=1e-6)
