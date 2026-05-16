"""Tests for collate_rows, smoothness_loss, and top-k MIL instance loss.  CPU-only."""

from __future__ import annotations

import torch
import torch.nn as nn

from mil.training import collate_rows, RowTensor
from mil.model import smoothness_loss


# ═══ collate_rows ═══

def test_collate_uniform():
    r1 = RowTensor(instances=torch.randn(5, 8), label=torch.tensor(0.0), temp_idx=torch.tensor(3))
    r2 = RowTensor(instances=torch.randn(5, 8), label=torch.tensor(1.0), temp_idx=torch.tensor(7))
    batch = collate_rows([r1, r2])
    assert batch["instances"].shape == (2, 5, 8)
    assert batch["mask"].shape == (2, 5)
    assert torch.all(batch["mask"] == 1.0)
    assert batch["label"].shape == (2,)
    assert batch["temp_idx"].shape == (2,)


def test_collate_variable_lengths():
    r1 = RowTensor(instances=torch.randn(3, 8), label=torch.tensor(0.0), temp_idx=torch.tensor(0))
    r2 = RowTensor(instances=torch.randn(7, 8), label=torch.tensor(1.0), temp_idx=torch.tensor(4))
    r3 = RowTensor(instances=torch.randn(5, 8), label=torch.tensor(0.0), temp_idx=torch.tensor(2))
    batch = collate_rows([r1, r2, r3])
    assert batch["instances"].shape == (3, 7, 8)
    assert torch.all(batch["mask"][0, :3] == 1.0)
    assert torch.all(batch["mask"][0, 3:] == 0.0)
    assert torch.all(batch["mask"][1, :7] == 1.0)
    assert torch.all(batch["mask"][2, :5] == 1.0)
    assert torch.all(batch["mask"][2, 5:] == 0.0)
    assert torch.all(batch["instances"][0, 3:, :] == 0.0)
    assert torch.all(batch["instances"][2, 5:, :] == 0.0)


def test_collate_single_segment():
    r = RowTensor(instances=torch.randn(1, 8), label=torch.tensor(0.0), temp_idx=torch.tensor(2))
    batch = collate_rows([r])
    assert batch["instances"].shape == (1, 1, 8)
    assert torch.all(batch["mask"] == 1.0)


def test_collate_empty_batch():
    import pytest
    with pytest.raises(ValueError):
        collate_rows([])


# ═══ smoothness_loss ═══

def test_smoothness_constant():
    loss = smoothness_loss(torch.ones(2, 5, 3))
    assert loss.item() == 0.0


def test_smoothness_single_segment():
    loss = smoothness_loss(torch.randn(3, 1, 5))
    assert loss.item() == 0.0


def test_smoothness_increasing():
    logits = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]])
    loss = smoothness_loss(logits)
    assert abs(loss.item() - 1.0) < 1e-6


# ═══ top-k MIL instance loss ═══

def _compute_mil_instance_loss(inst_logit, mask, y):
    device = inst_logit.device
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    for i in range(y.size(0)):
        n_valid = int(mask[i].sum().item())
        if n_valid == 0:
            continue
        if y[i].item() > 0.5:
            k = max(1, n_valid // 3)
            topk_logprobs, topk_idx = torch.topk(inst_logit[i, :n_valid], k)
            loss_pos = bce(topk_logprobs, torch.ones(k, device=device))
            all_idx = set(range(n_valid))
            rest_idx = torch.tensor(sorted(all_idx - set(topk_idx.tolist())), device=device)
            if len(rest_idx) > 0:
                rest_logits = inst_logit[i, rest_idx]
                loss_rest = bce(rest_logits, torch.zeros(len(rest_idx), device=device))
                total_loss += loss_pos + loss_rest
                total_count += k + len(rest_idx)
            else:
                total_loss += loss_pos
                total_count += k
        else:
            logits_i = inst_logit[i, :n_valid]
            loss_neg = bce(logits_i, torch.zeros(n_valid, device=device))
            total_loss += loss_neg
            total_count += n_valid
    return float(total_loss / max(1, total_count))


def test_mil_instance_loss_negative_bag():
    loss = _compute_mil_instance_loss(torch.tensor([[-5.0, -5.0, -5.0]]), torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([0.0]))
    assert loss < 0.1


def test_mil_instance_loss_negative_bag_high_logits():
    loss = _compute_mil_instance_loss(torch.tensor([[5.0, 5.0, 5.0]]), torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([0.0]))
    assert loss > 1.0


def test_mil_instance_loss_positive_bag():
    loss = _compute_mil_instance_loss(torch.tensor([[5.0, -5.0, -5.0]]), torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([1.0]))
    assert loss < 2.0


def test_mil_instance_loss_positive_bag_all_low():
    loss = _compute_mil_instance_loss(torch.tensor([[-5.0, -5.0, -5.0]]), torch.tensor([[1.0, 1.0, 1.0]]), torch.tensor([1.0]))
    assert loss > 1.0


def test_mil_instance_loss_positive_bag_single_instance():
    loss = _compute_mil_instance_loss(torch.tensor([[3.0]]), torch.tensor([[1.0]]), torch.tensor([1.0]))
    assert loss < 0.5


def test_mil_instance_loss_masked_instances():
    loss = _compute_mil_instance_loss(torch.tensor([[5.0, 3.0, -1.0]]), torch.tensor([[1.0, 1.0, 0.0]]), torch.tensor([0.0]))
    assert loss > 1.0


# ═══ pure (k=1) method ═══

def _compute_pure_instance_loss(inst_logit, mask, y):
    """Replicate the pure (k=1) MIL instance loss logic."""
    device = inst_logit.device
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    for i in range(y.size(0)):
        n_valid = int(mask[i].sum().item())
        if n_valid == 0:
            continue
        scores = inst_logit[i, :n_valid]
        if y[i].item() > 0.5:
            k = 1
            topk_logprobs, topk_idx = torch.topk(scores, k)
            loss_pos = bce(topk_logprobs, torch.ones(k, device=device))
            all_idx = set(range(n_valid))
            rest_idx = torch.tensor(sorted(all_idx - set(topk_idx.tolist())), device=device)
            if len(rest_idx) > 0:
                rest_logits = scores[rest_idx]
                loss_rest = bce(rest_logits, torch.zeros(len(rest_idx), device=device))
                total_loss += loss_pos.sum() + loss_rest.sum()
                total_count += k + len(rest_idx)
            else:
                total_loss += loss_pos.sum()
                total_count += k
        else:
            loss_neg = bce(scores, torch.zeros(n_valid, device=device))
            total_loss += loss_neg.sum()
            total_count += n_valid
    return float(total_loss / max(1, total_count))


def test_pure_only_top1_penalized():
    """Pure method: only the single highest score gets target=1."""
    inst_logit = torch.tensor([[8.0, 5.0, 3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    loss = _compute_pure_instance_loss(inst_logit, mask, y)
    # BCE(8,1)≈0.0003 + BCE(5,0)≈5.0 + BCE(3,0)≈3.1 → avg≈2.69
    assert 2.0 < loss < 3.5


def test_pure_single_instance():
    """Pure method with only 1 instance: top-1 is everything, no rest."""
    loss = _compute_pure_instance_loss(torch.tensor([[3.0]]), torch.tensor([[1.0]]), torch.tensor([1.0]))
    assert loss < 0.5


# ═══ soft pseudo-label method ═══

def _compute_spl_instance_loss(inst_logit, mask, y):
    """Replicate the soft pseudo-label instance loss logic."""
    device = inst_logit.device
    bce = nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    total_count = 0
    for i in range(y.size(0)):
        n_valid = int(mask[i].sum().item())
        if n_valid == 0:
            continue
        scores = inst_logit[i, :n_valid]
        if y[i].item() > 0.5:
            probs = torch.sigmoid(scores).detach()
            if probs.max() < 0.5:
                probs[probs.argmax()] = 0.5
            loss_val = bce(scores, probs)
            total_loss += loss_val.sum()
            total_count += n_valid
        else:
            loss_neg = bce(scores, torch.zeros(n_valid, device=device))
            total_loss += loss_neg.sum()
            total_count += n_valid
    return float(total_loss / max(1, total_count))


def test_spl_soft_targets():
    """High scores → targets near 1; low scores → targets near 0."""
    inst_logit = torch.tensor([[5.0, -2.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    loss = _compute_spl_instance_loss(inst_logit, mask, y)
    assert loss < 3.0  # Should be moderate — model is confident


def test_spl_anti_degeneration():
    """When all scores < 0 (all sigmoids < 0.5), max is clamped to 0.5."""
    inst_logit = torch.tensor([[-2.0, -3.0, -4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    loss = _compute_spl_instance_loss(inst_logit, mask, y)
    # Without clamp, all targets < 0.5 → very weak signal.  With clamp,
    # max target = 0.5 → model receives a minimum push.  Loss should exist.
    assert loss > 0.0


# ═══ contrastive method ═══

def _compute_ctr_instance_loss(inst_logit, mask, y):
    """Replicate the contrastive instance loss logic."""
    total_loss = 0.0
    total_count = 0
    for i in range(y.size(0)):
        n_valid = int(mask[i].sum().item())
        if n_valid == 0:
            continue
        scores = inst_logit[i, :n_valid]
        if y[i].item() > 0.5:
            loss_val = (torch.logsumexp(scores, dim=0) - scores.max()
                        + torch.nn.functional.softplus(-scores.max()))
            total_loss += loss_val
            total_count += 1
        else:
            loss_neg = scores.pow(2).mean()
            total_loss += loss_neg * n_valid
            total_count += n_valid
    return float(total_loss / max(1, total_count))


def test_ctr_positive_bag_low_loss_when_one_high():
    """One high score, others low → low contrastive loss."""
    scores = torch.tensor([[8.0, -2.0, -3.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    loss = _compute_ctr_instance_loss(scores, mask, y)
    # logsumexp≈8.0003, max=8, softplus(-8)≈0 → loss≈0.0006
    assert loss < 0.1


def test_ctr_positive_bag_penalized_when_max_negative():
    """When max score is negative, softplus term pushes it toward positive."""
    scores_neg = torch.tensor([[-3.0, -5.0, -7.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    loss_neg = _compute_ctr_instance_loss(scores_neg, mask, y)
    # softplus(3)≈3.05 → significant penalty for negative max
    # Same scores but max positive:
    scores_pos = torch.tensor([[3.0, 1.0, -1.0]])
    loss_pos = _compute_ctr_instance_loss(scores_pos, mask, y)
    # softplus(-3)≈0.05 → small penalty
    assert loss_neg > loss_pos, f"Negative max should be penalized: {loss_neg:.4f} > {loss_pos:.4f}"


def test_ctr_positive_bag_high_loss_when_all_similar():
    """Similar scores → high contrastive loss (no clear winner)."""
    scores = torch.tensor([[2.0, 1.5, 2.5]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([1.0])
    loss = _compute_ctr_instance_loss(scores, mask, y)
    # logsumexp([2, 1.5, 2.5]) ≈ 3.2, max = 2.5 → loss ≈ 0.7
    assert loss > 0.5


def test_ctr_negative_bag():
    """Negative bag: MSE pushes all scores toward 0."""
    scores = torch.tensor([[5.0, 3.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0]])
    y = torch.tensor([0.0])
    loss = _compute_ctr_instance_loss(scores, mask, y)
    # MSE: (25+9+1)/3 ≈ 11.7 → high loss for high scores
    assert loss > 5.0
