"""Tests for MILModel: shapes, full components, and attention behavior."""

import torch

from mil.model import MILModel, InstanceEncoder


def test_mil_forward_shapes():
    model = MILModel(input_dim=8, hidden_dim=16, aggregator="attention")
    x = torch.randn(4, 5, 8)
    out = model(x)

    assert out["bag_repr"].shape == (4, 16)
    assert out["bag_logit"].shape == (4,)
    assert out["attn_w"].shape == (4, 5)
    assert "inst_logit" not in out
    assert "encoder_out" not in out


def test_mil_forward_full_components():
    """MILModel with all components enabled: pos encoding + BiGRU + attention."""
    model = MILModel(
        input_dim=8, hidden_dim=16,
        aggregator="attention", use_position=True, use_gru=True,
    )
    x = torch.randn(4, 5, 8)
    out = model(x)

    assert out["bag_repr"].shape == (4, 16)
    assert out["bag_logit"].shape == (4,)
    assert out["attn_w"].shape == (4, 5)


def test_mil_forward_mean_aggregator():
    model = MILModel(input_dim=8, hidden_dim=16, aggregator="mean")
    x = torch.randn(4, 5, 8)
    out = model(x)
    assert out["bag_repr"].shape == (4, 16)
    # Mean aggregator: uniform weights
    assert torch.allclose(out["attn_w"], torch.full((4, 5), 0.2), atol=1e-6)


def test_mil_forward_max_aggregator():
    model = MILModel(input_dim=8, hidden_dim=16, aggregator="max")
    x = torch.randn(4, 5, 8)
    out = model(x)
    assert out["bag_repr"].shape == (4, 16)
    assert out["attn_w"].shape == (4, 5)


def test_mil_forward_no_position_no_gru():
    model = MILModel(
        input_dim=8, hidden_dim=16,
        aggregator="attention", use_position=False, use_gru=False,
    )
    x = torch.randn(4, 5, 8)
    out = model(x)
    assert out["bag_logit"].shape == (4,)
    assert out["attn_w"].shape == (4, 5)


def test_mil_forward_single_instance():
    """Single segment per sample (K=1) — edge case for softmax."""
    model = MILModel(input_dim=8, hidden_dim=16, aggregator="attention",
                     use_position=True, use_gru=True)
    x = torch.randn(2, 1, 8)
    out = model(x)
    assert out["bag_repr"].shape == (2, 16)
    assert out["bag_logit"].shape == (2,)
    assert out["attn_w"].shape == (2, 1)
    # Single-instance attention weight should be 1.0
    assert torch.allclose(out["attn_w"], torch.ones(2, 1), atol=1e-6)


def test_mil_forward_attention_bag_only():
    """Forward pass returns only bag_logit, bag_repr, and attn_w — no inst_head."""
    model = MILModel(input_dim=8, hidden_dim=16)
    x = torch.randn(3, 4, 8)
    out = model(x)
    assert set(out.keys()) == {"bag_repr", "bag_logit", "attn_w"}
    assert out["bag_logit"].shape == (3,)
    assert out["attn_w"].shape == (3, 4)
