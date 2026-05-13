"""Tests for MILModel: shapes, full components, and feature consistency."""

import torch

from mil.model import MILModel, InstanceEncoder


def test_mil_forward_shapes():
    model = MILModel(input_dim=8, hidden_dim=16, aggregator="attention")
    x = torch.randn(4, 5, 8)
    out = model(x)

    assert out["bag_repr"].shape == (4, 16)
    assert out["bag_logit"].shape == (4,)
    assert out["inst_logit"].shape == (4, 5)
    assert out["attn_w"].shape == (4, 5)


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
    assert out["inst_logit"].shape == (4, 5)
    assert out["attn_w"].shape == (4, 5)
    assert out["encoder_out"].shape == (4, 5, 16)


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
    assert out["inst_logit"].shape == (4, 5)
    assert out["encoder_out"].shape == (4, 5, 16)


def test_mil_forward_single_instance():
    """Single segment per sample (K=1) — edge case for softmax / topk."""
    model = MILModel(input_dim=8, hidden_dim=16, aggregator="attention",
                     use_position=True, use_gru=True)
    x = torch.randn(2, 1, 8)
    out = model(x)
    assert out["bag_repr"].shape == (2, 16)
    assert out["bag_logit"].shape == (2,)
    assert out["inst_logit"].shape == (2, 1)
    assert out["attn_w"].shape == (2, 1)
    # Single-instance attention weight should be 1.0
    assert torch.allclose(out["attn_w"], torch.ones(2, 1), atol=1e-6)


def test_dynamic_head_feature_consistency():
    """DynamicTempHead must receive the same features during training and eval.

    During training, inst_repr should come from out["encoder_out"], NOT from
    a bare mil.encoder(x) call.  The raw encoder features are different from
    encoder_out when position encoding and/or GRU are enabled.
    """
    model = MILModel(
        input_dim=8, hidden_dim=16,
        aggregator="attention", use_position=True, use_gru=True,
    )
    x = torch.randn(2, 5, 8)

    out = model(x)
    encoder_out = out["encoder_out"]  # encoder + pos + GRU → [B, K, hidden_dim]

    # The raw encoder output (what the BUGGY code used)
    raw_encoder = model.encoder(x)    # encoder only, no pos, no GRU

    # When position encoding + GRU are enabled, these SHOULD differ
    assert not torch.allclose(encoder_out, raw_encoder, atol=1e-4), (
        "encoder_out and raw encoder(x) should differ when pos+GRU are enabled"
    )

    # Verify they are the same ONLY when both pos and GRU are disabled
    model_plain = MILModel(
        input_dim=8, hidden_dim=16,
        aggregator="attention", use_position=False, use_gru=False,
    )
    out_plain = model_plain(x)
    raw_encoder_plain = model_plain.encoder(x)
    assert torch.allclose(out_plain["encoder_out"], raw_encoder_plain, atol=1e-4), (
        "encoder_out and raw encoder(x) should match when pos+GRU are disabled"
    )
