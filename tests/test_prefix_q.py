"""CPU tests for Prefix-Q labels, losses, and selector helpers."""

from __future__ import annotations

import torch

from mil.prefix_data import continuation_collate, oracle_temperature_stats
from mil.prefix_value import PrefixValueModel, masked_binomial_nll
from scripts.eval_q_selector import allowed_temperature_indices, select_q_temperature


def _cache_entry(sample_id: str) -> dict:
    return {
        "sample_id": sample_id,
        "features": torch.ones(3, 6, dtype=torch.float16),
        "token_mask": torch.ones(3, 3, dtype=torch.uint8),
        "terminal_target": 1.0,
    }


def test_prefix_q_forward_shapes_and_hidden_projection():
    torch.manual_seed(1)
    model = PrefixValueModel(
        token_dim=2, segment_size=3, hidden_dim=8, max_segments=8, n_temps=5,
    )
    features = torch.randn(2, 4, 6)
    token_mask = torch.ones(2, 4, 3)
    segment_mask = torch.ones(2, 4)
    output = model(features, token_mask, segment_mask)
    assert output["q_logits"].shape == (2, 4, 5)
    assert output["terminal_q_logits"].shape == (2, 5)
    projected = model.q_from_hidden(output["terminal_hidden"])
    assert torch.allclose(projected, output["terminal_q_logits"], atol=1e-6)


def test_prompt_aware_prefix_q_forward_shapes_and_hidden_projection():
    torch.manual_seed(2)
    model = PrefixValueModel(
        token_dim=2, segment_size=3, hidden_dim=8, max_segments=8,
        n_temps=5, prompt_dim=4,
    )
    features = torch.randn(2, 4, 6)
    token_mask = torch.ones(2, 4, 3)
    segment_mask = torch.ones(2, 4)
    prompt_hidden = torch.randn(2, 4)
    output = model(features, token_mask, segment_mask, prompt_hidden=prompt_hidden)
    assert output["q_logits"].shape == (2, 4, 5)
    assert output["terminal_q_logits"].shape == (2, 5)
    projected = model.q_from_hidden(output["terminal_hidden"])
    assert torch.allclose(projected, output["terminal_q_logits"], atol=1e-6)


def test_masked_binomial_nll_ignores_masked_temperatures():
    logits = torch.logit(torch.tensor([[0.75, 0.10, 0.90]]))
    n_correct = torch.tensor([[3.0, 0.0, 0.0]])
    n_total = torch.tensor([[4.0, 0.0, 0.0]])
    mask = torch.tensor([[1.0, 0.0, 0.0]])
    good = masked_binomial_nll(logits, n_correct, n_total, mask)
    bad = masked_binomial_nll(torch.logit(torch.tensor([[0.10, 0.10, 0.90]])), n_correct, n_total, mask)
    assert good < bad


def test_continuation_collate_builds_q_labels_from_continuations():
    records = [{
        "source_sample_id": "s1",
        "prefix_segments": 2,
        "n_correct": 3,
        "n_total": 4,
        "continuations": [
            {"temperature": 0.1, "correct": True},
            {"temperature": 0.1, "correct": False},
            {"temperature": 0.3, "correct": True},
            {"temperature": 0.3, "correct": True},
        ],
    }]
    batch = continuation_collate(
        {"s1": _cache_entry("s1")}, records, [0], temp_bins=[0.1, 0.2, 0.3],
    )
    assert batch["q_n_correct"].tolist() == [[1.0, 0.0, 2.0]]
    assert batch["q_n_total"].tolist() == [[2.0, 0.0, 2.0]]
    assert batch["q_mask"].tolist() == [[1.0, 0.0, 1.0]]
    assert batch["q_target"][0, 1].item() == 0.5


def test_oracle_temperature_stats_tie_breaks_to_lower_temperature():
    stats = oracle_temperature_stats({
        "continuations": [
            {"temperature": 0.1, "correct": True},
            {"temperature": 0.3, "correct": True},
        ],
    })
    assert stats["oracle_temperature"] == 0.1
    assert stats["oracle_success_rate"] == 1.0


def test_q_selector_respects_allowed_temperatures_and_tie_margin():
    temp_bins = [0.1, 0.2, 0.3, 0.4]
    allowed = allowed_temperature_indices(temp_bins, [0.1, 0.3, 0.4])
    assert allowed == [0, 2, 3]
    selected = select_q_temperature(
        [0.80, 0.99, 0.81, 0.70],
        temp_bins,
        allowed,
        tie_margin=0.02,
    )
    assert selected["best_temperature"] == 0.3
    assert selected["temperature"] == 0.1
