"""CPU tests for the complete prefix-value proposal."""

import pytest
import torch

from features.segmenter import build_masked_concat_segment_obs_from_lp
from mil.prefix_data import (
    build_ranking_pairs,
    continuation_collate,
    prefix_segment_specs,
    prefix_segment_counts,
    select_continuation_prefixes,
    terminal_collate,
)
from mil.prefix_value import (
    PrefixRecurrentState,
    PrefixValueModel,
    binomial_nll,
    paired_ranking_loss,
    potential_reward,
)
from ppo.model import PrefixPolicyValueNet
from ppo.prefix_rollout import PrefixRolloutEngine
from scripts.build_prefix_continuations import continuation_request_plan


def _lp(n_tokens: int, top_k: int = 8) -> torch.Tensor:
    tensor = torch.full((n_tokens, top_k + 1), -3.0)
    tensor[:, 0] = -0.5
    return tensor


def test_masked_concat_keeps_and_pads_tail():
    result = build_masked_concat_segment_obs_from_lp(
        _lp(5), ["x"] * 5, "xxxxx", segment_size=4, token_dim=6,
    )
    assert result.features.shape == (2, 24)
    assert result.token_mask.shape == (2, 4)
    assert torch.equal(result.token_mask[0], torch.ones(4))
    assert torch.equal(result.token_mask[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.all(result.features[1, 6:] == 0)


def test_prefix_model_padding_does_not_change_valid_outputs():
    torch.manual_seed(1)
    model = PrefixValueModel(token_dim=2, segment_size=3, hidden_dim=8, max_segments=8)
    model.eval()
    features = torch.randn(1, 2, 6)
    token_mask = torch.ones(1, 2, 3)
    single = model(features, token_mask, torch.ones(1, 2))

    padded_features = torch.cat([features, torch.randn(1, 2, 6)], dim=1)
    padded_token_mask = torch.cat([token_mask, torch.zeros(1, 2, 3)], dim=1)
    padded_segment_mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    padded = model(padded_features, padded_token_mask, padded_segment_mask)
    assert torch.allclose(
        single["value_logits"], padded["value_logits"][:, :2], atol=1e-6,
    )
    assert torch.allclose(single["terminal_logits"], padded["terminal_logits"], atol=1e-6)


def test_prefix_model_batch_matches_streaming_steps():
    torch.manual_seed(2)
    model = PrefixValueModel(token_dim=2, segment_size=3, hidden_dim=8, max_segments=8)
    model.eval()
    features = torch.randn(1, 4, 6)
    token_mask = torch.tensor([[[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]]], dtype=torch.float32)
    batch = model(features, token_mask, torch.ones(1, 4))["value_logits"][0]
    state = None
    streaming = []
    for index in range(4):
        logit, _, state = model.step(features[0, index], token_mask[0, index], state)
        streaming.append(logit.squeeze(0))
    assert torch.allclose(batch, torch.stack(streaming), atol=1e-5)


def test_prompt_aware_prefix_model_requires_prompt_hidden():
    model = PrefixValueModel(
        token_dim=2, segment_size=3, hidden_dim=8, max_segments=8, prompt_dim=5,
    )
    features = torch.randn(1, 2, 6)
    token_mask = torch.ones(1, 2, 3)
    with pytest.raises(ValueError, match="prompt_hidden"):
        model(features, token_mask, torch.ones(1, 2))


def test_prompt_hidden_changes_prefix_value_output():
    torch.manual_seed(5)
    model = PrefixValueModel(
        token_dim=2, segment_size=3, hidden_dim=8, max_segments=8, prompt_dim=5,
    )
    model.eval()
    features = torch.randn(1, 2, 6)
    token_mask = torch.ones(1, 2, 3)
    segment_mask = torch.ones(1, 2)
    prompt_a = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]])
    prompt_b = torch.tensor([[4.0, 1.0, 3.0, 0.0, 2.0]])
    logits_a = model(features, token_mask, segment_mask, prompt_hidden=prompt_a)["terminal_logits"]
    logits_b = model(features, token_mask, segment_mask, prompt_hidden=prompt_b)["terminal_logits"]
    assert not torch.allclose(logits_a, logits_b)


def test_prompt_aware_batch_matches_streaming_steps():
    torch.manual_seed(6)
    model = PrefixValueModel(
        token_dim=2, segment_size=3, hidden_dim=8, max_segments=8, prompt_dim=5,
    )
    model.eval()
    features = torch.randn(1, 4, 6)
    token_mask = torch.tensor([[[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]]], dtype=torch.float32)
    prompt_hidden = torch.randn(1, 5)
    batch = model(
        features, token_mask, torch.ones(1, 4), prompt_hidden=prompt_hidden,
    )["value_logits"][0]
    state = PrefixRecurrentState(model.initial_hidden(prompt_hidden), 0)
    streaming = []
    for index in range(4):
        logit, _, state = model.step(features[0, index], token_mask[0, index], state)
        streaming.append(logit.squeeze(0))
    assert torch.allclose(batch, torch.stack(streaming), atol=1e-5)


def test_prefix_collates_preserve_prompt_hidden():
    entry = {
        "sample_id": "s1",
        "features": torch.ones(3, 6, dtype=torch.float16),
        "token_mask": torch.ones(3, 3, dtype=torch.uint8),
        "terminal_target": 1.0,
        "prompt_hidden": torch.arange(5, dtype=torch.float16),
    }
    terminal = terminal_collate([entry], [0])
    assert terminal["prompt_hidden"].shape == (1, 5)
    assert terminal["prompt_hidden"].dtype == torch.float32

    continuation = continuation_collate(
        {"s1": entry},
        [{"source_sample_id": "s1", "prefix_segments": 2, "n_correct": 1, "n_total": 2}],
        [0],
    )
    assert torch.equal(continuation["prompt_hidden"], terminal["prompt_hidden"])


def test_binomial_nll_prefers_matching_probability():
    correct = torch.tensor([6.0])
    total = torch.tensor([8.0])
    matching = torch.logit(torch.tensor([0.75]))
    wrong = torch.logit(torch.tensor([0.10]))
    assert binomial_nll(matching, correct, total) < binomial_nll(wrong, correct, total)


def test_ranking_loss_prefers_correct_order():
    target_a = torch.tensor([0.8])
    target_b = torch.tensor([0.2])
    good = paired_ranking_loss(torch.tensor([2.0]), torch.tensor([-2.0]), target_a, target_b)
    bad = paired_ranking_loss(torch.tensor([-2.0]), torch.tensor([2.0]), target_a, target_b)
    assert good < bad


def test_potential_reward_terminal_and_nonterminal():
    nonterminal = potential_reward(0.4, 0.7, gamma=0.99, shaping_coef=0.15)
    terminal = potential_reward(0.7, 0.0, gamma=0.99, shaping_coef=0.15, terminal_reward=1.0)
    assert torch.allclose(nonterminal, torch.tensor(0.15 * (0.99 * 0.7 - 0.4)))
    assert torch.allclose(terminal, torch.tensor(1.0 - 0.15 * 0.7))


def test_prefix_selection_is_problem_grouped_and_excludes_full():
    rows = []
    for label, suffix in [(0, 0), (1, 1)]:
        rows.append({
            "sample_id": f"q1_t0.5_v{suffix}",
            "individual_label": label,
            "token_ids": list(range(20)),
        })
    selected = select_continuation_prefixes(rows, segment_size=4, sampling_seed=123)
    assert selected
    assert {row["problem_id"] for row in selected} == {"q1"}
    assert all(row["prefix_token_end"] < 20 for row in selected)
    counts = prefix_segment_counts(
        20, 4, sampling_seed=123, source_sample_id="q1_t0.5_v0",
    )
    assert counts == sorted(set(counts))
    assert all(1 <= count <= 4 for count in counts)
    assert {2, 3, 4}.issubset(set(counts))
    assert all("prefix_sampling_seed" in row for row in selected)


def test_prefix_sampling_is_seeded_and_records_strata():
    first = prefix_segment_specs(
        100, 4, sampling_seed=42, source_sample_id="trajectory_a",
    )
    second = prefix_segment_specs(
        100, 4, sampling_seed=42, source_sample_id="trajectory_a",
    )
    different = prefix_segment_specs(
        100, 4, sampling_seed=43, source_sample_id="trajectory_a",
    )
    assert first == second
    assert first != different
    assert [item["prefix_segments"] for item in first] == sorted({
        item["prefix_segments"] for item in first
    })
    random_quantiles = {
        source: q
        for item in first
        for source, q in zip(item["prefix_sources"], item["prefix_quantiles"])
        if source.startswith("random_")
    }
    assert 0.05 <= random_quantiles["random_early"] <= 0.30
    assert 0.30 <= random_quantiles["random_middle"] <= 0.65
    assert 0.65 <= random_quantiles["random_late"] <= 0.95
    sources = {
        source
        for item in first
        for source in item["prefix_sources"]
    }
    assert {"anchor_25", "anchor_50", "anchor_75", "anchor_penultimate"} <= sources


def test_continuation_request_plan_has_four_seeds_per_temperature():
    temperatures = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
    plan = continuation_request_plan(
        n_records=1,
        temperatures=temperatures,
        seeds_per_temperature=4,
        base_seed=42,
    )
    assert len(plan) == 32
    assert len({item["generation_seed"] for item in plan}) == 32
    for temp in temperatures:
        per_temp = [item for item in plan if item["temperature"] == temp]
        assert len(per_temp) == 4
        assert {item["seed_index"] for item in per_temp} == {0, 1, 2, 3}


def test_ranking_pairs_require_nonoverlapping_posteriors():
    records = [
        {"problem_id": "q", "source_sample_id": "a", "n_correct": 8, "n_total": 8},
        {"problem_id": "q", "source_sample_id": "b", "n_correct": 0, "n_total": 8},
        {"problem_id": "q", "source_sample_id": "c", "n_correct": 4, "n_total": 8},
    ]
    pairs = build_ranking_pairs(records, max_pairs_per_problem=64)
    assert (0, 1) in pairs


def test_synthetic_value_to_policy_smoke():
    torch.manual_seed(3)
    value_model = PrefixValueModel(token_dim=2, segment_size=3, hidden_dim=8, max_segments=8)
    policy = PrefixPolicyValueNet(hidden_dim=8, n_actions=4)
    features = torch.randn(2, 3, 6)
    token_mask = torch.ones(2, 3, 3)
    segment_mask = torch.ones(2, 3)
    output = value_model(features, token_mask, segment_mask)
    logits, critic = policy(output["terminal_hidden"].detach())
    assert logits.shape == (2, 4)
    assert critic.shape == (2,)


class _FakeTokenizer:
    eos_token_id = 999


class _FakeRunner:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.round = 0

    def build_math_messages(self, question, system_prompt=None):
        return [{"role": "user", "content": question}]

    def render_messages(self, messages):
        return "prompt:"

    def generate_with_features(self, prompts, temperatures, segment_size, **kwargs):
        terminal = self.round == 1
        self.round += 1
        outputs = []
        for _ in prompts:
            token_ids = [5, 999] if terminal else [1, 2, 3, 4]
            outputs.append({
                "token_ids": token_ids,
                "tokens": ["x"] * len(token_ids),
                "text": "\\boxed{1}" if terminal else "reason ",
                "finish_reason": "stop" if terminal else "length",
                "logprobs": _lp(len(token_ids)),
                "hidden_states": None,
                "prompt_hidden": torch.ones(5) if kwargs.get("return_prompt_hidden") else None,
            })
        return outputs


def test_continuation_value_policy_rollout_smoke():
    torch.manual_seed(4)
    value_model = PrefixValueModel(token_dim=6, segment_size=4, hidden_dim=8, max_segments=4)
    policy = PrefixPolicyValueNet(hidden_dim=8, n_actions=2)
    engine = PrefixRolloutEngine(
        runner=_FakeRunner(), value_model=value_model, calibration_temperature=1.0,
        device=torch.device("cpu"), temp_bins=[0.5, 1.0], segment_size=4,
        token_dim=6, top_k_logprobs=8, num_votes=2, max_new_tokens=8,
        gamma=0.99, shaping_coef=0.15, system_prompt="", use_math_chat=True,
    )
    result = engine.rollout(
        [{"question": "1?", "answer": "1"}], policy,
        stochastic=False, rng=__import__("random").Random(42), generation_seed=42,
    )
    assert result.majority_correct == [1]
    assert result.individual_correct == [[1, 1]]
    assert all(len(chain) == 1 for chain in result.transitions[0])
    assert all(chain[0].done for chain in result.transitions[0])


def test_prompt_aware_rollout_smoke():
    torch.manual_seed(7)
    value_model = PrefixValueModel(
        token_dim=6, segment_size=4, hidden_dim=8, max_segments=4, prompt_dim=5,
    )
    prompt_state = value_model.initial_hidden(torch.ones(1, 5))
    assert prompt_state is not None
    assert not torch.allclose(prompt_state, torch.zeros_like(prompt_state))
    policy = PrefixPolicyValueNet(hidden_dim=8, n_actions=2)
    engine = PrefixRolloutEngine(
        runner=_FakeRunner(), value_model=value_model, calibration_temperature=1.0,
        device=torch.device("cpu"), temp_bins=[0.5, 1.0], segment_size=4,
        token_dim=6, top_k_logprobs=8, num_votes=2, max_new_tokens=8,
        gamma=0.99, shaping_coef=0.15, system_prompt="", use_math_chat=True,
    )
    result = engine.rollout(
        [{"question": "1?", "answer": "1"}], policy,
        stochastic=False, rng=__import__("random").Random(42), generation_seed=42,
    )
    assert result.majority_correct == [1]
    assert all(len(chain) == 1 for chain in result.transitions[0])
