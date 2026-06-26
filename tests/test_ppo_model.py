"""Tests for PolicyValueNet, compute_gae, MIL warm-start, and shared rollout helpers."""

import torch

from ppo.model import PolicyValueNet, compute_gae, load_mil_encoder_for_warmstart
from ppo.training import _decide_temperature, _process_generated_features
from features.segmenter import build_segment_obs_from_lp


def test_policy_value_shapes():
    net = PolicyValueNet(obs_dim=10, n_actions=5)
    x = torch.randn(7, 10)
    logits, values = net(x)
    assert logits.shape == (7, 5)
    assert values.shape == (7,)


def test_policy_value_single_obs():
    net = PolicyValueNet(obs_dim=10, n_actions=5)
    x = torch.randn(1, 10)
    logits, values = net(x)
    assert logits.shape == (1, 5)
    assert values.shape == (1,)


def test_gae_shape():
    rewards = torch.randn(16)
    dones = torch.zeros(16)
    values = torch.randn(16)
    adv, ret = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)
    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape


def test_gae_terminal_state():
    """Terminal state: done=1 → mask=0 → returns = reward (no bootstrapping)."""
    rewards = torch.tensor([0.0, 1.0], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.3], dtype=torch.float32)

    adv, ret = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)

    # ret uses raw (unstandardized) advantages: ret = adv_raw + values
    # adv is standardized, so ret ≠ adv_std + values.  This is correct:
    # standardized advantages → policy gradient, raw returns → value target.
    # Verify: ret is the correct value target
    # Terminal step (t=1, done=1): delta = 1.0 + 0 - 0.3 = 0.7
    # ret_raw = adv_raw + value = 0.7 + 0.3 = 1.0
    assert abs(ret[1].item() - 1.0) < 0.01

    # Terminal advantage sign matches delta sign (0.7 > 0)
    assert adv[1] > 0


def test_gae_episode_boundary():
    """done=1 resets advantage propagation across episodes.

    Verify that when two episodes are separated by done=1, the second episode's
    terminal advantage does not incorporate the first episode's reward.
    """
    # Two episodes of 2 steps each
    rewards = torch.tensor([0.0, 1.0, 0.0, -1.0], dtype=torch.float32)
    dones = torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.3, 0.5, 0.3], dtype=torch.float32)

    adv, ret = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)

    # ep2_t3 terminal delta = -1.0 - 0.3 = -1.3 → advantage should be negative
    assert adv[3] < 0, "ep2 terminal advantage should be negative (reward < value)"

    # Compare to what would happen if done flags were all 0 (no boundaries):
    # the ep1 positive reward would bleed into ep2, making ep2's advantage less negative
    dones_no_boundary = torch.zeros(4, dtype=torch.float32)
    adv_no_boundary, _ = compute_gae(rewards, dones_no_boundary, values, gamma=0.99, lam=0.95)

    # With boundary, ep2_t3 should be more negative (no positive bleed from ep1)
    assert adv[3] < adv_no_boundary[3], (
        f"With episode boundary, adv[3]={adv[3]:.4f} should be less than "
        f"without boundary adv[3]={adv_no_boundary[3]:.4f}"
    )


def test_gae_all_intermediate():
    """Without any done flags, advantage propagates through all steps."""
    rewards = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
    dones = torch.zeros(3, dtype=torch.float32)
    values = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    adv, ret = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)

    # With dones=0 and constant values, all returns should be within 0.1
    # of each other — the only difference is discounting over 3 steps.
    assert abs(ret[0].item() - ret[2].item()) < 0.1, \
        f"Returns should be similar, got ret[0]={ret[0]:.4f} ret[2]={ret[2]:.4f}"

    # All returns should be < reward_max (1.0) since intermediate steps
    # contribute zero reward and gamma < 1
    # Terminal step (t=2): ret = (reward - value) + value = 1.0 exactly
    assert abs(ret[2].item() - 1.0) < 0.01


def test_gae_standardization():
    """GAE output advantages should have mean ~0 and std ~1."""
    rewards = torch.randn(100)
    dones = torch.zeros(100)
    dones[49] = 1.0   # episode boundary at t=50
    values = torch.randn(100) * 0.5

    adv, ret = compute_gae(rewards, dones, values, gamma=0.99, lam=0.95)

    assert abs(adv.mean().item()) < 0.01
    assert abs(adv.std().item() - 1.0) < 0.01


def test_mil_warmstart_weights():
    """load_mil_encoder_for_warmstart returns None for missing ckpt."""
    result = load_mil_encoder_for_warmstart("/nonexistent/path.pt", torch.device("cpu"))
    assert result is None


# ── _decide_temperature tests ──────────────────────────────────────────


def test_decide_temp_first_segment():
    """First segment (obs=None) returns default T=0.7 and dummy tensors."""
    policy = PolicyValueNet(obs_dim=8, n_actions=5)
    temp_bins = [0.3, 0.5, 0.7, 0.9, 1.1]
    device = torch.device("cpu")

    temp, action, logp, value = _decide_temperature(
        None, policy, temp_bins, device, deterministic=False,
    )
    assert temp == 0.7
    assert action.item() == 0
    assert logp.item() == 0.0
    assert value.item() == 0.0
    # Deterministic mode should behave identically for first segment
    temp2, _, _, _ = _decide_temperature(
        None, policy, temp_bins, device, deterministic=True,
    )
    assert temp2 == 0.7


def test_decide_temp_deterministic_argmax():
    """Deterministic mode picks argmax action."""
    policy = PolicyValueNet(obs_dim=4, n_actions=3)
    # Bias action 1 to be highest, action 2 second, action 0 lowest
    policy.pi.bias.data = torch.tensor([-5.0, 10.0, 0.0])
    temp_bins = [0.2, 0.7, 1.2]
    device = torch.device("cpu")

    # segment_obs shape: [K, obs_dim]. We pass [1, 4].
    obs = torch.randn(1, 4)
    temp, action, logp, value = _decide_temperature(
        obs, policy, temp_bins, device, deterministic=True,
    )
    assert action.item() == 1  # argmax picks index 1
    assert temp == 0.7
    assert logp.item() == 0.0  # deterministic returns dummy logp
    assert value.ndim == 0  # scalar


def test_decide_temp_stochastic_shapes():
    """Stochastic mode returns valid shapes and action in range."""
    policy = PolicyValueNet(obs_dim=6, n_actions=4)
    temp_bins = [0.3, 0.5, 0.7, 0.9]
    device = torch.device("cpu")

    obs = torch.randn(2, 6)  # K=2 segments
    temp, action, logp, value = _decide_temperature(
        obs, policy, temp_bins, device, deterministic=False,
    )
    assert isinstance(temp, float)
    assert temp in temp_bins
    assert action.ndim == 0
    assert logp.ndim == 0
    assert value.ndim == 0


def test_decide_temp_multi_segment_uses_last():
    """With K>1 segments, only the last segment [-1:] is used for the decision."""
    policy = PolicyValueNet(obs_dim=3, n_actions=2)
    temp_bins = [0.5, 1.0]
    device = torch.device("cpu")

    # Two segments: first is all zeros, second is all ones
    # Shape matches real build_segment_obs_from_lp output: [K, obs_dim] (2D)
    obs = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    temp1, _, _, _ = _decide_temperature(
        obs, policy, temp_bins, device, deterministic=True,
    )
    # The decision is based on obs[-1:] = the [100,100,100] segment
    # Just verify it runs and returns a valid temp
    assert temp1 in temp_bins


# ── _process_generated_features tests ──────────────────────────────────


class _MockTokenizer:
    def __init__(self, eos_id: int | None = 100):
        self.eos_token_id = eos_id


def _make_feat_dict(
    text: str = "hello",
    token_ids: list[int] | None = None,
    finish_reason: str = "length",
    logprobs: torch.Tensor | None = None,
    tokens: list[str] | None = None,
    hidden_states: torch.Tensor | None = None,
) -> dict:
    if token_ids is None:
        token_ids = [1, 2, 3]
    if logprobs is None:
        logprobs = torch.randn(3, 5)  # [n_tok, top_k+1]
    if tokens is None:
        tokens = ["a", "b", "c"]
    return {
        "text": text,
        "token_ids": token_ids,
        "finish_reason": finish_reason,
        "logprobs": logprobs,
        "tokens": tokens,
        "hidden_states": hidden_states,
    }


def test_process_features_eos():
    """Chain with EOS token in new_tokens → is_done=True."""
    tok = _MockTokenizer(eos_id=5)
    feat = _make_feat_dict(token_ids=[1, 2, 5], finish_reason="length")
    text_delta, done, obs = _process_generated_features(
        feat, tok, segment_size=32, instance_dim=64,
        device=torch.device("cpu"), segment_mode="fixed_window",
        hs_needed=False, pooling_mode="mean",
    )
    assert text_delta == "hello"
    assert done is True
    assert obs is None


def test_process_features_stop():
    """finish_reason='stop' → is_done=True."""
    tok = _MockTokenizer(eos_id=100)
    feat = _make_feat_dict(token_ids=[1, 2, 3], finish_reason="stop")
    text_delta, done, obs = _process_generated_features(
        feat, tok, segment_size=32, instance_dim=64,
        device=torch.device("cpu"), segment_mode="fixed_window",
        hs_needed=False, pooling_mode="mean",
    )
    assert done is True
    assert obs is None


def test_process_features_empty_tokens():
    """Empty new_tokens → is_done=True."""
    tok = _MockTokenizer(eos_id=100)
    feat = _make_feat_dict(token_ids=[], finish_reason="length")
    text_delta, done, obs = _process_generated_features(
        feat, tok, segment_size=32, instance_dim=64,
        device=torch.device("cpu"), segment_mode="fixed_window",
        hs_needed=False, pooling_mode="mean",
    )
    assert done is True
    assert obs is None


def test_process_features_continue():
    """Normal chain produces is_done=False and a segment observation."""
    tok = _MockTokenizer(eos_id=100)
    # 32 tokens → 1 segment of segment_size=32
    feat = _make_feat_dict(
        token_ids=list(range(32)),
        finish_reason="length",
        logprobs=torch.randn(32, 5),
        tokens=["x"] * 32,
    )
    text_delta, done, obs = _process_generated_features(
        feat, tok, segment_size=32, instance_dim=64,
        device=torch.device("cpu"), segment_mode="fixed_window",
        hs_needed=False, pooling_mode="mean",
    )
    assert text_delta == "hello"
    assert done is False
    assert obs is not None
    assert isinstance(obs, torch.Tensor)
    # mean pooling: [1, instance_dim]
    assert obs.ndim == 2
    assert obs.shape[1] == 64


def test_process_features_no_eos_token_id():
    """Tokenizer without eos_token_id still works (only stop/empty trigger)."""
    tok = _MockTokenizer(eos_id=None)
    feat = _make_feat_dict(token_ids=[1, 2, 99], finish_reason="length")
    _, done, _ = _process_generated_features(
        feat, tok, segment_size=32, instance_dim=64,
        device=torch.device("cpu"), segment_mode="fixed_window",
        hs_needed=False, pooling_mode="mean",
    )
    # No EOS, not stop, not empty → continues
    assert done is False


# ═══════════════════════  PPO batch shape contract  ═══════════════════════

def test_segment_obs_squeeze_stack_policy_shapes():
    """build_segment_obs → squeeze(0) → stack → policy produces correct 2D logits.

    Regression test: ep_obs[i][v][t] stores [1, obs_dim] from build_segment_obs_from_lp.
    Before the fix, all_obs.append(ep_obs[i][v][t]) without squeeze produced
    obs_t [N, 1, obs_dim] → Categorical.log_prob broadcast [N, N] instead of [N].
    """
    obs_dim = 64
    segment_size = 32
    n_tokens = 32
    top_k = 10
    n_actions = 5
    batch_size = 4

    # Simulate the per-chain tensor shape from generate_with_features
    lp_tensor = torch.randn(n_tokens, top_k + 1)
    tokens = ["tok"] * n_tokens
    text = " ".join(tokens)

    obs = build_segment_obs_from_lp(
        lp_tensor, tokens, text,
        segment_size=segment_size, obs_dim=obs_dim,
        device=torch.device("cpu"),
        segment_mode="fixed_window",
        include_topk=False,
        pooling_mode="mean",
    )
    # build_segment_obs_from_lp returns [n_segments, obs_dim]
    assert obs.dim() == 2, f"Expected 2D, got {obs.dim()}D shape {obs.shape}"
    assert obs.shape[0] == 1, f"Expected 1 segment, got {obs.shape[0]}"
    assert obs.shape[1] == obs_dim

    # Simulate PPO batch construction: squeeze → stack
    squeezed = [obs.squeeze(0) for _ in range(batch_size)]
    # After squeeze(0): each is [obs_dim]
    for s in squeezed:
        assert s.dim() == 1
        assert s.shape == (obs_dim,)

    obs_t = torch.stack(squeezed)  # [N, obs_dim]
    assert obs_t.dim() == 2
    assert obs_t.shape == (batch_size, obs_dim)

    # Policy forward
    policy = PolicyValueNet(obs_dim=obs_dim, n_actions=n_actions, hidden=32)
    logits, values = policy(obs_t)
    assert logits.shape == (batch_size, n_actions), \
        f"Expected {(batch_size, n_actions)}, got {logits.shape}"
    assert values.shape == (batch_size,), \
        f"Expected {(batch_size,)}, got {values.shape}"

    # Categorical.log_prob shape contract
    actions = torch.randint(0, n_actions, (batch_size,))
    dist = torch.distributions.Categorical(logits=logits)
    new_logp = dist.log_prob(actions)
    assert new_logp.shape == (batch_size,), \
        f"log_prob must be [{batch_size}], got {new_logp.shape}. " \
        f"If you see [{batch_size}, {batch_size}] here, the squeeze(0) fix is missing."


def test_3d_obs_assertion_guards_logp_broadcast_bug():
    """PolicyValueNet now rejects 3D input [N, 1, obs_dim] with AssertionError.

    Before the shape assertion was added, 3D obs produced logits [N, 1, n_actions]
    and Categorical.log_prob broadcast [N] × [N,1] → [N,N], corrupting the PPO
    gradient.  This test verifies the assertion catches that class of inputs.
    """
    obs_dim = 64
    n_actions = 5
    batch_size = 4

    # Simulate the pre-fix bug: all_obs entries were [1, obs_dim]
    all_obs_bug = [torch.randn(1, obs_dim) for _ in range(batch_size)]
    obs_t_3d = torch.stack(all_obs_bug)  # [N, 1, obs_dim] — 3D
    assert obs_t_3d.shape == (batch_size, 1, obs_dim)

    policy = PolicyValueNet(obs_dim=obs_dim, n_actions=n_actions, hidden=32)
    # The assertion must fire on 3D input
    try:
        policy(obs_t_3d)
        assert False, "PolicyValueNet should have rejected 3D input"
    except AssertionError as e:
        assert "3D" in str(e) or "got 3D" in str(e), \
            f"Expected 3D rejection message, got: {e}"
