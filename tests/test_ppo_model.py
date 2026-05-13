"""Tests for PolicyValueNet, compute_gae, and MIL warm-start mapping."""

import torch

from ppo.model import PolicyValueNet, compute_gae, load_mil_encoder_for_warmstart


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
