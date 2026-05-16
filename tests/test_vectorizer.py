"""Tests for token_to_vec, token_to_obs, mean_pool_obs, compute_entropy.  CPU-only."""

from __future__ import annotations

import torch

from features.vectorizer import token_to_vec, token_to_obs, mean_pool_obs, compute_entropy

OBS_DIM = 64


# ═══ token_to_vec ═══

def test_token_to_vec_normal():
    feat = {"logprob": -2.0, "entropy": 1.5, "topk_logprobs": [0.1] * 16}
    v = token_to_vec(feat, OBS_DIM)
    assert v.shape == (OBS_DIM,)
    assert abs(v[0].item() + 2.0) < 1e-6
    assert abs(v[1].item() - 1.5) < 1e-6
    assert abs(v[2].item() - 0.1) < 1e-6
    assert abs(v[17].item() - 0.1) < 1e-6
    assert torch.all(v[18:] == 0.0)


def test_token_to_vec_missing_fields():
    feat: dict = {}
    v = token_to_vec(feat, OBS_DIM)
    assert abs(v[0].item() + 20.0) < 1e-6
    assert abs(v[1].item()) < 1e-6
    assert v.shape == (OBS_DIM,)


def test_token_to_vec_truncation():
    feat = {"logprob": -1.0, "entropy": 0.5, "topk_logprobs": list(range(100))}
    v = token_to_vec(feat, 10)
    assert v.shape == (10,)
    assert abs(v[0].item() + 1.0) < 1e-6
    assert abs(v[1].item() - 0.5) < 1e-6


def test_token_to_vec_with_hidden():
    feat = {"logprob": -1.0, "entropy": 0.5, "topk_logprobs": [0.1], "hidden": [0.5] * 4}
    v = token_to_vec(feat, 64)
    assert abs(v[3].item() - 0.5) < 1e-6


def test_token_to_vec_extracted_hidden():
    """extracted parameter: tensor consumed inline, not stored in dict."""
    feat = {"logprob": -1.0, "entropy": 0.5}
    extracted = {"hidden": torch.tensor([0.3, 0.3, 0.3])}
    v = token_to_vec(feat, 10, extracted=extracted)
    assert abs(v[2].item() - 0.3) < 1e-6
    assert "hidden" not in feat


def test_token_to_vec_extracted_topk():
    feat = {"logprob": -2.0, "entropy": 1.0}
    extracted = {"topk_logprobs": torch.tensor([0.1, 0.2, 0.3])}
    v = token_to_vec(feat, 10, extracted=extracted)
    assert abs(v[2].item() - 0.1) < 1e-6
    assert abs(v[4].item() - 0.3) < 1e-6


# ═══ token_to_obs ═══

def test_token_to_obs_normal():
    obs = token_to_obs(logprob=-1.5, entropy_val=1.2, topk_logprobs=[0.1] * 16, obs_dim=OBS_DIM)
    assert obs.shape == (OBS_DIM,)
    assert abs(obs[0].item() + 1.5) < 1e-6
    assert abs(obs[1].item() - 1.2) < 1e-6


def test_token_to_obs_truncation():
    obs = token_to_obs(logprob=-1.0, entropy_val=0.5, topk_logprobs=list(range(100)), obs_dim=10)
    assert obs.shape == (10,)


# ═══ mean_pool_obs ═══

def test_mean_pool_obs():
    obs1 = torch.arange(OBS_DIM, dtype=torch.float32)
    obs2 = torch.arange(OBS_DIM, dtype=torch.float32) + 10.0
    pooled = mean_pool_obs([obs1, obs2], OBS_DIM)
    assert pooled.shape == (OBS_DIM,)
    assert abs(pooled[0].item() - 5.0) < 1e-6
    assert abs(pooled[10].item() - 15.0) < 1e-6


def test_mean_pool_obs_empty():
    pooled = mean_pool_obs([], OBS_DIM)
    assert pooled.shape == (OBS_DIM,)
    assert torch.all(pooled == 0.0)


# ═══ compute_entropy ═══

def test_entropy_uniform():
    ent = compute_entropy([-1.0986, -1.0986, -1.0986])
    assert abs(ent - 1.0986) < 0.01


def test_entropy_deterministic():
    ent = compute_entropy([0.0, -100.0, -100.0])
    assert ent < 0.01
