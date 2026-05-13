"""Tests for token_to_vec, token_to_obs, mean_pool_obs, compute_entropy.  CPU-only."""

from __future__ import annotations

from features.vectorizer import token_to_vec, token_to_obs, mean_pool_obs, compute_entropy

OBS_DIM = 64


# ═══ token_to_vec ═══

def test_token_to_vec_normal():
    feat = {"logprob": -2.0, "entropy": 1.5, "topk_logits": [0.1] * 16}
    v = token_to_vec(feat, OBS_DIM)
    assert len(v) == OBS_DIM
    assert v[0] == -2.0
    assert v[1] == 1.5
    assert v[2] == 0.1
    assert v[17] == 0.1
    assert all(x == 0.0 for x in v[18:])


def test_token_to_vec_missing_fields():
    feat: dict = {}
    v = token_to_vec(feat, OBS_DIM)
    assert v[0] == -20.0
    assert v[1] == 0.0
    assert len(v) == OBS_DIM


def test_token_to_vec_truncation():
    feat = {"logprob": -1.0, "entropy": 0.5, "topk_logits": list(range(100))}
    v = token_to_vec(feat, 10)
    assert len(v) == 10
    assert v[0] == -1.0
    assert v[1] == 0.5


def test_token_to_vec_with_hidden():
    feat = {"logprob": -1.0, "entropy": 0.5, "topk_logits": [0.1], "hidden": [0.5] * 4}
    v = token_to_vec(feat, 64)
    assert v[3] == 0.5


# ═══ token_to_obs ═══

def test_token_to_obs_normal():
    obs = token_to_obs(logprob=-1.5, entropy_val=1.2, topk_logits=[0.1] * 16, obs_dim=OBS_DIM)
    assert len(obs) == OBS_DIM
    assert obs[0] == -1.5
    assert obs[1] == 1.2


def test_token_to_obs_truncation():
    obs = token_to_obs(logprob=-1.0, entropy_val=0.5, topk_logits=list(range(100)), obs_dim=10)
    assert len(obs) == 10


# ═══ mean_pool_obs ═══

def test_mean_pool_obs():
    obs1 = [float(i) for i in range(OBS_DIM)]
    obs2 = [float(i + 10) for i in range(OBS_DIM)]
    pooled = mean_pool_obs([obs1, obs2], OBS_DIM)
    assert len(pooled) == OBS_DIM
    assert abs(pooled[0] - 5.0) < 1e-6
    assert abs(pooled[10] - 15.0) < 1e-6


def test_mean_pool_obs_empty():
    pooled = mean_pool_obs([], OBS_DIM)
    assert len(pooled) == OBS_DIM
    assert all(x == 0.0 for x in pooled)


# ═══ compute_entropy ═══

def test_entropy_uniform():
    ent = compute_entropy([-1.0986, -1.0986, -1.0986])
    assert abs(ent - 1.0986) < 0.01


def test_entropy_deterministic():
    ent = compute_entropy([0.0, -100.0, -100.0])
    assert ent < 0.01
