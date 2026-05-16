"""Token-level feature vector construction utilities.

Single source of truth for all token → fixed-dim-vector conversion
functions used across MIL training, PPO training, and online evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List


def token_to_vec(token_feat: Dict[str, Any], obs_dim: int) -> List[float]:
    """Convert a dict of per-token features into a fixed-dim observation vector.

    Merge order: logprob, entropy, topk_logprobs, hidden.  This groups
    semantically related features — logprob and entropy capture individual
    token certainty, top-k logprobs capture distribution shape, and hidden
    states (when available) add representation-level information.

    Defaults: logprob=-20.0 (~log(2e-9), near-zero probability — maximum
    uncertainty), entropy=0.0.  Missing optional fields (topk_logprobs, hidden)
    are treated as empty lists.  If the merged vector exceeds ``obs_dim``,
    trailing features are truncated; if shorter, zeros are appended.
    """
    base = [
        float(token_feat.get("logprob", -20.0)),
        float(token_feat.get("entropy", 0.0)),
    ]
    topk = token_feat.get("topk_logprobs") or []
    hidden = token_feat.get("hidden") or []
    merged = base + [float(x) for x in topk] + [float(x) for x in hidden]
    if len(merged) >= obs_dim:
        return merged[:obs_dim]
    return merged + [0.0] * (obs_dim - len(merged))


def token_to_obs(
    logprob: float,
    entropy_val: float,
    topk_logprobs: List[float],
    obs_dim: int,
) -> List[float]:
    """Convert scalar per-token features into a fixed-dim observation vector."""
    base = [float(logprob), float(entropy_val)]
    merged = base + [float(x) for x in (topk_logprobs or [])]
    if len(merged) >= obs_dim:
        return merged[:obs_dim]
    return merged + [0.0] * (obs_dim - len(merged))


def mean_pool_obs(token_obs_list: List[List[float]], obs_dim: int) -> List[float]:
    """Mean-pool a list of per-token observation vectors into one segment vector."""
    if not token_obs_list:
        return [0.0] * obs_dim
    avg = [0.0] * obs_dim
    for row in token_obs_list:
        for i, v in enumerate(row):
            avg[i] += v
    denom = float(len(token_obs_list))
    return [v / denom for v in avg]


def compute_entropy(logprobs: List[float]) -> float:
    """Entropy of a categorical distribution given log-probabilities."""
    probs = [math.exp(min(lp, 0.0)) for lp in logprobs]
    z = sum(probs) + 1e-12
    ent = 0.0
    for p in probs:
        pn = p / z
        if pn > 1e-12:
            ent -= pn * math.log(pn)
    return ent
