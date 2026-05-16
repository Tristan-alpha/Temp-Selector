"""Token-level feature vector construction utilities.

Single source of truth for all token → fixed-dim-vector conversion
functions used across MIL training, PPO training, and online evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


def token_to_vec(
    feat: Dict[str, Any],
    obs_dim: int,
    extracted: Optional[Dict[str, torch.Tensor]] = None,
) -> torch.Tensor:
    """Convert a dict of per-token features into a fixed-dim observation tensor.

    ``feat`` provides mandatory scalar fields (logprob, entropy).
    ``extracted`` provides optional per-token tensor fields (hidden,
    topk_logprobs) that are consumed inline and NOT stored in the dict.
    When absent, reads ``topk_logprobs`` / ``hidden`` from ``feat`` directly
    (backward compat for non-collate callers like PPO).

    Merge order: logprob, entropy, topk_logprobs, hidden.  Truncates or
    zero-pads to ``obs_dim``.
    """
    base = torch.tensor(
        [float(feat.get("logprob", -20.0)), float(feat.get("entropy", 0.0))],
        dtype=torch.float32,
    )
    parts = [base]

    if extracted is not None:
        for key in ("topk_logprobs", "hidden"):
            v = extracted.get(key)
            if v is not None:
                parts.append(v.float())
    else:
        for key in ("topk_logprobs", "hidden"):
            v = feat.get(key)
            if v is not None:
                parts.append(v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))

    merged = torch.cat(parts)
    if merged.shape[0] >= obs_dim:
        return merged[:obs_dim]
    return torch.cat([merged, torch.zeros(obs_dim - merged.shape[0])])


def token_to_obs(
    logprob: float,
    entropy_val: float,
    topk_logprobs: Optional[List[float]],
    obs_dim: int,
) -> torch.Tensor:
    """Convert scalar per-token features into a fixed-dim observation tensor."""
    base = torch.tensor([float(logprob), float(entropy_val)], dtype=torch.float32)
    if topk_logprobs:
        topk = torch.tensor([float(x) for x in topk_logprobs], dtype=torch.float32)
    else:
        topk = torch.zeros(0)
    merged = torch.cat([base, topk])
    if merged.shape[0] >= obs_dim:
        return merged[:obs_dim]
    return torch.cat([merged, torch.zeros(obs_dim - merged.shape[0])])


def mean_pool_obs(token_obs_list: List[torch.Tensor], obs_dim: int) -> torch.Tensor:
    """Mean-pool a list of per-token observation tensors into [obs_dim]."""
    if not token_obs_list:
        return torch.zeros(obs_dim)
    return torch.stack(token_obs_list).mean(dim=0)


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
