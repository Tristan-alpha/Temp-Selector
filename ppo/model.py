"""PPO policy/value network, GAE computation, and MIL warm-start utilities."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.pi(h), self.v(h).squeeze(-1)


def sample_action(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dist = torch.distributions.Categorical(logits=logits)
    a = dist.sample()
    lp = dist.log_prob(a)
    return a, lp


def compute_gae(rewards: torch.Tensor, dones: torch.Tensor, values: torch.Tensor,
                gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation (GAE-Lambda).

    Computes advantage estimates and returns for a flat sequence of
    transitions that may span multiple episodes.  done[t]=1 masks out
    future value bootstrapping, resetting advantage propagation at
    episode boundaries.  Output advantages are standardized to zero
    mean and unit variance for stable policy gradient updates.
    Returns are NOT standardized (they are value-function targets).
    """
    adv = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
        next_value = values[t]
    ret = adv + values
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv, ret


def load_mil_encoder_for_warmstart(mil_ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor] | None:
    """Load MIL checkpoint and extract InstanceEncoder weights for PPO backbone warm-start.

    MIL encoder is 2 layers (net.0, net.2).  PPO backbone may have 2 or 3
    layers (0, 2, [4]).  We map the first two and leave any extra layers
    randomly initialised.
    """
    try:
        ckpt = torch.load(mil_ckpt_path, map_location=device, weights_only=False)
    except FileNotFoundError:
        return None
    mil_state = ckpt.get("mil", {})
    encoder_mapping = {
        "encoder.net.0.weight": "backbone.0.weight",
        "encoder.net.0.bias": "backbone.0.bias",
        "encoder.net.2.weight": "backbone.2.weight",
        "encoder.net.2.bias": "backbone.2.bias",
    }
    warm_weights: Dict[str, torch.Tensor] = {}
    for src, dst in encoder_mapping.items():
        if src in mil_state:
            warm_weights[dst] = mil_state[src]
    return warm_weights if len(warm_weights) == 4 else None
