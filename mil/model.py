from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Add sinusoidal position information to instance representations."""

    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class InstanceEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionAggregator(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.attn(h).squeeze(-1)
        w = torch.softmax(score, dim=-1)
        bag = torch.sum(h * w.unsqueeze(-1), dim=1)
        return bag, w


class MILModel(nn.Module):
    """Multiple Instance Learning model for error localization in reasoning chains."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        aggregator: str = "attention",
        use_position: bool = True,
        use_gru: bool = True,
    ):
        super().__init__()
        self.encoder = InstanceEncoder(input_dim, hidden_dim)
        self.aggregator = aggregator
        self.use_gru = use_gru

        self.pos_encoder: Optional[SinusoidalPositionalEncoding] = None
        if use_position:
            self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim)

        self.gru: Optional[nn.GRU] = None
        self.gru_proj: Optional[nn.Linear] = None
        if use_gru:
            self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.attn_agg = AttentionAggregator(hidden_dim)
        self.bag_head = nn.Linear(hidden_dim, 1)
        self.inst_head = nn.Linear(hidden_dim, 1)

    def _aggregate(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.aggregator == "mean":
            bag = h.mean(dim=1)
            w = torch.full((h.size(0), h.size(1)), 1.0 / h.size(1), device=h.device)
            return bag, w
        if self.aggregator == "max":
            bag, _ = h.max(dim=1)
            score = torch.norm(h, dim=-1)
            w = torch.softmax(score, dim=-1)
            return bag, w
        return self.attn_agg(h)

    def forward(self, instances: torch.Tensor) -> dict:
        """Forward pass through the full MIL pipeline.

        Feature flow:
        1. InstanceEncoder: [B, K, input_dim] → [B, K, hidden_dim]
        2. PositionEncoding (optional): adds sinusoidal position information
        3. BiGRU (optional): bidirectional GRU captures error propagation
           across segments, then projected back to hidden_dim
        4. Aggregator: pool K segment features → single bag representation
           (attention: learned weights; mean: uniform; max: L2-norm softmax)
        5. bag_head: bag_repr → bag_logit (scalar: prob of wrong answer)
        6. inst_head: per-segment logit (scalar per segment: error score)

        Returns dict with:
          bag_logit  [B]:     whole-answer error logit (→ sigmoid → prob)
          inst_logit [B, K]:  per-segment error score (→ PPO shaping reward)
          attn_w     [B, K]:  segment attention weights (interpretable)
          bag_repr   [B, D]:  aggregated representation
          encoder_out [B, K, D]: features after encoder+pos+GRU (for temp heads)
        """
        h = self.encoder(instances)                           # [B, K, hidden_dim]

        if self.pos_encoder is not None:
            h = self.pos_encoder(h)

        if self.gru is not None:
            h, _ = self.gru(h)                                # [B, N, hidden_dim*2]
            h = self.gru_proj(h)                              # [B, N, hidden_dim]

        encoder_out = h
        bag_repr, attn_w = self._aggregate(h)
        bag_logit = self.bag_head(bag_repr).squeeze(-1)       # [B]
        inst_logit = self.inst_head(h).squeeze(-1)            # [B, N]
        return {
            "bag_repr": bag_repr,
            "bag_logit": bag_logit,
            "inst_logit": inst_logit,
            "attn_w": attn_w,
            "encoder_out": encoder_out,
        }


# ═══════════════════════════  auxiliary temperature heads  ═══════════════════

class GlobalTempHead(nn.Module):
    def __init__(self, hidden_dim: int, n_bins: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, n_bins)

    def forward(self, bag_repr: torch.Tensor) -> torch.Tensor:
        return self.classifier(bag_repr)


class DynamicTempHead(nn.Module):
    def __init__(self, hidden_dim: int, n_bins: int):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_bins)

    def forward(self, inst_repr: torch.Tensor, h0: Optional[torch.Tensor] = None) -> torch.Tensor:
        out, _ = self.rnn(inst_repr, h0)
        return self.classifier(out)


def smoothness_loss(logits: torch.Tensor) -> torch.Tensor:
    if logits.size(1) < 2:
        return torch.tensor(0.0, device=logits.device)
    diff = logits[:, 1:, :] - logits[:, :-1, :]
    return (diff * diff).mean()
