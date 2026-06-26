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
    """Linear attention: score = w^T · h (original simple pooler)."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        score = self.attn(h).squeeze(-1)
        w = torch.softmax(score, dim=-1)
        bag = torch.sum(h * w.unsqueeze(-1), dim=1)
        return bag, w


class GatedAttentionAggregator(nn.Module):
    """Gated attention (Ilse et al. 2018, Eq. 10):

        score = w^T · (tanh(V·h) ⊙ sigmoid(U·h))

    The gating mechanism provides non-linear interaction between
    feature dimensions — tanh captures transformed features, sigmoid
    selectively suppresses or enhances each dimension.
    """

    def __init__(self, hidden_dim: int, gate_dim: int | None = None):
        super().__init__()
        gd = gate_dim or max(hidden_dim // 2, 32)
        self.V = nn.Linear(hidden_dim, gd)
        self.U = nn.Linear(hidden_dim, gd)
        self.w = nn.Linear(gd, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.tanh(self.V(h))         # [B, K, gate_dim]
        b = torch.sigmoid(self.U(h))       # [B, K, gate_dim]
        score = self.w(a * b).squeeze(-1)  # [B, K]
        w = torch.softmax(score, dim=-1)
        bag = torch.sum(h * w.unsqueeze(-1), dim=1)
        return bag, w


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: Euclidean projection onto the probability simplex (Martins & Astudillo 2016).

    Unlike softmax, sparsemax produces sparse outputs — most entries are
    exactly zero.  No temperature parameter; sparsity is determined by the
    logit values themselves.
    """
    z = logits.clone()
    sorted_z, _ = z.sort(dim=dim, descending=True)
    cssv = sorted_z.cumsum(dim=dim)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    cond = sorted_z * k > (cssv - 1)
    k_star = cond.long().sum(dim=dim).clamp(min=1)
    idx = k_star.unsqueeze(dim) - 1
    cssv_k = cssv.gather(dim, idx)
    tau = (cssv_k - 1) / k_star.unsqueeze(dim).float()
    return (z - tau).clamp(min=0)


class GatedSparsemaxAggregator(nn.Module):
    """Gated + sparsemax attention for sparse MIL pooling.

    Gating provides non-linear interaction; sparsemax zeros out irrelevant
    segments, giving max-pool-like behaviour while keeping attention
    weights for PPO credit assignment.
    """

    def __init__(self, hidden_dim: int, gate_dim: int | None = None):
        super().__init__()
        gd = gate_dim or max(hidden_dim // 2, 32)
        self.V = nn.Linear(hidden_dim, gd)
        self.U = nn.Linear(hidden_dim, gd)
        self.w = nn.Linear(gd, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = torch.tanh(self.V(h))         # [B, K, gate_dim]
        b = torch.sigmoid(self.U(h))       # [B, K, gate_dim]
        score = self.w(a * b).squeeze(-1)  # [B, K]
        w = _sparsemax(score, dim=-1)
        bag = torch.sum(h * w.unsqueeze(-1), dim=1)
        return bag, w


class MultiHeadSparsemaxAggregator(nn.Module):
    """Multi-head gated sparsemax: H independent heads, each with its own
    gating + sparsemax.  Heads attend to different error patterns; final
    representation concatenates all heads and projects back to hidden_dim.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, gate_dim: int | None = None):
        super().__init__()
        self.num_heads = num_heads
        gd = gate_dim or max(hidden_dim // 2, 32)
        self.heads = nn.ModuleList([
            GatedSparsemaxAggregator(hidden_dim, gate_dim=gd)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bags: list[torch.Tensor] = []
        attns: list[torch.Tensor] = []
        for head in self.heads:
            bag_i, w_i = head(h)                         # [B, D], [B, K]
            bags.append(bag_i)
            attns.append(w_i)
        bag = self.proj(torch.cat(bags, dim=-1))         # [B, D]
        attn = torch.stack(attns, dim=1).mean(dim=1)     # [B, K] — mean over heads
        return bag, attn


class MILModel(nn.Module):
    """Multiple Instance Learning model for error localization in reasoning chains."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        aggregator: str = "attention",
        use_position: bool = True,
        use_gru: bool = True,
        gated_attention: bool = False,
        num_heads: int = 1,
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

        self.attn_agg: AttentionAggregator | GatedAttentionAggregator | GatedSparsemaxAggregator | MultiHeadSparsemaxAggregator
        if aggregator == "sparsemax":
            self.attn_agg = GatedSparsemaxAggregator(hidden_dim)
        elif aggregator == "multihead_sparsemax":
            self.attn_agg = MultiHeadSparsemaxAggregator(hidden_dim, num_heads=num_heads)
        elif gated_attention:
            self.attn_agg = GatedAttentionAggregator(hidden_dim)
        else:
            self.attn_agg = AttentionAggregator(hidden_dim)
        self.num_heads = num_heads
        self.bag_head = nn.Linear(hidden_dim, 1)

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

        Returns dict with:
          bag_logit  [B]:     whole-answer error logit (→ sigmoid → prob)
          attn_w     [B, K]:  segment attention weights (interpretable)
          bag_repr   [B, D]:  aggregated representation
        """
        assert instances.dim() == 3, \
            f"MILModel: expected 3D input [B, K, D], got {instances.shape}"
        h = self.encoder(instances)                           # [B, K, hidden_dim]

        if self.pos_encoder is not None:
            h = self.pos_encoder(h)

        if self.gru is not None:
            h, _ = self.gru(h)                                # [B, N, hidden_dim*2]
            h = self.gru_proj(h)                              # [B, N, hidden_dim]

        bag_repr, attn_w = self._aggregate(h)
        bag_logit = self.bag_head(bag_repr).squeeze(-1)       # [B]
        return {
            "bag_repr": bag_repr,
            "bag_logit": bag_logit,
            "attn_w": attn_w,
        }


