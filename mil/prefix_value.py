"""Causal, mask-aware prefix value model and its training objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from mil.model import SinusoidalPositionalEncoding


@dataclass
class PrefixRecurrentState:
    hidden: torch.Tensor
    position: int


class PrefixValueModel(nn.Module):
    """Estimate the probability that a generated prefix ends correctly."""

    def __init__(self, token_dim: int = 64, segment_size: int = 64,
                 hidden_dim: int = 1024, max_segments: int = 8192,
                 n_temps: int = 0, prompt_dim: int = 0,
                 prompt_integration: str = "none"):
        super().__init__()
        self.token_dim = int(token_dim)
        self.segment_size = int(segment_size)
        self.hidden_dim = int(hidden_dim)
        self.n_temps = int(n_temps)
        self.prompt_dim = int(prompt_dim)
        self.prompt_integration = str(prompt_integration or "none")
        if self.prompt_dim < 0:
            raise ValueError("prompt_dim must be non-negative")
        if self.prompt_dim > 0 and self.prompt_integration == "none":
            self.prompt_integration = "gru_init"
        if self.prompt_dim > 0 and self.prompt_integration != "gru_init":
            raise ValueError("prompt-aware PVM currently supports prompt_integration='gru_init' only")
        self.feature_dim = self.token_dim * self.segment_size
        self.input_dim = self.feature_dim + self.segment_size

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.pos_encoder = SinusoidalPositionalEncoding(hidden_dim, max_len=max_segments)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.q_head = nn.Linear(hidden_dim, self.n_temps) if self.n_temps > 0 else None
        self.prompt_projector = (
            nn.Sequential(
                nn.LayerNorm(self.prompt_dim),
                nn.Linear(self.prompt_dim, hidden_dim),
                nn.Tanh(),
            )
            if self.prompt_dim > 0 else None
        )

    def _masked_input(self, features: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if features.shape[:-1] != token_mask.shape[:-1]:
            raise ValueError("features and token_mask leading dimensions must match")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature dim {self.feature_dim}, got {features.shape[-1]}"
            )
        if token_mask.shape[-1] != self.segment_size:
            raise ValueError(
                f"expected token mask dim {self.segment_size}, got {token_mask.shape[-1]}"
            )
        expanded_mask = token_mask.repeat_interleave(self.token_dim, dim=-1)
        return torch.cat([features * expanded_mask, token_mask], dim=-1)

    def initial_hidden(self, prompt_hidden: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Project a prompt representation into the GRU initial hidden state."""
        if self.prompt_projector is None:
            return None
        if prompt_hidden is None:
            raise ValueError("prompt_hidden is required when prompt_dim > 0")
        if prompt_hidden.ndim != 2:
            raise ValueError("prompt_hidden must have shape [B, prompt_dim]")
        if prompt_hidden.shape[-1] != self.prompt_dim:
            raise ValueError(
                f"expected prompt_hidden dim {self.prompt_dim}, got {prompt_hidden.shape[-1]}"
            )
        return self.prompt_projector(prompt_hidden).unsqueeze(0)

    def forward(self, features: torch.Tensor, token_mask: torch.Tensor,
                segment_mask: torch.Tensor,
                prompt_hidden: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Encode complete prefixes while excluding padded segments from the GRU."""
        if features.ndim != 3 or token_mask.ndim != 3 or segment_mask.ndim != 2:
            raise ValueError("expected features [B,K,D], token_mask [B,K,S], segment_mask [B,K]")
        lengths = segment_mask.sum(dim=1).long()
        if torch.any(lengths <= 0):
            raise ValueError("every prefix must contain at least one valid segment")

        x = self.encoder(self._masked_input(features, token_mask))
        x = self.pos_encoder(x)
        packed = pack_padded_sequence(
            x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False,
        )
        packed_out, hidden = self.gru(packed, self.initial_hidden(prompt_hidden))
        encoded, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=features.shape[1],
        )
        encoded = encoded * segment_mask.unsqueeze(-1)
        value_logits = self.value_head(encoded).squeeze(-1)
        value_logits = value_logits * segment_mask
        q_logits = self.q_head(encoded) if self.q_head is not None else None
        if q_logits is not None:
            q_logits = q_logits * segment_mask.unsqueeze(-1)
        terminal_idx = lengths - 1
        terminal_logits = value_logits.gather(1, terminal_idx.unsqueeze(1)).squeeze(1)
        terminal_hidden = encoded.gather(
            1, terminal_idx.view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
        ).squeeze(1)
        result = {
            "value_logits": value_logits,
            "hidden_states": encoded,
            "terminal_logits": terminal_logits,
            "terminal_hidden": terminal_hidden,
            "gru_hidden": hidden,
        }
        if q_logits is not None:
            terminal_q_logits = q_logits.gather(
                1,
                terminal_idx.view(-1, 1, 1).expand(-1, 1, q_logits.shape[-1]),
            ).squeeze(1)
            result["q_logits"] = q_logits
            result["terminal_q_logits"] = terminal_q_logits
        return result

    def step(self, segment_feature: torch.Tensor, token_mask: torch.Tensor,
             state: Optional[PrefixRecurrentState] = None
             ) -> Tuple[torch.Tensor, torch.Tensor, PrefixRecurrentState]:
        """Advance one segment and return logit, encoded state, and recurrent state."""
        if segment_feature.ndim == 1:
            segment_feature = segment_feature.unsqueeze(0)
        if token_mask.ndim == 1:
            token_mask = token_mask.unsqueeze(0)
        position = 0 if state is None else state.position
        hidden = None if state is None else state.hidden
        logits, encoded, next_hidden = self.step_batch(
            segment_feature, token_mask, hidden=hidden, position=position,
        )
        return logits, encoded, PrefixRecurrentState(next_hidden, position + 1)

    def step_batch(self, segment_features: torch.Tensor, token_masks: torch.Tensor,
                   hidden: Optional[torch.Tensor], position: int
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance a batch of chains at the same segment position."""
        if segment_features.ndim != 2 or token_masks.ndim != 2:
            raise ValueError("step_batch expects [B,D] features and [B,S] token masks")
        if position >= self.pos_encoder.pe.shape[1]:
            raise ValueError(f"prefix position {position} exceeds positional encoding capacity")
        x = self.encoder(self._masked_input(segment_features, token_masks))
        x = x + self.pos_encoder.pe[:, position, :]
        encoded, next_hidden = self.gru(x.unsqueeze(1), hidden)
        encoded = encoded[:, 0, :]
        logits = self.value_head(encoded).squeeze(-1)
        return logits, encoded, next_hidden

    def q_from_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return per-temperature Q logits from encoded prefix hidden states."""
        if self.q_head is None:
            raise RuntimeError("PrefixValueModel was created without a Q head")
        if hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"expected hidden dim {self.hidden_dim}, got {hidden.shape[-1]}")
        return self.q_head(hidden)


def calibrated_probability(logits: torch.Tensor, temperature: float | torch.Tensor = 1.0) -> torch.Tensor:
    t = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device).clamp_min(1e-4)
    return torch.sigmoid(logits / t)


def binomial_nll(logits: torch.Tensor, n_correct: torch.Tensor,
                 n_total: torch.Tensor) -> torch.Tensor:
    """Mean per-rollout binomial negative log likelihood."""
    n_total = n_total.to(logits.dtype).clamp_min(1.0)
    n_correct = n_correct.to(logits.dtype)
    loss = -(n_correct * F.logsigmoid(logits) +
             (n_total - n_correct) * F.logsigmoid(-logits)) / n_total
    return loss.mean()


def masked_binomial_nll(logits: torch.Tensor, n_correct: torch.Tensor,
                        n_total: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean binomial NLL over valid per-temperature labels."""
    n_total = n_total.to(logits.dtype).clamp_min(1.0)
    n_correct = n_correct.to(logits.dtype)
    mask = mask.to(logits.dtype)
    loss = -(n_correct * F.logsigmoid(logits) +
             (n_total - n_correct) * F.logsigmoid(-logits)) / n_total
    valid = mask > 0
    if not torch.any(valid):
        return logits.new_zeros(())
    return loss[valid].mean()


def paired_ranking_loss(logits_a: torch.Tensor, logits_b: torch.Tensor,
                        target_a: torch.Tensor, target_b: torch.Tensor) -> torch.Tensor:
    """Weighted RankNet objective for already-filtered, non-tied pairs."""
    direction = torch.sign(target_a - target_b)
    weight = torch.abs(target_a - target_b)
    valid = direction != 0
    if not torch.any(valid):
        return logits_a.new_zeros(())
    return (weight[valid] * F.softplus(
        -direction[valid] * (logits_a[valid] - logits_b[valid])
    )).mean()


def potential_reward(phi_before: torch.Tensor | float,
                     phi_after: torch.Tensor | float,
                     gamma: float, shaping_coef: float,
                     terminal_reward: torch.Tensor | float | None = None
                     ) -> torch.Tensor:
    """Potential shaping for non-terminal or absorbing terminal transitions."""
    before = torch.as_tensor(phi_before, dtype=torch.float32)
    if terminal_reward is not None:
        terminal = torch.as_tensor(terminal_reward, dtype=torch.float32)
        return terminal - shaping_coef * before
    after = torch.as_tensor(phi_after, dtype=torch.float32)
    return shaping_coef * (gamma * after - before)
