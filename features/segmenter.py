"""Token sequence segmentation and per-segment feature pooling."""

from __future__ import annotations

from typing import Iterable, List

import torch

from features.schema import Segment, clamp_segment


# ═══════════════════════════════════════════════════════════════
# Segment boundary strategies
# ═══════════════════════════════════════════════════════════════

def fixed_window_segment(n_tokens: int, window_size: int) -> List[Segment]:
    spans: List[Segment] = []
    if n_tokens <= 0:
        return spans
    sid = 0
    for start in range(0, n_tokens, max(1, window_size)):
        end = min(n_tokens, start + max(1, window_size))
        spans.append(Segment(segment_id=sid, start=start, end=end))
        sid += 1
    return spans


def step_segment(tokens: List[str], response: str, max_window: int = 256) -> List[Segment]:
    """Segment by double-newline (step boundaries).

    Falls back to fixed-window if no ``\\n\\n`` delimiters are found.
    """
    if not response or "\n\n" not in response:
        return fixed_window_segment(len(tokens), max_window)

    cum_char = [0]
    for t in tokens:
        cum_char.append(cum_char[-1] + len(t))
    n_chars = cum_char[-1]

    spans: List[Segment] = []
    sid = 0
    char_pos = 0
    while char_pos < n_chars:
        next_break = response.find("\n\n", char_pos)
        if next_break == -1:
            next_break = n_chars
        st = _char_to_token(cum_char, char_pos)
        ed = _char_to_token(cum_char, min(next_break, n_chars))
        ed = max(st + 1, min(ed + 1, len(tokens)))
        if st < ed:
            spans.append(Segment(segment_id=sid, start=st, end=ed))
            sid += 1
        char_pos = next_break + 2

    return spans if spans else fixed_window_segment(len(tokens), max_window)


def _char_to_token(cum_char: List[int], char_pos: int) -> int:
    lo, hi = 0, len(cum_char) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum_char[mid] <= char_pos:
            lo = mid + 1
        else:
            hi = mid
    return max(0, lo - 1)


# ═══════════════════════════════════════════════════════════════
# Per-segment feature pooling
# ═══════════════════════════════════════════════════════════════

def segment_pooling(
    token_tensor: torch.Tensor,
    spans: List[Segment],
    obs_dim: int,
    mode: str = "mean",
    segment_size: int = 32,
) -> torch.Tensor:
    """Aggregate per-token feature vectors into per-segment instance vectors.

    ``token_tensor``: [n_tokens, obs_dim] float32.
    Returns [n_segments, obs_dim] float32.

    ``"mean"`` — average over tokens via ``chunk.mean(dim=0)``.
    ``"concat"`` — flatten tokens, zero-pad or truncate to ``segment_size × obs_dim``.

    Span boundaries are clamped: start is clipped to [0, n_tokens], end is
    clipped to [start+1, n_tokens] so that every span produces at least one
    token.  An empty input returns a single zero-vector.
    """
    n_tokens = token_tensor.shape[0]
    out: List[torch.Tensor] = []
    for s in spans:
        st = max(0, min(s.start, n_tokens))
        ed = max(st + 1, min(s.end, n_tokens))
        chunk = token_tensor[st:ed]
        if chunk.shape[0] == 0:
            out.append(torch.zeros(obs_dim))
            continue
        if mode == "concat":
            flat = chunk.reshape(-1)
            target_len = segment_size * obs_dim
            if flat.shape[0] < target_len:
                flat = torch.cat([flat, torch.zeros(target_len - flat.shape[0])])
            out.append(flat[:target_len])
        else:  # mean
            out.append(chunk.mean(dim=0))
    if not out:
        return torch.zeros(1, obs_dim)
    return torch.stack(out)


# ═══════════════════════════════════════════════════════════════
# Shared helper for PPO feature extraction
# ═══════════════════════════════════════════════════════════════

def build_segment_obs_from_lp(
    lp_tensor: torch.Tensor,
    tokens: List[str],
    text: str,
    segment_size: int,
    obs_dim: int,
    device: torch.device | None = None,
    extra_parts: List[torch.Tensor] | None = None,
    segment_mode: str = "fixed_window",
) -> torch.Tensor:
    """Convert ``generate_with_features`` logprob tensor into segment obs.

    ``lp_tensor``: [n_tok, top_k+1] where col 0 = sampled logprob, cols 1: = top-k.
    Returns [n_segments, obs_dim].
    """
    n_tok = lp_tensor.shape[0]
    if n_tok == 0:
        return torch.zeros(1, obs_dim, device=device)

    base = torch.zeros(n_tok, 2, dtype=torch.float32)
    base[:, 0] = lp_tensor[:, 0].float()
    lp = lp_tensor[:, 1:].float()
    base[:, 1] = -(torch.exp(lp) * lp).sum(dim=1)

    parts = [base]
    if extra_parts:
        parts.extend(extra_parts)
    tok_vecs = torch.cat(parts, dim=1)
    if tok_vecs.shape[1] < obs_dim:
        tok_vecs = torch.cat([
            tok_vecs, torch.zeros(n_tok, obs_dim - tok_vecs.shape[1]),
        ], dim=1)
    else:
        tok_vecs = tok_vecs[:, :obs_dim]

    spans = build_segments(tokens=tokens, mode=segment_mode,
                           segment_size=segment_size, response=text)
    return segment_pooling(tok_vecs.to(device) if device is not None else tok_vecs,
                           spans, obs_dim, mode="mean",
                           segment_size=segment_size)


# ═══════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════

def build_segments(
    tokens: List[str],
    mode: str,
    segment_size: int,
    response: str = "",
) -> List[Segment]:
    if mode == "fixed_window":
        return fixed_window_segment(len(tokens), segment_size)
    if mode == "step":
        return step_segment(tokens, response=response, max_window=segment_size)
    if mode == "punctuation":
        return _punctuation_segment(tokens, max_window=segment_size)
    raise ValueError(f"Unknown segment mode: {mode}")


def _punctuation_segment(tokens: Iterable[str], max_window: int = 64) -> List[Segment]:
    token_list = list(tokens)
    spans: List[Segment] = []
    start = 0
    sid = 0
    punct = {".", "!", "?", ";", "\n"}
    for i, t in enumerate(token_list):
        is_break = t.strip() in punct or (i - start + 1) >= max(1, max_window)
        if is_break:
            s, e = clamp_segment(start, i + 1, len(token_list))
            if e > s:
                spans.append(Segment(segment_id=sid, start=s, end=e))
                sid += 1
            start = i + 1
    if start < len(token_list):
        s, e = clamp_segment(start, len(token_list), len(token_list))
        if e > s:
            spans.append(Segment(segment_id=sid, start=s, end=e))
    return spans
