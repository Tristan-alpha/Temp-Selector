"""Token sequence segmentation and per-segment feature pooling."""

from __future__ import annotations

from typing import Iterable, List, Optional

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
    dev = token_tensor.device
    out: List[torch.Tensor] = []
    for s in spans:
        st = max(0, min(s.start, n_tokens))
        ed = max(st + 1, min(s.end, n_tokens))
        chunk = token_tensor[st:ed]
        if chunk.shape[0] == 0:
            out.append(torch.zeros(obs_dim, device=dev))
            continue
        if mode == "concat":
            if chunk.shape[0] < segment_size:
                continue  # drop incomplete segment
            flat = chunk.reshape(-1)
            out.append(flat)
        else:  # mean
            out.append(chunk.mean(dim=0))
    if not out:
        return torch.zeros(1, obs_dim, device=dev)
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
    include_topk: bool = False,
    pooling_mode: str = "mean",
) -> torch.Tensor:
    """Convert ``generate_with_features`` logprob tensor into segment obs.

    ``lp_tensor``: [n_tok, top_k+1] where col 0 = sampled logprob, cols 1: = top-k.
    Returns [n_segments, obs_dim].

    When ``include_topk``, the full top-k logprobs are appended to the feature
    vector (2 + top_k dims).  When False, only logprob + entropy form the base
    (2 dims), with ``extra_parts`` appended before padding.
    """
    n_tok = lp_tensor.shape[0]
    if n_tok == 0:
        return torch.zeros(1, obs_dim, device=device)

    lp = lp_tensor[:, 1:].float()
    sampled = lp_tensor[:, 0:1].float()                        # [n_tok, 1]
    entropy = -(torch.exp(lp) * lp).sum(dim=1, keepdim=True)  # [n_tok, 1]

    if include_topk:
        parts = [sampled, entropy, lp]                         # [n_tok, 2+top_k]
    else:
        parts = [torch.cat([sampled, entropy], dim=1)]         # [n_tok, 2]
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
                           spans, obs_dim, mode=pooling_mode,
                           segment_size=segment_size)


# ═══════════════════════════════════════════════════════════════
# Batched GPU variant for online eval
# ═══════════════════════════════════════════════════════════════

def batch_build_segment_obs_from_lp(
    lp_tensors: List[torch.Tensor],
    tokens_list: List[List[str]],
    texts: List[str],
    segment_size: int,
    obs_dim: int,
    device: torch.device,
    extra_tensors: Optional[List[torch.Tensor]] = None,
    segment_mode: str = "fixed_window",
    include_topk: bool = False,
    pooling_mode: str = "mean",
) -> List[torch.Tensor]:
    """GPU-batched version of ``build_segment_obs_from_lp`` for eval.

    Stacks per-chain logprob tensors, runs ``exp`` / ``cat`` / ``truncate`` on
    ``device``, then pools into per-chain observation vectors.  Falls back to
    per-chain CPU calls when ``device`` is CPU or only a single chain is given.

    ``lp_tensors``: each ``[n_tok_i, top_k+1]`` — active chains only (EOS
    chains already filtered by caller).

    ``extra_tensors``: optional hidden states, one per chain (same length as
    ``lp_tensors``), for ``feature_mode=hidden_states``.

    Returns a list of ``[n_segments_i, obs_dim]`` tensors on CPU.
    """
    B = len(lp_tensors)
    if B == 0:
        return []
    if device.type == "cpu" or B == 1:
        return [
            build_segment_obs_from_lp(
                lp_tensors[i], tokens_list[i], texts[i],
                segment_size, obs_dim, device=device,
                extra_parts=[extra_tensors[i]] if extra_tensors and extra_tensors[i] is not None else None,
                segment_mode=segment_mode, include_topk=include_topk,
                pooling_mode=pooling_mode,
            )
            for i in range(B)
        ]

    # ---- Pad to uniform n_tok, stack on GPU ----
    n_toks = [t.shape[0] for t in lp_tensors]
    max_tok = max(n_toks)
    need_pad = any(n != max_tok for n in n_toks)

    if not need_pad:
        stacked = torch.stack([t.to(device, non_blocking=True) for t in lp_tensors])
    else:
        padded = torch.zeros(B, max_tok, lp_tensors[0].shape[1], device=device)
        for i, t in enumerate(lp_tensors):
            padded[i, :n_toks[i]] = t.to(device)
        stacked = padded

    # ---- Token-level math on GPU ----
    lp = stacked[:, :, 1:].float()                                  # [B, T, top_k]
    sampled = stacked[:, :, 0:1].float()                            # [B, T, 1]
    entropy = -(torch.exp(lp) * lp).sum(dim=2, keepdim=True)       # [B, T, 1]

    if include_topk:
        parts = [sampled, entropy, lp]                              # [B, T, 2+top_k]
    else:
        parts = [torch.cat([sampled, entropy], dim=2)]              # [B, T, 2]

    if extra_tensors is not None:
        if need_pad:
            extra_padded = torch.zeros(B, max_tok, extra_tensors[0].shape[1], device=device)
            for i, e in enumerate(extra_tensors):
                extra_padded[i, :n_toks[i]] = e.to(device)
            parts.append(extra_padded)
        else:
            parts.append(torch.stack([e.to(device, non_blocking=True) for e in extra_tensors]))

    tok_vecs = torch.cat(parts, dim=2)                               # [B, T, D']
    D = tok_vecs.shape[2]
    if D < obs_dim:
        tok_vecs = torch.cat(
            [tok_vecs, torch.zeros(B, max_tok, obs_dim - D, device=device)], dim=2)
    elif D > obs_dim:
        tok_vecs = tok_vecs[:, :, :obs_dim]                          # [B, T, obs_dim]

    # ---- Pooling ----
    if segment_mode == "fixed_window" and pooling_mode == "mean":
        # All active chains have the same span: [Segment(0, max_tok)].
        # Mean-pool over the token dim on GPU.
        obs_gpu = tok_vecs.mean(dim=1)                                # [B, obs_dim]
        return [obs_gpu[i:i+1].cpu() for i in range(B)]              # each [1, obs_dim]

    if segment_mode == "fixed_window" and pooling_mode == "concat":
        obs_gpu = tok_vecs.reshape(B, -1)                             # [B, T*obs_dim]
        return [obs_gpu[i:i+1].cpu() for i in range(B)]              # each [1, T*obs_dim]

    # step / punctuation mode: pool per-chain on CPU (spans differ)
    tok_cpu = tok_vecs.cpu()
    results: List[torch.Tensor] = []
    for i in range(B):
        n = n_toks[i]
        spans = build_segments(tokens=tokens_list[i], mode=segment_mode,
                               segment_size=segment_size, response=texts[i])
        obs = segment_pooling(tok_cpu[i, :n], spans, obs_dim,
                              mode=pooling_mode, segment_size=segment_size)
        results.append(obs)
    return results


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
