"""Token sequence segmentation and per-segment feature pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch

from features.schema import Segment, clamp_segment


@dataclass
class MaskedSegmentFeatures:
    """Fixed-width concat features plus the per-token validity mask."""

    features: torch.Tensor    # [K, segment_size * token_dim]
    token_mask: torch.Tensor  # [K, segment_size]

    def to(self, device: torch.device | str) -> "MaskedSegmentFeatures":
        return MaskedSegmentFeatures(
            features=self.features.to(device),
            token_mask=self.token_mask.to(device),
        )


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
            flat = torch.zeros(segment_size * obs_dim, device=dev)
            n_take = min(chunk.shape[0], segment_size)
            flat[:n_take * obs_dim] = chunk[:n_take].reshape(-1)
            out.append(flat)
        else:  # mean
            out.append(chunk.mean(dim=0))
    if not out:
        out_dim = obs_dim * segment_size if mode == "concat" else obs_dim
        return torch.zeros(1, out_dim, device=dev)
    result = torch.stack(out)
    assert result.dim() == 2, f"segment_pooling: expected 2D [K, D], got {result.shape}"
    assert result.shape[-1] == (obs_dim * segment_size if mode == "concat" else obs_dim), \
        f"segment_pooling: expected last dim {obs_dim * segment_size if mode == 'concat' else obs_dim}, got {result.shape[-1]}"
    return result


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
    if pooling_mode == "concat" and segment_mode in ("step", "punctuation"):
        raise ValueError(
            f"pooling_mode='concat' is incompatible with segment_mode='{segment_mode}'. "
            f"Use segment_mode='fixed_window' with concat pooling."
        )

    n_tok = lp_tensor.shape[0]
    if n_tok == 0:
        return torch.zeros(1, obs_dim, device=device)

    lp = lp_tensor[:, 1:].float()
    sampled = lp_tensor[:, 0:1].float()                        # [n_tok, 1]
    # Top-k pseudo-entropy: computed over top-k logprobs only, not the full
    # vocabulary (~128K).  Summing over a subset under-estimates true entropy,
    # but full-distribution logprobs are prohibitively expensive (memory).
    # The downstream model receives the full top-k distribution (4096 dims)
    # which provides richer uncertainty signal than a single entropy scalar.
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
    result = segment_pooling(tok_vecs.to(device) if device is not None else tok_vecs,
                           spans, obs_dim, mode=pooling_mode,
                           segment_size=segment_size)
    assert result.dim() == 2, \
        f"build_segment_obs_from_lp: expected 2D [K, obs_dim], got {result.shape}"
    return result


def build_masked_concat_segment_obs_from_lp(
    lp_tensor: torch.Tensor,
    tokens: List[str],
    text: str,
    segment_size: int = 64,
    token_dim: int = 64,
    device: torch.device | None = None,
    segment_mode: str = "fixed_window",
) -> MaskedSegmentFeatures:
    """Build mask-aware fixed-width concat features for the prefix value path.

    Each response segment is represented by exactly ``segment_size`` token
    slots.  Incomplete final segments are zero-padded and retained.  The
    returned token mask is the source of truth for distinguishing real token
    features from padding.
    """
    if segment_size <= 0 or token_dim <= 0:
        raise ValueError("segment_size and token_dim must be positive")

    target_device = device if device is not None else lp_tensor.device
    n_tok = int(lp_tensor.shape[0])
    if n_tok == 0:
        return MaskedSegmentFeatures(
            features=torch.zeros(1, segment_size * token_dim, device=target_device),
            token_mask=torch.zeros(1, segment_size, device=target_device),
        )

    lp = lp_tensor[:, 1:].float()
    sampled = lp_tensor[:, 0:1].float()
    entropy = -(torch.exp(lp) * lp).sum(dim=1, keepdim=True)
    tok_vecs = torch.cat([sampled, entropy, lp], dim=1)
    if tok_vecs.shape[1] < token_dim:
        tok_vecs = torch.cat(
            [tok_vecs, tok_vecs.new_zeros(n_tok, token_dim - tok_vecs.shape[1])],
            dim=1,
        )
    else:
        tok_vecs = tok_vecs[:, :token_dim]
    tok_vecs = tok_vecs.to(target_device)

    spans = build_segments(
        tokens=tokens,
        mode=segment_mode,
        segment_size=segment_size,
        response=text,
    )
    if not spans:
        spans = [Segment(segment_id=0, start=0, end=n_tok)]

    feature_rows: List[torch.Tensor] = []
    mask_rows: List[torch.Tensor] = []
    for span in spans:
        start = max(0, min(span.start, n_tok))
        end = max(start, min(span.end, n_tok))
        chunk = tok_vecs[start:end][:segment_size]
        valid = int(chunk.shape[0])
        padded = tok_vecs.new_zeros(segment_size, token_dim)
        mask = tok_vecs.new_zeros(segment_size)
        if valid:
            padded[:valid] = chunk
            mask[:valid] = 1.0
        feature_rows.append(padded.reshape(-1))
        mask_rows.append(mask)

    return MaskedSegmentFeatures(
        features=torch.stack(feature_rows),
        token_mask=torch.stack(mask_rows),
    )


def batch_build_masked_concat_segment_obs_from_lp(
    lp_tensors: List[torch.Tensor],
    tokens_list: List[List[str]],
    texts: List[str],
    segment_size: int,
    token_dim: int,
    device: torch.device,
    segment_mode: str = "fixed_window",
) -> List[MaskedSegmentFeatures]:
    """Batched dispatcher for the mask-aware concat representation.

    The per-chain implementation is intentionally shared with offline feature
    extraction so online and offline semantics cannot diverge.
    """
    if not (len(lp_tensors) == len(tokens_list) == len(texts)):
        raise ValueError("lp_tensors, tokens_list, and texts must have equal length")
    return [
        build_masked_concat_segment_obs_from_lp(
            lp_tensor=lp,
            tokens=tokens,
            text=text,
            segment_size=segment_size,
            token_dim=token_dim,
            device=device,
            segment_mode=segment_mode,
        )
        for lp, tokens, text in zip(lp_tensors, tokens_list, texts)
    ]


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
    # Top-k pseudo-entropy (same caveat as per-chain version above).
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
        # Pad or truncate to segment_size tokens so output dim always
        # matches segment_pooling concat: segment_size * obs_dim.
        T = tok_vecs.shape[1]
        if T < segment_size:
            pad = torch.zeros(B, segment_size - T, obs_dim, device=tok_vecs.device)
            tok_vecs = torch.cat([tok_vecs, pad], dim=1)
        elif T > segment_size:
            tok_vecs = tok_vecs[:, :segment_size, :]
        obs_gpu = tok_vecs.reshape(B, segment_size * obs_dim)        # [B, segment_size * obs_dim]
        return [obs_gpu[i:i+1].cpu() for i in range(B)]              # each [1, segment_size * obs_dim]

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
    # Shape contract: each element is [n_segments_i, obs_dim] (2D)
    for i, r in enumerate(results):
        assert r.dim() == 2, \
            f"batch_build_segment_obs_from_lp: chain {i} expected 2D, got {r.shape}"
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
        is_break = t.strip() in punct or "\n" in t or (i - start + 1) >= max(1, max_window)
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
