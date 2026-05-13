"""Token sequence segmentation and per-segment feature pooling."""

from __future__ import annotations

from typing import Iterable, List

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
    token_vecs: List[List[float]],
    spans: List[Segment],
    obs_dim: int,
    mode: str = "mean",
    segment_size: int = 32,
) -> List[List[float]]:
    """Aggregate per-token feature vectors into per-segment instance vectors.

    ``"mean"`` — average over tokens (works for any segment mode).
    ``"concat"`` — flatten tokens, zero-pad or truncate to `segment_size × obs_dim`.
        Only makes sense with fixed_window segmentation (equal token counts).

    Span boundaries are clamped: start is clipped to [0, n_tokens], end is
    clipped to [start+1, n_tokens] so that every span produces at least one
    token.  An empty input returns a single zero-vector to satisfy the
    expectation that a bag always has at least one instance.
    ``"concat"`` — concatenate token vectors; shorter segments are zero-padded
    to *segment_size × obs_dim*.  Only makes sense with ``fixed_window``.
    """
    out: List[List[float]] = []
    for s in spans:
        st, ed = s.start, s.end
        st = max(0, min(st, len(token_vecs)))
        ed = max(st + 1, min(ed, len(token_vecs)))
        chunk = token_vecs[st:ed]
        if not chunk:
            out.append([0.0] * obs_dim)
            continue
        if mode == "concat":
            flat = [v for row in chunk for v in row]
            target_len = segment_size * obs_dim
            if len(flat) < target_len:
                flat += [0.0] * (target_len - len(flat))
            out.append(flat[:target_len])
        else:  # mean
            avg = [0.0] * obs_dim
            for row in chunk:
                for i, v in enumerate(row):
                    avg[i] += v
            denom = float(len(chunk))
            out.append([v / denom for v in avg])
    if not out:
        out = [[0.0] * obs_dim]
    return out


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
