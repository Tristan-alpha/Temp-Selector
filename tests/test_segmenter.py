"""Tests for segment_pooling, step_segment, and batch_build_segment_obs_from_lp.  CPU-only."""

import torch

from features.schema import Segment
from features.segmenter import build_segments, segment_pooling
from features.segmenter import batch_build_segment_obs_from_lp, build_segment_obs_from_lp

OBS_DIM = 64


def _t(rows, dim=OBS_DIM):
    """Build [n_tokens, dim] tensor from row values."""
    return torch.tensor([[float(v)] * dim for v in rows], dtype=torch.float32)


def test_segment_pooling_single_segment():
    t = _t([0, 1, 2, 3])
    spans = [Segment(start=0, end=4, segment_id=0)]
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (1, OBS_DIM)
    assert abs(out[0, 0].item() - 1.5) < 1e-6


def test_segment_pooling_multiple_segments():
    t = _t([0, 1, 2, 3, 4, 5, 6, 7])
    spans = [
        Segment(start=0, end=3, segment_id=0),
        Segment(start=3, end=5, segment_id=1),
        Segment(start=5, end=8, segment_id=2),
    ]
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (3, OBS_DIM)
    assert abs(out[0, 0].item() - 1.0) < 1e-6
    assert abs(out[1, 0].item() - 3.5) < 1e-6
    assert abs(out[2, 0].item() - 6.0) < 1e-6


def test_segment_pooling_zero_span_clamped():
    t = torch.ones(4, OBS_DIM)
    spans = [Segment(start=0, end=0, segment_id=0)]
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (1, OBS_DIM)
    assert out[0, 0].item() == 1.0


def test_segment_pooling_no_spans():
    t = torch.ones(4, OBS_DIM)
    spans: list = []
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (1, OBS_DIM)
    assert torch.all(out == 0.0)


def test_segment_pooling_span_exceeds_tokens():
    t = torch.ones(4, OBS_DIM)
    spans = [Segment(start=0, end=100, segment_id=0)]
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (1, OBS_DIM)
    assert abs(out[0, 0].item() - 1.0) < 1e-6


def test_segment_pooling_negative_start():
    t = torch.ones(4, OBS_DIM)
    spans = [Segment(start=-5, end=4, segment_id=0)]
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (1, OBS_DIM)
    assert abs(out[0, 0].item() - 1.0) < 1e-6


def test_segment_pooling_no_tokens():
    t = torch.zeros(0, OBS_DIM)
    spans = [Segment(start=0, end=4, segment_id=0)]
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (1, OBS_DIM)
    assert torch.all(out == 0.0)


def test_segment_pooling_concat():
    dim = 4
    seg_size = 3
    t = torch.tensor([[float(i + j) for j in range(dim)] for i in range(3)], dtype=torch.float32)
    spans = [Segment(start=0, end=3, segment_id=0)]
    out = segment_pooling(t, spans, dim, mode="concat", segment_size=seg_size)
    assert out.shape == (1, seg_size * dim)
    assert out[0, 0].item() == 0.0
    assert out[0, 3].item() == 3.0
    assert out[0, 4].item() == 1.0


def test_segment_pooling_concat_padding():
    """concat: segment with < segment_size tokens is zero-padded to segment_size."""
    dim = 2
    seg_size = 5
    t = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    spans = [Segment(start=0, end=1, segment_id=0)]
    out = segment_pooling(t, spans, dim, mode="concat", segment_size=seg_size)
    # 1 token × 2 dims = 2 real values, padded to 5*2 = 10
    assert out.shape == (1, seg_size * dim), f"expected (1, {seg_size * dim}), got {out.shape}"
    # First 2 elements preserve real features
    assert out[0, 0].item() == 1.0
    assert out[0, 1].item() == 2.0
    # Remaining 8 elements are zero-padded
    assert out[0, 2:].eq(0).all()

def test_segment_pooling_concat_padding_mixed():
    """concat: shorter-than-segment_size chunks are zero-padded, longer ones kept as-is.

    A bag with spans [0:2] (zero-padded to 5 tokens) and [2:7] (kept, 5 tokens >= seg_size=5)
    should produce two output segments of shape [seg_size * dim] each."""
    dim = 3
    seg_size = 5
    t = torch.randn(7, dim)
    spans = [
        Segment(start=0, end=2, segment_id=0),   # 2 tokens → zero-padded
        Segment(start=2, end=7, segment_id=1),   # 5 tokens → kept
    ]
    out = segment_pooling(t, spans, dim, mode="concat", segment_size=seg_size)
    assert out.shape == (2, seg_size * dim)
    # First segment: first 2*t_dim elements = real, rest = zeros
    assert torch.allclose(out[0, :2*dim], t[0:2].reshape(-1))
    assert out[0, 2*dim:].eq(0).all()
    # Second segment: all 5*t_dim elements = real (from first 5 tokens of span)
    assert torch.allclose(out[1], t[2:7][:seg_size].reshape(-1))


def test_segment_pooling_concat_truncation():
    """concat: 3 tokens >= segment_size=2, truncated to first 2 tokens."""
    dim = 2
    seg_size = 2
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    spans = [Segment(start=0, end=3, segment_id=0)]
    out = segment_pooling(t, spans, dim, mode="concat", segment_size=seg_size)
    # Truncated to segment_size=2 tokens × 2 dims = 4
    assert out.shape == (1, 4)
    assert out[0, 0].item() == 1.0
    assert out[0, 1].item() == 2.0
    assert out[0, 2].item() == 3.0
    assert out[0, 3].item() == 4.0


def test_segment_pooling_from_build_segments():
    """Integration: build_segments produces Segment objects consumed by segment_pooling."""
    tokens = ["Step", " one", ".\n", "\n", "Step", " two", "."]
    response = "Step one.\n\nStep two."
    spans = build_segments(tokens=tokens, response=response, mode="step", segment_size=256)
    t = _t(range(len(tokens)))
    out = segment_pooling(t, spans, OBS_DIM)
    assert out.shape == (len(spans), OBS_DIM)


def test_step_segment_basic():
    from features.segmenter import step_segment
    tokens = ["Step", " one", ".\n", "\n", "Step", " two", "."]
    response = "Step one.\n\nStep two."
    spans = step_segment(tokens, response)
    assert len(spans) >= 2, f"Expected at least 2 segments, got {len(spans)}"
    assert spans[0].start == 0
    assert spans[-1].end == len(tokens)


def test_punctuation_segment_basic():
    """Segment at punctuation boundaries."""
    from features.segmenter import _punctuation_segment
    tokens = ["Hello", ".", " ", "World", "!"]
    spans = _punctuation_segment(tokens, max_window=64)
    assert len(spans) == 2, f"Expected 2 segments, got {len(spans)}"
    assert spans[0].start == 0 and spans[0].end == 2   # "Hello ."
    assert spans[1].start == 2 and spans[1].end == 5    # " World !"


def test_punctuation_segment_newline_token():
    """Standalone newline token is treated as a boundary, even after strip()."""
    from features.segmenter import _punctuation_segment
    tokens = ["Step", " one", ".\n", "\n", "Step", " two", "."]
    spans = _punctuation_segment(tokens, max_window=64)
    # ".\n" (period+newline) and "\n" (standalone newline) both trigger breaks
    assert len(spans) >= 2, f"Expected >=2 segments, got {len(spans)}"


def test_concat_with_step_mode_raises():
    """concat + step/punctuation is forbidden."""
    import pytest
    from features.segmenter import build_segment_obs_from_lp
    lp = torch.randn(5, 5)
    with pytest.raises(ValueError, match="concat.*incompatible"):
        build_segment_obs_from_lp(
            lp, ["a"] * 5, "a a a a a",
            4, 8, device=torch.device("cpu"),
            segment_mode="step", pooling_mode="concat",
        )


def test_concat_with_fixed_window_allowed():
    """concat + fixed_window should work (allowed combination)."""
    from features.segmenter import build_segment_obs_from_lp
    lp = torch.randn(4, 5)
    out = build_segment_obs_from_lp(
        lp, ["a"] * 4, "a a a a",
        4, 8, device=torch.device("cpu"),
        segment_mode="fixed_window", pooling_mode="concat",
    )
    assert out.ndim == 2  # [K, seg_size * obs_dim]


def test_punctuation_segment_max_window():
    """max_window forces a break when no punctuation is found."""
    from features.segmenter import _punctuation_segment
    tokens = ["a"] * 100
    spans = _punctuation_segment(tokens, max_window=10)
    assert len(spans) > 1, f"max_window=10 should split, got {len(spans)}"
    for sp in spans:
        assert sp.end - sp.start <= 10, f"segment too long: {sp}"


# ── batch_build_segment_obs_from_lp tests ──

def _make_lp_tensor(n_tok, top_k=8):
    """Synthetic logprob tensor: [n_tok, top_k+1], col 0 = sampled."""
    lp = torch.randn(n_tok, top_k + 1, dtype=torch.float32) * 0.1
    lp[:, 0] = -0.5  # sampled logprob
    return lp


def test_batch_build_obs_matches_per_chain_mean():
    """CPU path: batch output matches per-chain for fixed_window+mean."""
    seg_size = 4
    obs_dim = 10
    N = 3
    lp_tensors = [_make_lp_tensor(seg_size, top_k=6) for _ in range(N)]
    tokens_list = [["a"] * seg_size for _ in range(N)]
    texts = ["a a a a"] * N
    device = torch.device("cpu")

    batch_out = batch_build_segment_obs_from_lp(
        lp_tensors, tokens_list, texts,
        seg_size, obs_dim, device,
        include_topk=True, pooling_mode="mean",
    )
    per_chain = [
        build_segment_obs_from_lp(
            lp_tensors[i], tokens_list[i], texts[i],
            seg_size, obs_dim,
            include_topk=True, pooling_mode="mean",
        )
        for i in range(N)
    ]
    assert len(batch_out) == N
    for bo, pc in zip(batch_out, per_chain):
        assert torch.allclose(bo, pc, atol=1e-5), f"batch vs per-chain mismatch"


def test_batch_build_obs_matches_per_chain_concat():
    """CPU path: batch output matches per-chain for fixed_window+concat."""
    seg_size = 3
    obs_dim = 8
    N = 2
    lp_tensors = [_make_lp_tensor(seg_size, top_k=6) for _ in range(N)]
    tokens_list = [["x"] * seg_size for _ in range(N)]
    texts = ["x x x"] * N
    device = torch.device("cpu")

    batch_out = batch_build_segment_obs_from_lp(
        lp_tensors, tokens_list, texts,
        seg_size, obs_dim, device,
        include_topk=False, pooling_mode="concat",
    )
    per_chain = [
        build_segment_obs_from_lp(
            lp_tensors[i], tokens_list[i], texts[i],
            seg_size, obs_dim,
            include_topk=False, pooling_mode="concat",
        )
        for i in range(N)
    ]
    assert len(batch_out) == N
    for bo, pc in zip(batch_out, per_chain):
        assert torch.allclose(bo, pc, atol=1e-5)


def test_batch_build_obs_single_chain():
    """Single chain falls through to per-chain path."""
    device = torch.device("cpu")
    lp = _make_lp_tensor(4, top_k=4)
    out = batch_build_segment_obs_from_lp(
        [lp], [["a"] * 4], ["a a a a"], 4, 8, device,
        include_topk=True,
    )
    assert len(out) == 1
    assert out[0].shape == (1, 8)


def test_batch_build_obs_empty():
    """Empty input returns empty list."""
    out = batch_build_segment_obs_from_lp(
        [], [], [], 4, 8, torch.device("cpu"),
    )
    assert out == []


# ── concat fast path pad/truncate tests ──

def test_batch_build_obs_concat_pads_short():
    """Concat mode: when all n_tok < segment_size, output is zero-padded to segment_size * obs_dim per segment."""
    seg_size = 8
    obs_dim = 10
    N = 3
    lp_tensors = [_make_lp_tensor(3, top_k=6) for _ in range(N)]
    tokens_list = [["a"] * 3 for _ in range(N)]
    texts = ["a a a"] * N
    device = torch.device("cpu")

    out = batch_build_segment_obs_from_lp(
        lp_tensors, tokens_list, texts,
        seg_size, obs_dim, device,
        include_topk=False, pooling_mode="concat",
    )
    # fixed_window: 1 segment per chain. concat output = [1, seg_size * obs_dim].
    expected_dim = seg_size * obs_dim
    for o in out:
        assert o.shape == (1, expected_dim), f"expected (1, {expected_dim}), got {o.shape}"
        assert not o[0, :3 * obs_dim].eq(0).all(), "first 3*obs_dim should contain real features"
        assert o[0, 3 * obs_dim:].eq(0).all(), "remainder should be zero-padded"


def test_batch_build_obs_concat_output_dim_always_segment_size_times_obs_dim():
    """Concat output per-segment dim is always segment_size * obs_dim, even with exact or excess tokens."""
    seg_size = 4
    obs_dim = 6
    N = 2
    # n_tok == segment_size (exact match — the common case)
    lp_tensors = [_make_lp_tensor(seg_size, top_k=4) for _ in range(N)]
    tokens_list = [["x"] * seg_size for _ in range(N)]
    texts = ["x " * seg_size] * N
    device = torch.device("cpu")

    out = batch_build_segment_obs_from_lp(
        lp_tensors, tokens_list, texts,
        seg_size, obs_dim, device,
        include_topk=False, pooling_mode="concat",
    )
    expected_dim = seg_size * obs_dim
    for o in out:
        assert o.shape[1] == expected_dim, f"per-segment dim mismatch: expected {expected_dim}, got {o.shape[1]}"


def test_batch_build_obs_concat_matches_per_chain_mixed_tokens():
    """Concat batch output matches per-chain build_segment_obs_from_lp with mixed token counts."""
    seg_size = 5
    obs_dim = 8
    n_toks_list = [2, 5]  # short and exact
    lp_tensors = [_make_lp_tensor(n, top_k=4) for n in n_toks_list]
    tokens_list = [["t"] * n for n in n_toks_list]
    texts = ["t " * n for n in n_toks_list]
    device = torch.device("cpu")

    batch_out = batch_build_segment_obs_from_lp(
        lp_tensors, tokens_list, texts,
        seg_size, obs_dim, device,
        include_topk=False, pooling_mode="concat",
    )
    per_chain = [
        build_segment_obs_from_lp(
            lp_tensors[i], tokens_list[i], texts[i],
            seg_size, obs_dim,
            include_topk=False, pooling_mode="concat",
        )
        for i in range(len(n_toks_list))
    ]
    assert len(batch_out) == len(per_chain)
    for bo, pc in zip(batch_out, per_chain):
        assert bo.shape == pc.shape, f"shape mismatch: {bo.shape} vs {pc.shape}"
        assert torch.allclose(bo, pc, atol=1e-5), f"batch vs per-chain mismatch"
