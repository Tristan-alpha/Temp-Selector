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
    """concat: segment with < segment_size tokens is dropped; empty → zero vector."""
    dim = 2
    seg_size = 5
    t = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    spans = [Segment(start=0, end=1, segment_id=0)]
    out = segment_pooling(t, spans, dim, mode="concat", segment_size=seg_size)
    assert out.shape == (1, dim)  # dropped, fallback zero


def test_segment_pooling_concat_truncation():
    """concat: 3 tokens >= segment_size=2, kept as-is (no more zero-padding/truncation)."""
    dim = 2
    seg_size = 2
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
    spans = [Segment(start=0, end=3, segment_id=0)]
    out = segment_pooling(t, spans, dim, mode="concat", segment_size=seg_size)
    assert out.shape == (1, 6)  # 3 tokens × 2 dims, no truncation to 4


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
