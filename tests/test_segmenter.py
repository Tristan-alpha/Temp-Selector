"""Tests for segment_pooling and step_segment.  CPU-only."""

from features.schema import Segment
from features.segmenter import build_segments, segment_pooling

OBS_DIM = 64


def test_segment_pooling_single_segment():
    vecs = [[float(i)] * OBS_DIM for i in range(4)]
    spans = [Segment(start=0, end=4, segment_id=0)]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 1
    assert abs(out[0][0] - 1.5) < 1e-6


def test_segment_pooling_multiple_segments():
    vecs = [[float(i)] * OBS_DIM for i in range(8)]
    spans = [
        Segment(start=0, end=3, segment_id=0),
        Segment(start=3, end=5, segment_id=1),
        Segment(start=5, end=8, segment_id=2),
    ]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 3
    assert abs(out[0][0] - 1.0) < 1e-6
    assert abs(out[1][0] - 3.5) < 1e-6
    assert abs(out[2][0] - 6.0) < 1e-6


def test_segment_pooling_zero_span_clamped():
    vecs = [[1.0] * OBS_DIM] * 4
    spans = [Segment(start=0, end=0, segment_id=0)]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 1
    assert out[0][0] == 1.0


def test_segment_pooling_no_spans():
    vecs = [[1.0] * OBS_DIM] * 4
    spans: list = []
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 1
    assert all(x == 0.0 for x in out[0])


def test_segment_pooling_span_exceeds_tokens():
    vecs = [[1.0] * OBS_DIM] * 4
    spans = [Segment(start=0, end=100, segment_id=0)]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 1
    assert abs(out[0][0] - 1.0) < 1e-6


def test_segment_pooling_negative_start():
    vecs = [[1.0] * OBS_DIM] * 4
    spans = [Segment(start=-5, end=4, segment_id=0)]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 1
    assert abs(out[0][0] - 1.0) < 1e-6


def test_segment_pooling_no_tokens():
    vecs: list = []
    spans = [Segment(start=0, end=4, segment_id=0)]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == 1
    assert all(x == 0.0 for x in out[0])


def test_segment_pooling_concat():
    dim = 4
    seg_size = 3
    vecs = [[float(i + j) for j in range(dim)] for i in range(3)]
    spans = [Segment(start=0, end=3, segment_id=0)]
    out = segment_pooling(vecs, spans, dim, mode="concat", segment_size=seg_size)
    assert len(out) == 1
    assert len(out[0]) == seg_size * dim
    assert out[0][0] == 0.0
    assert out[0][3] == 3.0
    assert out[0][4] == 1.0


def test_segment_pooling_concat_padding():
    dim = 2
    seg_size = 5
    vecs = [[1.0, 2.0]]
    spans = [Segment(start=0, end=1, segment_id=0)]
    out = segment_pooling(vecs, spans, dim, mode="concat", segment_size=seg_size)
    assert len(out) == 1
    assert len(out[0]) == seg_size * dim
    assert out[0][0] == 1.0
    assert out[0][1] == 2.0
    assert all(x == 0.0 for x in out[0][2:])


def test_segment_pooling_concat_truncation():
    dim = 2
    seg_size = 2
    vecs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    spans = [Segment(start=0, end=3, segment_id=0)]
    out = segment_pooling(vecs, spans, dim, mode="concat", segment_size=seg_size)
    assert len(out) == 1
    assert len(out[0]) == seg_size * dim


def test_segment_pooling_from_build_segments():
    """Integration: build_segments produces Segment objects consumed by segment_pooling."""
    tokens = ["Step", " one", ".\n", "\n", "Step", " two", "."]
    response = "Step one.\n\nStep two."
    spans = build_segments(tokens=tokens, response=response, mode="step", segment_size=256)
    vecs = [[float(i)] * OBS_DIM for i in range(len(tokens))]
    out = segment_pooling(vecs, spans, OBS_DIM)
    assert len(out) == len(spans)
    assert len(out[0]) == OBS_DIM


def test_step_segment_basic():
    from features.segmenter import step_segment
    tokens = ["Step", " one", ".\n", "\n", "Step", " two", "."]
    response = "Step one.\n\nStep two."
    spans = step_segment(tokens, response)
    assert len(spans) >= 2, f"Expected at least 2 segments, got {len(spans)}"
    assert spans[0].start == 0
    assert spans[-1].end == len(tokens)
