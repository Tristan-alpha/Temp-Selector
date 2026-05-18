from features.schema import Segment, coerce_label


def test_label_coerce():
    assert coerce_label(True) == 1
    assert coerce_label(False) == 0
    assert coerce_label(3) == 1
    assert coerce_label(0) == 0


def test_segment():
    s = Segment(segment_id=0, start=0, end=5)
    assert s.segment_id == 0
    assert s.start == 0
    assert s.end == 5
