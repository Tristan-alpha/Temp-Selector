from features.schema import BagSample, Segment, TokenFeature, coerce_label


def test_label_coerce():
    assert coerce_label(True) == 1
    assert coerce_label(False) == 0
    assert coerce_label(3) == 1
    assert coerce_label(0) == 0


def test_sample_to_dict():
    sample = BagSample(
        sample_id="s1",
        prompt="p",
        response="r",
        label=1,
        temperature=0.7,
        token_features=[TokenFeature(token_id=1, text="a", logprob=-1.2, entropy=0.3)],
        segment_spans=[Segment(segment_id=0, start=0, end=1)],
        metadata={"k": "v"},
    )
    d = sample.to_dict()
    assert d["sample_id"] == "s1"
    assert d["label"] == 1
    assert len(d["token_features"]) == 1
