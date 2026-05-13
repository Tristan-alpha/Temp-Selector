"""Tests for sample_prefix and row_group_key.  CPU-only."""

from utils.jsonl import sample_prefix, row_group_key


def test_sample_prefix_old_format():
    assert sample_prefix("q1_t0.2") == "q1"


def test_sample_prefix_vote_format():
    assert sample_prefix("q1_t0.2_v0") == "q1"
    assert sample_prefix("q1_t0.2_v2") == "q1"


def test_sample_prefix_multiple_underscores():
    assert sample_prefix("a_b_c_t0.5_v0") == "a_b_c"


def test_sample_prefix_no_temp_marker():
    assert sample_prefix("plain_id") == "plain_id"


def test_sample_prefix_different_temps_same_group():
    assert sample_prefix("q1_t0.2_v0") == sample_prefix("q1_t1.5_v0") == "q1"


def test_group_same_prompt():
    r1 = {"sample_id": "q1_t0.2_v0"}
    r2 = {"sample_id": "q1_t0.4_v1"}
    assert row_group_key(r1, "sample_prefix") == row_group_key(r2, "sample_prefix")


def test_group_different_prompts():
    r1 = {"sample_id": "q1_t0.2_v0"}
    r2 = {"sample_id": "q2_t0.2_v0"}
    assert row_group_key(r1, "sample_prefix") != row_group_key(r2, "sample_prefix")


def test_group_by_none():
    r1 = {"sample_id": "q1_t0.2_v0"}
    r2 = {"sample_id": "q1_t0.2_v1"}
    assert row_group_key(r1, "none") != row_group_key(r2, "none")


def test_group_by_question():
    r1 = {"sample_id": "q1", "question": "What is 2+2?"}
    r2 = {"sample_id": "q2", "question": "What is 2+2?"}
    assert row_group_key(r1, "question") == row_group_key(r2, "question")


def test_group_unknown_mode_raises():
    try:
        row_group_key({"sample_id": "x"}, "invalid")
        assert False, "Should raise"
    except ValueError:
        pass
