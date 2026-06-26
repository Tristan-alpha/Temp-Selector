"""Tests for collate_fn, segment cache, and feature construction.  CPU-only."""

from __future__ import annotations

import torch

from mil.utils import (make_collate_fn, make_cached_collate_fn, SegmentCacheDataset,
                       _build_cache_path, _load_or_build_segment_cache)
from features.segmenter import build_segment_obs_from_lp


# ═══ helpers ═══

def _make_row(tokens, _full_ids=None, _prompt_len=0, temperature=0.7):
    row = {
        "token_ids": [42] * len(tokens),
        "tokens": tokens,
        "text": " ".join(tokens),
        "temperature": temperature,
        "individual_label": 0,
        "_full_ids": _full_ids or ([0] * _prompt_len + [42] * len(tokens)),
        "_prompt_len": _prompt_len,
    }
    return row


# ═══ make_collate_fn / collate_fn ═══

def test_collate_uniform():
    collate = make_collate_fn(feature_mode="topk_logprobs", instance_dim=4098,
                              segment_mode="fixed_window", segment_size=64)
    r1 = _make_row(["hello", "world"])
    r2 = _make_row(["foo", "bar"])
    batch = collate([r1, r2])
    assert "instances" in batch
    assert "mask" in batch
    assert "label" in batch


def test_collate_empty_batch():
    collate = make_collate_fn(feature_mode="topk_logprobs", instance_dim=4098,
                              segment_mode="fixed_window", segment_size=64)
    batch = collate([])
    assert "instances" in batch
    assert batch["instances"].numel() == 0
    assert "mask" in batch
    assert "label" in batch


# ═══ make_cached_collate_fn ═══

def test_cached_collate_fn_basic():
    cache = [
        {"instances": torch.randn(3, 4098), "label": 0.0, "temp_idx": 2},
        {"instances": torch.randn(5, 4098), "label": 1.0, "temp_idx": 5},
        {"instances": torch.randn(2, 4098), "label": 0.0, "temp_idx": 3},
    ]
    collate = make_cached_collate_fn(cache, instance_dim=4098)
    batch = collate([0, 2])
    assert batch["instances"].shape == (2, 3, 4098)  # max_k=3
    assert batch["mask"].shape == (2, 3)
    assert batch["mask"][0, :3].sum() == 3  # row 0 has 3 valid


def test_cached_collate_fn_single_row():
    cache = [{"instances": torch.randn(4, 4098), "label": 1.0, "temp_idx": 1}]
    collate = make_cached_collate_fn(cache, instance_dim=4098)
    batch = collate([0])
    assert batch["instances"].shape == (1, 4, 4098)


def test_segment_cache_dataset():
    """SegmentCacheDataset + make_cached_collate_fn: indices → cache lookup."""
    cache = [
        {"instances": torch.randn(3, 4098), "label": 0.0, "temp_idx": 2},
        {"instances": torch.randn(5, 4098), "label": 1.0, "temp_idx": 5},
        {"instances": torch.randn(2, 4098), "label": 0.0, "temp_idx": 3},
    ]
    ds = SegmentCacheDataset(len(cache))
    assert len(ds) == 3
    assert ds[1] == 1  # SegmentCacheDataset returns the index itself

    collate = make_cached_collate_fn(cache, instance_dim=4098)
    batch = collate([0, 2])
    assert batch["instances"].shape == (2, 3, 4098)
    assert batch["mask"][0, :3].sum() == 3


# ═══ concat pooling end-to-end ═══

def test_build_segment_obs_concat():
    """build_segment_obs_from_lp with concat pooling produces flat per-segment vectors."""
    lp_tensor = torch.randn(3, 5) * 0.1
    lp_tensor[:, 0] = -0.5
    tokens = ["A"] * 3
    text = "A A A"
    obs = build_segment_obs_from_lp(
        lp_tensor, tokens, text, segment_size=3, obs_dim=4,
        segment_mode="fixed_window", pooling_mode="concat",
    )
    assert obs.shape == (1, 12)  # 1 segment × (3 tokens × 4 dims)


def test_build_segment_obs_concat_vs_mean_shape():
    """concat returns [K, seg*obs_dim] while mean returns [K, obs_dim]."""
    lp = torch.randn(4, 5) * 0.1
    lp[:, 0] = -0.5
    tokens = ["a"] * 4
    text = "a a a a"
    concat_obs = build_segment_obs_from_lp(
        lp, tokens, text, segment_size=4, obs_dim=8,
        segment_mode="fixed_window", pooling_mode="concat",
    )
    mean_obs = build_segment_obs_from_lp(
        lp, tokens, text, segment_size=4, obs_dim=8,
        segment_mode="fixed_window", pooling_mode="mean",
    )
    assert concat_obs.shape == (1, 32)  # 1 seg × (4 tok × 8 dim)
    assert mean_obs.shape == (1, 8)


def test_collate_fn_concat_pooling_mode_accepted():
    """make_collate_fn accepts pooling_mode='concat' without error."""
    collate = make_collate_fn(
        feature_mode="topk_logprobs",
        instance_dim=64,
        segment_mode="fixed_window",
        segment_size=64,
        pooling_mode="concat",
    )
    r1 = _make_row(["hello", "world"])
    batch = collate([r1])
    assert "instances" in batch
    assert "mask" in batch


def test_collate_fn_concat_empty_tokens():
    """Concatenate pooling with zero-token rows must not crash on dim mismatch.

    Regression test: before the fix, empty rows got torch.zeros(1, instance_dim)
    but concat pooling produces [1, segment_size * instance_dim].  Mixed batches
    triggered a shape mismatch at assignment time.
    """
    collate = make_collate_fn(
        feature_mode="topk_logprobs",
        instance_dim=64,
        segment_mode="fixed_window",
        segment_size=64,
        pooling_mode="concat",
    )
    # Row with zero tokens
    r_empty = _make_row([""], _full_ids=[0], _prompt_len=1)
    r_empty["token_ids"] = []
    r_empty["tokens"] = []
    r_empty["text"] = ""
    # Normal row
    r_full = _make_row(["hello", "world"])
    # Should not crash with either order of first row
    batch1 = collate([r_empty, r_full])
    assert batch1["instances"].shape[2] == 64 * 64  # segment_size * instance_dim
    batch2 = collate([r_full, r_empty])
    assert batch2["instances"].shape[2] == 64 * 64


def test_collate_fn_concat_all_empty():
    """All rows empty with concat pooling — batch builds successfully."""
    collate = make_collate_fn(
        feature_mode="topk_logprobs",
        instance_dim=64,
        segment_mode="fixed_window",
        segment_size=32,
        pooling_mode="concat",
    )
    r_empty = _make_row([""], _full_ids=[0], _prompt_len=1)
    r_empty["token_ids"] = []
    r_empty["tokens"] = []
    r_empty["text"] = ""
    batch = collate([r_empty, r_empty])
    assert batch["instances"].shape[2] == 32 * 64  # segment_size * instance_dim


# ═══ segment cache helpers ═══

def test_build_cache_path_concat():
    path = _build_cache_path("datasets/cache", "train", "fixed_window",
                             "concat", "topk_logprobs", 64, 64)
    assert path == "datasets/cache/train-fixed_window-concat-topk_logprobs-64-64.safetensors"


def test_build_cache_path_mean():
    path = _build_cache_path("/tmp/cache", "val", "fixed_window",
                             "mean", "hidden_states", 128, 32)
    assert path == "/tmp/cache/val-fixed_window-mean-hidden_states-128-32.safetensors"


def test_build_cache_path_different_splits():
    p1 = _build_cache_path("cache", "train", "fw", "m", "hs", 1, 1)
    p2 = _build_cache_path("cache", "val", "fw", "m", "hs", 1, 1)
    assert p1 != p2


def test_cache_hit_safetensors(tmp_path):
    """safetensors cache hit → loaded without vLLM call."""
    import os, logging
    entries = [
        {"instances": torch.randn(3, 64), "label": 0.0, "temp_idx": 2},
        {"instances": torch.randn(5, 64), "label": 1.0, "temp_idx": 5},
    ]
    cache_path = os.path.join(str(tmp_path), "test.safetensors")
    from mil.utils import _pack_segment_cache
    from safetensors.torch import save_file
    packed = _pack_segment_cache(entries)
    save_file(packed, cache_path)

    dummy_log = logging.getLogger("test_cache_hit_sf")
    dummy_log.addHandler(logging.NullHandler())

    def _fail_collate(rows):
        raise RuntimeError("collate_fn should not be called on cache hit")
    result = _load_or_build_segment_cache([], _fail_collate, cache_path, 1000, dummy_log)
    assert len(result) == 2
    assert result[0]["label"] == 0.0


def test_cache_hit_legacy_pt(tmp_path):
    """Legacy .pt cache hit → loaded via torch.load fallback."""
    import os, logging
    entries = [
        {"instances": torch.randn(3, 64), "label": 0.0, "temp_idx": 2},
        {"instances": torch.randn(5, 64), "label": 1.0, "temp_idx": 5},
    ]
    # Write .pt but pass .safetensors path → checks legacy fallback
    pt_path = os.path.join(str(tmp_path), "test.pt")
    torch.save(entries, pt_path)

    sf_path = os.path.join(str(tmp_path), "test.safetensors")
    dummy_log = logging.getLogger("test_cache_hit_legacy")
    dummy_log.addHandler(logging.NullHandler())

    def _fail_collate(rows):
        raise RuntimeError("collate_fn should not be called on cache hit")
    # Pass .safetensors path, but .pt exists at same stem → fallback
    result = _load_or_build_segment_cache([], _fail_collate, sf_path, 1000, dummy_log)
    assert len(result) == 2
    assert result[0]["label"] == 0.0


def test_cache_miss_saves(tmp_path):
    """Cache file missing → collate_fn called, saved as .safetensors."""
    import os, logging
    cache_path = os.path.join(str(tmp_path), "build.safetensors")

    dummy_log = logging.getLogger("test_cache_miss")
    dummy_log.addHandler(logging.NullHandler())

    def _build_collate(batch_rows):
        inst = torch.randn(len(batch_rows), 4, 64)
        mask = torch.ones(len(batch_rows), 4)
        label = torch.tensor([float(r.get("individual_label", 0)) for r in batch_rows])
        return {"instances": inst, "mask": mask, "label": label, "temp_idx": torch.zeros(len(batch_rows), dtype=torch.long)}

    rows = [{"individual_label": 0}, {"individual_label": 1}]
    result = _load_or_build_segment_cache(rows, _build_collate, cache_path, 1000, dummy_log)
    assert len(result) == 2
    assert os.path.exists(cache_path)

    # Reload from cache
    result2 = _load_or_build_segment_cache(rows, _build_collate, cache_path, 1000, dummy_log)
    assert len(result2) == 2
