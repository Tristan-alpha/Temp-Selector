"""Tests for hybrid JSONL + safetensors I/O helpers.  CPU-only."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from utils.dataset_io import (
    hidden_path,
    has_hidden_sidecar,
    read_hidden_offsets,
    split_hidden_sidecar,
    write_hidden_sidecar,
    write_jsonl,
    load_jsonl,
)


def _temp_dataset() -> str:
    """Return path to a temporary .jsonl file (does not create it)."""
    return os.path.join(tempfile.mkdtemp(), "test.jsonl")


def test_hidden_path():
    assert hidden_path("/data/train.jsonl") == "/data/train.jsonl.hidden.safetensors"
    assert hidden_path("relative.jsonl") == "relative.jsonl.hidden.safetensors"


def test_has_hidden_sidecar():
    path = _temp_dataset()
    assert not has_hidden_sidecar(path)
    # Create sidecar
    hs_path = hidden_path(path)
    Path(hs_path).parent.mkdir(parents=True, exist_ok=True)
    Path(hs_path).touch()
    assert has_hidden_sidecar(path)
    os.unlink(hs_path)


def test_read_hidden_offsets():
    assert read_hidden_offsets({}) == (-1, 0)
    assert read_hidden_offsets({"_hidden_offset": 10, "_hidden_count": 256}) == (10, 256)


def test_write_and_read_sidecar():
    path = _temp_dataset()
    t1 = torch.randn(3, 8, dtype=torch.float32)
    t2 = torch.randn(2, 8, dtype=torch.float32)
    write_hidden_sidecar(path, [t1, t2])
    assert has_hidden_sidecar(path)

    import safetensors
    with safetensors.safe_open(hidden_path(path), framework="pt") as f:
        hs = f.get_slice("hidden_states")
        assert hs.get_shape() == [5, 8]
        # Verify first row of t1
        chunk = hs[0:3, :]
        assert torch.allclose(chunk, t1, atol=1e-6)


def test_write_sidecar_with_none_tensors():
    path = _temp_dataset()
    t1 = torch.randn(3, 4, dtype=torch.float32)
    write_hidden_sidecar(path, [None, t1, None])
    assert has_hidden_sidecar(path)
    import safetensors
    with safetensors.safe_open(hidden_path(path), framework="pt") as f:
        hs = f.get_slice("hidden_states")
        assert hs.get_shape() == [3, 4]


def test_write_sidecar_empty():
    path = _temp_dataset()
    write_hidden_sidecar(path, [None, None])
    assert not has_hidden_sidecar(path)


def test_sidecar_with_bf16():
    path = _temp_dataset()
    t1 = torch.randn(2, 16, dtype=torch.bfloat16)
    write_hidden_sidecar(path, [t1])
    import safetensors
    with safetensors.safe_open(hidden_path(path), framework="pt") as f:
        hs = f.get_slice("hidden_states")
        assert hs.get_shape() == [2, 16]
        chunk = hs[:, :]
        assert chunk.dtype == torch.bfloat16
        assert torch.allclose(chunk.float(), t1.float(), atol=1e-2)


def test_split_hidden_sidecar():
    path = _temp_dataset()
    t1 = torch.randn(3, 8, dtype=torch.float32)
    t2 = torch.randn(2, 8, dtype=torch.float32)
    t3 = torch.randn(4, 8, dtype=torch.float32)
    write_hidden_sidecar(path, [t1, t2, t3])

    # Create JSONL rows with offsets
    rows = [
        {"sample_id": "a", "_hidden_offset": 0, "_hidden_count": 3},
        {"sample_id": "b", "_hidden_offset": 3, "_hidden_count": 2},
        {"sample_id": "c", "_hidden_offset": 5, "_hidden_count": 4},
    ]
    split_path = _temp_dataset()
    split_hidden_sidecar(path, [(split_path, rows[:2])])

    import safetensors
    with safetensors.safe_open(hidden_path(split_path), framework="pt") as f:
        hs = f.get_slice("hidden_states")
        assert hs.get_shape() == [5, 8]  # rows a+b
        assert torch.allclose(hs[0:3, :], t1, atol=1e-6)
        assert torch.allclose(hs[3:5, :], t2, atol=1e-6)


def test_split_sidecar_no_source():
    """split_hidden_sidecar is a no-op when source has no sidecar."""
    path = _temp_dataset()
    rows = [{"sample_id": "x"}]
    split_path = _temp_dataset()
    split_hidden_sidecar(path, [(split_path, rows)])
    assert not has_hidden_sidecar(split_path)


def test_load_write_jsonl_round_trip():
    path = _temp_dataset()
    rows = [{"a": 1, "b": [1, 2, 3]}, {"a": 2, "b": []}]
    write_jsonl(path, rows)
    loaded = load_jsonl(path)
    assert loaded == rows
