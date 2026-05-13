"""Dataset I/O: hybrid JSONL + safetensors hidden-state sidecar format.

A dataset consists of one or two files:

    dataset.jsonl                  -- JSONL rows (no hidden state vectors)
    dataset.jsonl.hidden.safetensors -- hidden states [total_tokens, hidden_dim]

JSONL rows may have ``_hidden_offset`` and ``_hidden_count`` keys pointing
into the safetensors tensor.  When absent, no hidden states are loaded.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional, Tuple

import numpy as np
import safetensors.torch
import torch

HIDDEN_SUFFIX = ".hidden.safetensors"


def hidden_path(dataset_path: str) -> str:
    """Derive the safetensors sidecar path from the JSONL dataset path."""
    return str(Path(dataset_path)) + HIDDEN_SUFFIX


def has_hidden_sidecar(dataset_path: str) -> bool:
    """Check whether a companion safetensors file exists."""
    return Path(hidden_path(dataset_path)).exists()


def write_hidden_sidecar(
    dataset_path: str,
    tensors: List[Optional[torch.Tensor]],
) -> None:
    """Write hidden states safetensors from per-sample tensors.

    Each element is either a ``torch.Tensor`` of shape ``[n_tokens, hidden_dim]``
    (native dtype, typically bf16) or ``None`` for samples without hidden states.
    Valid tensors are concatenated along axis 0 into a single ``"hidden_states"``
    tensor.
    """
    hpath = hidden_path(dataset_path)
    valid: List[torch.Tensor] = [t for t in tensors if t is not None]
    if not valid:
        Path(hpath).unlink(missing_ok=True)
        return
    stacked = torch.cat(valid, dim=0)
    safetensors.torch.save_file({"hidden_states": stacked}, hpath)


def read_hidden_offsets(row: Dict[str, Any]) -> Tuple[int, int]:
    """Extract ``(_hidden_offset, _hidden_count)`` from a row dict.

    Returns ``(-1, 0)`` if the keys are absent (legacy row).
    """
    offset = row.get("_hidden_offset", -1)
    count = row.get("_hidden_count", 0)
    return int(offset) if offset is not None else -1, int(count) if count is not None else 0


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read all JSONL rows into memory."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows as JSONL (one JSON object per line)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_hidden_sidecar(
    source_dataset_path: str,
    splits: List[Tuple[str, List[Dict[str, Any]]]],
) -> None:
    """Split the hidden sidecar from a source dataset into output splits.

    Reads the source safetensors once via mmap and writes sidecar files
    for each output path, slicing out hidden states for the rows assigned
    to each split.

    Args:
        source_dataset_path: Path to the source ``.jsonl`` (sidecar derived
            from this path).
        splits: List of ``(output_jsonl_path, rows)`` tuples.
    """
    hpath = hidden_path(source_dataset_path)
    if not Path(hpath).exists():
        return

    with safetensors.safe_open(hpath, framework="pt") as f:
        hs = f.get_slice("hidden_states")

        for out_path, rows in splits:
            tensors: List[Optional[torch.Tensor]] = []
            for row in rows:
                offset, count = read_hidden_offsets(row)
                if offset >= 0 and count > 0:
                    # slice creates a copy (not mmap-referencing)
                    tensors.append(hs[offset:offset + count, :].clone())
                else:
                    tensors.append(None)
            write_hidden_sidecar(out_path, tensors)
