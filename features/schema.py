from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Segment:
    segment_id: int
    start: int
    end: int


def coerce_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if value > 0 else 0
    raise ValueError(f"Unsupported label type: {type(value)}")


def clamp_segment(start: int, end: int, n_tokens: int) -> Tuple[int, int]:
    start = max(0, min(start, n_tokens))
    end = max(start, min(end, n_tokens))
    return start, end
