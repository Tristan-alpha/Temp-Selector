from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TokenFeature:
    token_id: int
    text: str
    logprob: float
    entropy: float
    topk_logprobs: Optional[List[float]] = None
    hidden: Optional[List[float]] = None


@dataclass
class Segment:
    segment_id: int
    start: int
    end: int


@dataclass
class BagSample:
    sample_id: str                    # unique identifier
    prompt: str                       # original math problem
    response: str                     # LLM-generated answer text
    label: int                        # 0 = correct (negative bag), 1 = error (positive bag)
                                      #   FLIPPED from standard MIL convention — see PIPELINE.md
    temperature: float                # generation temperature used
    token_features: List[TokenFeature]  # per-token features (logprob, entropy, top-k logits)
    metadata: Dict[str, Any]            # vote_id, individual_correct, gold_answer, etc.
    segment_spans: List[Segment] = field(default_factory=list)  # computed by BagDataset at load time

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return payload


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
