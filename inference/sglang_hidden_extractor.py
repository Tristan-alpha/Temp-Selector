"""SGLang per-token hidden state extraction — thin wrapper around Engine.generate.

Usage:
    engine = sglang.Engine(model_path=..., ...)
    extractor = SGLangHiddenStateExtractor(engine)
    hs_tensors = extractor.extract(prompts=["What is x?"], responses=["x = 3"])
    # hs_tensors[i] = torch.Tensor of shape [n_response_tokens, hidden_size]
"""

from __future__ import annotations

from typing import List

import torch


class SGLangHiddenStateExtractor:
    """Extract per-token hidden states from an existing SGLang Engine.

    The engine is provided externally — this class does not create or
    manage the engine lifecycle.  No internal batching; the caller is
    responsible for splitting large datasets into manageable chunks.
    """

    def __init__(self, engine):
        self._engine = engine

    def extract(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[torch.Tensor]:
        """Return per-response hidden state tensors (native dtype).

        Each returned tensor has shape ``[n_response_tokens, hidden_size]``.
        """
        full_texts = [p + r for p, r in zip(prompts, responses)]
        batch_params = [{"max_new_tokens": 1, "temperature": 0.0}] * len(full_texts)

        outputs = self._engine.generate(
            full_texts, batch_params,
            return_hidden_states=True,
        )

        results: List[torch.Tensor] = []
        for prompt, output in zip(prompts, outputs):
            meta = output.get("meta_info", {})
            hs = meta.get("hidden_states")
            prompt_len = len(self._tokenize(prompt))
            if hs is None:
                results.append(torch.zeros(1, self._hidden_size(output), dtype=torch.float32))
                continue
            hs_chunk = hs[prompt_len:]
            if not isinstance(hs_chunk, torch.Tensor):
                hs_chunk = torch.tensor(hs_chunk)
            results.append(hs_chunk)
        return results

    def _tokenize(self, text: str) -> List[int]:
        tokenizer = self._engine.tokenizer_manager.tokenizer
        return tokenizer.encode(text)

    def _hidden_size(self, output: dict) -> int:
        hs = output.get("meta_info", {}).get("hidden_states")
        if hs is not None:
            if isinstance(hs, torch.Tensor) and hs.ndim >= 2:
                return hs.shape[1]
            if hasattr(hs, "__len__") and len(hs) > 0:
                first = hs[0]
                if hasattr(first, "__len__"):
                    return len(first)
        return 4096
