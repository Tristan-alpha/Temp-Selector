"""Two-pass hidden state extraction using vLLM's speculative decoding trick.

Pass 1: vLLM generates responses normally (fast).
Pass 2: prompt + response is fed back through vLLM prefill with
        ``extract_hidden_states`` speculative config, which captures
        per-token hidden states from the KV cache and saves them to
        safetensors files.  We read those files, slice off the prompt
        portion, and return per-token hidden states for the response.

Usage:
    extractor = VLLMHiddenStateExtractor(model_path, layer_ids, tmp_dir)
    hidden_states = extractor.extract(prompts=["What is x?"],
                                      responses=["x = 3"])
    # hidden_states[0] = torch.Tensor of shape [n_tokens, hidden_size]
"""

from __future__ import annotations

import atexit
import os
import tempfile
from typing import List

import safetensors
import torch


class VLLMHiddenStateExtractor:
    """Two-pass extractor for per-token hidden states of generated text."""

    def __init__(
        self,
        model_name_or_path: str,
        layer_ids: List[int],
        storage_dir: str | None = None,
        tensor_parallel_size: int | str = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.layer_ids = layer_ids
        self._storage_dir = storage_dir or tempfile.mkdtemp(prefix="hs_")
        self._tensor_parallel_size = tensor_parallel_size
        self._gpu_memory_utilization = gpu_memory_utilization
        self._max_model_len = max_model_len
        self._llm = None
        self._tokenizer = None
        self._cleaned = False
        atexit.register(self.cleanup)

    def _lazy_init(self):
        if self._llm is not None:
            return
        from vllm import LLM

        tp = self._tensor_parallel_size
        if isinstance(tp, str) and tp == "auto":
            import os as _os
            visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            tp = max(1, len([d for d in visible.split(",") if d.strip() and d.strip() != "-1"])) if visible else 1
        self._tensor_parallel_size = tp

        self._llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=tp,
            max_model_len=self._max_model_len or 32768,
            gpu_memory_utilization=self._gpu_memory_utilization,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": self.layer_ids,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {
                    "shared_storage_path": self._storage_dir,
                },
            },
        )
        self._tokenizer = self._llm.get_tokenizer()

    def extract(
        self,
        prompts: List[str],
        responses: List[str],
        debug_token_check: bool = False,
    ) -> List[torch.Tensor]:
        """Extract per-token hidden states for each response.

        Returns a list of length len(prompts), where each element is a
        tensor of shape [n_response_tokens, hidden_size] in the model's
        native dtype (typically bf16 for Qwen3-8B).

        When ``debug_token_check=True``, checks whether tokenizing
        ``prompt`` and ``response`` separately and then concatenating
        produces the same token IDs as tokenizing ``prompt+response``
        as one string.  A mismatch means there is a BPE boundary merge
        that could cause hidden-state / token misalignment.
        """
        self._lazy_init()

        full_texts = [p + r for p, r in zip(prompts, responses)]

        # Batch generate with max_tokens=1 — triggers prefill + hidden save
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=1, temperature=0.0)
        outputs = self._llm.generate(full_texts, params)

        hidden_size = self._llm.llm_engine.model_config.hf_config.hidden_size

        tokenizer = self._tokenizer  # local ref for the loop

        results: List[torch.Tensor] = []
        mismatches = 0
        for i, (prompt, _, output) in enumerate(zip(prompts, responses, outputs)):
            prompt_len = len(tokenizer.encode(prompt))
            hs_path = output.kv_transfer_params.get("hidden_states_path")

            vllm_resp_ids = output.prompt_token_ids[prompt_len:] if output.prompt_token_ids else []

            if debug_token_check:
                # Compare: tokenize(prompt) + tokenize(response) vs tokenize(prompt+response)
                prompt_ids = tokenizer.encode(prompt)
                resp_ids = tokenizer.encode(responses[i])
                separate_ids = prompt_ids + resp_ids
                full_ids = tokenizer.encode(prompt + responses[i])
                if separate_ids != full_ids:
                    mismatches += 1
                    diff_at = None
                    for k in range(min(len(separate_ids), len(full_ids))):
                        if separate_ids[k] != full_ids[k]:
                            diff_at = k
                            break
                    print(
                        f"[TOKEN_CHECK] boundary_merge sample={i} "
                        f"prompt_tokens={len(prompt_ids)} resp_tokens={len(resp_ids)} "
                        f"separate_total={len(separate_ids)} full_total={len(full_ids)} "
                        f"first_diff_at={diff_at}"
                    )

            if hs_path is None or not os.path.exists(hs_path):
                # Fallback: return zeros for each response token
                n_tokens = max(len(vllm_resp_ids), 1)
                results.append(torch.zeros(n_tokens, hidden_size, dtype=torch.float32))
                continue

            with safetensors.safe_open(hs_path, framework="pt") as f:
                hidden_states = f.get_tensor("hidden_states")  # [num_tokens, num_heads, head_size]
                # Flatten last two dims: [num_tokens, hidden_size]
                hidden_states = hidden_states.reshape(hidden_states.shape[0], -1)

            # Slice off prompt tokens; keep as tensor (native dtype)
            results.append(hidden_states[prompt_len:])

        if debug_token_check and mismatches > 0:
            print(f"[TOKEN_CHECK] DONE: {mismatches}/{len(prompts)} samples have token mismatches")
        elif debug_token_check:
            print(f"[TOKEN_CHECK] DONE: all {len(prompts)} samples OK — token IDs match")

        return results

    def __del__(self):
        self.cleanup()

    def sleep(self, level: int = 1):
        """Offload model weights to CPU so another LLM can use the GPU."""
        self._lazy_init()
        self._llm.sleep(level=level)

    def wake_up(self):
        """Reload model weights to GPU after :meth:`sleep`."""
        self._lazy_init()
        self._llm.wake_up()

    def cleanup(self):
        """Remove temporary storage directory.  Idempotent — safe to call
        multiple times (including from atexit / __del__ after explicit cleanup)."""
        if self._cleaned:
            return
        self._cleaned = True
        import shutil
        if os.path.exists(self._storage_dir):
            shutil.rmtree(self._storage_dir, ignore_errors=True)
