"""SGLang feature exporter — single-engine generation + hidden state extraction.

SGLang natively supports ``return_hidden_states=True`` so hidden state
extraction happens inline during generation.  No second engine instance
or speculative decoding trick is needed.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import torch

from features.schema import TokenFeature

DEFAULT_MATH_SYSTEM_PROMPT = (
    "You are a math reasoning assistant.\n\n"
    "Formatting rules:\n"
    "- Solve the problem step by step.\n"
    "- Each step must be written as a separate paragraph.\n"
    "- Separate every step with exactly two newline characters.\n"
    "- Do not use numbering, bullets, or any markers at the start of a step.\n"
    "- Each step must be a complete sentence that describes one reasoning move.\n"
    "- Inline LaTeX expressions should use $...$.\n"
    "- The final paragraph must include the final boxed answer written as \\boxed{}.\n"
    "- Do not include any explanations, headers, or summaries outside the steps.\n"
    "- The response must end immediately after the final paragraph containing \\boxed{}."
)


class SGLangRunner:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        parallel_size: int | str = "auto",
        gpu_memory_utilization: float = 0.90,
        feature_mode: str = "basic",
        log_level: str = "warn",
        engine_preset: str = "decode",
        base_gpu_id: int = 0,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.parallel_size = parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.feature_mode = feature_mode
        self.log_level = log_level
        self.engine_preset = engine_preset
        self.base_gpu_id = base_gpu_id
        self._engine = None
        self._tokenizer = None
        self._resolve_device_count()

    def _resolve_device_count(self):
        tp = self.parallel_size
        if isinstance(tp, int):
            self._tp = max(1, tp)
            return
        if isinstance(tp, str) and tp == "auto":
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            n = max(1, len([d for d in visible.split(",") if d.strip() and d.strip() != "-1"])) if visible else 1
            self._tp = max(1, n - self.base_gpu_id)
            return
        self._tp = max(1, int(tp))

    @property
    def tokenizer(self):
        """The tokenizer from the engine."""
        self._lazy_init()
        return self._tokenizer

    def generate_raw(self, prompt, sampling_params,
                     return_logprob=False, top_logprobs_num=None,
                     return_hidden_states=False):
        """Forward to ``engine.generate()``.  For per-segment PPO generation."""
        self._lazy_init()
        return self._engine.generate(
            prompt, sampling_params,
            return_logprob=return_logprob,
            top_logprobs_num=top_logprobs_num,
            return_hidden_states=return_hidden_states,
        )

    def _lazy_init(self):
        if self._engine is not None:
            return
        from sglang import Engine

        # Shared args for both presets
        engine_kwargs: Dict[str, Any] = dict(
            model_path=self.model_name_or_path,
            base_gpu_id=self.base_gpu_id,
            dp_size=self._tp,
            tp_size=1,
            mem_fraction_static=self.gpu_memory_utilization,
            context_length=self.max_new_tokens + 2048,  # prompt headroom + output
            schedule_policy="lpm",  # APC: prompts share common question prefix
            pre_warm_nccl=True,
            enable_tokenizer_batch_encode=True,
            random_seed=42,
            log_level=self.log_level,
            enable_return_hidden_states=self.feature_mode in {"hidden_states", "all"},
            stream_interval=32,
        )

        if self.engine_preset == "prefill":
            # Prefill-heavy: feature extraction with max_new_tokens=1.
            # Single decode step → no need for CUDA graphs, mixed chunk,
            # or multi-step decode batching.
            # Higher conservativeness to avoid OOM on large prefill batches.
            engine_kwargs.update(
                max_running_requests=32,
                schedule_conservativeness=0.3,
                enable_mixed_chunk=False,
                num_continuous_decode_steps=1,
                max_prefill_tokens=16384,
                # chunked_prefill_size=-1,
                disable_cuda_graph=True,
                disable_piecewise_cuda_graph=True,
                skip_tokenizer_init=True,
            )
        else:
            # Decode-heavy: generation with many decode steps (build_dataset, PPO).
            engine_kwargs.update(
                cuda_graph_max_bs=512,
                max_running_requests=512,
                schedule_conservativeness=0.3,
                enable_mixed_chunk=True,
                num_continuous_decode_steps=8,
            )

        self._engine = Engine(**engine_kwargs)
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    def build_math_messages(self, question: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt or DEFAULT_MATH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

    def render_messages(self, messages: List[Dict[str, str]]) -> str:
        self._lazy_init()
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except Exception:
                pass
        chunks = []
        for m in messages:
            role = m.get("role", "user").upper()
            content = m.get("content", "")
            chunks.append(f"[{role}]\n{content}")
        chunks.append("[ASSISTANT]\n")
        return "\n\n".join(chunks)

    # ------------------------------------------------------------------
    # Multi-temperature batching
    # ------------------------------------------------------------------

    def export_token_features_multi_temp(
        self,
        prompts: List[str],
        temperatures: List[float],
        top_k_logprobs: int = 16,
        use_math_chat_prompt: bool = False,
        system_prompt: Optional[str] = None,
        num_votes: int = 1,
    ) -> List[Dict[str, Any]]:
        if not prompts or not temperatures:
            return []

        self._lazy_init()

        rendered_prompts = list(prompts)
        if use_math_chat_prompt:
            rendered_prompts = [
                self.render_messages(self.build_math_messages(question=p, system_prompt=system_prompt))
                for p in prompts
            ]

        need_hidden = self.feature_mode in {"hidden_states", "all"}
        need_logprobs = self.feature_mode in {"topk_logprobs", "all"}
        top_k = top_k_logprobs if need_logprobs else 1

        n_temps = len(temperatures)
        n_prompts = len(rendered_prompts)
        total = n_prompts * n_temps
        logger = __import__('logging').getLogger(__name__)
        logger.info("multi_temp start n_prompts=%d n_temps=%d total_requests=%d",
                      n_prompts, n_temps, total)

        # Interleave + replicate: prompt0@T0×V, prompt0@T1×V, ..., prompt1@T0×V, ...
        # SGLang warns that n>1 is suboptimal for single batches; replicating
        # requests explicitly with n=1 is faster.
        all_prompts: List[str] = []
        all_params: List[Dict[str, Any]] = []
        for rp in rendered_prompts:
            for temp in temperatures:
                for _ in range(num_votes):
                    all_prompts.append(rp)
                    all_params.append({"max_new_tokens": self.max_new_tokens, "temperature": temp, "n": 1,
                                       "top_p": 1.0, "top_k": -1})

        outputs = self._engine.generate(
            all_prompts, all_params,
            return_logprob=True,
            top_logprobs_num=top_k,
            return_hidden_states=need_hidden,
        )

        # Parse: each output is a single vote (n=1).  Ordering matches the
        # interleave pattern above.
        n_per_temp = num_votes
        payloads: List[Dict[str, Any]] = []
        for req_idx, batch_output in enumerate(outputs):
            prompt_idx = req_idx // (n_temps * n_per_temp)
            temp_idx = (req_idx // n_per_temp) % n_temps
            payload = self._build_feature_payload(
                rendered_prompt=rendered_prompts[prompt_idx],
                output=batch_output,
                temperature=temperatures[temp_idx],
                vote_idx=0,
                num_votes=1,
            )
            payloads.append(payload)

        logger.info("multi_temp done total_outputs=%d", len(payloads))
        return payloads

    # ------------------------------------------------------------------
    # Single-temperature batch (API backend compat)
    # ------------------------------------------------------------------

    def export_token_features_batch(
        self,
        prompts: List[str],
        temperature: float,
        top_k_logprobs: int = 16,
        use_math_chat_prompt: bool = False,
        system_prompt: Optional[str] = None,
        num_votes: int = 1,
    ) -> List[Dict[str, Any]]:
        return self.export_token_features_multi_temp(
            prompts=prompts,
            temperatures=[temperature],
            top_k_logprobs=top_k_logprobs,
            use_math_chat_prompt=use_math_chat_prompt,
            system_prompt=system_prompt,
            num_votes=num_votes,
        )

    # ------------------------------------------------------------------
    # Payload construction
    # ------------------------------------------------------------------

    def _build_feature_payload(
        self,
        rendered_prompt: str,
        output: Dict[str, Any],
        temperature: float,
        vote_idx: int = 0,
        num_votes: int = 1,
    ) -> Dict[str, Any]:
        meta = output.get("meta_info", {})
        output_ids = output.get("output_ids", [])

        # SGLang returns per-token logprobs as lists of (logprob, token_id, text_or_None) tuples
        logprob_tuples = meta.get("output_token_logprobs") or []
        top_logprob_tuples = meta.get("output_top_logprobs") or []

        # For multi-vote, SGLang interleaves outputs. Extract the right vote.
        n_per_vote = len(output_ids) // num_votes if num_votes > 0 else len(output_ids)
        if num_votes > 1:
            start_idx = vote_idx * n_per_vote
        else:
            start_idx = 0

        vote_ids = output_ids[start_idx : start_idx + n_per_vote]

        features: List[TokenFeature] = []
        for pos, tid in enumerate(vote_ids):
            global_pos = start_idx + pos
            lp = logprob_tuples[global_pos][0] if global_pos < len(logprob_tuples) else -20.0

            topk_list: Optional[List[float]] = None
            if self.feature_mode in {"topk_logprobs", "all"} and global_pos < len(top_logprob_tuples):
                topk_list = [t[0] for t in top_logprob_tuples[global_pos]]

            probs = [math.exp(v) for v in (topk_list or [lp])]
            z = sum(probs) + 1e-12
            ent = 0.0
            for p in probs:
                pn = p / z
                ent -= pn * math.log(max(pn, 1e-12))

            token_text = ""
            if self._tokenizer is not None:
                try:
                    token_text = self._tokenizer.decode([tid])
                except Exception:
                    token_text = ""

            features.append(TokenFeature(
                token_id=int(tid),
                text=token_text,
                logprob=float(lp),
                entropy=float(ent),
                topk_logprobs=topk_list,
            ))

        response_text = output.get("text", "")
        return {
            "prompt": rendered_prompt,
            "response": response_text,
            "token_features": features,
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Hidden state & logprob extraction
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Extraction (pre-tokenized, skip SGLang internal tokenization)
    # ------------------------------------------------------------------

    def extract_hidden_from_ids(
        self,
        full_ids: List[List[int]],
        prompt_lens: List[int],
    ) -> List[torch.Tensor]:
        """Return per-response hidden states using pre-tokenized IDs.

        SGLang returns hidden states as a list of per-token tensors
        (``List[Tensor]``).  We slice at ``[prompt_len - 1:]`` because
        ``hs[i]`` produces the logits for ``token[i+1]``.
        """
        self._lazy_init()

        outputs = self._engine.generate(
            input_ids=full_ids,
            sampling_params=[{"max_new_tokens": 0, "temperature": 1.0,
                              "top_p": 1.0, "top_k": -1}] * len(full_ids),
            return_hidden_states=True,
        )

        results: List[torch.Tensor] = []
        for p_len, output in zip(prompt_lens, outputs):
            hs_list = output.get("meta_info", {}).get("hidden_states")
            if not hs_list:
                results.append(torch.zeros(1, 4096, dtype=torch.float32))
                continue
            # hs_list is List[Tensor] — one per token position
            chunk = hs_list[max(0, p_len - 1):]
            if not chunk:
                results.append(torch.zeros(1, 4096, dtype=torch.float32))
                continue
            stacked = torch.stack([t.unsqueeze(0) if t.dim() == 1 else t for t in chunk])
            results.append(stacked)
        return results

    def extract_logprobs_from_ids(
        self,
        full_ids: List[List[int]],
        prompt_lens: List[int],
        temperatures: Optional[List[float]] = None,
        top_k: int = 4096,
    ) -> List[torch.Tensor]:
        """Return per-response top-k logprob tensors using pre-tokenized IDs.

        ``logprob_start_len`` is set per-sample to ``prompt_lens[i]`` so
        SGLang only returns logprobs for response tokens — no post-hoc
        slicing needed.
        """
        self._lazy_init()

        batch_params = [{"max_new_tokens": 0, "temperature": 1.0,
                         "top_p": 1.0, "top_k": -1}] * len(full_ids)
        if temperatures is not None:
            batch_params = [{"max_new_tokens": 0, "temperature": t,
                             "top_p": 1.0, "top_k": -1} for t in temperatures]
        outputs = self._engine.generate(
            input_ids=full_ids,
            sampling_params=batch_params,
            return_logprob=True,
            logprob_start_len=prompt_lens,
            top_logprobs_num=top_k,
        )

        results: List[torch.Tensor] = []
        for output in outputs:
            meta = output.get("meta_info", {})
            tp_list = meta.get("input_top_logprobs", [])
            if not tp_list or all(x is None for x in tp_list):
                results.append(torch.zeros(1, top_k))
                continue
            lp_tensor = torch.tensor([[t[0] for t in (row or [])] for row in tp_list])
            results.append(lp_tensor)
        return results

