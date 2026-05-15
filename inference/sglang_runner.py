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


class SGLangFeatureExporter:
    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        tensor_parallel_size: int | str = "auto",
        gpu_memory_utilization: float = 0.90,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._engine = None
        self._tokenizer = None
        self._resolve_device_count()

    def _resolve_device_count(self):
        tp = self.tensor_parallel_size
        if isinstance(tp, int):
            self._tp = max(1, tp)
            return
        if isinstance(tp, str) and tp == "auto":
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            self._tp = max(1, len([d for d in visible.split(",") if d.strip() and d.strip() != "-1"])) if visible else 1
            return
        self._tp = max(1, int(tp))

    def _lazy_init(self):
        if self._engine is not None:
            return
        from sglang import Engine

        self._engine = Engine(
            model_path=self.model_name_or_path,
            tp_size=self._tp,
            mem_fraction_static=self.gpu_memory_utilization,
            context_length=self.max_new_tokens + 2048,  # prompt headroom + output
            schedule_conservativeness=0.3,  # aggressive batching → higher throughput
            enable_mixed_chunk=True,         # overlap prefill + decode in same batch
            random_seed=42,
            log_level="info",
            enable_return_hidden_states=True,
        )
        self._tokenizer = self._engine.tokenizer_manager.tokenizer

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
        feature_mode: str = "basic",
        top_k_logits: int = 16,
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

        need_hidden = feature_mode in {"hidden_states", "all"}
        need_logprobs = feature_mode in {"topk_logits", "all"}
        top_k = top_k_logits if need_logprobs else 1

        n_temps = len(temperatures)
        n_prompts = len(rendered_prompts)
        total = n_prompts * n_temps
        logger = __import__('logging').getLogger(__name__)
        logger.info("multi_temp start n_prompts=%d n_temps=%d total_requests=%d",
                      n_prompts, n_temps, total)

        # Interleave: prompt0@T0, prompt0@T1, ..., prompt0@Tk, prompt1@T0, ...
        # Same ordering as vLLM for radix cache prefix sharing.
        all_prompts: List[str] = []
        all_params: List[Dict[str, Any]] = []
        for rp in rendered_prompts:
            for temp in temperatures:
                all_prompts.append(rp)
                all_params.append({"max_new_tokens": self.max_new_tokens, "temperature": temp, "n": num_votes})

        outputs = self._engine.generate(
            all_prompts, all_params,
            return_logprob=True,
            top_logprobs_num=top_k,
            return_hidden_states=need_hidden,
        )

        # Parse: SGLang expands `n=num_votes` into flat list.
        # n_reqs original requests → n_reqs × num_votes outputs.
        n_reqs = len(all_prompts)
        payloads: List[Dict[str, Any]] = []
        for req_idx in range(n_reqs):
            prompt_idx = req_idx // n_temps
            temp_idx = req_idx % n_temps
            for v_idx in range(num_votes):
                batch_output = outputs[req_idx * num_votes + v_idx]
                payload = self._build_feature_payload(
                    rendered_prompt=rendered_prompts[prompt_idx],
                    output=batch_output,
                    feature_mode=feature_mode,
                    temperature=temperatures[temp_idx],
                    vote_idx=0,   # each output is a single vote already
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
        feature_mode: str = "basic",
        top_k_logits: int = 16,
        use_math_chat_prompt: bool = False,
        system_prompt: Optional[str] = None,
        num_votes: int = 1,
    ) -> List[Dict[str, Any]]:
        return self.export_token_features_multi_temp(
            prompts=prompts,
            temperatures=[temperature],
            feature_mode=feature_mode,
            top_k_logits=top_k_logits,
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
        feature_mode: str,
        temperature: float,
        vote_idx: int = 0,
        num_votes: int = 1,
    ) -> Dict[str, Any]:
        meta = output.get("meta_info", {})
        output_ids = output.get("output_ids", [])

        # SGLang stores per-token logprobs as flat lists
        logprob_vals = meta.get("output_token_logprobs_val", [])
        top_logprob_vals = meta.get("output_top_logprobs_val", [])
        top_logprob_idx = meta.get("output_top_logprobs_idx", [])

        # For multi-vote, SGLang interleaves outputs. Extract the right vote.
        n_per_vote = len(output_ids) // num_votes if num_votes > 0 else len(output_ids)
        if num_votes > 1:
            start_idx = vote_idx * n_per_vote
            end_idx = start_idx + n_per_vote
        else:
            start_idx = 0
            end_idx = len(output_ids)

        vote_ids = output_ids[start_idx:end_idx]

        # Logprob values are per-token; top-k values are grouped
        top_k_num = len(top_logprob_vals) // max(1, len(output_ids)) if output_ids else 0

        features: List[TokenFeature] = []
        for pos, tid in enumerate(vote_ids):
            global_pos = start_idx + pos
            logprob = logprob_vals[global_pos] if global_pos < len(logprob_vals) else -20.0

            # Extract top-k logprobs for this token
            topk_list: Optional[List[float]] = None
            if top_k_num > 0 and feature_mode in {"topk_logits", "all"}:
                topk_start = global_pos * top_k_num
                topk_end = topk_start + top_k_num
                topk_list = top_logprob_vals[topk_start:topk_end]

            # Compute entropy from top-k probs
            probs = [math.exp(v) for v in (topk_list or [logprob])]
            z = sum(probs) + 1e-12
            entropy = 0.0
            for p in probs:
                pn = p / z
                entropy -= pn * math.log(max(pn, 1e-12))

            token_text = ""
            if self._tokenizer is not None:
                try:
                    token_text = self._tokenizer.decode([tid])
                except Exception:
                    token_text = ""

            features.append(TokenFeature(
                token_id=int(tid),
                text=token_text,
                logprob=float(logprob),
                entropy=float(entropy),
                topk_logits=topk_list,
            ))

        response_text = output.get("text", "")
        return {
            "prompt": rendered_prompt,
            "response": response_text,
            "token_features": features,
        }

    # ------------------------------------------------------------------
    # Per-sample hidden state extraction (mirrors vLLM extractor interface)
    # ------------------------------------------------------------------

    def extract_hidden_states(
        self,
        prompts: List[str],
        responses: List[str],
    ) -> List[torch.Tensor]:
        """Return per-response hidden state tensors (native dtype).

        Uses the existing engine — no second instance needed.
        """
        self._lazy_init()

        results: List[torch.Tensor] = []
        for prompt, response in zip(prompts, responses):
            full_text = prompt + response
            prompt_len = len(self._tokenizer.encode(prompt))
            output = self._engine.generate(
                full_text,
                {"max_new_tokens": 1, "temperature": 0.0},
                return_hidden_states=True,
            )
            if isinstance(output, list):
                output = output[0]
            hs = output.get("meta_info", {}).get("hidden_states")
            if hs is None:
                output_ids = output.get("output_ids", [])
                n_tokens = prompt_len + len(output_ids)
                results.append(torch.zeros(n_tokens, self._hidden_size(), dtype=torch.float32))
                continue

            # Slice off prompt tokens; convert to tensor if SGLang returns a list
            hs_chunk = hs[prompt_len:]
            if not isinstance(hs_chunk, torch.Tensor):
                hs_chunk = torch.tensor(hs_chunk)
            results.append(hs_chunk)
        return results

    def _hidden_size(self) -> int:
        """Return the model's hidden size."""
        self._lazy_init()
        if hasattr(self._engine, "model_config"):
            return self._engine.model_config.hidden_size
        return 4096
