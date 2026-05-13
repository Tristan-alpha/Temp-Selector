from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from features.schema import TokenFeature


DEFAULT_MATH_SYSTEM_PROMPT = (
    "You are a math reasoning assistant.\n"
    "\n"
    "Formatting rules:\n"
    "- Solve the problem step by step.\n"
    "- Each step must be written as a separate paragraph.\n"
    "- Separate every step with exactly two newline characters.\n"
    "- Do not use numbering, bullets, or any markers at the start of a step.\n"
    "- Each step must be a complete sentence that describes one reasoning move.\n"
    "- Inline LaTeX expressions should use $...$.\n"
    "- The final paragraph must include the final boxed answer written as \\boxed{}.\n"
    "- Do not include any explanations, headers, or summaries outside the steps.\n"
    "- The response must end immediately after the final paragraph containing \\boxed{}.\n"
)


@dataclass
class GenerationOutput:
    text: str
    tokens: List[str]
    token_ids: List[int]
    logprobs: List[float]
    topk_logits: Optional[List[List[float]]]
    hidden_states: Optional[List[List[float]]]


class VLLMFeatureExporter:
    def __init__(self, model_name_or_path: str, max_new_tokens: int = 256,
                 tensor_parallel_size: int | str | None = "auto",
                 gpu_memory_utilization: float = 0.90):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None
        self._tokenizer = None

    def _resolve_tensor_parallel_size(self) -> int:
        if isinstance(self.tensor_parallel_size, int):
            return max(1, self.tensor_parallel_size)

        if isinstance(self.tensor_parallel_size, str) and self.tensor_parallel_size.lower() != "auto":
            try:
                return max(1, int(self.tensor_parallel_size))
            except ValueError:
                return 1

        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible:
            devices = [d.strip() for d in visible.split(",") if d.strip() and d.strip() != "-1"]
            if devices:
                return max(1, len(devices))

        try:
            import torch

            return max(1, int(torch.cuda.device_count()))
        except Exception:
            return 1

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vLLM is required for online feature export.") from exc
        tp_size = self._resolve_tensor_parallel_size()
        max_model_len = self.max_new_tokens + 2048  # prompt headroom + output
        self._llm = LLM(model=self.model_name_or_path,
                        tensor_parallel_size=tp_size,
                        max_model_len=max_model_len,
                        gpu_memory_utilization=self.gpu_memory_utilization)
        self._tokenizer = self._llm.get_tokenizer()

    def build_math_messages(self, question: str, system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
        return [
            {
                "role": "system",
                "content": system_prompt or DEFAULT_MATH_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": question,
            },
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

    def generate(self, prompt: str, temperature: float, top_k_logits: int = 16) -> GenerationOutput:
        return self.generate_batch(prompts=[prompt], temperature=temperature, top_k_logits=top_k_logits)[0]

    def _to_generation_output(self, out: Any, top_k_logits: int) -> GenerationOutput:
        token_ids = out.token_ids
        logprobs = []
        topk = []
        tokens = []
        for idx, tid in enumerate(token_ids):
            token_str = ""
            if self._tokenizer is not None:
                try:
                    token_str = self._tokenizer.decode([tid])
                except Exception:
                    token_str = ""
            tokens.append(token_str)
            if out.logprobs and idx < len(out.logprobs) and out.logprobs[idx] is not None:
                lp_dict = out.logprobs[idx]
                lp = lp_dict.get(tid)
                if lp is None:
                    vals = list(lp_dict.values())
                    lp = max(vals) if vals else -20.0
                logprobs.append(float(lp.logprob if hasattr(lp, "logprob") else lp))
                vals = [float(v.logprob if hasattr(v, "logprob") else v) for v in lp_dict.values()]
                vals = sorted(vals, reverse=True)[:top_k_logits]
                topk.append(vals)
            else:
                logprobs.append(-20.0)
                topk.append([-20.0] * top_k_logits)

        return GenerationOutput(
            text=out.text,
            tokens=tokens,
            token_ids=token_ids,
            logprobs=logprobs,
            topk_logits=topk,
            hidden_states=None,
        )

    def generate_batch(self, prompts: List[str], temperature: float, top_k_logits: int = 16, num_votes: int = 1) -> List[GenerationOutput]:
        if not prompts:
            return []

        self._lazy_init()
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            n=num_votes,
            temperature=temperature,
            max_tokens=self.max_new_tokens,
            logprobs=max(1, top_k_logits),
        )
        outputs = self._llm.generate(prompts, sampling_params)

        parsed: List[GenerationOutput] = []
        for req_out in outputs:
            for out in req_out.outputs:
                parsed.append(self._to_generation_output(out=out, top_k_logits=top_k_logits))
        return parsed

    def _build_feature_payload(
        self,
        rendered_prompt: str,
        gen: GenerationOutput,
        feature_mode: str,
        temperature: float,
    ) -> Dict[str, Any]:
        features: List[TokenFeature] = []
        for i, tid in enumerate(gen.token_ids):
            logprob = gen.logprobs[i]
            dist = gen.topk_logits[i] if gen.topk_logits else [logprob]
            probs = [math.exp(v) for v in dist]
            z = sum(probs) + 1e-12
            entropy = 0.0
            for p in probs:
                pn = p / z
                entropy -= pn * math.log(max(pn, 1e-12))

            # basic / hidden_states: logprob + entropy only
            # topk_logits / all: + top-16 logprob values
            if feature_mode in {"topk_logits", "all"}:
                tk_logits = dist
            else:
                tk_logits = None

            features.append(
                TokenFeature(
                    token_id=int(tid),
                    text=gen.tokens[i] if i < len(gen.tokens) else "",
                    logprob=float(logprob),
                    entropy=float(entropy),
                    topk_logits=tk_logits,
                    hidden=None,  # requires speculative decoding backend
                )
            )

        return {
            "prompt": rendered_prompt,
            "response": gen.text,
            "token_features": features,
            "temperature": temperature,
        }

    # ------------------------------------------------------------------
    # Multi-temperature batch (APC-optimised)
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
        """Generate completions for *all temperatures* in a single vLLM call.

        Prompts are interleaved so that the same prompt appears consecutively
        for every temperature — APC shares the prompt KV-cache across all
        temperature variants automatically.

        Returns one payload per (prompt, temperature, vote).
        Ordering: prompt0@T0 (×num_votes), prompt0@T1 (×num_votes), ...,
                  prompt1@T0 (×num_votes), ...
        """
        if not prompts or not temperatures:
            return []

        self._lazy_init()
        from vllm import SamplingParams

        n_temps = len(temperatures)

        # Render prompts *once* per input prompt
        rendered_prompts = list(prompts)
        if use_math_chat_prompt:
            rendered_prompts = [
                self.render_messages(self.build_math_messages(question=p, system_prompt=system_prompt))
                for p in prompts
            ]

        # Interleave: prompt0@T0, prompt0@T1, ..., prompt0@Tk, prompt1@T0, ...
        all_prompts: List[str] = []
        all_params: List[SamplingParams] = []
        for rp in rendered_prompts:
            for temp in temperatures:
                all_prompts.append(rp)
                all_params.append(SamplingParams(
                    n=num_votes,
                    temperature=temp,
                    max_tokens=self.max_new_tokens,
                    logprobs=max(1, top_k_logits),
                ))

        outputs = self._llm.generate(all_prompts, all_params)

        # Parse outputs: each request produces num_votes GenerationOutputs
        gens: List[GenerationOutput] = []
        for req_out in outputs:
            for out in req_out.outputs:
                gens.append(self._to_generation_output(out=out, top_k_logits=top_k_logits))

        # Build payloads
        payloads: List[Dict[str, Any]] = []
        for i, gen in enumerate(gens):
            prompt_idx = i // (n_temps * num_votes)
            temp_idx = (i // num_votes) % n_temps
            payloads.append(self._build_feature_payload(
                rendered_prompt=rendered_prompts[prompt_idx],
                gen=gen,
                feature_mode=feature_mode,
                temperature=temperatures[temp_idx],
            ))
        return payloads

    # ------------------------------------------------------------------
    # Single-temperature (legacy, kept for API backend compatibility)
    # ------------------------------------------------------------------

    def export_token_features(
        self,
        prompt: str,
        temperature: float,
        feature_mode: str,
        top_k_logits: int = 16,
        use_math_chat_prompt: bool = False,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.export_token_features_batch(
            prompts=[prompt],
            temperature=temperature,
            feature_mode=feature_mode,
            top_k_logits=top_k_logits,
            use_math_chat_prompt=use_math_chat_prompt,
            system_prompt=system_prompt,
        )[0]

    def export_token_features_batch(
        self,
        prompts: List[str],
        temperature: float,
        feature_mode: str,
        top_k_logits: int = 16,
        use_math_chat_prompt: bool = False,
        system_prompt: Optional[str] = None,
        num_votes: int = 1,
    ) -> List[Dict[str, Any]]:
        if not prompts:
            return []

        rendered_prompts = list(prompts)
        if use_math_chat_prompt:
            rendered_prompts = [
                self.render_messages(self.build_math_messages(question=p, system_prompt=system_prompt)) for p in prompts
            ]

        gens = self.generate_batch(prompts=rendered_prompts, temperature=temperature, top_k_logits=top_k_logits, num_votes=num_votes)
        payloads = []
        for i, gen in enumerate(gens):
            prompt_idx = i // num_votes
            rendered_prompt = rendered_prompts[prompt_idx]
            payloads.append(
                self._build_feature_payload(
                    rendered_prompt=rendered_prompt,
                    gen=gen,
                    feature_mode=feature_mode,
                    temperature=temperature,
                )
            )
        return payloads
