"""OpenAI-compatible API backend for online feature export.

Replaces vLLM in Stage 1 (build_dataset) so the pipeline can use
cloud-hosted models (e.g. Bailian / DashScope) instead of a local GPU.
The public interface mirrors VLLMFeatureExporter — downstream code
does not need to change.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from features.schema import TokenFeature


@dataclass
class GenerationOutput:
    text: str
    tokens: List[str]
    token_ids: List[int]
    logprobs: List[float]
    topk_logits: Optional[List[List[float]]]
    hidden_states: Optional[List[List[float]]]


class APIFeatureExporter:
    """OpenAI-compatible chat-completions backend.

    Parameters
    ----------
    model_name_or_path : str
        Model name as known by the API (e.g. ``"qwen3-8b"``).
    max_new_tokens : int
        Max tokens to generate per request.
    base_url : str
        API endpoint, e.g. ``"https://dashscope.aliyuncs.com/compatible-mode/v1"``.
    api_key : str or None
        API key.  If ``None``, reads ``DASHSCOPE_API_KEY`` from the environment.
    max_concurrent : int
        Max in-flight API requests (thread-pool size).
    max_retries : int
        Retries on transient HTTP / rate-limit errors.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_new_tokens: int = 256,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key: Optional[str] = None,
        max_concurrent: int = 16,
        max_retries: int = 3,
    ):
        self.model_name = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError(
                "API key not found. Set DASHSCOPE_API_KEY environment variable "
                "or pass api_key=..."
            )
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    # ------------------------------------------------------------------
    # Public API (same signatures as VLLMFeatureExporter)
    # ------------------------------------------------------------------

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
        """Generate completions for every prompt and extract token features.

        Returns one payload dict per *completion* (len = len(prompts) × num_votes).
        """
        if not prompts:
            return []

        # API supports n ∈ [1, 4] for Qwen3; split into sub-batches if needed.
        api_n_max = 4
        all_gens: List[GenerationOutput] = []

        for batch_start in range(0, num_votes, api_n_max):
            batch_n = min(api_n_max, num_votes - batch_start)
            gens = self._generate_batch(
                prompts=prompts,
                temperature=temperature,
                top_k_logits=top_k_logits,
                n=batch_n,
                system_prompt=system_prompt if use_math_chat_prompt else None,
            )
            all_gens.extend(gens)

        # Build payloads — one per (prompt_idx, vote_idx)
        payloads: List[Dict[str, Any]] = []
        for i, gen in enumerate(all_gens):
            prompt_idx = i // num_votes
            rendered_prompt = prompts[prompt_idx]  # kept as-is for metadata
            payloads.append(
                self._build_feature_payload(
                    rendered_prompt=rendered_prompt,
                    gen=gen,
                    feature_mode=feature_mode,
                    temperature=temperature,
                )
            )
        return payloads

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _generate_batch(
        self,
        prompts: List[str],
        temperature: float,
        top_k_logits: int,
        n: int,
        system_prompt: Optional[str],
    ) -> List[GenerationOutput]:
        """Call the API concurrently for all prompts, each with n completions."""
        client = self._get_client()
        # API top_logprobs is capped at 5; our downstream code pads to obs_dim
        api_top_logprobs = min(top_k_logits, 5)

        def _call_one(idx: int, prompt: str) -> List[GenerationOutput]:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            for attempt in range(self.max_retries):
                try:
                    resp = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=self.max_new_tokens,
                        n=n,
                        logprobs=True,
                        top_logprobs=api_top_logprobs,
                        extra_body={"enable_thinking": False},
                    )
                    outs: List[GenerationOutput] = []
                    for choice in resp.choices:
                        outs.append(self._choice_to_output(choice, api_top_logprobs))
                    return outs
                except Exception as exc:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(2 ** attempt)
            return []  # unreachable

        # Concurrent dispatch
        results: Dict[int, List[GenerationOutput]] = {}
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as pool:
            futures = {pool.submit(_call_one, i, p): i for i, p in enumerate(prompts)}
            for fut in as_completed(futures):
                idx = futures[fut]
                results[idx] = fut.result()

        # Flatten in original order
        flat: List[GenerationOutput] = []
        for i in range(len(prompts)):
            flat.extend(results.get(i, []))
        return flat

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _choice_to_output(choice: Any, top_k: int) -> GenerationOutput:
        """Convert an OpenAI chat completion choice to GenerationOutput."""
        text = choice.message.content or ""
        tokens: List[str] = []
        token_ids: List[int] = []
        logprobs: List[float] = []
        all_topk: List[List[float]] = []

        lp_content = getattr(choice, "logprobs", None)
        if lp_content is not None:
            lp_content = lp_content.content  # List[ChatCompletionTokenLogprob]
        if lp_content:
            for i, item in enumerate(lp_content):
                tokens.append(item.token)
                token_ids.append(i)  # API does not expose token ids
                logprobs.append(float(item.logprob))
                if item.top_logprobs:
                    all_topk.append([float(t.logprob) for t in item.top_logprobs])
                else:
                    all_topk.append([float(item.logprob)] * top_k)
        else:
            # Fallback: split text into characters if logprobs unavailable
            for i, ch in enumerate(text):
                tokens.append(ch)
                token_ids.append(i)
                logprobs.append(-20.0)
                all_topk.append([-20.0] * top_k)

        return GenerationOutput(
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            logprobs=logprobs,
            topk_logits=all_topk,
            hidden_states=None,
        )

    # ------------------------------------------------------------------
    # Feature payload (same logic as VLLMFeatureExporter._build_feature_payload)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_feature_payload(
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
            "feature_mode": feature_mode,
            "temperature": temperature,
        }
