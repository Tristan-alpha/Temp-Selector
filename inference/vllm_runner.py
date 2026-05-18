from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

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
    topk_logprobs: Optional[List[List[float]]]
    hidden_states: Optional[List[List[float]]]


class _LogprobsComputeFn:
    """Callable for ``llm.apply_model()`` — single chunk, must be picklable."""

    def __init__(self, hidden_states_cpu, token_ids_cpu, k, temperature_cpu=None):
        self.hidden_states_cpu = hidden_states_cpu
        self.token_ids_cpu = token_ids_cpu
        self.k = k
        self.temperature_cpu = temperature_cpu

    def __call__(self, model):
        from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
        dev = next(model.parameters()).device
        h = self.hidden_states_cpu.to(dev, non_blocking=True)
        ids = self.token_ids_cpu.to(dev, non_blocking=True)

        normed = model.model.norm(h)
        logits = model.compute_logits(normed)
        if self.temperature_cpu is not None:
            t = self.temperature_cpu.to(dev, non_blocking=True)
            logits = logits / t
        result = compute_topk_logprobs(logits, self.k, ids)
        return result.logprobs.cpu()


class VLLMFeatureExporter:
    def __init__(self, model_name_or_path: str, max_new_tokens: int = 256,
                 parallel_size: int | None = None,
                 gpu_memory_utilization: float = 0.90,
                 feature_mode: str = "basic",
                 max_logprobs: int = 4096,
                 reserve_training_gpu: bool = False):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.parallel_size = parallel_size if isinstance(parallel_size, int) else None
        self.gpu_memory_utilization = gpu_memory_utilization
        self.feature_mode = feature_mode
        self.max_logprobs = max_logprobs
        self.reserve_training_gpu = reserve_training_gpu
        self._llm = None
        self._tokenizer = None
        self._hs_tmpdir: Optional[str] = None  # for hidden state extraction

    def _cleanup_hs_tmpdir(self):
        import shutil
        if self._hs_tmpdir and os.path.isdir(self._hs_tmpdir):
            shutil.rmtree(self._hs_tmpdir, ignore_errors=True)

    def _resolve_parallel_size(self) -> int:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available for vLLM")

        tp = self.parallel_size if self.parallel_size is not None else n_gpus

        if self.reserve_training_gpu:
            tp -= 1

        if tp <= 0:
            raise RuntimeError(
                f"No GPUs left for vLLM after training reservation "
                f"(total={n_gpus}, reserve_training_gpu=True)"
            )
        return tp

    @property
    def tokenizer(self):
        """The tokenizer from the LLM (lazy-init)."""
        self._lazy_init()
        return self._tokenizer

    def generate_with_features(
        self,
        prompts: List[str],
        temperatures: List[float],
        segment_size: int,
        top_k: int = 4096,
        return_hidden: bool = False,
        n: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate ``segment_size`` tokens per prompt and return per-token features.

        Returns one dict per prompt with keys: token_ids, tokens, text,
        all_texts (list of all ``n`` output texts), logprobs (tensor
        [n_tok, top_k+1] from first output), hidden_states (tensor or None),
        finish_reason.
        """
        self._lazy_init()
        from vllm import SamplingParams
        from safetensors import safe_open

        params = [SamplingParams(n=n, temperature=t, max_tokens=segment_size,
                                  logprobs=top_k, top_p=1.0, top_k=0)
                  for t in temperatures]
        outputs = self._llm.generate(prompts, params, use_tqdm=False)

        results: List[Dict[str, Any]] = []
        for out in outputs:
            o0 = out.outputs[0]
            token_ids = o0.token_ids
            n_tok = len(token_ids)
            tokens = [self._tokenizer.decode([tid]) if self._tokenizer else ""
                      for tid in token_ids]
            finish_reason = getattr(o0, "finish_reason", None)
            all_texts = [o.text for o in out.outputs]

            lp_tensor: Optional[torch.Tensor] = None
            if o0.logprobs and n_tok > 0:
                lp_rows = []
                for idx, tid in enumerate(token_ids):
                    lp_dict = o0.logprobs[idx] if idx < len(o0.logprobs) else None
                    if lp_dict is None:
                        lp_rows.append(torch.full((top_k + 1,), -20.0))
                        continue
                    sampled = lp_dict.get(tid)
                    sampled_lp = float(sampled.logprob if hasattr(sampled, "logprob") else sampled) \
                        if sampled is not None else -20.0
                    vals = sorted(
                        [float(v.logprob if hasattr(v, "logprob") else v) for v in lp_dict.values()],
                        reverse=True,
                    )[:top_k]
                    row = [sampled_lp] + vals
                    if len(row) < top_k + 1:
                        row += [-20.0] * (top_k + 1 - len(row))
                    lp_rows.append(torch.tensor(row, dtype=torch.float32))
                lp_tensor = torch.stack(lp_rows)  # [n_tok, top_k+1]

            hs_tensor: Optional[torch.Tensor] = None
            if return_hidden:
                hs_path = out.kv_transfer_params.get("hidden_states_path")
                if hs_path is not None:
                    with safe_open(hs_path, "pt") as f:
                        hs = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
                    os.remove(hs_path)
                    hs_1d = hs[:, -1, :]                       # [seq_len, hidden_dim]
                    hs_tensor = hs_1d[-n_tok:].float()         # [n_tok, hidden_dim]

            results.append({
                "token_ids": token_ids,
                "tokens": tokens,
                "text": o0.text,
                "all_texts": all_texts,
                "logprobs": lp_tensor,
                "hidden_states": hs_tensor,
                "finish_reason": finish_reason,
            })
        return results

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vLLM is required for online feature export.") from exc

        tp_size = self._resolve_parallel_size()
        max_model_len = self.max_new_tokens + 2048

        llm_kwargs: Dict[str, Any] = dict(
            model=self.model_name_or_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_logprobs=self.max_logprobs,
        )

        need_hidden = self.feature_mode != "basic"

        if need_hidden:
            import tempfile, atexit
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(self.model_name_or_path)
            # eagle_aux_hidden_state_layer_ids is 1-indexed; last layer = num_layers
            last_layer_id = hf_cfg.num_hidden_layers
            self._hs_tmpdir = tempfile.mkdtemp(prefix="vllm_hs_")
            atexit.register(self._cleanup_hs_tmpdir)
            llm_kwargs.update(
                speculative_config={
                    "method": "extract_hidden_states",
                    "num_speculative_tokens": 1,
                    "draft_model_config": {
                        "hf_config": {
                            "eagle_aux_hidden_state_layer_ids": [last_layer_id],
                        }
                    },
                },
                kv_transfer_config={
                    "kv_connector": "ExampleHiddenStatesConnector",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "shared_storage_path": self._hs_tmpdir,
                    },
                },
            )

        self._llm = LLM(**llm_kwargs)
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

    def _to_generation_output(self, out: Any, top_k_logprobs: int) -> GenerationOutput:
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
                vals = sorted(vals, reverse=True)[:top_k_logprobs]
                topk.append(vals)
            else:
                logprobs.append(-20.0)
                topk.append([-20.0] * top_k_logprobs)

        return GenerationOutput(
            text=out.text,
            tokens=tokens,
            token_ids=token_ids,
            logprobs=logprobs,
            topk_logprobs=topk,
            hidden_states=None,
        )

    # ------------------------------------------------------------------
    # Extraction (pre-tokenized, for MIL training collate_fn)
    # ------------------------------------------------------------------

    def extract_from_ids(
        self,
        full_ids: List[List[int]],
        prompt_lens: List[int],
        temperatures: Optional[List[float]] = None,
        top_k: int = 4096,
        return_logprobs: bool = False,
        return_hidden: bool = False,
        device: torch.device | None = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Return per-response logprob and/or hidden tensors from pre-tokenized IDs.

        Uses a single ``llm.generate()`` call to get hidden states, then
        optionally computes logprobs via ``apply_model``.  Logprob chunks
        are concatenated on ``device`` (or CPU if ``None``).
        """
        self._lazy_init()
        from vllm import SamplingParams
        from safetensors import safe_open

        if temperatures is not None:
            params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0, temperature=t)
                      for t in temperatures]
        else:
            params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0)] * len(full_ids)
        outputs = self._llm.generate(full_ids, params)

        logprob_results: List[torch.Tensor] = []
        hidden_results: List[torch.Tensor] = []
        for i, (out, p_len) in enumerate(zip(outputs, prompt_lens)):
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                logger.warning("extract_from_ids: no hidden_states_path for sample %d (p_len=%d). "
                               "Is extract_hidden_states configured in _lazy_init?", i, p_len)
                if return_logprobs:
                    logprob_results.append(torch.zeros(1, top_k + 1))
                if return_hidden:
                    hidden_results.append(torch.zeros(1, 4096))
                continue
            with safe_open(hs_path, "pt") as f:
                hs = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
            os.remove(hs_path)

            token_ids = full_ids[i][p_len:]
            n_resp = len(token_ids)
            if n_resp == 0:
                logger.warning("extract_from_ids: zero response tokens for sample %d (p_len=%d)", i, p_len)
                if return_logprobs:
                    logprob_results.append(torch.zeros(1, top_k + 1))
                if return_hidden:
                    hidden_results.append(torch.zeros(1, 4096))
                continue

            hs_1d = hs[:, -1, :]                               # [seq_len, hidden_dim]
            resp_hs = hs_1d[max(0, p_len - 1):][:n_resp]       # [R, hidden_dim]

            if return_logprobs:
                t_cpu = (torch.tensor(temperatures[i], dtype=torch.float32)
                         if temperatures is not None else None)
                tid_tensor = torch.tensor(token_ids, dtype=torch.long)
                hs_cpu = resp_hs.cpu()

                CHUNK = 1024
                lp_chunks: List[torch.Tensor] = []
                for start in range(0, n_resp, CHUNK):
                    end = min(start + CHUNK, n_resp)
                    raw = self._llm.apply_model(
                        _LogprobsComputeFn(hs_cpu[start:end],
                                           tid_tensor[start:end],
                                           top_k, t_cpu)
                    )[0]
                    lp_chunks.append(raw)

                cat_device = device if device is not None else torch.device("cpu")
                lp = torch.cat([c.to(cat_device) for c in lp_chunks], dim=0)
                logprob_results.append(lp.float())               # [R, top_k+1]
            if return_hidden:
                hidden_results.append(resp_hs.float())

        result: Dict[str, List[torch.Tensor]] = {}
        if return_logprobs:
            result["logprobs"] = logprob_results
        if return_hidden:
            result["hidden"] = hidden_results
        return result

    def _build_feature_payload(
        self,
        rendered_prompt: str,
        gen: GenerationOutput,
        temperature: float,
    ) -> Dict[str, Any]:
        features: List[TokenFeature] = []
        for i, tid in enumerate(gen.token_ids):
            logprob = gen.logprobs[i]
            dist = gen.topk_logprobs[i] if gen.topk_logprobs else [logprob]
            probs = [math.exp(v) for v in dist]
            z = sum(probs) + 1e-12
            entropy = 0.0
            for p in probs:
                pn = p / z
                entropy -= pn * math.log(max(pn, 1e-12))

            # basic / hidden_states: logprob + entropy only
            # topk_logprobs / all: + top-16 logprob values
            if self.feature_mode in {"topk_logprobs", "all"}:
                tk_logits = dist
            else:
                tk_logits = None

            features.append(
                TokenFeature(
                    token_id=int(tid),
                    text=gen.tokens[i] if i < len(gen.tokens) else "",
                    logprob=float(logprob),
                    entropy=float(entropy),
                    topk_logprobs=tk_logits,
                    hidden=None,  # requires speculative decoding backend
                )
            )

        return {
            "prompt": rendered_prompt,
            "response": gen.text,
            "token_features": features,
            "temperature": temperature,
        }

    def export_token_features_multi_temp(
        self,
        prompts: List[str],
        temperatures: List[float],
        top_k_logprobs: int = 16,
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
        need_logprobs = self.feature_mode in {"topk_logprobs", "all"}
        _req_logprobs = max(1, top_k_logprobs) if need_logprobs else None
        for rp in rendered_prompts:
            for temp in temperatures:
                sp_kwargs: Dict[str, Any] = dict(
                    n=num_votes, temperature=temp,
                    max_tokens=self.max_new_tokens,
                    top_p=1.0, top_k=0,
                )
                if _req_logprobs is not None:
                    sp_kwargs["logprobs"] = _req_logprobs
                all_prompts.append(rp)
                all_params.append(SamplingParams(**sp_kwargs))

        outputs = self._llm.generate(all_prompts, all_params)

        # Parse outputs: each request produces num_votes GenerationOutputs
        gens: List[GenerationOutput] = []
        for req_out in outputs:
            for out in req_out.outputs:
                gens.append(self._to_generation_output(out=out, top_k_logprobs=top_k_logprobs))

        # Build payloads
        payloads: List[Dict[str, Any]] = []
        for i, gen in enumerate(gens):
            prompt_idx = i // (n_temps * num_votes)
            temp_idx = (i // num_votes) % n_temps
            payloads.append(self._build_feature_payload(
                rendered_prompt=rendered_prompts[prompt_idx],
                gen=gen,
                temperature=temperatures[temp_idx],
            ))
        return payloads
