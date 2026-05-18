from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

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
    """Callable for ``llm.apply_model()`` — must be picklable (no closures)."""

    def __init__(self, hidden_states_cpu, token_ids_cpu, k, temperature_cpu=None):
        self.hidden_states_cpu = hidden_states_cpu
        self.token_ids_cpu = token_ids_cpu
        self.k = k
        self.temperature_cpu = temperature_cpu

    def __call__(self, model):
        from vllm.v1.worker.gpu.sample.logprob import compute_topk_logprobs
        import torch as _t
        dev = next(model.parameters()).device
        h = self.hidden_states_cpu.to(dev, non_blocking=True)
        ids = self.token_ids_cpu.to(dev, non_blocking=True)
        n, = h.shape[:1]
        if self.temperature_cpu is not None:
            t = self.temperature_cpu.to(dev, non_blocking=True)

        CHUNK_SIZE = 1024
        lp_chunks, id_chunks = [], []
        for start in range(0, n, CHUNK_SIZE):
            end = start + CHUNK_SIZE
            normed = model.model.norm(h[start:end])
            logits = model.compute_logits(normed)
            if self.temperature_cpu is not None:
                logits = logits / t
            result = compute_topk_logprobs(logits, self.k, ids[start:end])
            lp_chunks.append(result.logprobs.cpu())
            id_chunks.append(result.logprob_token_ids.cpu())

        lp = _t.cat(lp_chunks, dim=0) if len(lp_chunks) > 1 else lp_chunks[0]
        ids_out = _t.cat(id_chunks, dim=0) if len(id_chunks) > 1 else id_chunks[0]
        return _t.stack([lp, ids_out.float()])


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

    def generate_raw(self, prompts, sampling_params):
        """Forward to ``llm.generate()``.  For per-segment PPO generation."""
        self._lazy_init()
        return self._llm.generate(prompts, sampling_params, use_tqdm=False)

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

    def extract_logprobs_from_ids(
        self,
        full_ids: List[List[int]],
        prompt_lens: List[int],
        temperatures: Optional[List[float]] = None,
        top_k: int = 4096,
    ) -> List[torch.Tensor]:
        """Return per-response top-k logprob tensors via apply_model + compute_logits."""
        self._lazy_init()
        from vllm import SamplingParams
        from safetensors import safe_open

        sp = SamplingParams(max_tokens=1, top_p=1.0, top_k=0)
        if temperatures is not None:
            params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0, temperature=t)
                      for t in temperatures]
        else:
            params = [sp] * len(full_ids)
        outputs = self._llm.generate(full_ids, params)

        results: List[torch.Tensor] = []
        for i, (out, p_len) in enumerate(zip(outputs, prompt_lens)):
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                results.append(torch.zeros(1, top_k))
                continue
            with safe_open(hs_path, "pt") as f:
                hs = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
            os.remove(hs_path)
            resp_hs = hs[max(0, p_len - 1):, -1, :]  # [R+1, hidden_dim]

            if resp_hs.shape[0] == 0:
                results.append(torch.zeros(1, top_k))
                continue

            token_ids = torch.tensor(full_ids[i][p_len:], dtype=torch.long)
            resp_hs = resp_hs[: len(token_ids)]  # trim extra hidden state
            t_cpu = (torch.tensor(temperatures[i], dtype=torch.float32)
                     if temperatures is not None else None)
            raw = self._llm.apply_model(
                _LogprobsComputeFn(resp_hs.cpu(), token_ids, top_k, t_cpu)
            )[0]
            results.append(raw[0].float())  # logprobs [R, top_k+1]
        return results

    def extract_hidden_from_ids(
        self,
        full_ids: List[List[int]],
        prompt_lens: List[int],
    ) -> List[torch.Tensor]:
        """Return per-response hidden states via speculative extract_hidden_states."""
        self._lazy_init()
        from vllm import SamplingParams
        from safetensors import safe_open

        outputs = self._llm.generate(
            full_ids,
            [SamplingParams(max_tokens=1, top_p=1.0, top_k=0)] * len(full_ids),
        )

        results: List[torch.Tensor] = []
        for out, p_len in zip(outputs, prompt_lens):
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                results.append(torch.zeros(1, 4096))
                continue
            with safe_open(hs_path, "pt") as f:
                hs = f.get_tensor("hidden_states")  # [seq_len, 1, hidden_dim]
            os.remove(hs_path)
            hs_1d = hs[:, -1, :]                           # [seq_len, hidden_dim]
            resp_hs = hs_1d[max(0, p_len - 1):]             # response portion
            results.append(resp_hs.float())
        return results

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
