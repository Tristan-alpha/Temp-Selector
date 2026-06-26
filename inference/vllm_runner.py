from __future__ import annotations

import logging
import os
import socket
from typing import Any, Dict, List, Optional

# vLLM defaults to fork (envs.py:68) but CUDA requires spawn when used as
# a library.  Set early, before any torch.cuda call — vllm_runner is
# typically imported by training scripts that later touch CUDA.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch

logger = logging.getLogger(__name__)

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


def _load_hidden_states_file(path: str) -> Dict[str, torch.Tensor]:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector as connector,
        )
        load = getattr(connector, "load_hidden_states", None)
        if callable(load):
            return load(path)
    except ImportError:
        pass

    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as f:
        return {key: f.get_tensor(key) for key in f.keys()}


def _cleanup_hidden_states_file(path: str) -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1 import (
            example_hidden_states_connector as connector,
        )
        cleanup = getattr(connector, "cleanup_hidden_states", None)
        if callable(cleanup):
            cleanup(path)
            return
    except ImportError:
        pass

    for candidate in (path, f"{path}.lock"):
        try:
            os.remove(candidate)
        except FileNotFoundError:
            pass


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
                 reserve_training_gpu: bool = False,
                 max_batch_size: int | None = None,
                 enforce_eager: bool = False,
                 enable_prefix_caching: bool | None = None,
                 kv_port: int | None = None):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.parallel_size = parallel_size if isinstance(parallel_size, int) else None
        self.gpu_memory_utilization = gpu_memory_utilization
        self.reserve_training_gpu = reserve_training_gpu
        self.max_batch_size = (
            max_batch_size if isinstance(max_batch_size, int) and max_batch_size > 0 else None
        )
        self.enforce_eager = enforce_eager
        self.enable_prefix_caching = enable_prefix_caching
        self.kv_port = kv_port if isinstance(kv_port, int) and kv_port > 0 else None
        self._llm = None
        self._tokenizer = None
        self._hs_tmpdir: Optional[str] = None  # for hidden state extraction

    def _cleanup_hs_tmpdir(self):
        import shutil
        if self._hs_tmpdir and os.path.isdir(self._hs_tmpdir):
            shutil.rmtree(self._hs_tmpdir, ignore_errors=True)

    def reset_prefix_cache(self, reset_connector: bool = True) -> bool:
        """Clear vLLM prefix-cache state between independent rollout batches."""
        if self._llm is None:
            return False
        engine = getattr(self._llm, "llm_engine", None)
        reset = getattr(engine, "reset_prefix_cache", None)
        if not callable(reset):
            return False
        try:
            return bool(reset(
                reset_running_requests=False,
                reset_connector=reset_connector,
            ))
        except TypeError:
            return bool(reset())
        except Exception:
            logger.exception("failed to reset vLLM prefix cache")
            return False

    def _resolve_parallel_size(self) -> int:
        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            raise RuntimeError("No GPUs available for vLLM")

        if self.parallel_size is not None:
            tp = self.parallel_size
        else:
            tp = n_gpus
            if self.reserve_training_gpu:
                tp -= 1

        if self.reserve_training_gpu and self.parallel_size is not None and tp >= n_gpus:
            raise RuntimeError(
                f"parallel_size={tp} leaves no GPU for training reservation "
                f"(total={n_gpus}, reserve_training_gpu=True)"
            )

        if tp <= 0:
            raise RuntimeError(
                f"No GPUs left for vLLM after training reservation "
                f"(total={n_gpus}, reserve_training_gpu=True)"
            )
        return tp

    def _resolve_kv_port(self) -> int:
        if self.kv_port is not None:
            return self.kv_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

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
        return_logprobs: bool = False,
        return_hidden: bool = False,
        return_prompt_hidden: bool = False,
        device: torch.device | None = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        if len(prompts) != len(temperatures):
            raise ValueError("prompts and temperatures must match length")
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError("seeds must match prompts length")

        max_batch = self.max_batch_size
        if max_batch is not None and len(prompts) > max_batch:
            logger.info(
                "generate_with_features micro_batching total=%d max_batch_size=%d",
                len(prompts), max_batch,
            )
            results: List[Dict[str, Any]] = []
            for start in range(0, len(prompts), max_batch):
                end = min(start + max_batch, len(prompts))
                results.extend(self._generate_with_features_once(
                    prompts[start:end],
                    temperatures[start:end],
                    segment_size,
                    top_k=top_k,
                    return_logprobs=return_logprobs,
                    return_hidden=return_hidden,
                    return_prompt_hidden=return_prompt_hidden,
                    device=device,
                    seeds=None if seeds is None else seeds[start:end],
                ))
            return results
        return self._generate_with_features_once(
            prompts, temperatures, segment_size,
            top_k=top_k, return_logprobs=return_logprobs,
            return_hidden=return_hidden,
            return_prompt_hidden=return_prompt_hidden,
            device=device, seeds=seeds,
        )

    def _generate_with_features_once(
        self,
        prompts: List[str],
        temperatures: List[float],
        segment_size: int,
        top_k: int = 4096,
        return_logprobs: bool = False,
        return_hidden: bool = False,
        return_prompt_hidden: bool = False,
        device: torch.device | None = None,
        seeds: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """Generate ``segment_size`` tokens per prompt and return per-token features.

        Single-pass: ``include_output_tokens`` makes the connector export hidden
        states for ``all_token_ids[:-1]`` — covering every response token we need
        logprobs for.  No second ``llm.generate()`` / ``extract_from_ids`` call.

        Returns one dict per prompt with keys: token_ids, tokens, text,
        logprobs (tensor [n_tok, top_k+1] or None), hidden_states
        (tensor or None), finish_reason.
        """
        self._lazy_init()
        from vllm import SamplingParams

        if return_prompt_hidden and self.enable_prefix_caching is not False:
            raise RuntimeError("return_prompt_hidden requires enable_prefix_caching=False")
        need_hs = return_logprobs or return_hidden or return_prompt_hidden

        # ── Single pass: generate + hidden states ──
        params = [SamplingParams(
            temperature=t, max_tokens=segment_size, top_p=1.0, top_k=0,
            seed=None if seeds is None else seeds[i],
            extra_args={"kv_transfer_params": {"include_output_tokens": True}} if need_hs else None,
        ) for i, t in enumerate(temperatures)]
        outputs = self._llm.generate(prompts, params, use_tqdm=False)

        results: List[Dict[str, Any]] = []
        for i, out in enumerate(outputs):
            o0 = out.outputs[0]
            token_ids = o0.token_ids

            # ── Hidden states & logprobs ──
            lp_tensor: torch.Tensor | None = None
            hs_tensor: torch.Tensor | None = None
            prompt_hs_tensor: torch.Tensor | None = None
            hs_path = out.kv_transfer_params.get("hidden_states_path")

            if need_hs and hs_path is not None:
                try:
                    data = _load_hidden_states_file(hs_path)
                finally:
                    if hs_path is not None:
                        try:
                            _cleanup_hidden_states_file(hs_path)
                        except Exception:
                            pass

                hs = data["hidden_states"]              # [seq_len, 1, hidden_dim]

                hs_1d = hs[:, -1, :]                    # [seq_len, hidden_dim]
                # Take the last len(token_ids) hidden states — they always
                # correspond to the newly generated tokens.  Avoids relying
                # on prompt length, which is unreliable when APC evicts
                # cached prefix hidden states.
                if len(token_ids) > 0 and hs_1d.shape[0] >= len(token_ids):
                    resp_hs = hs_1d[-len(token_ids):]
                elif len(token_ids) > 0:
                    saved_tok = data.get("token_ids")
                    logger.warning(
                        "generate_with_features: hs_1d too short for token_ids "
                        "(hs=%d, tok=%d, p_len=%d, saved_tok_len=%s). "
                        "Logprobs will be None for this response.",
                        hs_1d.shape[0], len(token_ids),
                        len(out.prompt_token_ids),
                        saved_tok.shape[0] if saved_tok is not None else "N/A")
                    if return_prompt_hidden:
                        raise RuntimeError("hidden states are too short to extract prompt_hidden")
                    resp_hs = hs_1d[:0]  # empty → logprobs = None
                else:
                    resp_hs = hs_1d[:0]

                cat_device = device if device is not None else torch.device("cpu")
                if return_prompt_hidden and resp_hs.shape[0] > 0:
                    prompt_hs_tensor = resp_hs[0].float().to(cat_device)

                if return_logprobs and resp_hs.shape[0] > 0:
                    t_cpu = torch.tensor(temperatures[i], dtype=torch.float32)
                    tid_tensor = torch.tensor(token_ids, dtype=torch.long)
                    hs_cpu = resp_hs.cpu()

                    CHUNK = 1024
                    lp_chunks: List[torch.Tensor] = []
                    for start in range(0, len(token_ids), CHUNK):
                        end = min(start + CHUNK, len(token_ids))
                        raw = self._llm.apply_model(
                            _LogprobsComputeFn(hs_cpu[start:end],
                                               tid_tensor[start:end],
                                               top_k, t_cpu)
                        )[0]
                        lp_chunks.append(raw)

                    lp_tensor = torch.cat([c.to(cat_device) for c in lp_chunks], dim=0).float()
                if return_hidden and resp_hs.shape[0] > 0:
                    hs_tensor = resp_hs.float().to(cat_device)
            elif return_prompt_hidden:
                raise RuntimeError(
                    "generate_with_features: missing hidden_states_path while return_prompt_hidden=True"
                )

            results.append({
                "token_ids": token_ids,
                "tokens": [self._tokenizer.decode([tid]) if self._tokenizer else ""
                          for tid in token_ids],
                "text": o0.text,
                "finish_reason": getattr(o0, "finish_reason", None),
                "logprobs": lp_tensor,
                "hidden_states": hs_tensor,
                "prompt_hidden": prompt_hs_tensor,
            })
        return results

    def _lazy_init(self) -> None:
        if self._llm is not None:
            return
        # vLLM 0.18 requires this for apply_model callables. The callable used
        # here is defined locally and only receives tensors prepared by this
        # process.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError("vLLM is required for online feature export.") from exc

        tp_size = self._resolve_parallel_size()
        max_model_len = self.max_new_tokens + 2048

        import tempfile, atexit
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(self.model_name_or_path)
        last_layer_id = hf_cfg.num_hidden_layers  # 1-indexed
        self._hs_tmpdir = tempfile.mkdtemp(prefix="vllm_hs_", dir="/dev/shm")
        atexit.register(self._cleanup_hs_tmpdir)
        kv_port = self._resolve_kv_port()

        llm_kwargs: Dict[str, Any] = dict(
            model=self.model_name_or_path,
            tensor_parallel_size=tp_size,
            max_model_len=max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enforce_eager=self.enforce_eager,
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
                "kv_port": kv_port,
                "kv_connector_extra_config": {
                    "shared_storage_path": self._hs_tmpdir,
                },
            },
        )
        if self.enable_prefix_caching is not None:
            llm_kwargs["enable_prefix_caching"] = self.enable_prefix_caching

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
        return_prompt_hidden: bool = False,
        device: torch.device | None = None,
    ) -> Dict[str, List[torch.Tensor]]:
        if len(full_ids) != len(prompt_lens):
            raise ValueError("full_ids and prompt_lens must match length")
        if temperatures is not None and len(temperatures) != len(full_ids):
            raise ValueError("temperatures must match full_ids length")

        max_batch = self.max_batch_size
        if max_batch is not None and len(full_ids) > max_batch:
            logger.info(
                "extract_from_ids micro_batching total=%d max_batch_size=%d",
                len(full_ids), max_batch,
            )
            logprob_results: List[torch.Tensor] = []
            hidden_results: List[torch.Tensor] = []
            prompt_hidden_results: List[torch.Tensor] = []
            for start in range(0, len(full_ids), max_batch):
                end = min(start + max_batch, len(full_ids))
                extracted = self._extract_from_ids_once(
                    full_ids[start:end],
                    prompt_lens[start:end],
                    None if temperatures is None else temperatures[start:end],
                    top_k=top_k,
                    return_logprobs=return_logprobs,
                    return_hidden=return_hidden,
                    return_prompt_hidden=return_prompt_hidden,
                    device=device,
                )
                if return_logprobs:
                    logprob_results.extend(extracted.get("logprobs", []))
                if return_hidden:
                    hidden_results.extend(extracted.get("hidden", []))
                if return_prompt_hidden:
                    prompt_hidden_results.extend(extracted.get("prompt_hidden", []))
            result: Dict[str, List[torch.Tensor]] = {}
            if return_logprobs:
                result["logprobs"] = logprob_results
            if return_hidden:
                result["hidden"] = hidden_results
            if return_prompt_hidden:
                result["prompt_hidden"] = prompt_hidden_results
            return result
        return self._extract_from_ids_once(
            full_ids, prompt_lens, temperatures,
            top_k=top_k, return_logprobs=return_logprobs,
            return_hidden=return_hidden,
            return_prompt_hidden=return_prompt_hidden,
            device=device,
        )

    def _extract_from_ids_once(
        self,
        full_ids: List[List[int]],
        prompt_lens: List[int],
        temperatures: Optional[List[float]] = None,
        top_k: int = 4096,
        return_logprobs: bool = False,
        return_hidden: bool = False,
        return_prompt_hidden: bool = False,
        device: torch.device | None = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Return per-response logprob and/or hidden tensors from pre-tokenized IDs.

        Uses a single ``llm.generate()`` call to get hidden states, then
        optionally computes logprobs via ``apply_model``.  Logprob chunks
        are concatenated on ``device`` (or CPU if ``None``).

        Only supports ``enable_prefix_caching=False`` — APC would skip
        recomputation of cached prompt tokens, causing hidden-states /
        token-ids length mismatches (vLLM issue #44485).
        """
        assert self.enable_prefix_caching is False, (
            "extract_from_ids requires enable_prefix_caching=False. "
            "APC would cause hidden_states length mismatches."
        )
        self._lazy_init()
        from vllm import SamplingParams

        if temperatures is not None:
            params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0, temperature=t)
                      for t in temperatures]
        else:
            params = [SamplingParams(max_tokens=1, top_p=1.0, top_k=0)] * len(full_ids)
        outputs = self._llm.generate(full_ids, params, use_tqdm=False)

        logprob_results: List[torch.Tensor] = []
        hidden_results: List[torch.Tensor] = []
        prompt_hidden_results: List[torch.Tensor] = []
        cat_device = device if device is not None else torch.device("cpu")
        for i, (out, p_len) in enumerate(zip(outputs, prompt_lens)):
            hs_path = out.kv_transfer_params.get("hidden_states_path")
            if hs_path is None:
                if return_prompt_hidden:
                    raise RuntimeError(
                        f"extract_from_ids: no hidden_states_path for sample {i} with return_prompt_hidden=True"
                    )
                logger.warning("extract_from_ids: no hidden_states_path for sample %d (p_len=%d). "
                               "Is extract_hidden_states configured in _lazy_init?", i, p_len)
                if return_logprobs:
                    logprob_results.append(torch.zeros(1, top_k + 1))
                if return_hidden:
                    hidden_results.append(torch.zeros(1, 4096))
                continue
            try:
                data = _load_hidden_states_file(hs_path)
                hs = data["hidden_states"]  # [seq_len, 1, hidden_dim]
            finally:
                try:
                    _cleanup_hidden_states_file(hs_path)
                except Exception:
                    pass

            token_ids = full_ids[i][p_len:]
            n_resp = len(token_ids)
            if n_resp == 0:
                if return_prompt_hidden:
                    raise RuntimeError(
                        f"extract_from_ids: zero response tokens for sample {i} with return_prompt_hidden=True"
                    )
                logger.warning("extract_from_ids: zero response tokens for sample %d (p_len=%d)", i, p_len)
                if return_logprobs:
                    logprob_results.append(torch.zeros(1, top_k + 1))
                if return_hidden:
                    hidden_results.append(torch.zeros(1, 4096))
                continue

            hs_1d = hs[:, -1, :]                               # [seq_len, hidden_dim]
            # The last (n_resp+1) hidden states cover response
            # positions [p_len-1, ..., p_len+R-1]; drop the final
            # entry (it predicts the llm.generate dummy token).
            resp_hs = hs_1d[-(n_resp + 1):-1]                   # [R, hidden_dim]
            if return_prompt_hidden:
                if resp_hs.shape[0] != n_resp:
                    raise RuntimeError(
                        "extract_from_ids: hidden states are too short to extract prompt_hidden "
                        f"for sample {i} (hs={hs_1d.shape[0]}, response={n_resp})"
                    )
                prompt_hidden_results.append(resp_hs[0].float().to(cat_device))

            cat_device = device if device is not None else torch.device("cpu")

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

                lp = torch.cat([c.to(cat_device) for c in lp_chunks], dim=0)
                logprob_results.append(lp.float())               # [R, top_k+1]
            if return_hidden:
                hidden_results.append(resp_hs.float().to(cat_device))

        result: Dict[str, List[torch.Tensor]] = {}
        if return_logprobs:
            result["logprobs"] = logprob_results
        if return_hidden:
            result["hidden"] = hidden_results
        if return_prompt_hidden:
            result["prompt_hidden"] = prompt_hidden_results
        return result
