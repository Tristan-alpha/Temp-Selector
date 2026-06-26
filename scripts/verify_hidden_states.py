#!/usr/bin/env python3
"""Verify ``include_output_tokens`` — single-pass hidden-state export for vLLM.

Background
----------
``VLLMFeatureExporter`` currently uses two passes:
  Pass 1: ``llm.generate()`` → tokens + speculative-decode hidden states
  Pass 2: ``extract_from_ids(full_ids)`` → logprobs via apply_model

Since vLLM has ``include_output_tokens`` (``example_hidden_states_connector.py``
line 475), passing ``SamplingParams(extra_args={"kv_transfer_params":
{"include_output_tokens": True}})`` makes the connector save hidden states for
**all** tokens in ``all_token_ids[:-1]`` (prompt + all generated tokens except
the very last one, which was never input to a forward pass).

Crucially, the missing last token does NOT matter for logprob extraction:
  hs[prompt_len-1]  → predicts resp[0]
  hs[prompt_len]    → predicts resp[1]
  ...
  hs[prompt_len+resp_len-2] → predicts resp[resp_len-1]

So ``hs[prompt_len-1:][:resp_len]`` gives exactly ``resp_len`` hidden states,
each mapping to one response-token logprob.  Two-pass is unnecessary.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python scripts/verify_hidden_states.py \\
        --model Qwen/Qwen2.5-0.5B-Instruct --max-tokens 64
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path
from typing import List

# vLLM imports before torch — prevents CUDA from initialising before
# vLLM's engine‑core fork (default method: fork; safe when CUDA is cold).
from vllm import LLM, SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.example_hidden_states_connector import (
    load_hidden_states, cleanup_hidden_states,
)
from transformers import AutoConfig
from safetensors.torch import load_file as safetensors_load

import torch


def _count_lockfiles(d: str) -> int:
    return len(list(Path(d).glob("*.lock")))


def _make_llm(model_name: str, max_tokens: int, gpu_mem: float,
              tp_size: int, hs_dir: str) -> LLM:
    hf_cfg = AutoConfig.from_pretrained(model_name)
    last_layer_id = hf_cfg.num_hidden_layers

    return LLM(
        model=model_name,
        tensor_parallel_size=tp_size,
        max_model_len=max_tokens + 2048,
        gpu_memory_utilization=gpu_mem,
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
                "shared_storage_path": hs_dir,
            },
        },
    )


# ═══════════════════════════════════════════════════════════════════
# Test 1 — include_output_tokens: false vs true
# ═══════════════════════════════════════════════════════════════════

def test_include_output_tokens(llm: LLM, prompt: str, max_tokens: int) -> dict:
    """Compare hs coverage with and without include_output_tokens."""
    prompt_len = len(llm.get_tokenizer().encode(prompt))
    results = {}

    for label, extra in [
        ("without", {}),
        ("with",    {"kv_transfer_params": {"include_output_tokens": True}}),
    ]:
        params = SamplingParams(
            temperature=0.7, max_tokens=max_tokens, top_p=1.0, top_k=0,
            extra_args=extra,
        )
        out = llm.generate([prompt], params, use_tqdm=False)[0]
        resp_len = len(out.outputs[0].token_ids)

        hs_path = out.kv_transfer_params.get("hidden_states_path")
        hs_seq = 0
        dim = "-"
        n_resp_hs = 0
        if hs_path:
            data = load_hidden_states(hs_path)
            cleanup_hidden_states(hs_path)
            hs = data["hidden_states"]
            hs_seq = hs.shape[0]
            dim = str(hs.shape[-1])
            hs_1d = hs[:, -1, :]
            n_resp_hs = len(hs_1d[max(0, prompt_len - 1):][:resp_len])

        ok = n_resp_hs >= resp_len
        print(f"  {label:>8}s:  prompt={prompt_len}  resp={resp_len}  "
              f"hs_seq={hs_seq}  dim={dim}  resp_hs_available={n_resp_hs}/{resp_len}"
              f"  {'✅' if ok else '❌'}")
        results[label] = {"ok": ok, "n_resp_hs": n_resp_hs, "resp_len": resp_len}

    return results


# ═══════════════════════════════════════════════════════════════════
# Test 2 — Lock file lifecycle
# ═══════════════════════════════════════════════════════════════════

def test_lockfile_leak(llm: LLM, prompts: List[str], max_tokens: int, hs_dir: str) -> None:
    """Track .lock files through generate → cleanup."""
    n_before = _count_lockfiles(hs_dir)

    params = [SamplingParams(temperature=0.7, max_tokens=max_tokens,
                              top_p=1.0, top_k=0) for _ in prompts]
    outputs = llm.generate(prompts, params, use_tqdm=False)
    n_after_gen = _count_lockfiles(hs_dir)

    for out in outputs:
        p = out.kv_transfer_params.get("hidden_states_path")
        if p is None:
            continue
        load_hidden_states(p)
        cleanup_hidden_states(p)

    n_after = _count_lockfiles(hs_dir)
    leaked = n_after - n_before

    print(f"  lockfiles: {n_before} → {n_after_gen} (after gen)"
          f" → {n_after} (after cleanup)")
    if leaked:
        print(f"  ❌ {leaked} lockfiles leaked")
        for lf in Path(hs_dir).glob("*.lock"):
            print(f"     leftover: {lf}")
            lf.unlink()
    else:
        print("  ✅ no leakage")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify include_output_tokens for single-pass hs export")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    args = parser.parse_args()

    # torch.cuda.device_count() delayed until AFTER LLM() — see import note above.
    hs_dir = tempfile.mkdtemp(prefix="vllm_hs_", dir="/dev/shm")
    print(f"model: {args.model}")
    print(f"hs_dir: {hs_dir}")

    llm = _make_llm(args.model, args.max_tokens, args.gpu_mem, 1, hs_dir)
    print(f"GPUs: {torch.cuda.device_count()}")

    prompt = "What is 2 + 2?  Think step by step."

    print(f"\n{'='*60}")
    print("Test 1 — include_output_tokens: False vs True")
    print("{'='*60}")
    results = test_include_output_tokens(llm, prompt, args.max_tokens)

    if results.get("with", {}).get("ok"):
        print("\n  ✅ include_output_tokens=True gives enough hs for all response"
              " logprobs — Pass 2 can be eliminated.")

    print(f"\n{'='*60}")
    print("Test 2 — Lock file lifecycle")
    print("{'='*60}")
    test_lockfile_leak(llm, [prompt], args.max_tokens, hs_dir)

    del llm
    gc.collect()
    import shutil
    shutil.rmtree(hs_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print("Done.")
    print("{'='*60}")


if __name__ == "__main__":
    main()
