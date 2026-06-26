#!/usr/bin/env python3
"""Benchmark condition 2: hidden states only — speculative decode ON, greedy, no segmentation.

Raw ``LLM()`` with ``speculative_config`` + ``kv_transfer_config`` matching
``VLLMFeatureExporter._lazy_init()``.  Greedy decoding (temperature=0),
full response in one shot.  Hidden states are discarded.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from transformers import AutoConfig
from vllm import LLM, SamplingParams

from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
from utils.answer_verifier import extract_answer, verify_answer_by_value
from utils.jsonl import sample_prefix


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_prompts(path: str, max_samples: int = 0) -> List[Dict[str, Any]]:
    seen: set = set()
    prompts: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            prefix = sample_prefix(sid)
            if prefix in seen:
                continue
            seen.add(prefix)
            prompts.append({
                "question": row.get("prompt", ""),
                "answer": row.get("metadata", {}).get("gold_answer", ""),
            })
            if max_samples > 0 and len(prompts) >= max_samples:
                break
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hidden states only: speculative decode ON, greedy, full generation in one shot")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--data", required=True, help="JSONL dataset with prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default=None, help="Save results JSON here")
    parser.add_argument("--parallel-size", type=int, default=None)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    inf_cfg = cfg["inference"]
    model_path = inf_cfg["model_name_or_path"]
    max_new_tokens = int(inf_cfg["max_new_tokens"])
    use_math_chat = bool(inf_cfg.get("use_math_chat_prompt", True))
    system_prompt = inf_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT)
    gpu_mem = float(inf_cfg.get("gpu_memory_utilization", 0.90))
    seed = args.seed

    prompts_data = _load_prompts(args.data, max_samples=args.max_samples)
    N = len(prompts_data)
    print(f"Loaded {N} unique prompts")

    n_gpus = torch.cuda.device_count()
    tp = args.parallel_size if args.parallel_size is not None else n_gpus
    max_model_len = max_new_tokens + 2048

    # Same speculative + kv_transfer config as VLLMFeatureExporter._lazy_init()
    hf_cfg = AutoConfig.from_pretrained(model_path)
    last_layer_id = hf_cfg.num_hidden_layers
    hs_tmpdir = tempfile.mkdtemp(prefix="vllm_hs_bench_", dir="/dev/shm")
    atexit.register(lambda: shutil.rmtree(hs_tmpdir, ignore_errors=True))

    print(f"Initialising LLM with speculative_config (extract_hidden_states), "
          f"tp={tp}, greedy...")
    t0 = time.perf_counter()
    # enable_chunked_prefill=False is required by extract_hidden_states in this
    # vLLM version — it is a known confound vs baseline, documented in the
    # comparison report.
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        enable_chunked_prefill=False,
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
                "shared_storage_path": hs_tmpdir,
            },
        },
    )
    tokenizer = llm.get_tokenizer()
    print(f"LLM ready in {time.perf_counter() - t0:.1f}s")

    # Render prompts
    rendered: List[str] = []
    for p in prompts_data:
        q = p["question"]
        if use_math_chat:
            msgs = [
                {"role": "system", "content": system_prompt or DEFAULT_MATH_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            try:
                rp = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                rp = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{q}\n\n[ASSISTANT]\n"
        else:
            rp = q
        rendered.append(rp)

    params = SamplingParams(
        temperature=0, max_tokens=max_new_tokens,
        top_p=1.0, top_k=0, seed=seed,
    )

    print(f"Generating {N} prompts (greedy, speculative decode ON, single shot)...")
    t1 = time.perf_counter()
    outputs = llm.generate(rendered, [params] * N, use_tqdm=True)

    # Clean up hidden state tempfiles written by kv_transfer
    for out in outputs:
        hs_path = out.kv_transfer_params.get("hidden_states_path") if out.kv_transfer_params else None
        if hs_path is not None:
            try:
                os.remove(hs_path)
            except OSError:
                pass

    gen_time = time.perf_counter() - t1
    print(f"Generation done in {gen_time:.1f}s ({gen_time / N:.2f}s/prompt)")

    results: List[Dict[str, Any]] = []
    n_correct = 0
    for i, pdata in enumerate(prompts_data):
        o0 = outputs[i].outputs[0]
        response_text = o0.text
        token_ids = o0.token_ids
        correct = verify_answer_by_value(extract_answer(response_text), pdata["answer"])
        if correct:
            n_correct += 1
        results.append({
            "sample_id": f"{sample_prefix(str(i))}_greedy",
            "prompt_idx": i,
            "question": pdata["question"],
            "gold_answer": pdata["answer"],
            "response": response_text,
            "token_ids": token_ids,
            "correct": correct,
            "extracted_answer": extract_answer(response_text),
        })

    summary = {
        "script": "hidden_states",
        "config": args.config,
        "data": args.data,
        "seed": seed,
        "n_prompts": N,
        "temperature": 0,
        "max_new_tokens": max_new_tokens,
        "note": "enable_chunked_prefill=False (forced by extract_hidden_states compat)",
        "n_correct": n_correct,
        "n_total": N,
        "accuracy": n_correct / max(1, N),
        "gen_time_s": gen_time,
    }
    output = {"summary": summary, "results": results}

    print(f"\nHidden states (greedy): accuracy={summary['accuracy']:.4f}  "
          f"correct={n_correct}/{N}  time={gen_time:.1f}s")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")

    shutil.rmtree(hs_tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
