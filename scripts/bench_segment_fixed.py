#!/usr/bin/env python3
"""Benchmark: segment-by-segment with block alignment + fixed batch size.

Applies two fixes to eliminate nondeterminism from the segment path:

1. **Block alignment** — pads prompt token IDs so ``prompt_len % block_size == 0``,
   guaranteeing every segment boundary lands on a full block.  No partial blocks
   → no re-prefill of decode-generated tokens → APC hits 100%.

2. **Fixed batch size** — pads every round to the same number of requests with
   cheap 1-token dummy prompts.  Same batch size every round → attention kernel
   (FlashDecoding / Split-KV) makes identical parallelisation choices → no
   batch-dependent numeric drift.

Run with --no-align and/or --no-fix-batch to isolate each fix.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from vllm import LLM, SamplingParams

from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
from utils.answer_verifier import extract_answer, verify_answer_by_value
from utils.jsonl import sample_prefix

BLOCK_SIZE = 16  # vLLM default KV cache block size (tokens)


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


def _align_prompt_ids(ids: List[int]) -> List[int]:
    """Pre-pend pad tokens so len(ids) is a multiple of BLOCK_SIZE."""
    pad_needed = (BLOCK_SIZE - len(ids) % BLOCK_SIZE) % BLOCK_SIZE
    if pad_needed == 0:
        return ids
    # Use the first token of the prompt as pad (minimally invasive — it's
    # self-attended to by the rest of the prompt and doesn't change semantics).
    pad_token = ids[0]
    return [pad_token] * pad_needed + ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Segment by token IDs with block alignment + fixed batch size")
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument("--data", required=True, help="JSONL dataset with prompts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default=None, help="Save results JSON here")
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--no-align", action="store_true",
                        help="Disable block alignment (isolate fixed-batch effect)")
    parser.add_argument("--no-fix-batch", action="store_true",
                        help="Disable fixed batch size (isolate alignment effect)")
    args = parser.parse_args()

    do_align = not args.no_align
    do_fix_batch = not args.no_fix_batch

    cfg = _load_config(args.config)
    inf_cfg = cfg["inference"]
    model_path = inf_cfg["model_name_or_path"]
    max_new_tokens = int(inf_cfg["max_new_tokens"])
    use_math_chat = bool(inf_cfg.get("use_math_chat_prompt", True))
    system_prompt = inf_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT)
    gpu_mem = float(inf_cfg.get("gpu_memory_utilization", 0.90))
    segment_size = int(cfg["data"]["segment_size"])
    seed = args.seed

    prompts_data = _load_prompts(args.data, max_samples=args.max_samples)
    N = len(prompts_data)
    print(f"Loaded {N} unique prompts")
    print(f"Block alignment: {'ON' if do_align else 'OFF'}")
    print(f"Fixed batch size: {'ON' if do_fix_batch else 'OFF'}")

    n_gpus = torch.cuda.device_count()
    tp = args.parallel_size if args.parallel_size is not None else n_gpus
    max_model_len = max_new_tokens + 2048

    print(f"Initialising raw LLM (tp={tp}, greedy, NO speculative decode)...")
    t0 = time.perf_counter()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
    )
    tokenizer = llm.get_tokenizer()
    print(f"LLM ready in {time.perf_counter() - t0:.1f}s")

    # Tokenize prompts once
    prompt_ids_list: List[List[int]] = []
    for p in prompts_data:
        q = p["question"]
        if use_math_chat:
            msgs = [
                {"role": "system", "content": system_prompt or DEFAULT_MATH_SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ]
            try:
                ids = tokenizer.apply_chat_template(
                    msgs, tokenize=True, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                rp = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{q}\n\n[ASSISTANT]\n"
                ids = tokenizer.encode(rp, add_special_tokens=False)
        else:
            ids = tokenizer.encode(q, add_special_tokens=True)

        if do_align:
            ids = _align_prompt_ids(ids)

        prompt_ids_list.append(ids)

    # Print alignment info
    if do_align:
        lengths = [len(ids) for ids in prompt_ids_list]
        aligned = sum(1 for l in lengths if l % BLOCK_SIZE == 0)
        print(f"Aligned prompts: {aligned}/{N} (all should be divisible by {BLOCK_SIZE})")
        if aligned != N:
            bad = [l for l in lengths if l % BLOCK_SIZE != 0]
            print(f"  Unaligned lengths: {bad[:5]}...")

    # Dummy prompt for batch padding — single token, max_tokens=1, finishes instantly
    if do_fix_batch:
        dummy_prompt = [tokenizer.encode("A", add_special_tokens=False)[-1]]

    max_rounds = max_new_tokens // segment_size
    eos_token_id = tokenizer.eos_token_id

    generated_ids: List[List[int]] = [[] for _ in range(N)]
    generated_text: List[str] = [""] * N
    active: List[bool] = [True] * N

    total_requests = 0
    total_dummy_requests = 0
    gen_time_total = 0.0

    print(f"Segment-by-token-IDs (greedy): segment_size={segment_size}, "
          f"max_rounds={max_rounds}, N={N}")

    for round_idx in range(max_rounds):
        round_prompts: List[List[int]] = []
        round_params: List[SamplingParams] = []
        round_map: List[int] = []  # real prompt indices
        round_dummy_start: int = -1  # first dummy index

        for i in range(N):
            if not active[i]:
                continue
            input_ids = prompt_ids_list[i] + generated_ids[i]
            round_prompts.append(input_ids)
            round_params.append(SamplingParams(
                temperature=0, max_tokens=segment_size,
                top_p=1.0, top_k=0, seed=seed,
            ))
            round_map.append(i)

        n_real = len(round_prompts)
        if n_real == 0:
            break

        # Pad batch to constant size N
        if do_fix_batch and n_real < N:
            round_dummy_start = n_real
            pad_needed = N - n_real
            dummy_params = SamplingParams(
                temperature=0, max_tokens=1,
            )
            for _ in range(pad_needed):
                round_prompts.append(list(dummy_prompt))
                round_params.append(dummy_params)
            total_dummy_requests += pad_needed

        t_round0 = time.perf_counter()
        outputs = llm.generate(round_prompts, round_params, use_tqdm=False)
        gen_time_total += time.perf_counter() - t_round0
        total_requests += len(round_prompts)

        for j, i in enumerate(round_map):
            o0 = outputs[j].outputs[0]
            new_ids = o0.token_ids
            new_text = o0.text
            finish_reason = getattr(o0, "finish_reason", None)

            generated_ids[i].extend(new_ids)
            generated_text[i] += new_text

            if ((eos_token_id is not None and eos_token_id in new_ids) or
                finish_reason == "stop" or not new_ids):
                active[i] = False

        n_active = sum(active)
        if round_idx < 3 or n_active == 0:
            real_str = f"real={n_real}" + (
                f" dummy={total_dummy_requests}" if round_dummy_start >= 0 else "")
            print(f"  round={round_idx:3d}  active_after={n_active:4d}  "
                  f"batch=({real_str})")

        if n_active == 0:
            break

    print(f"Generation done: {total_requests} requests "
          f"({total_dummy_requests} dummy), {gen_time_total:.1f}s total")

    # Score
    results: List[Dict[str, Any]] = []
    n_correct = 0
    for i, pdata in enumerate(prompts_data):
        correct = verify_answer_by_value(extract_answer(generated_text[i]), pdata["answer"])
        if correct:
            n_correct += 1
        results.append({
            "sample_id": f"{sample_prefix(str(i))}_fixed",
            "prompt_idx": i,
            "question": pdata["question"],
            "gold_answer": pdata["answer"],
            "response": generated_text[i],
            "token_ids": generated_ids[i],
            "correct": correct,
            "extracted_answer": extract_answer(generated_text[i]),
        })

    script_label = "segment_fixed"
    if not do_align:
        script_label += "_noalign"
    if not do_fix_batch:
        script_label += "_nofixbatch"

    summary = {
        "script": script_label,
        "config": args.config,
        "data": args.data,
        "seed": seed,
        "n_prompts": N,
        "temperature": 0,
        "segment_size": segment_size,
        "max_new_tokens": max_new_tokens,
        "block_alignment": do_align,
        "fixed_batch_size": do_fix_batch,
        "n_correct": n_correct,
        "n_total": N,
        "accuracy": n_correct / max(1, N),
        "gen_time_s": gen_time_total,
        "total_requests": total_requests,
        "total_dummy_requests": total_dummy_requests,
    }
    output = {"summary": summary, "results": results}

    print(f"\nSegment (fixed): accuracy={summary['accuracy']:.4f}  "
          f"correct={n_correct}/{N}  time={gen_time_total:.1f}s")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
