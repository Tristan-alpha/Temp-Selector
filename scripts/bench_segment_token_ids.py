#!/usr/bin/env python3
"""Benchmark condition 4: segment-by-segment with token IDs — greedy, no re-tokenization.

Same as ``bench_segment_only.py`` except that context is passed as raw token IDs
instead of text strings.  This guarantees perfect token alignment across segment
boundaries.
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
        description="Segment by token IDs: raw LLM, greedy, segment-by-segment, no text re-tokenization")
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
    segment_size = int(cfg["data"]["segment_size"])
    seed = args.seed

    prompts_data = _load_prompts(args.data, max_samples=args.max_samples)
    N = len(prompts_data)
    print(f"Loaded {N} unique prompts")

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

    # Tokenize prompts ONCE
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
        prompt_ids_list.append(ids)

    max_rounds = max_new_tokens // segment_size
    eos_token_id = tokenizer.eos_token_id

    # Per-prompt state
    generated_ids: List[List[int]] = [[] for _ in range(N)]
    generated_text: List[str] = [""] * N  # built from decoded token IDs, for scoring
    active: List[bool] = [True] * N

    total_requests = 0
    gen_time_total = 0.0

    print(f"Segment-by-token-IDs (greedy): segment_size={segment_size}, "
          f"max_rounds={max_rounds}, N={N}")

    for round_idx in range(max_rounds):
        round_prompts: List[List[int]] = []  # token ID lists
        round_map: List[int] = []

        for i in range(N):
            if not active[i]:
                continue
            # Raw token IDs — NO text concatenation, NO re-tokenization
            input_ids = prompt_ids_list[i] + generated_ids[i]
            round_prompts.append(input_ids)
            round_map.append(i)

        if not round_prompts:
            break

        params = SamplingParams(
            temperature=0, max_tokens=segment_size,
            top_p=1.0, top_k=0, seed=seed,
        )

        t_round0 = time.perf_counter()
        outputs = llm.generate(round_prompts, [params] * len(round_prompts), use_tqdm=False)
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
            print(f"  round={round_idx:3d}  active_after={n_active:4d}  "
                  f"batch={len(round_prompts)}")

        if n_active == 0:
            break

    print(f"Generation done: {total_requests} segment requests, "
          f"{gen_time_total:.1f}s total")

    # Score
    results: List[Dict[str, Any]] = []
    n_correct = 0
    for i, pdata in enumerate(prompts_data):
        correct = verify_answer_by_value(extract_answer(generated_text[i]), pdata["answer"])
        if correct:
            n_correct += 1
        results.append({
            "sample_id": f"{sample_prefix(str(i))}_greedy",
            "prompt_idx": i,
            "question": pdata["question"],
            "gold_answer": pdata["answer"],
            "response": generated_text[i],
            "token_ids": generated_ids[i],
            "correct": correct,
            "extracted_answer": extract_answer(generated_text[i]),
        })

    summary = {
        "script": "segment_token_ids",
        "config": args.config,
        "data": args.data,
        "seed": seed,
        "n_prompts": N,
        "temperature": 0,
        "segment_size": segment_size,
        "max_new_tokens": max_new_tokens,
        "n_correct": n_correct,
        "n_total": N,
        "accuracy": n_correct / max(1, N),
        "gen_time_s": gen_time_total,
        "total_segment_requests": total_requests,
    }
    output = {"summary": summary, "results": results}

    print(f"\nSegment (greedy, token IDs): accuracy={summary['accuracy']:.4f}  "
          f"correct={n_correct}/{N}  time={gen_time_total:.1f}s")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
