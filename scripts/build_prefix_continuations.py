#!/usr/bin/env python3
"""Generate continuation-based soft labels for selected reasoning prefixes."""

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

from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
from mil.prefix_data import select_continuation_prefixes
from utils.answer_verifier import extract_final_answer, verify_answer
from utils.jsonl import load_jsonl, write_jsonl


DEFAULT_TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
DEFAULT_SEEDS_PER_TEMPERATURE = 4


def continuation_generation_seed(base_seed: int, record_idx: int, temp_idx: int,
                                 seed_index: int, n_temperatures: int,
                                 seeds_per_temperature: int) -> int:
    return (
        int(base_seed) +
        record_idx * n_temperatures * seeds_per_temperature +
        temp_idx * seeds_per_temperature +
        seed_index
    )


def continuation_request_plan(n_records: int, temperatures: List[float],
                              seeds_per_temperature: int,
                              base_seed: int) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    for record_idx in range(n_records):
        for temp_idx, temp in enumerate(temperatures):
            for seed_index in range(seeds_per_temperature):
                generation_seed = continuation_generation_seed(
                    base_seed, record_idx, temp_idx, seed_index,
                    len(temperatures), seeds_per_temperature,
                )
                plan.append({
                    "record_idx": record_idx,
                    "temperature": float(temp),
                    "temperature_index": temp_idx,
                    "seed_index": seed_index,
                    "generation_seed": generation_seed,
                })
    return plan


def _render_prompt(tokenizer, row: Dict[str, Any], system_prompt: str,
                   use_math_chat: bool) -> str:
    rendered = row.get("metadata", {}).get("rendered_prompt")
    if rendered:
        return str(rendered)
    question = str(row.get("prompt", row.get("question", row.get("problem", ""))))
    if not use_math_chat:
        return question
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    except Exception:
        return f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{question}\n\n[ASSISTANT]\n"


def generate_labels(config_path: str, input_path: str, output_path: str,
                    parallel_size: int | None = None,
                    max_records: int | None = None) -> None:
    from vllm import LLM, SamplingParams

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    rows = load_jsonl(input_path)
    segment_size = int(cfg["data"].get("segment_size", 64))
    continuation_cfg = cfg.get("prefix_value", {}).get("continuations", {})
    seed = int(cfg.get("seed", 42))
    prefix_sampling_seed = int(continuation_cfg.get("prefix_sampling_seed", seed))
    specs = select_continuation_prefixes(
        rows, segment_size, sampling_seed=prefix_sampling_seed,
    )
    if max_records is not None:
        specs = specs[:max_records]

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs available for continuation generation")
    tp_size = parallel_size if parallel_size is not None else n_gpus
    inference_cfg = cfg["inference"]
    temperatures = [float(x) for x in continuation_cfg.get(
        "temperatures", DEFAULT_TEMPERATURES,
    )]
    seeds_per_temperature = int(continuation_cfg.get(
        "seeds_per_temperature", DEFAULT_SEEDS_PER_TEMPERATURE,
    ))
    if seeds_per_temperature < 1:
        raise ValueError("prefix_value.continuations.seeds_per_temperature must be >= 1")
    continuation_max_tokens = int(continuation_cfg.get(
        "max_new_tokens", inference_cfg.get("max_new_tokens", 8192),
    ))
    response_token_budget = int(inference_cfg.get("max_new_tokens", 8192))
    model_path = inference_cfg["model_name_or_path"]
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        max_model_len=response_token_budget + 2048,
        gpu_memory_utilization=float(inference_cfg.get("gpu_memory_utilization", 0.90)),
    )
    tokenizer = llm.get_tokenizer()
    use_math_chat = bool(inference_cfg.get("use_math_chat_prompt", True))
    system_prompt = inference_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT)

    by_id = {str(row.get("sample_id", "")): row for row in rows}
    prompts: List[str] = []
    params: List[SamplingParams] = []
    request_map: List[Dict[str, Any]] = []
    prepared: List[Dict[str, Any]] = []
    for record_idx, spec in enumerate(specs):
        row = by_id[spec["source_sample_id"]]
        prefix_ids = list(row.get("token_ids", []))[:int(spec["prefix_token_end"])]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
        rendered = _render_prompt(tokenizer, row, system_prompt, use_math_chat)
        prepared.append({**spec, "prefix_text": prefix_text})
        remaining_tokens = max(
            1, min(continuation_max_tokens,
                   response_token_budget - int(spec["prefix_token_end"])),
        )
        for request in continuation_request_plan(
            1, temperatures, seeds_per_temperature, seed,
        ):
            request = {
                **request,
                "record_idx": record_idx,
                "generation_seed": continuation_generation_seed(
                    seed, record_idx, int(request["temperature_index"]),
                    int(request["seed_index"]), len(temperatures),
                    seeds_per_temperature,
                ),
            }
            temp = float(request["temperature"])
            prompts.append(rendered + prefix_text)
            params.append(SamplingParams(
                n=1, temperature=temp, max_tokens=remaining_tokens,
                top_p=1.0, top_k=-1,
                seed=int(request["generation_seed"]),
            ))
            request_map.append({**request, "prefix_text": prefix_text})

    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True) if prompts else []
    elapsed = time.perf_counter() - started
    continuation_results: List[List[Dict[str, Any]]] = [[] for _ in prepared]
    for out, request in zip(outputs, request_map):
        record_idx = int(request["record_idx"])
        temp = float(request["temperature"])
        prefix_text = str(request["prefix_text"])
        generated = out.outputs[0]
        full_response = prefix_text + generated.text
        row = by_id[prepared[record_idx]["source_sample_id"]]
        gold = str(row.get("metadata", {}).get("gold_answer", ""))
        is_correct = verify_answer(full_response, gold)
        continuation_results[record_idx].append({
            "temperature": temp,
            "temperature_index": int(request["temperature_index"]),
            "seed_index": int(request["seed_index"]),
            "generation_seed": int(request["generation_seed"]),
            "correct": bool(is_correct),
            "extracted_answer": extract_final_answer(full_response),
            "generated_tokens": len(generated.token_ids),
            "finish_reason": getattr(generated, "finish_reason", None),
        })

    output_rows: List[Dict[str, Any]] = []
    for spec, continuations in zip(prepared, continuation_results):
        output_rows.append({
            **spec,
            "n_correct": sum(1 for item in continuations if item["correct"]),
            "n_total": len(continuations),
            "continuations": continuations,
            "generation": {
                "seed": seed,
                "prefix_sampling_seed": prefix_sampling_seed,
                "seeds_per_temperature": seeds_per_temperature,
                "model_name_or_path": model_path,
                "temperatures": temperatures,
                "max_new_tokens": continuation_max_tokens,
                "response_token_budget": response_token_budget,
            },
        })
    write_jsonl(output_path, output_rows)

    metadata_path = str(Path(output_path).with_suffix(".meta.json"))
    Path(metadata_path).write_text(json.dumps({
        "input_path": input_path,
        "output_path": output_path,
        "n_source_rows": len(rows),
        "n_prefixes": len(output_rows),
        "n_continuations": len(outputs),
        "elapsed_seconds": elapsed,
        "temperatures": temperatures,
        "seeds_per_temperature": seeds_per_temperature,
        "seed": seed,
        "prefix_sampling_seed": prefix_sampling_seed,
    }, indent=2), encoding="utf-8")
    print(f"prefixes={len(output_rows)} continuations={len(outputs)} elapsed={elapsed:.1f}s")
    print(f"labels={output_path} metadata={metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    input_path = args.input or paths[f"{args.split}_dataset"]
    output_path = args.output or paths[f"{args.split}_continuations"]
    generate_labels(
        args.config, input_path, output_path,
        parallel_size=args.parallel_size, max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
