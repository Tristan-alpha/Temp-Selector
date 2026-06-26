#!/usr/bin/env python3
"""Extend existing prefix-continuation labels with additional seeds."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml

from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
from scripts.build_prefix_continuations import (
    DEFAULT_TEMPERATURES,
    continuation_temperature_summary,
)
from utils.answer_verifier import extract_final_answer, verify_answer
from utils.jsonl import load_jsonl, write_jsonl


DEFAULT_TARGET_SEEDS_PER_TEMPERATURE = 32
DEFAULT_APPEND_SEED_OFFSET = 10_000_000


def append_generation_seed(base_seed: int, record_idx: int, temp_idx: int,
                           seed_index: int, n_temperatures: int,
                           target_seeds_per_temperature: int,
                           append_seed_offset: int) -> int:
    return (
        int(append_seed_offset) +
        int(base_seed) +
        int(record_idx) * int(n_temperatures) * int(target_seeds_per_temperature) +
        int(temp_idx) * int(target_seeds_per_temperature) +
        int(seed_index)
    )


def continuation_seed_key(item: Dict[str, Any]) -> Tuple[int, int]:
    if "temperature_index" not in item:
        raise ValueError(f"continuation is missing temperature_index: {item}")
    if "seed_index" not in item:
        raise ValueError(f"continuation is missing seed_index: {item}")
    return int(item["temperature_index"]), int(item["seed_index"])


def existing_seed_keys(record: Dict[str, Any]) -> set[Tuple[int, int]]:
    keys: set[Tuple[int, int]] = set()
    for item in record.get("continuations", []):
        key = continuation_seed_key(item)
        if key in keys:
            raise ValueError(
                "duplicate continuation for "
                f"source_sample_id={record.get('source_sample_id')} key={key}"
            )
        keys.add(key)
    return keys


def missing_request_plan(records: Sequence[Dict[str, Any]],
                         temperatures: Sequence[float],
                         target_seeds_per_temperature: int,
                         base_seed: int,
                         append_seed_offset: int,
                         record_index_offset: int = 0) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []
    n_temperatures = len(temperatures)
    for record_idx, record in enumerate(records):
        global_record_idx = int(record_index_offset) + record_idx
        present = existing_seed_keys(record)
        for temp_idx, temp in enumerate(temperatures):
            for seed_index in range(target_seeds_per_temperature):
                key = (temp_idx, seed_index)
                if key in present:
                    continue
                plan.append({
                    "record_idx": record_idx,
                    "global_record_idx": global_record_idx,
                    "temperature": float(temp),
                    "temperature_index": temp_idx,
                    "seed_index": seed_index,
                    "generation_seed": append_generation_seed(
                        base_seed,
                        global_record_idx,
                        temp_idx,
                        seed_index,
                        n_temperatures,
                        target_seeds_per_temperature,
                        append_seed_offset,
                    ),
                })
    return plan


def _sorted_continuations(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        (dict(item) for item in items),
        key=lambda item: (
            int(item.get("temperature_index", 0)),
            int(item.get("seed_index", 0)),
            int(item.get("generation_seed", 0)),
        ),
    )


def recompute_record(record: Dict[str, Any], continuations: List[Dict[str, Any]],
                     target_seeds_per_temperature: int,
                     temperatures: Sequence[float],
                     generation_update: Dict[str, Any]) -> Dict[str, Any]:
    ordered = _sorted_continuations(continuations)
    summary = continuation_temperature_summary(ordered)
    out = dict(record)
    out["n_correct"] = sum(1 for item in ordered if item.get("correct", False))
    out["n_total"] = len(ordered)
    out.update(summary)
    out["continuations"] = ordered
    generation = dict(record.get("generation", {}))
    generation.update({
        "seeds_per_temperature": int(target_seeds_per_temperature),
        "temperatures": [float(temp) for temp in temperatures],
        **generation_update,
    })
    out["generation"] = generation
    return out


def extend_records_with_results(records: Sequence[Dict[str, Any]],
                                request_map: Sequence[Dict[str, Any]],
                                generated_results: Sequence[Dict[str, Any]],
                                target_seeds_per_temperature: int,
                                temperatures: Sequence[float],
                                generation_update: Dict[str, Any]) -> List[Dict[str, Any]]:
    by_record: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for request, result in zip(request_map, generated_results):
        by_record[int(request["record_idx"])].append(result)

    extended: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        continuations = list(record.get("continuations", [])) + by_record.get(idx, [])
        extended.append(recompute_record(
            record,
            continuations,
            target_seeds_per_temperature,
            temperatures,
            generation_update,
        ))
    return extended


def validate_extended_records(records: Sequence[Dict[str, Any]],
                              temperatures: Sequence[float],
                              target_seeds_per_temperature: int) -> Dict[str, Any]:
    expected_seed_indices = set(range(int(target_seeds_per_temperature)))
    expected_n_total = len(temperatures) * int(target_seeds_per_temperature)
    n_total_distribution: Dict[str, int] = defaultdict(int)
    errors: List[str] = []
    for record_idx, record in enumerate(records):
        n_total = int(record.get("n_total", -1))
        n_total_distribution[str(n_total)] += 1
        if n_total != expected_n_total:
            errors.append(f"record {record_idx} n_total={n_total} expected={expected_n_total}")
        grouped: Dict[int, set[int]] = defaultdict(set)
        for item in record.get("continuations", []):
            temp_idx, seed_index = continuation_seed_key(item)
            grouped[temp_idx].add(seed_index)
        for temp_idx in range(len(temperatures)):
            seeds = grouped.get(temp_idx, set())
            if seeds != expected_seed_indices:
                errors.append(
                    f"record {record_idx} temp_idx={temp_idx} seeds="
                    f"{sorted(seeds)} expected={sorted(expected_seed_indices)}"
                )
    return {
        "passed": not errors,
        "errors": errors[:20],
        "n_errors": len(errors),
        "n_records": len(records),
        "expected_n_total": expected_n_total,
        "n_total_distribution": dict(sorted(n_total_distribution.items())),
    }


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


def _batched(items: Sequence[Dict[str, Any]],
             records_per_batch: int) -> Iterable[List[Dict[str, Any]]]:
    if records_per_batch < 1:
        raise ValueError("records_per_batch must be >= 1")
    for start in range(0, len(items), records_per_batch):
        yield list(items[start:start + records_per_batch])


def _requests_by_record(plan: Sequence[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for request in plan:
        grouped[int(request["record_idx"])].append(dict(request))
    return grouped


def default_checkpoint_dir(output_path: str | Path) -> Path:
    return Path(f"{output_path}.batches")


def batch_checkpoint_paths(checkpoint_dir: str | Path,
                           record_start: int,
                           record_end: int) -> Tuple[Path, Path]:
    checkpoint_root = Path(checkpoint_dir)
    stem = f"batch_{int(record_start)}_{int(record_end)}"
    return checkpoint_root / f"{stem}.jsonl", checkpoint_root / f"{stem}.meta.json"


def progress_path(output_path: str | Path) -> Path:
    return Path(f"{output_path}.progress.json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_jsonl_atomic(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(f"{output.name}.tmp.{os.getpid()}")
    write_jsonl(str(tmp), rows)
    os.replace(tmp, output)


def write_json_atomic(path: str | Path, data: Dict[str, Any]) -> None:
    _atomic_write_text(Path(path), json.dumps(data, indent=2),)


def write_progress(output_path: str | Path,
                   checkpoint_dir: str | Path,
                   status: str,
                   completed_batches: Sequence[Dict[str, Any]],
                   total_records: int,
                   total_missing_continuations: int,
                   started_at: float) -> Dict[str, Any]:
    completed_records = sum(
        int(batch["record_end"]) - int(batch["record_start"])
        for batch in completed_batches
    )
    completed_generated = sum(
        int(batch.get("generated_count", 0))
        for batch in completed_batches
    )
    progress = {
        "status": status,
        "output_path": str(output_path),
        "checkpoint_dir": str(checkpoint_dir),
        "completed_batches": list(completed_batches),
        "n_completed_batches": len(completed_batches),
        "n_completed_records": completed_records,
        "n_total_records": int(total_records),
        "n_completed_generated_continuations": completed_generated,
        "n_total_missing_continuations": int(total_missing_continuations),
        "elapsed_seconds": time.perf_counter() - started_at,
        "updated_at_unix": time.time(),
    }
    write_json_atomic(progress_path(output_path), progress)
    return progress


def write_batch_checkpoint(checkpoint_dir: str | Path,
                           record_start: int,
                           record_end: int,
                           records: Sequence[Dict[str, Any]],
                           validation: Dict[str, Any],
                           generated_count: int,
                           elapsed_seconds: float,
                           extra_metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
    labels_path, meta_path = batch_checkpoint_paths(
        checkpoint_dir, record_start, record_end,
    )
    write_jsonl_atomic(labels_path, records)
    metadata = {
        "mode": "batch_checkpoint",
        "record_start": int(record_start),
        "record_end": int(record_end),
        "n_prefixes": len(records),
        "n_generated_continuations": int(generated_count),
        "elapsed_seconds": float(elapsed_seconds),
        "labels_path": str(labels_path),
        "metadata_path": str(meta_path),
        "validation": validation,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json_atomic(meta_path, metadata)
    return metadata


def load_batch_checkpoint(checkpoint_dir: str | Path,
                          record_start: int,
                          record_end: int,
                          temperatures: Sequence[float],
                          target_seeds_per_temperature: int,
                          expected_records: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]] | None:
    labels_path, meta_path = batch_checkpoint_paths(
        checkpoint_dir, record_start, record_end,
    )
    if not labels_path.exists() or not meta_path.exists():
        return None
    records = load_jsonl(str(labels_path))
    if len(records) != int(expected_records):
        return None
    validation = validate_extended_records(
        records, temperatures, target_seeds_per_temperature,
    )
    if not validation["passed"]:
        return None
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    meta_validation = metadata.get("validation", {})
    if meta_validation and not bool(meta_validation.get("passed", False)):
        return None
    return records, metadata


def extend_labels(config_path: str, split: str, existing_path: str,
                  output_path: str, target_seeds_per_temperature: int,
                  append_seed_offset: int, parallel_size: int | None = None,
                  max_records: int | None = None,
                  record_start: int = 0,
                  record_end: int | None = None,
                  records_per_batch: int = 32,
                  overwrite: bool = False,
                  save_generated_text: bool = False,
                  resume: bool = False,
                  checkpoint_dir: str | None = None) -> Dict[str, Any]:
    output = Path(output_path)
    if output.exists() and not overwrite:
        if not resume:
            raise FileExistsError(f"output already exists: {output}")
        existing = load_jsonl(str(output))
        validation = validate_extended_records(
            existing,
            cfg_temperatures_from_config(config_path),
            target_seeds_per_temperature,
        )
        if validation["passed"]:
            return {
                "mode": "resume_existing_output",
                "output_path": str(output),
                "validation": validation,
            }
        raise FileExistsError(f"output already exists and is not a valid completed output: {output}")
    if Path(existing_path).resolve() == output.resolve():
        raise ValueError("output_path must differ from existing_path")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    input_path = paths[f"{split}_dataset"]
    rows = load_jsonl(input_path)
    all_records = load_jsonl(existing_path)
    if record_start < 0:
        raise ValueError("--record-start must be >= 0")
    effective_end = len(all_records) if record_end is None else int(record_end)
    if effective_end < record_start:
        raise ValueError("--record-end must be >= --record-start")
    records = all_records[record_start:effective_end]
    if max_records is not None:
        records = records[:max_records]
        effective_end = record_start + len(records)

    continuation_cfg = cfg.get("prefix_value", {}).get("continuations", {})
    seed = int(cfg.get("seed", 42))
    prefix_sampling_seed = int(continuation_cfg.get("prefix_sampling_seed", seed))
    inference_cfg = cfg["inference"]
    temperatures = [
        float(x) for x in continuation_cfg.get("temperatures", DEFAULT_TEMPERATURES)
    ]
    if target_seeds_per_temperature < 1:
        raise ValueError("--target-seeds-per-temperature must be >= 1")
    continuation_max_tokens = int(continuation_cfg.get(
        "max_new_tokens", inference_cfg.get("max_new_tokens", 8192),
    ))
    response_token_budget = int(inference_cfg.get("max_new_tokens", 8192))
    model_path = inference_cfg["model_name_or_path"]

    request_plan = missing_request_plan(
        records,
        temperatures,
        target_seeds_per_temperature,
        seed,
        append_seed_offset,
        record_index_offset=record_start,
    )
    requests_by_record = _requests_by_record(request_plan)
    existing_count = sum(len(record.get("continuations", [])) for record in records)
    if not request_plan:
        generation_update = {
            "append_from_existing": existing_path,
            "append_seed_offset": int(append_seed_offset),
            "target_seeds_per_temperature": int(target_seeds_per_temperature),
            "generated_missing_continuations": 0,
        }
        extended = [
            recompute_record(
                record,
                list(record.get("continuations", [])),
                target_seeds_per_temperature,
                temperatures,
                generation_update,
            )
            for record in records
        ]
        write_jsonl_atomic(output, extended)
        validation = validate_extended_records(
            extended, temperatures, target_seeds_per_temperature,
        )
        metadata = _write_metadata(
            cfg, input_path, existing_path, str(output), records, existing_count,
            0, temperatures, target_seeds_per_temperature, seed,
            prefix_sampling_seed, append_seed_offset, 0.0, validation,
            max_records=max_records,
            record_start=record_start,
            record_end=effective_end,
        )
        print(
            "prefixes=%d existing=%d generated=0 total=%d elapsed=0.0s validation=%s"
            % (
                len(extended),
                existing_count,
                existing_count,
                "passed" if validation["passed"] else "failed",
            )
        )
        print(f"labels={output_path} metadata={metadata['metadata_path']}")
        if not validation["passed"]:
            raise RuntimeError(f"extended continuation validation failed: {validation}")
        return metadata

    by_id = {str(row.get("sample_id", "")): row for row in rows}

    started = time.perf_counter()
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else default_checkpoint_dir(output)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    generation_update = {
        "append_from_existing": existing_path,
        "append_seed_offset": int(append_seed_offset),
        "target_seeds_per_temperature": int(target_seeds_per_temperature),
        "generated_missing_continuations": len(request_plan),
        "prefix_sampling_seed": prefix_sampling_seed,
        "model_name_or_path": model_path,
        "max_new_tokens": continuation_max_tokens,
        "response_token_budget": response_token_budget,
        "save_generated_text": bool(save_generated_text),
        "incremental_write": True,
        "checkpoint_dir": str(checkpoint_root),
    }
    completed_batches: List[Dict[str, Any]] = []
    batch_records_by_start: Dict[int, List[Dict[str, Any]]] = {}
    llm = None
    tokenizer = None
    SamplingParams = None
    use_math_chat = bool(inference_cfg.get("use_math_chat_prompt", True))
    system_prompt = inference_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT)

    for batch_record_start in range(0, len(records), records_per_batch):
        batch_record_end = min(len(records), batch_record_start + records_per_batch)
        global_batch_start = int(record_start) + batch_record_start
        global_batch_end = int(record_start) + batch_record_end
        completed = (
            load_batch_checkpoint(
                checkpoint_root,
                global_batch_start,
                global_batch_end,
                temperatures,
                target_seeds_per_temperature,
                expected_records=batch_record_end - batch_record_start,
            )
            if resume else None
        )
        if completed is not None:
            checkpoint_records, checkpoint_meta = completed
            batch_records_by_start[batch_record_start] = checkpoint_records
            completed_batches.append({
                "record_start": global_batch_start,
                "record_end": global_batch_end,
                "generated_count": int(checkpoint_meta.get("n_generated_continuations", 0)),
                "labels_path": checkpoint_meta.get("labels_path"),
                "metadata_path": checkpoint_meta.get("metadata_path"),
                "resumed": True,
            })
            write_progress(
                output,
                checkpoint_root,
                "partial",
                completed_batches,
                len(records),
                len(request_plan),
                started,
            )
            print(
                f"resume batch records={global_batch_start}:{global_batch_end} "
                f"path={checkpoint_meta.get('labels_path')}"
            )
            continue

        if llm is None:
            from vllm import LLM, SamplingParams as VLLMSamplingParams

            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                raise RuntimeError("No GPUs available for continuation extension")
            tp_size = parallel_size if parallel_size is not None else n_gpus
            SamplingParams = VLLMSamplingParams
            llm = LLM(
                model=model_path,
                tensor_parallel_size=tp_size,
                max_model_len=response_token_budget + 2048,
                gpu_memory_utilization=float(inference_cfg.get("gpu_memory_utilization", 0.90)),
            )
            tokenizer = llm.get_tokenizer()

        batch_started = time.perf_counter()
        batch_requests = [
            request
            for record_idx in range(batch_record_start, batch_record_end)
            for request in requests_by_record.get(record_idx, [])
        ]
        prompts: List[str] = []
        params: List[SamplingParams] = []
        request_map: List[Dict[str, Any]] = []
        for request in batch_requests:
            record_idx = int(request["record_idx"])
            record = records[record_idx]
            row = by_id[str(record["source_sample_id"])]
            prefix_text = str(record.get("prefix_text", ""))
            if not prefix_text:
                prefix_ids = list(row.get("token_ids", []))[:int(record["prefix_token_end"])]
                prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
            rendered = _render_prompt(tokenizer, row, system_prompt, use_math_chat)
            remaining_tokens = max(
                1,
                min(
                    continuation_max_tokens,
                    response_token_budget - int(record["prefix_token_end"]),
                ),
            )
            prompts.append(rendered + prefix_text)
            params.append(SamplingParams(
                n=1,
                temperature=float(request["temperature"]),
                max_tokens=remaining_tokens,
                top_p=1.0,
                top_k=-1,
                seed=int(request["generation_seed"]),
            ))
            request_map.append({**request, "prefix_text": prefix_text})

        outputs = llm.generate(prompts, params, use_tqdm=True) if prompts else []
        generated_by_record: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for out, request in zip(outputs, request_map):
            record_idx = int(request["record_idx"])
            record = records[record_idx]
            row = by_id[str(record["source_sample_id"])]
            generated = out.outputs[0]
            full_response = str(request["prefix_text"]) + generated.text
            gold = str(row.get("metadata", {}).get("gold_answer", ""))
            is_correct = verify_answer(full_response, gold)
            result = {
                "temperature": float(request["temperature"]),
                "temperature_index": int(request["temperature_index"]),
                "seed_index": int(request["seed_index"]),
                "generation_seed": int(request["generation_seed"]),
                "correct": bool(is_correct),
                "extracted_answer": extract_final_answer(full_response),
                "generated_tokens": len(generated.token_ids),
                "finish_reason": getattr(generated, "finish_reason", None),
            }
            if save_generated_text:
                result["generated_text"] = generated.text
                result["full_response_text"] = full_response
            generated_by_record[record_idx].append(result)

        batch_records: List[Dict[str, Any]] = []
        for record_idx in range(batch_record_start, batch_record_end):
            record = records[record_idx]
            continuations = (
                list(record.get("continuations", [])) +
                generated_by_record.get(record_idx, [])
            )
            batch_records.append(recompute_record(
                record,
                continuations,
                target_seeds_per_temperature,
                temperatures,
                generation_update,
            ))
        batch_validation = validate_extended_records(
            batch_records, temperatures, target_seeds_per_temperature,
        )
        if not batch_validation["passed"]:
            raise RuntimeError(
                f"batch validation failed for records {global_batch_start}:{global_batch_end}: "
                f"{batch_validation}"
            )
        batch_elapsed = time.perf_counter() - batch_started
        batch_meta = write_batch_checkpoint(
            checkpoint_root,
            global_batch_start,
            global_batch_end,
            batch_records,
            batch_validation,
            generated_count=len(batch_requests),
            elapsed_seconds=batch_elapsed,
            extra_metadata={
                "save_generated_text": bool(save_generated_text),
                "target_seeds_per_temperature": int(target_seeds_per_temperature),
                "temperatures": [float(temp) for temp in temperatures],
            },
        )
        batch_records_by_start[batch_record_start] = batch_records
        completed_batches.append({
            "record_start": global_batch_start,
            "record_end": global_batch_end,
            "generated_count": len(batch_requests),
            "labels_path": batch_meta["labels_path"],
            "metadata_path": batch_meta["metadata_path"],
            "resumed": False,
        })
        write_progress(
            output,
            checkpoint_root,
            "partial",
            completed_batches,
            len(records),
            len(request_plan),
            started,
        )
        print(
            f"batch records={global_batch_start}:{global_batch_end} "
            f"generated={len(batch_requests)} elapsed={batch_elapsed:.1f}s "
            f"labels={batch_meta['labels_path']}"
        )

    elapsed = time.perf_counter() - started
    extended_records: List[Dict[str, Any]] = []
    for batch_record_start in range(0, len(records), records_per_batch):
        if batch_record_start not in batch_records_by_start:
            raise RuntimeError(f"missing completed batch at local record {batch_record_start}")
        extended_records.extend(batch_records_by_start[batch_record_start])

    validation = validate_extended_records(
        extended_records, temperatures, target_seeds_per_temperature,
    )
    if not validation["passed"]:
        raise RuntimeError(f"extended continuation validation failed: {validation}")
    write_jsonl_atomic(output, extended_records)
    metadata = _write_metadata(
        cfg, input_path, existing_path, str(output), records, existing_count,
        len(request_plan), temperatures, target_seeds_per_temperature, seed,
        prefix_sampling_seed, append_seed_offset, elapsed, validation,
        max_records=max_records,
        record_start=record_start,
        record_end=effective_end,
    )
    write_progress(
        output,
        checkpoint_root,
        "complete",
        completed_batches,
        len(records),
        len(request_plan),
        started,
    )
    print(
        "prefixes=%d existing=%d generated=%d total=%d elapsed=%.1fs validation=%s"
        % (
            len(extended_records),
            existing_count,
            len(request_plan),
            existing_count + len(request_plan),
            elapsed,
            "passed" if validation["passed"] else "failed",
        )
    )
    print(f"labels={output_path} metadata={metadata['metadata_path']}")
    return metadata


def cfg_temperatures_from_config(config_path: str) -> List[float]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    continuation_cfg = cfg.get("prefix_value", {}).get("continuations", {})
    return [
        float(x) for x in continuation_cfg.get("temperatures", DEFAULT_TEMPERATURES)
    ]


def _write_metadata(cfg: Dict[str, Any], input_path: str, existing_path: str,
                    output_path: str, records: Sequence[Dict[str, Any]],
                    existing_count: int, generated_count: int,
                    temperatures: Sequence[float],
                    target_seeds_per_temperature: int, seed: int,
                    prefix_sampling_seed: int, append_seed_offset: int,
                    elapsed: float, validation: Dict[str, Any],
                    max_records: int | None = None,
                    record_start: int = 0,
                    record_end: int | None = None) -> Dict[str, Any]:
    metadata_path = str(Path(output_path).with_suffix(".meta.json"))
    metadata = {
        "mode": "append_from_existing",
        "config": cfg,
        "input_path": input_path,
        "existing_path": existing_path,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "max_records": max_records,
        "record_start": int(record_start),
        "record_end": None if record_end is None else int(record_end),
        "n_source_rows": None,
        "n_prefixes": len(records),
        "n_existing_continuations": existing_count,
        "n_generated_continuations": generated_count,
        "n_continuations": existing_count + generated_count,
        "elapsed_seconds": elapsed,
        "temperatures": [float(temp) for temp in temperatures],
        "target_seeds_per_temperature": int(target_seeds_per_temperature),
        "seed": int(seed),
        "prefix_sampling_seed": int(prefix_sampling_seed),
        "append_seed_offset": int(append_seed_offset),
        "validation": validation,
    }
    try:
        metadata["n_source_rows"] = len(load_jsonl(input_path))
    except FileNotFoundError:
        metadata["n_source_rows"] = None
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--existing", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--target-seeds-per-temperature",
        type=int,
        default=DEFAULT_TARGET_SEEDS_PER_TEMPERATURE,
    )
    parser.add_argument("--append-seed-offset", type=int, default=DEFAULT_APPEND_SEED_OFFSET)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--record-end", type=int, default=None)
    parser.add_argument("--records-per-batch", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-generated-text", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()
    extend_labels(
        config_path=args.config,
        split=args.split,
        existing_path=args.existing,
        output_path=args.output,
        target_seeds_per_temperature=args.target_seeds_per_temperature,
        append_seed_offset=args.append_seed_offset,
        parallel_size=args.parallel_size,
        max_records=args.max_records,
        record_start=args.record_start,
        record_end=args.record_end,
        records_per_batch=args.records_per_batch,
        overwrite=args.overwrite,
        save_generated_text=args.save_generated_text,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
