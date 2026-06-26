#!/usr/bin/env python3
"""Generate segment-level candidate records for latent path analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
TF_MIL_ROOT = EXPERIMENT_ROOT.parent
SRC = EXPERIMENT_ROOT / "src"
for path in (SRC, TF_MIL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data.segment_records import write_json  # noqa: E402
from features.segmenter import build_masked_concat_segment_obs_from_lp  # noqa: E402
from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT, VLLMFeatureExporter  # noqa: E402
from mil.prefix_data import pad_prefix_entries  # noqa: E402
from mil.prefix_value import PrefixValueModel, calibrated_probability  # noqa: E402
from utils.answer_verifier import extract_final_answer, verify_answer  # noqa: E402


DEFAULT_CONFIG = TF_MIL_ROOT / "configs" / "training" / "min_pvm_ppo_500_seed42.yaml"
DEFAULT_OUTPUT_DIR = EXPERIMENT_ROOT / "outputs" / "segment_latent_paths"
DEFAULT_TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--continuations", type=Path, default=None)
    parser.add_argument("--source-dataset", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--pvm-checkpoint", type=Path, default=None)
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "segment_candidate_records.jsonl",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT_DIR / "generation_manifest.json")
    parser.add_argument("--max-prefixes", type=int, default=0)
    parser.add_argument("--start-prefix", type=int, default=0)
    parser.add_argument("--stop-prefix", type=int, default=0)
    parser.add_argument("--append-output", action="store_true")
    parser.add_argument("--prefix-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--temperatures", type=float, nargs="+", default=DEFAULT_TEMPERATURES)
    parser.add_argument("--seeds-per-temperature", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=0)
    parser.add_argument("--segment-size", type=int, default=0)
    parser.add_argument("--top-k-logprobs", type=int, default=64)
    parser.add_argument("--feature-temperature", type=float, default=1.0)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-micro-batch-size", type=int, default=None)
    parser.add_argument("--no-greedy", action="store_true")
    return parser.parse_args()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def resolve_path(path: str | Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    if value.is_absolute():
        return value
    return root / value


def prefix_id(row: Mapping[str, Any], record_index: int) -> str:
    return (
        f"{row.get('problem_id', '')}::"
        f"{row.get('source_sample_id', '')}::"
        f"tok{int(row.get('prefix_token_end', 0))}::"
        f"rec{int(record_index)}"
    )


def generation_seed(base_seed: int, prefix_index: int, request_offset: int, requests_per_prefix: int) -> int:
    return int(base_seed) + int(prefix_index) * int(requests_per_prefix) + int(request_offset)


def load_value_model(cfg: Mapping[str, Any], checkpoint_path: Path, device: torch.device) -> tuple[PrefixValueModel, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = cfg["prefix_value"]["model"]
    model = PrefixValueModel(
        token_dim=int(cfg["data"]["instance_dim"]),
        segment_size=int(cfg["data"]["segment_size"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        max_segments=int(model_cfg.get("max_segments", 8192)),
        n_temps=int(model_cfg.get("n_temps", 0)),
        prompt_dim=int(model_cfg.get("prompt_dim", 0) or 0),
        prompt_integration=str(model_cfg.get("prompt_integration", "none")),
    ).to(device)
    model.load_state_dict(checkpoint["prefix_value"], strict=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, float(checkpoint.get("calibration_temperature", 1.0))


@torch.no_grad()
def score_prefixes(
    records: Sequence[Mapping[str, Any]],
    *,
    cache_by_id: Mapping[str, Mapping[str, Any]],
    model: PrefixValueModel,
    calibration_temperature: float,
    device: torch.device,
    batch_size: int,
) -> list[float]:
    scores: list[float] = []
    for start in range(0, len(records), int(batch_size)):
        batch_records = records[start:start + int(batch_size)]
        batch = pad_prefix_entries([
            (dict(cache_by_id[str(row["source_sample_id"])]), int(row["prefix_segments"]))
            for row in batch_records
        ])
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        output = model(
            tensor_batch["features"],
            tensor_batch["token_mask"],
            tensor_batch["segment_mask"],
            prompt_hidden=tensor_batch.get("prompt_hidden"),
        )
        probs = calibrated_probability(output["terminal_logits"], calibration_temperature)
        scores.extend(float(value) for value in probs.detach().cpu().tolist())
    return scores


def assign_groups(scores: Sequence[float]) -> list[str]:
    ordered = sorted(range(len(scores)), key=lambda idx: (float(scores[idx]), idx))
    groups = [""] * len(scores)
    n = len(scores)
    for rank, idx in enumerate(ordered):
        if rank < n / 3:
            groups[idx] = "low"
        elif rank < 2 * n / 3:
            groups[idx] = "mid"
        else:
            groups[idx] = "high"
    return groups


def render_prompt(tokenizer, source_row: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    rendered = source_row.get("metadata", {}).get("rendered_prompt")
    if rendered:
        return str(rendered)
    question = str(source_row.get("prompt", source_row.get("question", source_row.get("problem", ""))))
    if not bool(cfg["inference"].get("use_math_chat_prompt", True)):
        return question
    messages = [
        {
            "role": "system",
            "content": str(cfg["inference"].get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT)),
        },
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        return (
            f"[SYSTEM]\n{messages[0]['content']}\n\n"
            f"[USER]\n{question}\n\n[ASSISTANT]\n"
        )


def candidate_requests(
    *,
    record: Mapping[str, Any],
    record_index: int,
    source_row: Mapping[str, Any],
    rendered_prompt: str,
    prefix_score: float,
    pvm_group: str,
    temperatures: Sequence[float],
    seeds_per_temperature: int,
    base_seed: int,
    include_greedy: bool,
) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    requests_per_prefix = (1 if include_greedy else 0) + len(temperatures) * int(seeds_per_temperature)
    current_offset = 0
    base = {
        "record_index": int(record_index),
        "prefix_id": prefix_id(record, record_index),
        "problem_id": str(record.get("problem_id", "")),
        "source_sample_id": str(record["source_sample_id"]),
        "prefix_text": str(record.get("prefix_text", "")),
        "prefix_token_end": int(record["prefix_token_end"]),
        "prefix_segments": int(record["prefix_segments"]),
        "prefix_pvm_score": float(prefix_score),
        "pvm_group": pvm_group,
        "source_individual_label": int(record.get("source_individual_label", -1)),
        "rendered_prompt": rendered_prompt,
        "gold_answer": str(source_row.get("metadata", {}).get("gold_answer", "")),
        "prefix_token_ids": [
            int(item)
            for item in list(source_row.get("token_ids", []))[: int(record["prefix_token_end"])]
        ],
    }
    if include_greedy:
        requests.append({
            **base,
            "candidate_role": "greedy",
            "temperature": 0.0,
            "seed_index": 0,
            "generation_seed": generation_seed(base_seed, record_index, current_offset, requests_per_prefix),
            "temperature_index": -1,
        })
        current_offset += 1
    for temp_idx, temp in enumerate(temperatures):
        for seed_idx in range(int(seeds_per_temperature)):
            requests.append({
                **base,
                "candidate_role": "sample",
                "temperature": float(temp),
                "seed_index": int(seed_idx),
                "generation_seed": generation_seed(base_seed, record_index, current_offset, requests_per_prefix),
                "temperature_index": int(temp_idx),
            })
            current_offset += 1
    return requests


def _tokenizer_decode(tokenizer, token_ids: Sequence[int]) -> str:
    if not token_ids:
        return ""
    try:
        return str(tokenizer.decode(list(token_ids), skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(list(token_ids)))


def generate_candidates_for_requests(
    *,
    llm,
    tokenizer,
    requests: Sequence[Mapping[str, Any]],
    response_token_budget: int,
    max_new_tokens: int,
    segment_size: int,
) -> list[dict[str, Any]]:
    from vllm import SamplingParams

    prompts: list[str] = []
    params: list[SamplingParams] = []
    for request in requests:
        prompts.append(str(request["rendered_prompt"]) + str(request["prefix_text"]))
        remaining = max(
            1,
            min(int(max_new_tokens), int(response_token_budget) - int(request["prefix_token_end"])),
        )
        params.append(SamplingParams(
            n=1,
            temperature=float(request["temperature"]),
            max_tokens=remaining,
            top_p=1.0,
            top_k=0,
            seed=int(request["generation_seed"]),
        ))

    outputs = llm.generate(prompts, params, use_tqdm=True) if prompts else []
    candidate_rows: list[dict[str, Any]] = []
    for request, out in zip(requests, outputs):
        generated = out.outputs[0]
        token_ids = [int(item) for item in list(generated.token_ids)]
        segment_token_ids = token_ids[: int(segment_size)]
        segment_text = _tokenizer_decode(tokenizer, segment_token_ids)
        continuation_text = str(generated.text)
        full_response = str(request["prefix_text"]) + continuation_text
        correct = verify_answer(full_response, str(request["gold_answer"]))
        candidate_rows.append({
            key: value
            for key, value in dict(request).items()
            if key not in {"rendered_prompt", "gold_answer"}
        })
        candidate_rows[-1].update({
            "candidate_id": (
                f"{request['prefix_id']}::"
                f"{request['candidate_role']}::"
                f"T{float(request['temperature']):.1f}::"
                f"s{int(request['seed_index'])}"
            ),
            "segment_text": segment_text,
            "segment_token_ids": segment_token_ids,
            "correct": bool(correct),
            "extracted_answer": extract_final_answer(full_response),
            "generated_tokens": len(token_ids),
            "finish_reason": getattr(generated, "finish_reason", None),
            "continuation_text": continuation_text,
        })
    return candidate_rows


@torch.no_grad()
def score_child_candidates(
    *,
    runner: VLLMFeatureExporter,
    rows: list[dict[str, Any]],
    model: PrefixValueModel,
    calibration_temperature: float,
    device: torch.device,
    segment_size: int,
    token_dim: int,
    top_k: int,
    feature_temperature: float,
    score_batch_size: int,
) -> None:
    if not rows:
        return
    tokenizer = runner.tokenizer
    prompt_lens: list[int] = []
    full_ids: list[list[int]] = []
    response_ids_list: list[list[int]] = []
    response_texts: list[str] = []
    response_tokens: list[list[str]] = []
    for row in rows:
        rendered = str(row.pop("_rendered_prompt_for_scoring"))
        prompt_ids = [int(item) for item in tokenizer(rendered, add_special_tokens=False).input_ids]
        response_ids = [int(item) for item in row.get("prefix_token_ids", [])] + [
            int(item) for item in row.get("segment_token_ids", [])
        ]
        prompt_lens.append(len(prompt_ids))
        full_ids.append(prompt_ids + response_ids)
        response_ids_list.append(response_ids)
        response_texts.append(_tokenizer_decode(tokenizer, response_ids))
        response_tokens.append([
            str(item)
            for item in tokenizer.convert_ids_to_tokens(response_ids)
        ])

    extracted = runner.extract_from_ids(
        full_ids,
        prompt_lens,
        temperatures=[float(feature_temperature)] * len(full_ids),
        top_k=int(top_k),
        return_logprobs=True,
        return_hidden=False,
        device=device,
    )
    entries: list[dict[str, torch.Tensor]] = []
    for logprobs, response_ids, tokens, text in zip(
        extracted["logprobs"], response_ids_list, response_tokens, response_texts,
    ):
        masked = build_masked_concat_segment_obs_from_lp(
            logprobs[: len(response_ids)],
            tokens,
            text,
            segment_size=int(segment_size),
            token_dim=int(token_dim),
            device=device,
            segment_mode="fixed_window",
        )
        entries.append({
            "features": masked.features.detach().cpu().to(torch.float16),
            "token_mask": masked.token_mask.detach().cpu().to(torch.uint8),
        })

    scores: list[float] = []
    for start in range(0, len(entries), int(score_batch_size)):
        batch_entries = entries[start:start + int(score_batch_size)]
        batch = pad_prefix_entries([(entry, None) for entry in batch_entries])
        tensor_batch = {
            key: value.to(device)
            for key, value in batch.items()
            if isinstance(value, torch.Tensor)
        }
        output = model(
            tensor_batch["features"],
            tensor_batch["token_mask"],
            tensor_batch["segment_mask"],
            prompt_hidden=tensor_batch.get("prompt_hidden"),
        )
        probs = calibrated_probability(output["terminal_logits"], calibration_temperature)
        scores.extend(float(value) for value in probs.detach().cpu().tolist())

    for row, score in zip(rows, scores):
        row["child_pvm_score"] = float(score)
        row["feature_temperature"] = float(feature_temperature)


def write_jsonl_rows(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def strip_runtime_fields(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.pop("_rendered_prompt_for_scoring", None)
        stripped.append(item)
    return stripped


def init_plain_llm(cfg: Mapping[str, Any], args: argparse.Namespace, max_new_tokens: int):
    from vllm import LLM

    inference_cfg = cfg["inference"]
    return LLM(
        model=str(inference_cfg["model_name_or_path"]),
        tensor_parallel_size=(
            int(args.parallel_size)
            if args.parallel_size is not None else max(1, torch.cuda.device_count())
        ),
        max_model_len=int(max_new_tokens) + 2048,
        gpu_memory_utilization=(
            float(args.gpu_memory_utilization)
            if args.gpu_memory_utilization is not None
            else float(inference_cfg.get("gpu_memory_utilization", 0.80))
        ),
        enforce_eager=bool(inference_cfg.get("vllm_enforce_eager", False)),
        enable_prefix_caching=bool(inference_cfg.get("enable_prefix_caching", False)),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    continuation_path = resolve_path(args.continuations or paths["val_continuations"], TF_MIL_ROOT)
    source_dataset_path = resolve_path(args.source_dataset or paths["val_dataset"], TF_MIL_ROOT)
    feature_cache_path = resolve_path(args.feature_cache or paths["val_feature_cache"], TF_MIL_ROOT)
    checkpoint_path = resolve_path(args.pvm_checkpoint or paths["prefix_value_ckpt"], TF_MIL_ROOT)
    assert continuation_path is not None
    assert source_dataset_path is not None
    assert feature_cache_path is not None
    assert checkpoint_path is not None

    continuation_rows = read_jsonl(continuation_path)
    if int(args.max_prefixes) > 0:
        continuation_rows = continuation_rows[: int(args.max_prefixes)]
    source_rows = read_jsonl(source_dataset_path)
    source_by_id = {str(row["sample_id"]): row for row in source_rows}
    missing_sources = sorted({
        str(row["source_sample_id"])
        for row in continuation_rows
        if str(row["source_sample_id"]) not in source_by_id
    })
    if missing_sources:
        raise RuntimeError(f"source dataset missing {len(missing_sources)} ids: {missing_sources[:5]}")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise RuntimeError("segment candidate generation requires at least one GPU")
    device = torch.device("cuda:0")

    inference_cfg = cfg["inference"]
    segment_size = int(args.segment_size or cfg["data"]["segment_size"])
    response_token_budget = int(inference_cfg.get("max_new_tokens", 8192))
    max_new_tokens = int(args.max_new_tokens or response_token_budget)
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()

    if args.stage in {"generate", "all"}:
        cache = torch.load(feature_cache_path, map_location="cpu", weights_only=False)
        cache_by_id = {str(entry["sample_id"]): entry for entry in cache}
        missing_cache = sorted({
            str(row["source_sample_id"])
            for row in continuation_rows
            if str(row["source_sample_id"]) not in cache_by_id
        })
        if missing_cache:
            raise RuntimeError(f"feature cache missing {len(missing_cache)} ids: {missing_cache[:5]}")
        model, calibration_temperature = load_value_model(cfg, checkpoint_path, device)
        prefix_scores = score_prefixes(
            continuation_rows,
            cache_by_id=cache_by_id,
            model=model,
            calibration_temperature=calibration_temperature,
            device=device,
            batch_size=int(args.score_batch_size),
        )
        pvm_groups = assign_groups(prefix_scores)
        del model
        torch.cuda.empty_cache()

        llm = init_plain_llm(cfg, args, max_new_tokens)
        tokenizer = llm.get_tokenizer()
        started = time.perf_counter()
        written = sum(1 for _ in output_path.open("r", encoding="utf-8")) if args.append_output and output_path.exists() else 0
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() and not args.append_output:
            output_path.unlink()

        include_greedy = not bool(args.no_greedy)
        prefix_batch_size = max(1, int(args.prefix_batch_size))
        start_prefix = max(0, int(args.start_prefix))
        stop_prefix = int(args.stop_prefix) if int(args.stop_prefix) > 0 else len(continuation_rows)
        stop_prefix = min(stop_prefix, len(continuation_rows))
        if start_prefix >= stop_prefix:
            raise ValueError(f"empty prefix range: start_prefix={start_prefix} stop_prefix={stop_prefix}")
        for start in range(start_prefix, stop_prefix, prefix_batch_size):
            end = min(start + prefix_batch_size, stop_prefix)
            batch_records = continuation_rows[start:end]
            requests: list[dict[str, Any]] = []
            for local_offset, record in enumerate(batch_records):
                record_index = start + local_offset
                source_row = source_by_id[str(record["source_sample_id"])]
                rendered = render_prompt(tokenizer, source_row, cfg)
                requests.extend(candidate_requests(
                    record=record,
                    record_index=record_index,
                    source_row=source_row,
                    rendered_prompt=rendered,
                    prefix_score=prefix_scores[record_index],
                    pvm_group=pvm_groups[record_index],
                    temperatures=[float(item) for item in args.temperatures],
                    seeds_per_temperature=int(args.seeds_per_temperature),
                    base_seed=int(args.seed),
                    include_greedy=include_greedy,
                ))
            candidate_rows = generate_candidates_for_requests(
                llm=llm,
                tokenizer=tokenizer,
                requests=requests,
                response_token_budget=response_token_budget,
                max_new_tokens=max_new_tokens,
                segment_size=segment_size,
            )
            write_jsonl_rows(output_path, candidate_rows, append=args.append_output or written > 0)
            written += len(candidate_rows)
            elapsed = time.perf_counter() - started
            print(
                f"generate chunk prefixes={start}:{end} wrote={len(candidate_rows)} "
                f"total={written} elapsed={elapsed:.1f}s",
                flush=True,
            )

        elapsed = time.perf_counter() - started
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage": "generate",
            "config": str(args.config.resolve()),
            "continuations": str(continuation_path),
            "source_dataset": str(source_dataset_path),
            "feature_cache": str(feature_cache_path),
            "pvm_checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "n_prefixes": len(continuation_rows),
            "start_prefix": start_prefix,
            "stop_prefix": stop_prefix,
            "n_candidates": written,
            "appended_to_existing_output": bool(args.append_output),
            "include_greedy": include_greedy,
            "temperatures": [float(item) for item in args.temperatures],
            "seeds_per_temperature": int(args.seeds_per_temperature),
            "seed": int(args.seed),
            "segment_size": segment_size,
            "max_new_tokens": max_new_tokens,
            "response_token_budget": response_token_budget,
            "calibration_temperature": float(calibration_temperature),
            "parallel_size": args.parallel_size,
            "gpu_memory_utilization": (
                float(args.gpu_memory_utilization)
                if args.gpu_memory_utilization is not None
                else float(inference_cfg.get("gpu_memory_utilization", 0.80))
            ),
            "elapsed_seconds": elapsed,
        }
        write_json(manifest_path, manifest)
        print(f"wrote unscored segment candidates to {output_path}")
        print(f"manifest: {manifest_path}")
        print(f"elapsed_seconds={elapsed:.1f}")
        if args.stage == "generate":
            return
        del llm
        torch.cuda.empty_cache()

    if args.stage in {"score", "all"}:
        rows = read_jsonl(output_path)
        model, calibration_temperature = load_value_model(cfg, checkpoint_path, device)
        runner = VLLMFeatureExporter(
            model_name_or_path=str(inference_cfg["model_name_or_path"]),
            max_new_tokens=max_new_tokens,
            parallel_size=args.parallel_size,
            gpu_memory_utilization=(
                float(args.gpu_memory_utilization)
                if args.gpu_memory_utilization is not None
                else float(inference_cfg.get("gpu_memory_utilization", 0.80))
            ),
            reserve_training_gpu=False,
            max_batch_size=(
                int(args.vllm_micro_batch_size)
                if args.vllm_micro_batch_size is not None
                else inference_cfg.get("vllm_micro_batch_size")
            ),
            enforce_eager=bool(inference_cfg.get("vllm_enforce_eager", False)),
            enable_prefix_caching=False,
        )
        runner._lazy_init()
        started = time.perf_counter()
        scored_rows: list[dict[str, Any]] = []
        batch_size = max(1, int(args.prefix_batch_size) * ((0 if args.no_greedy else 1) + len(args.temperatures) * int(args.seeds_per_temperature)))
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = [dict(row) for row in rows[start:end]]
            for row in batch_rows:
                source_row = source_by_id[str(row["source_sample_id"])]
                row["_rendered_prompt_for_scoring"] = render_prompt(runner.tokenizer, source_row, cfg)
            score_child_candidates(
                runner=runner,
                rows=batch_rows,
                model=model,
                calibration_temperature=calibration_temperature,
                device=device,
                segment_size=segment_size,
                token_dim=int(cfg["data"]["instance_dim"]),
                top_k=int(args.top_k_logprobs),
                feature_temperature=float(args.feature_temperature),
                score_batch_size=int(args.score_batch_size),
            )
            scored_rows.extend(strip_runtime_fields(batch_rows))
            print(f"score candidates={start}:{end} total_scored={len(scored_rows)}", flush=True)

        tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
        write_jsonl_rows(tmp_path, scored_rows, append=False)
        tmp_path.replace(output_path)
        elapsed = time.perf_counter() - started
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "stage": "score",
            "config": str(args.config.resolve()),
            "continuations": str(continuation_path),
            "source_dataset": str(source_dataset_path),
            "feature_cache": str(feature_cache_path),
            "pvm_checkpoint": str(checkpoint_path),
            "output": str(output_path),
            "n_candidates": len(scored_rows),
            "temperatures": [float(item) for item in args.temperatures],
            "seeds_per_temperature": int(args.seeds_per_temperature),
            "seed": int(args.seed),
            "segment_size": segment_size,
            "max_new_tokens": max_new_tokens,
            "top_k_logprobs": int(args.top_k_logprobs),
            "feature_temperature": float(args.feature_temperature),
            "calibration_temperature": float(calibration_temperature),
            "parallel_size": args.parallel_size,
            "gpu_memory_utilization": (
                float(args.gpu_memory_utilization)
                if args.gpu_memory_utilization is not None
                else float(inference_cfg.get("gpu_memory_utilization", 0.80))
            ),
            "elapsed_seconds": elapsed,
        }
        write_json(manifest_path, manifest)
        print(f"wrote scored segment candidates to {output_path}")
        print(f"manifest: {manifest_path}")
        print(f"elapsed_seconds={elapsed:.1f}")


if __name__ == "__main__":
    main()
