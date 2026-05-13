#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Allow direct execution: python scripts/build_dataset.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from features.schema import BagSample
from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
from inference.vllm_runner import VLLMFeatureExporter
from utils.answer_verifier import verify_answer, self_consistency_correct
from utils.dataset_io import write_hidden_sidecar, write_jsonl
from utils.exp_logger import log_exception, setup_experiment_logger
from utils.jsonl import add_groupby_arg, split_by_group


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str, logger=None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
                if logger:
                    logger.warning("skipping malformed JSON at line %d (%.80s...)", line_no, line)
    if skipped and logger:
        logger.warning("skipped %d malformed lines out of %d total", skipped, line_no)
    return rows


def _write_split_with_sidecar(
    out_path: str,
    rows: List[Dict[str, Any]],
    hidden_tensors: List[Any],
) -> None:
    """Write a JSONL file + hidden safetensors sidecar for a split.

    Each row dict must have a temporary ``_idx`` key pointing into
    ``hidden_tensors``.  The key is removed before writing.
    Offsets are recomputed for this split.
    """
    tensors_for_sidecar: List[Any] = []
    current_offset = 0
    for row in rows:
        idx: int = row.pop("_idx")
        ht = hidden_tensors[idx]
        if ht is not None:
            row["_hidden_offset"] = current_offset
            row["_hidden_count"] = int(ht.shape[0])
            tensors_for_sidecar.append(ht)
            current_offset += int(ht.shape[0])
    write_jsonl(out_path, rows)
    if tensors_for_sidecar:
        write_hidden_sidecar(out_path, tensors_for_sidecar)


def build_dataset(
    config_path: str,
    input_path: str,
    output_path: str,
    train_out: str = "",
    val_out: str = "",
    test_out: str = "",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_seed: int = 42,
    group_by: str = "sample_prefix",
    backend: str = "vllm",
    run_name: str | None = None,
    log_dir: str = "logs",
) -> None:
    cfg = load_config(config_path)
    logger, log_path, final_run_name = setup_experiment_logger(
        component="build_dataset",
        run_name=run_name,
        log_dir=log_dir,
        config=cfg,
    )
    logger.info("input_path=%s output_path=%s backend=%s", input_path, output_path, backend)
    prompts = load_prompts(input_path, logger=logger)
    logger.info("n_input_rows=%d", len(prompts))

    inf_cfg = cfg["inference"]

    if backend == "api":
        from inference.api_runner import APIFeatureExporter

        api_cfg = inf_cfg.get("api", {})
        exporter = APIFeatureExporter(
            model_name_or_path=inf_cfg["model_name_or_path"],
            max_new_tokens=inf_cfg["max_new_tokens"],
            base_url=api_cfg.get("base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
            api_key=api_cfg.get("api_key") or None,
            max_concurrent=int(api_cfg.get("max_concurrent", 16)),
            max_retries=int(api_cfg.get("max_retries", 3)),
        )
        logger.info("backend=api base_url=%s max_concurrent=%d", exporter.base_url, exporter.max_concurrent)
    else:
        exporter = VLLMFeatureExporter(
            model_name_or_path=inf_cfg["model_name_or_path"],
            max_new_tokens=inf_cfg["max_new_tokens"],
            tensor_parallel_size=inf_cfg.get("tensor_parallel_size", "auto"),
            gpu_memory_utilization=float(inf_cfg.get("gpu_memory_utilization", 0.90)),
        )
        logger.info("backend=vllm")

    rows_prepared: List[Dict[str, Any]] = []
    for row in prompts:
        question = row.get("question") or row.get("problem") or row.get("prompt")
        if question is None:
            raise ValueError("Each input row must contain 'question' or 'problem' (or legacy 'prompt').")
        gold_answer = row.get("answer")
        if gold_answer is None:
            raise ValueError("Each input row must contain gold 'answer' for auto-labeling.")
        sample_base = row.get("sample_id") or row.get("unique_id") or str(uuid.uuid4())
        rows_prepared.append(
            {
                "question": question,
                "gold_answer": str(gold_answer),
                "sample_base": sample_base,
                "source": row.get("source", "unknown"),
                "subject": row.get("subject", ""),
                "level": row.get("level", ""),
            }
        )

    num_votes = int(inf_cfg.get("num_votes", 1))
    if num_votes < 1:
        raise ValueError(f"num_votes must be >= 1, got {num_votes}")
    logger.info("num_votes=%d majority_voting=%s", num_votes, "enabled" if num_votes > 1 else "disabled")

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    all_temps = [float(t) for t in inf_cfg["temperature_grid"]]
    n_temps = len(all_temps)
    total_rows = len(rows_prepared)
    all_questions = [r["question"] for r in rows_prepared]

    n_samples = 0
    n_positive = 0
    n_individual_correct = 0

    # ------------------------------------------------------------------
    # Generate all temperatures in one vLLM call (APC shares prompt KV)
    # or per-temperature for API backend.
    # ------------------------------------------------------------------
    if backend == "vllm":
        logger.info("multi_temp start n_prompts=%d n_temps=%d total_requests=%d",
                     total_rows, n_temps, total_rows * n_temps)
        exported_batch = exporter.export_token_features_multi_temp(
            prompts=all_questions,
            temperatures=all_temps,
            feature_mode=inf_cfg["feature_mode"],
            top_k_logits=int(inf_cfg["top_k_logits"]),
            use_math_chat_prompt=bool(inf_cfg.get("use_math_chat_prompt", True)),
            system_prompt=inf_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT),
            num_votes=num_votes,
        )
        logger.info("multi_temp done total_outputs=%d", len(exported_batch))

        # --- hidden state extraction (two-pass: prefill prompt+response) ---
        fmode = inf_cfg.get("feature_mode", "basic")
        if fmode in {"hidden_states", "all"}:
            # Save TP size and free the generation LLM before creating the extractor
            tp_size = exporter.tensor_parallel_size if hasattr(exporter, "tensor_parallel_size") else 1
            del exporter
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()

            from inference.vllm_hidden_extractor import VLLMHiddenStateExtractor

            layer_ids = inf_cfg.get("eagle_aux_hidden_state_layer_ids", [28])
            extractor = VLLMHiddenStateExtractor(
                model_name_or_path=inf_cfg["model_name_or_path"],
                layer_ids=[int(x) for x in layer_ids],
                tensor_parallel_size=tp_size,
            )
            prompt_response_pairs = [
                (ex["prompt"], ex["response"])
                for ex in exported_batch
            ]
            debug_tok = bool(inf_cfg.get("debug_token_check", False))
            logger.info("extracting hidden states for %d responses ...", len(prompt_response_pairs))
            all_hidden = extractor.extract(
                [p for p, _ in prompt_response_pairs],
                [r for _, r in prompt_response_pairs],
                debug_token_check=debug_tok,
            )
            extractor.cleanup()
            logger.info("hidden state extraction done")

        # ------------------------------------------------------------------
        # Build sample dicts.  For hidden_states/all mode we collect them
        # in memory, split by group, and write train/val/test directly.
        # For basic/topk_logits we write all_dataset.jsonl inline.
        # ------------------------------------------------------------------
        if fmode in {"hidden_states", "all"}:
            # --- Merged build+split: write train/val/test + sidecars ---
            all_sample_dicts: List[Dict[str, Any]] = []
            all_hidden_tensors: List[Any] = []
            batch_idx = 0

            for q_idx, row_obj in enumerate(rows_prepared):
                prompt_start = q_idx * n_temps * num_votes
                for t_idx, temp in enumerate(all_temps):
                    vote_start = prompt_start + t_idx * num_votes
                    vote_exports = exported_batch[vote_start : vote_start + num_votes]
                    vote_correct = [
                        1 if verify_answer(prediction=ex["response"], gold=row_obj["gold_answer"]) else 0
                        for ex in vote_exports
                    ]
                    n_individual_correct += sum(vote_correct)
                    n_correct = sum(vote_correct)
                    responses = [ex["response"] for ex in vote_exports]
                    majority_correct = self_consistency_correct(responses, row_obj["gold_answer"])
                    majority_label = 0 if majority_correct else 1

                    for v, exported in enumerate(vote_exports):
                        n_samples += 1
                        n_positive += (1 - majority_label)
                        token_features = exported["token_features"]
                        vid_suffix = f"_v{v}" if num_votes > 1 else ""

                        # Collect hidden tensor for this sample (native dtype, not JSONL)
                        ht = all_hidden[batch_idx] if batch_idx < len(all_hidden) else None
                        batch_idx += 1

                        sample = BagSample(
                            sample_id=f"{row_obj['sample_base']}_t{temp}{vid_suffix}",
                            prompt=row_obj["question"],
                            response=exported["response"],
                            label=majority_label,
                            temperature=temp,
                            token_features=token_features,
                            segment_spans=[],
                            metadata={
                                "feature_mode": inf_cfg["feature_mode"],
                                "source": row_obj["source"],
                                "subject": row_obj["subject"],
                                "level": row_obj["level"],
                                "gold_answer": row_obj["gold_answer"],
                                "rendered_prompt": exported.get("prompt", row_obj["question"]),
                                "vote_id": v,
                                "num_votes": num_votes,
                                "individual_correct": bool(vote_correct[v]),
                                "votes_correct": n_correct,
                                "votes_total": num_votes,
                            },
                        )
                        d = sample.to_binary_dict()
                        d["_idx"] = len(all_sample_dicts)
                        all_sample_dicts.append(d)
                        all_hidden_tensors.append(ht)

            # Split by group (same logic as split_jsonl.py)
            if not (0.0 < val_ratio < 1.0) or not (0.0 < test_ratio < 1.0) or (val_ratio + test_ratio >= 1.0):
                raise ValueError(f"Invalid split ratios: val={val_ratio} test={test_ratio}")
            train_val_rows, test_rows = split_by_group(all_sample_dicts, test_ratio, split_seed, group_by)
            val_frac = val_ratio / (1.0 - test_ratio)
            train_rows, val_rows = split_by_group(train_val_rows, val_frac, split_seed + 1, group_by)

            logger.info("split train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))

            _write_split_with_sidecar(train_out, train_rows, all_hidden_tensors)
            _write_split_with_sidecar(val_out, val_rows, all_hidden_tensors)
            _write_split_with_sidecar(test_out, test_rows, all_hidden_tensors)
        else:
            # --- Legacy flow: write all_dataset.jsonl inline (no sidecar) ---
            with out_file.open("w", encoding="utf-8") as f:
                for q_idx, row_obj in enumerate(rows_prepared):
                    prompt_start = q_idx * n_temps * num_votes
                    for t_idx, temp in enumerate(all_temps):
                        vote_start = prompt_start + t_idx * num_votes
                        vote_exports = exported_batch[vote_start : vote_start + num_votes]
                        vote_correct = [
                            1 if verify_answer(prediction=ex["response"], gold=row_obj["gold_answer"]) else 0
                            for ex in vote_exports
                        ]
                        n_individual_correct += sum(vote_correct)
                        n_correct = sum(vote_correct)
                        responses = [ex["response"] for ex in vote_exports]
                        majority_correct = self_consistency_correct(responses, row_obj["gold_answer"])
                        majority_label = 0 if majority_correct else 1

                        for v, exported in enumerate(vote_exports):
                            n_samples += 1
                            n_positive += (1 - majority_label)
                            token_features = exported["token_features"]
                            vid_suffix = f"_v{v}" if num_votes > 1 else ""
                            sample = BagSample(
                                sample_id=f"{row_obj['sample_base']}_t{temp}{vid_suffix}",
                                prompt=row_obj["question"],
                                response=exported["response"],
                                label=majority_label,
                                temperature=temp,
                                token_features=token_features,
                                segment_spans=[],
                                metadata={
                                    "feature_mode": inf_cfg["feature_mode"],
                                    "source": row_obj["source"],
                                    "subject": row_obj["subject"],
                                    "level": row_obj["level"],
                                    "gold_answer": row_obj["gold_answer"],
                                    "rendered_prompt": exported.get("prompt", row_obj["question"]),
                                    "vote_id": v,
                                    "num_votes": num_votes,
                                    "individual_correct": bool(vote_correct[v]),
                                    "votes_correct": n_correct,
                                    "votes_total": num_votes,
                                },
                            )
                            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
    else:
        # API backend: per-temperature loop (no APC benefit)
        with out_file.open("w", encoding="utf-8") as f:
            for t_idx, temp in enumerate(all_temps):
                logger.info("temperature=%.4f [%d/%d] start", temp, t_idx + 1, n_temps)

                exported_batch = exporter.export_token_features_batch(
                    prompts=all_questions,
                    temperature=temp,
                    feature_mode=inf_cfg["feature_mode"],
                    top_k_logits=int(inf_cfg["top_k_logits"]),
                    use_math_chat_prompt=bool(inf_cfg.get("use_math_chat_prompt", True)),
                    system_prompt=inf_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT),
                    num_votes=num_votes,
                )

                expected_size = total_rows * num_votes
                if len(exported_batch) != expected_size:
                    raise RuntimeError(
                        f"Unexpected generation size at temperature={temp}: "
                        f"got {len(exported_batch)}, expected {expected_size}"
                    )

                for q_idx, row_obj in enumerate(rows_prepared):
                    vote_exports = exported_batch[q_idx * num_votes : (q_idx + 1) * num_votes]
                    # Per-vote correctness (auxiliary statistic only — does NOT
                    # determine the bag label.  Label is determined by
                    # self-consistency majority voting below.)
                    vote_correct = [
                        1 if verify_answer(prediction=ex["response"], gold=row_obj["gold_answer"]) else 0
                        for ex in vote_exports
                    ]
                    n_individual_correct += sum(vote_correct)
                    n_correct = sum(vote_correct)
                    # Self-consistency: extract answer from each response,
                    # find modal (plurality) answer, compare to gold.
                    responses = [ex["response"] for ex in vote_exports]
                    majority_correct = self_consistency_correct(responses, row_obj["gold_answer"])
                    majority_label = 0 if majority_correct else 1  # 0=correct, 1=error

                    for v, exported in enumerate(vote_exports):
                        n_samples += 1
                        n_positive += (1 - majority_label)
                        token_features = exported["token_features"]
                        vid_suffix = f"_v{v}" if num_votes > 1 else ""
                        sample = BagSample(
                            sample_id=f"{row_obj['sample_base']}_t{temp}{vid_suffix}",
                            prompt=row_obj["question"],
                            response=exported["response"],
                            label=majority_label,
                            temperature=temp,
                            token_features=token_features,
                            segment_spans=[],  # computed by BagDataset at load time
                            metadata={
                                "feature_mode": inf_cfg["feature_mode"],
                                "source": row_obj["source"],
                                "subject": row_obj["subject"],
                                "level": row_obj["level"],
                                "gold_answer": row_obj["gold_answer"],
                                "rendered_prompt": exported.get("prompt", row_obj["question"]),
                                "vote_id": v,
                                "num_votes": num_votes,
                                "individual_correct": bool(vote_correct[v]),
                                "votes_correct": n_correct,
                                "votes_total": num_votes,
                            },
                        )
                        f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

                logger.info("temperature=%.4f [%d/%d] done emitted=%d", temp, t_idx + 1, n_temps, total_rows * num_votes)

    logger.info(
        "dataset_done run_name=%s n_samples=%d positive_ratio=%.6f individual_accuracy=%.4f num_votes=%d log_path=%s",
        final_run_name,
        n_samples,
        (float(n_positive) / max(1, n_samples)),
        (float(n_individual_correct) / max(1, n_samples)),
        num_votes,
        log_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None, help="Override paths.raw_input from config")
    parser.add_argument("--output", default=None, help="Override paths.all_dataset from config")
    parser.add_argument("--train-output", default=None, help="Override paths.train_dataset from config")
    parser.add_argument("--val-output", default=None, help="Override paths.val_dataset from config")
    parser.add_argument("--test-output", default=None, help="Override paths.test_dataset from config")
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--backend", default="vllm", choices=["vllm", "api"],
                        help="Inference backend: vllm (local GPU) or api (DashScope / OpenAI-compatible)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    add_groupby_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_path = args.input or cfg["paths"]["raw_input"]
    output_path = args.output or cfg["paths"]["all_dataset"]
    paths_cfg = cfg.get("paths", {})
    train_out = args.train_output or paths_cfg.get("train_dataset", str(Path(output_path).parent / "train.jsonl"))
    val_out = args.val_output or paths_cfg.get("val_dataset", str(Path(output_path).parent / "val.jsonl"))
    test_out = args.test_output or paths_cfg.get("test_dataset", str(Path(output_path).parent / "test.jsonl"))
    try:
        build_dataset(
            args.config, input_path, output_path,
            train_out=train_out, val_out=val_out, test_out=test_out,
            val_ratio=args.val_ratio, test_ratio=args.test_ratio,
            split_seed=args.split_seed, group_by=args.group_by,
            backend=args.backend, run_name=args.run_name, log_dir=args.log_dir,
        )
    except Exception as exc:
        cfg = load_config(args.config)
        logger, _, _ = setup_experiment_logger(
            component="build_dataset",
            run_name=args.run_name,
            log_dir=args.log_dir,
            config=cfg,
        )
        log_exception(logger, exc)
        raise


if __name__ == "__main__":
    main()
