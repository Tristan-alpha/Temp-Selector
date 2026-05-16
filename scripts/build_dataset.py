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
from inference.sglang_runner import SGLangRunner
from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
from inference.vllm_runner import VLLMFeatureExporter
from utils.answer_verifier import verify_answer, self_consistency_correct
from utils.jsonl import write_jsonl
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

    if backend == "vllm":
        exporter = VLLMFeatureExporter(
            model_name_or_path=inf_cfg["model_name_or_path"],
            max_new_tokens=inf_cfg["max_new_tokens"],
            parallel_size=inf_cfg.get("parallel_size", "auto"),
            gpu_memory_utilization=float(inf_cfg.get("gpu_memory_utilization", 0.90)),
            feature_mode=inf_cfg.get("feature_mode", "basic"),
        )
        logger.info("backend=vllm")
    else:
        # sglang (default)
        exporter = SGLangRunner(
            model_name_or_path=inf_cfg["model_name_or_path"],
            max_new_tokens=inf_cfg["max_new_tokens"],
            parallel_size=inf_cfg.get("parallel_size", "auto"),
            gpu_memory_utilization=float(inf_cfg.get("gpu_memory_utilization", 0.90)),
            feature_mode=inf_cfg.get("feature_mode", "basic"),
            log_level="info",
        )
        logger.info("backend=sglang")

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
    # Generate all temperatures.  vLLM uses APC for KV-cache sharing;
    # SGLang uses radix cache; API backend uses per-temperature loop.
    # ------------------------------------------------------------------
    if backend in ("vllm", "sglang"):
        logger.info("multi_temp start n_prompts=%d n_temps=%d total_requests=%d backend=%s",
                     total_rows, n_temps, total_rows * n_temps, backend)
        exported_batch = exporter.export_token_features_multi_temp(
            prompts=all_questions,
            temperatures=all_temps,
            top_k_logits=int(inf_cfg["top_k_logits"]),
            use_math_chat_prompt=bool(inf_cfg.get("use_math_chat_prompt", True)),
            system_prompt=inf_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT),
            num_votes=num_votes,
        )
        logger.info("multi_temp done total_outputs=%d", len(exported_batch))

        # ------------------------------------------------------------------
        fmode = inf_cfg.get("feature_mode", "basic")

        # Build sample dicts.  For hidden_states/all mode we collect them
        # in memory, split by group, and write train/val/test directly.
        # For basic/topk_logits we write all_dataset.jsonl inline.
        # ------------------------------------------------------------------
        if fmode in {"hidden_states", "all"}:
            # --- Merged build+split: write train/val/test JSONL (no safetensors sidecar).
            #     Hidden states are extracted on-demand during MIL training via
            #     SGLangRunner. ---
            all_sample_dicts: List[Dict[str, Any]] = []

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
                        all_sample_dicts.append(sample.to_dict())

            # Split by group (same logic as split_jsonl.py)
            if not (0.0 < val_ratio < 1.0) or not (0.0 < test_ratio < 1.0) or (val_ratio + test_ratio >= 1.0):
                raise ValueError(f"Invalid split ratios: val={val_ratio} test={test_ratio}")
            train_val_rows, test_rows = split_by_group(all_sample_dicts, test_ratio, split_seed, group_by)
            val_frac = val_ratio / (1.0 - test_ratio)
            train_rows, val_rows = split_by_group(train_val_rows, val_frac, split_seed + 1, group_by)

            logger.info("split train=%d val=%d test=%d", len(train_rows), len(val_rows), len(test_rows))

            write_jsonl(train_out, train_rows)
            write_jsonl(val_out, val_rows)
            write_jsonl(test_out, test_rows)
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
    parser.add_argument("--backend", default=None, choices=["sglang", "vllm"],
                        help="Inference backend (default: read from config, fallback sglang)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    add_groupby_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    backend = args.backend or cfg["inference"].get("backend", "sglang")
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
            backend=backend, run_name=args.run_name, log_dir=args.log_dir,
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
