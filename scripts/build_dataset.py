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
from vllm import LLM, SamplingParams

from inference.vllm_runner import DEFAULT_MATH_SYSTEM_PROMPT
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
    run_name: str | None = None,
    log_dir: str = "logs",
    parallel_size: int | None = None,
) -> None:
    cfg = load_config(config_path)
    logger, log_path, final_run_name = setup_experiment_logger(
        component="build_dataset",
        run_name=run_name,
        log_dir=log_dir,
        config=cfg,
    )
    logger.info("input_path=%s output_path=%s", input_path, output_path)
    prompts = load_prompts(input_path, logger=logger)
    logger.info("n_input_rows=%d", len(prompts))

    inf_cfg = cfg["inference"]

    # Raw LLM — no speculative decode overhead for generation-only use
    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        raise RuntimeError("No GPUs available")
    tp_size = parallel_size if parallel_size is not None else n_gpus
    if tp_size < 1:
        raise RuntimeError(f"Invalid parallel_size: {parallel_size}")

    model_path = inf_cfg["model_name_or_path"]
    max_new_tokens = int(inf_cfg["max_new_tokens"])
    max_model_len = max_new_tokens + 2048
    gpu_mem = float(inf_cfg.get("gpu_memory_utilization", 0.90))
    llm = LLM(model=model_path, tensor_parallel_size=tp_size,
              max_model_len=max_model_len, gpu_memory_utilization=gpu_mem)
    tokenizer = llm.get_tokenizer()

    use_math_chat = bool(inf_cfg.get("use_math_chat_prompt", True))
    system_prompt = inf_cfg.get("system_prompt", DEFAULT_MATH_SYSTEM_PROMPT)

    rows_prepared: List[Dict[str, Any]] = []
    for row in prompts:
        question = row.get("question") or row.get("problem") or row.get("prompt")
        if question is None:
            raise ValueError("Each input row must contain 'question' or 'problem' (or legacy 'prompt').")
        gold_answer = row.get("answer")
        if gold_answer is None:
            raise ValueError("Each input row must contain gold 'answer' for auto-labeling.")
        sample_base = row.get("sample_id") or row.get("unique_id") or str(uuid.uuid4())
        rows_prepared.append({
            "question": question,
            "gold_answer": str(gold_answer),
            "sample_base": sample_base,
            "source": row.get("source", "unknown"),
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
        })

    num_votes = int(inf_cfg.get("num_votes", 1))
    if num_votes < 1:
        raise ValueError(f"num_votes must be >= 1, got {num_votes}")
    logger.info("num_votes=%d majority_voting=%s", num_votes, "enabled" if num_votes > 1 else "disabled")

    all_temps = [float(t) for t in inf_cfg["temperature_grid"]]
    n_temps = len(all_temps)
    total_rows = len(rows_prepared)

    # Render prompts
    rendered_prompts: List[str] = []
    for q in rows_prepared:
        question = q["question"]
        if use_math_chat:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            try:
                rp = tokenizer.apply_chat_template(messages, tokenize=False,
                                                    add_generation_prompt=True,
                                                    enable_thinking=False)
            except Exception:
                rp = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{question}\n\n[ASSISTANT]\n"
        else:
            rp = question
        rendered_prompts.append(rp)

    # Interleave: prompt0@T0, prompt0@T1, ..., prompt1@T0, ...
    all_prompts: List[str] = []
    all_params: List[SamplingParams] = []
    for rp in rendered_prompts:
        for temp in all_temps:
            all_prompts.append(rp)
            all_params.append(SamplingParams(n=num_votes, temperature=temp,
                                              max_tokens=max_new_tokens,
                                              top_p=1.0, top_k=0))

    logger.info("multi_temp start n_prompts=%d n_temps=%d total_requests=%d",
                 total_rows, n_temps, len(all_prompts))
    outputs = llm.generate(all_prompts, all_params, use_tqdm=True)
    logger.info("multi_temp done total_outputs=%d", len(outputs))

    # Build sample dicts
    all_sample_dicts: List[Dict[str, Any]] = []
    n_samples = 0
    n_positive = 0
    n_individual_correct = 0

    for q_idx, row_obj in enumerate(rows_prepared):
        for t_idx, temp in enumerate(all_temps):
            req_idx = q_idx * n_temps + t_idx
            req_out = outputs[req_idx]

            vote_texts: List[str] = []
            vote_token_ids: List[List[int]] = []
            vote_tokens: List[List[str]] = []
            vote_correct: List[int] = []

            for v, out in enumerate(req_out.outputs):
                vote_texts.append(out.text)
                vote_ids = out.token_ids
                vote_token_ids.append(vote_ids)
                vote_tokens.append([tokenizer.decode([tid]) if tokenizer else ""
                                     for tid in vote_ids])
                vote_correct.append(
                    1 if verify_answer(prediction=out.text, gold=row_obj["gold_answer"]) else 0)

            n_individual_correct += sum(vote_correct)
            n_correct = sum(vote_correct)
            majority_correct = self_consistency_correct(vote_texts, row_obj["gold_answer"])
            majority_label = 0 if majority_correct else 1

            for v in range(num_votes):
                n_samples += 1
                if majority_label == 0:
                    n_positive += 1
                vid_suffix = f"_v{v}" if num_votes > 1 else ""
                sample = {
                    "sample_id": f"{row_obj['sample_base']}_t{temp}{vid_suffix}",
                    "prompt": row_obj["question"],
                    "response": vote_texts[v],
                    "label": majority_label,
                    "temperature": temp,
                    "token_ids": vote_token_ids[v],
                    "tokens": vote_tokens[v],
                    "metadata": {
                        "source": row_obj["source"],
                        "subject": row_obj["subject"],
                        "level": row_obj["level"],
                        "gold_answer": row_obj["gold_answer"],
                        "rendered_prompt": rendered_prompts[q_idx],
                        "vote_id": v,
                        "num_votes": num_votes,
                        "individual_correct": bool(vote_correct[v]),
                        "votes_correct": n_correct,
                        "votes_total": num_votes,
                    },
                }
                all_sample_dicts.append(sample)

    # Split and write
    if train_out and val_out and test_out:
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
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with out_file.open("w", encoding="utf-8") as f:
            for row in all_sample_dicts:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

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
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--parallel-size", type=int, default=None)
    add_groupby_arg(parser)
    args = parser.parse_args()
    cfg = load_config(args.config)
    input_path = args.input or cfg["paths"]["raw_input"]
    output_path = args.output or cfg["paths"]["all_dataset"]
    paths_cfg = cfg.get("paths", {})
    train_out = args.train_output or paths_cfg.get("train_dataset", str(Path(output_path).parent / "train.jsonl"))
    val_out = args.val_output or paths_cfg.get("val_dataset", str(Path(output_path).parent / "val.jsonl"))
    test_out = args.test_output or paths_cfg.get("test_dataset", str(Path(output_path).parent / "test.jsonl"))
    split_cfg = cfg.get("split", {})
    val_ratio = args.val_ratio if args.val_ratio is not None else float(split_cfg.get("val_ratio", 0.1))
    test_ratio = args.test_ratio if args.test_ratio is not None else float(split_cfg.get("test_ratio", 0.1))
    split_seed = args.split_seed if args.split_seed is not None else int(cfg.get("seed", 42))
    try:
        build_dataset(
            args.config, input_path, output_path,
            train_out=train_out, val_out=val_out, test_out=test_out,
            val_ratio=val_ratio, test_ratio=test_ratio,
            split_seed=split_seed, group_by=args.group_by,
            run_name=args.run_name, log_dir=args.log_dir,
            parallel_size=args.parallel_size,
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
