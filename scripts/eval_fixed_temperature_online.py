#!/usr/bin/env python3
"""Online evaluation for a fixed-temperature segment rollout."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.vllm_runner import VLLMFeatureExporter
from ppo.model import PrefixPolicyValueNet
from ppo.prefix_rollout import PrefixRolloutEngine
from ppo.prefix_training import _load_value_model
from scripts.eval_q_selector import load_eval_prompts
from utils.calibration import (
    binary_nll,
    brier_score,
    expected_calibration_error,
)


def _temperature_distribution(temperatures: List[List[float]]) -> Dict[str, int]:
    flat = [float(temp) for row in temperatures for temp in row]
    return {str(temp): count for temp, count in sorted(Counter(flat).items())}


def summarize_fixed_rollout(method: str, seed: int, fixed_temperature: float,
                            rollout: Any, prompts: List[Dict[str, Any]],
                            n_votes: int, elapsed: float) -> Dict[str, Any]:
    n_prompts = len(prompts)
    total_individual = sum(sum(row) for row in rollout.individual_correct)
    total_tokens = sum(sum(row) for row in rollout.token_counts)
    total_segments = sum(sum(row) for row in rollout.segment_counts)
    confidences = [float(value) for value in rollout.sc_confidences]
    correctness = [int(value) for value in rollout.majority_correct]
    pass_at_1_total = sum(row[0] for row in rollout.individual_correct if row)
    predictions: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        predictions.append({
            "problem_id": prompt["problem_id"],
            "majority_correct": int(rollout.majority_correct[idx]),
            "individual_correct": [int(value) for value in rollout.individual_correct[idx]],
            "extracted_answers": rollout.extracted_answers[idx],
            "majority_answer": rollout.majority_answers[idx],
            "majority_count": int(rollout.majority_counts[idx]),
            "sc_confidence": float(rollout.sc_confidences[idx]),
            "answer_entropy": float(rollout.answer_entropies[idx]),
            "temperatures": rollout.temperatures[idx],
            "segment_counts": rollout.segment_counts[idx],
            "token_counts": rollout.token_counts[idx],
        })
    return {
        "method": method,
        "seed": int(seed),
        "fixed_temperature": float(fixed_temperature),
        "n_prompts": n_prompts,
        "num_votes": int(n_votes),
        "majority_accuracy": sum(correctness) / max(1, n_prompts),
        "pass_at_1_accuracy": pass_at_1_total / max(1, n_prompts),
        "individual_accuracy": total_individual / max(1, n_prompts * n_votes),
        "ece": expected_calibration_error(confidences, correctness),
        "brier": brier_score(confidences, correctness),
        "nll": binary_nll(confidences, correctness),
        "mean_confidence": sum(confidences) / max(1, len(confidences)),
        "mean_answer_entropy": sum(rollout.answer_entropies) / max(1, len(rollout.answer_entropies)),
        "average_tokens": total_tokens / max(1, n_prompts * n_votes),
        "mean_tokens_per_vote": total_tokens / max(1, n_prompts * n_votes),
        "mean_segments_per_vote": total_segments / max(1, n_prompts * n_votes),
        "total_tokens": total_tokens,
        "selected_temperature_distribution": _temperature_distribution(rollout.temperatures),
        "wall_seconds": float(elapsed),
        "predictions": predictions,
    }


@torch.no_grad()
def evaluate_fixed_temperature(config_path: str, input_path: str | None = None,
                               temperature: float = 1.0, seed: int = 42,
                               parallel_size: int | None = None,
                               max_prompts: int | None = None,
                               gpu_memory_utilization: float | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts, data_path = load_eval_prompts(cfg, input_path=input_path, max_prompts=max_prompts)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise RuntimeError("Fixed-temperature online evaluation requires GPUs")
    device = torch.device(f"cuda:{n_gpu - 1}")
    value_model, calibration_temperature = _load_value_model(cfg, device)
    temp_bins = [float(value) for value in cfg["data"]["temp_bins"]]
    runner = VLLMFeatureExporter(
        model_name_or_path=cfg["inference"]["model_name_or_path"],
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        parallel_size=parallel_size,
        gpu_memory_utilization=(
            float(gpu_memory_utilization)
            if gpu_memory_utilization is not None
            else float(cfg["inference"].get("gpu_memory_utilization", 0.90))
        ),
        reserve_training_gpu=False,
        max_batch_size=cfg["inference"].get("vllm_micro_batch_size"),
        enforce_eager=bool(cfg["inference"].get("vllm_enforce_eager", False)),
        enable_prefix_caching=cfg["inference"].get("enable_prefix_caching"),
    )
    training_cfg = cfg.get("ppo", {}).get("training", {})
    engine = PrefixRolloutEngine(
        runner=runner,
        value_model=value_model,
        calibration_temperature=calibration_temperature,
        device=device,
        temp_bins=temp_bins,
        segment_size=int(cfg["data"]["segment_size"]),
        token_dim=int(cfg["data"]["instance_dim"]),
        top_k_logprobs=int(cfg["inference"]["top_k_logprobs"]),
        num_votes=int(cfg["inference"]["num_votes"]),
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        gamma=float(training_cfg.get("gamma", 0.99)),
        shaping_coef=float(training_cfg.get("shaping_coef", 0.0)),
        system_prompt=str(cfg["inference"].get("system_prompt", "")),
        use_math_chat=bool(cfg["inference"].get("use_math_chat_prompt", True)),
    )
    policy = PrefixPolicyValueNet(value_model.hidden_dim, len(temp_bins)).to(device)
    policy.eval()
    started = time.perf_counter()
    rollout = engine.rollout(
        prompts,
        policy,
        stochastic=False,
        rng=random.Random(seed),
        collect_transitions=False,
        generation_seed=seed,
        fixed_temperature=float(temperature),
    )
    elapsed = time.perf_counter() - started
    result = summarize_fixed_rollout(
        f"fixed_temperature_{float(temperature):.1f}",
        seed,
        float(temperature),
        rollout,
        prompts,
        int(cfg["inference"]["num_votes"]),
        elapsed,
    )
    result.update({
        "config": config_path,
        "input_path": data_path,
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate_fixed_temperature(
        args.config,
        input_path=args.input,
        temperature=args.temperature,
        seed=args.seed,
        parallel_size=args.parallel_size,
        max_prompts=args.max_prompts,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    compact = {key: value for key, value in metrics.items() if key != "predictions"}
    print(json.dumps(compact, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
