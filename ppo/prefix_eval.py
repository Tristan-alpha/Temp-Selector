"""Online evaluation for the complete prefix-value PPO proposal."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml

from features.dataset_eval import load_temperature_labels
from inference.vllm_runner import VLLMFeatureExporter
from ppo.model import PrefixPolicyValueNet
from ppo.prefix_rollout import PrefixRolloutEngine
from ppo.prefix_training import _load_value_model, load_prompts
from utils.exp_logger import setup_experiment_logger


def select_best_fixed_temperature(validation_dataset_path: str,
                                  temp_bins: List[float]) -> float:
    labels = load_temperature_labels(validation_dataset_path)
    default = temp_bins[len(temp_bins) // 2]
    if not labels:
        return float(default)
    accuracy = {
        float(temp): sum(values) / len(values)
        for temp, values in labels.items()
        if values
    }
    if not accuracy:
        return float(default)
    candidates = [float(temp) for temp in sorted(temp_bins) if float(temp) in accuracy]
    if not candidates:
        candidates = sorted(accuracy)
    return float(max(candidates, key=lambda temp: accuracy[temp]))


def _summarize_rollout(method: str, seed: int, rollout,
                       prompts: List[Dict[str, Any]], n_votes: int,
                       elapsed: float) -> Dict[str, Any]:
    n_prompts = len(prompts)
    total_individual = sum(sum(row) for row in rollout.individual_correct)
    total_tokens = sum(sum(row) for row in rollout.token_counts)
    total_segments = sum(sum(row) for row in rollout.segment_counts)
    predictions: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        predictions.append({
            "problem_id": prompt["problem_id"],
            "majority_correct": rollout.majority_correct[idx],
            "individual_correct": rollout.individual_correct[idx],
            "extracted_answers": rollout.extracted_answers[idx],
            "majority_answer": rollout.majority_answers[idx],
            "majority_count": rollout.majority_counts[idx],
            "sc_confidence": rollout.sc_confidences[idx],
            "answer_entropy": rollout.answer_entropies[idx],
            "temperatures": rollout.temperatures[idx],
            "segment_counts": rollout.segment_counts[idx],
            "token_counts": rollout.token_counts[idx],
        })
    return {
        "method": method,
        "seed": seed,
        "n_prompts": n_prompts,
        "num_votes": n_votes,
        "majority_accuracy": sum(rollout.majority_correct) / max(1, n_prompts),
        "individual_accuracy": total_individual / max(1, n_prompts * n_votes),
        "mean_tokens_per_vote": total_tokens / max(1, n_prompts * n_votes),
        "mean_segments_per_vote": total_segments / max(1, n_prompts * n_votes),
        "total_tokens": total_tokens,
        "wall_seconds": elapsed,
        "predictions": predictions,
    }


def _compact_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key, value in metrics.items():
        if key == "predictions":
            continue
        if isinstance(value, dict):
            compact[key] = _compact_metrics(value)
        else:
            compact[key] = value
    return compact


def evaluate(config_path: str, seed: int, parallel_size: int | None = None,
             include_random: bool = False) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts = load_prompts(cfg["paths"]["test_dataset"])
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise RuntimeError("Prefix PPO evaluation requires GPUs")
    device = torch.device(f"cuda:{n_gpu - 1}")
    value_model, calibration_temperature = _load_value_model(cfg, device)
    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    best_fixed_temp = select_best_fixed_temperature(cfg["paths"]["val_dataset"], temp_bins)
    policy = PrefixPolicyValueNet(value_model.hidden_dim, len(temp_bins)).to(device)
    checkpoint = torch.load(cfg["paths"]["ppo_ckpt"], map_location=device, weights_only=False)
    policy.load_state_dict(checkpoint["prefix_policy_value"])
    policy.eval()

    runner = VLLMFeatureExporter(
        model_name_or_path=cfg["inference"]["model_name_or_path"],
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        parallel_size=parallel_size,
        gpu_memory_utilization=float(cfg["inference"].get("gpu_memory_utilization", 0.90)),
        reserve_training_gpu=True,
        max_batch_size=cfg["inference"].get("vllm_micro_batch_size"),
        enforce_eager=bool(cfg["inference"].get("vllm_enforce_eager", False)),
        enable_prefix_caching=cfg["inference"].get("enable_prefix_caching"),
    )
    training_cfg = cfg["ppo"]["training"]
    engine = PrefixRolloutEngine(
        runner=runner, value_model=value_model,
        calibration_temperature=calibration_temperature, device=device,
        temp_bins=temp_bins, segment_size=int(cfg["data"]["segment_size"]),
        token_dim=int(cfg["data"]["instance_dim"]),
        top_k_logprobs=int(cfg["inference"]["top_k_logprobs"]),
        num_votes=int(cfg["inference"]["num_votes"]),
        max_new_tokens=int(cfg["inference"]["max_new_tokens"]),
        gamma=float(training_cfg["gamma"]),
        shaping_coef=float(training_cfg["shaping_coef"]),
        system_prompt=str(cfg["inference"].get("system_prompt", "")),
        use_math_chat=bool(cfg["inference"].get("use_math_chat_prompt", True)),
    )
    started = time.perf_counter()
    rollout = engine.rollout(
        prompts, policy, stochastic=False, rng=random.Random(seed),
        collect_transitions=False,
        generation_seed=seed,
    )
    elapsed = time.perf_counter() - started
    n_votes = int(cfg["inference"]["num_votes"])
    result = _summarize_rollout(
        "full_prefix_value", seed, rollout, prompts, n_votes, elapsed,
    )

    if hasattr(runner, "reset_prefix_cache"):
        runner.reset_prefix_cache(reset_connector=True)
    fixed_started = time.perf_counter()
    fixed_rollout = engine.rollout(
        prompts, policy, stochastic=False, rng=random.Random(seed),
        collect_transitions=False,
        generation_seed=seed,
        fixed_temperature=best_fixed_temp,
    )
    fixed_elapsed = time.perf_counter() - fixed_started
    fixed_metrics = _summarize_rollout(
        "validation_best_fixed", seed, fixed_rollout, prompts, n_votes, fixed_elapsed,
    )
    result["best_fixed_temperature"] = best_fixed_temp
    result["best_fixed_selection"] = {
        "source": "validation",
        "validation_dataset": cfg["paths"]["val_dataset"],
    }
    result["best_fixed"] = fixed_metrics
    result["comparison"] = {
        "majority_accuracy_delta": (
            result["majority_accuracy"] - fixed_metrics["majority_accuracy"]
        ),
        "individual_accuracy_delta": (
            result["individual_accuracy"] - fixed_metrics["individual_accuracy"]
        ),
        "total_token_delta": result["total_tokens"] - fixed_metrics["total_tokens"],
        "mean_segments_per_vote_delta": (
            result["mean_segments_per_vote"] -
            fixed_metrics["mean_segments_per_vote"]
        ),
    }
    if include_random:
        if hasattr(runner, "reset_prefix_cache"):
            runner.reset_prefix_cache(reset_connector=True)
        random_started = time.perf_counter()
        random_rollout = engine.rollout(
            prompts, policy, stochastic=False, rng=random.Random(seed),
            collect_transitions=False, generation_seed=seed,
            random_temperature=True,
        )
        random_elapsed = time.perf_counter() - random_started
        result["random_temperature_per_segment"] = _summarize_rollout(
            "random_temperature_per_segment", seed, random_rollout,
            prompts, n_votes, random_elapsed,
        )
    result["single_seed_evidence"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--include-random", action="store_true")
    args = parser.parse_args()
    logger, _, _ = setup_experiment_logger(
        component="prefix_online_eval", run_name=args.run_name,
        log_dir=args.log_dir, config={"config": args.config, "seed": args.seed},
    )
    metrics = evaluate(args.config, args.seed, args.parallel_size, args.include_random)
    output = args.output
    if output is None:
        output = f"results/full_seed{args.seed}.json"
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("metrics=%s", json.dumps(_compact_metrics(metrics)))
    print(json.dumps(_compact_metrics(metrics), indent=2))
    print(f"output={path}")


if __name__ == "__main__":
    main()
