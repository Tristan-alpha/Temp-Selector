"""PPO training over frozen causal Prefix Value Model hidden states."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
import yaml

from features.dataset_eval import load_temperature_labels
from inference.vllm_runner import VLLMFeatureExporter
from mil.prefix_value import PrefixValueModel
from ppo.model import PrefixPolicyValueNet, compute_gae
from ppo.prefix_rollout import PrefixRolloutEngine
from utils.exp_logger import log_exception, setup_experiment_logger
from utils.jsonl import sample_prefix


def load_prompts(dataset_path: str) -> List[Dict[str, str]]:
    seen = set()
    prompts: List[Dict[str, str]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            pid = sample_prefix(str(row.get("sample_id", "")))
            if pid in seen:
                continue
            seen.add(pid)
            prompts.append({
                "problem_id": pid,
                "question": str(row.get("prompt", "")),
                "answer": str(row.get("metadata", {}).get("gold_answer", "")),
            })
    return prompts


def _load_value_model(cfg: Dict[str, Any], device: torch.device
                      ) -> tuple[PrefixValueModel, float]:
    checkpoint = torch.load(
        cfg["paths"]["prefix_value_ckpt"], map_location=device, weights_only=False,
    )
    model = PrefixValueModel(
        token_dim=int(cfg["data"]["instance_dim"]),
        segment_size=int(cfg["data"]["segment_size"]),
        hidden_dim=int(cfg["prefix_value"]["model"]["hidden_dim"]),
        max_segments=int(cfg["prefix_value"]["model"].get("max_segments", 8192)),
    ).to(device)
    model.load_state_dict(checkpoint["prefix_value"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, float(checkpoint.get("calibration_temperature", 1.0))


def _cpu_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _checkpoint_variants(output: Path) -> tuple[Path, Path]:
    suffix = output.suffix or ".pt"
    stem = output.name[:-len(output.suffix)] if output.suffix else output.name
    latest = output.with_name(f"{stem}.latest{suffix}")
    best = output.with_name(f"{stem}.best{suffix}")
    return latest, best


def _build_checkpoint(
    *,
    cfg: Dict[str, Any],
    policy_state: Dict[str, torch.Tensor],
    iteration: int,
    best_validation_accuracy: float,
    current_validation_accuracy: float | None,
    current_train_accuracy: float | None,
    patience: int,
    run_name: str,
) -> Dict[str, Any]:
    return {
        "prefix_policy_value": policy_state,
        "prefix_value_ckpt": cfg["paths"]["prefix_value_ckpt"],
        "best_validation_accuracy": best_validation_accuracy,
        "current_validation_accuracy": current_validation_accuracy,
        "current_train_accuracy": current_train_accuracy,
        "iteration": iteration,
        "patience": patience,
        "run_name": run_name,
        "config": cfg,
    }


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    vx = [x - mean_x for x in xs]
    vy = [y - mean_y for y in ys]
    denom_x = sum(x * x for x in vx)
    denom_y = sum(y * y for y in vy)
    if denom_x <= 0.0 or denom_y <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(vx, vy)) / ((denom_x * denom_y) ** 0.5)


def _bucket_temperature_stats(transitions: List[Any]) -> Dict[str, Any]:
    if not transitions:
        return {"count": 0, "mean_temperature": 0.0}
    temps = [float(t.temperature) for t in transitions]
    return {
        "count": len(transitions),
        "mean_temperature": _mean(temps),
        "min_temperature": min(temps),
        "max_temperature": max(temps),
    }


def _rollout_diagnostics(rollout) -> Dict[str, Any]:
    flat = []
    by_stage: Dict[str, List[Any]] = defaultdict(list)
    for prompt_chains in rollout.transitions:
        for chain in prompt_chains:
            chain_len = max(1, len(chain))
            for pos, transition in enumerate(chain):
                flat.append(transition)
                q = (pos + 1) / chain_len
                if q < 0.33:
                    stage = "early"
                elif q < 0.66:
                    stage = "middle"
                else:
                    stage = "late"
                by_stage[stage].append(transition)
    if not flat:
        return {"n_transitions": 0}

    final_correct = [float(t.final_correct) for t in flat]
    phi_before = [float(t.phi_before) for t in flat]
    phi_after = [float(t.phi_after) for t in flat]
    phi_delta = [after - before for before, after in zip(phi_before, phi_after)]
    rewards = [float(t.reward) for t in flat]
    terminal_rewards = [float(t.terminal_reward) for t in flat]
    shaping_rewards = [float(t.shaping_reward) for t in flat]
    positive_delta = [t for t, delta in zip(flat, phi_delta) if delta > 0.0]
    nonpositive_delta = [t for t, delta in zip(flat, phi_delta) if delta <= 0.0]

    ordered = sorted(flat, key=lambda t: float(t.phi_before))
    third = max(1, len(ordered) // 3)
    phi_buckets = {
        "low": ordered[:third],
        "mid": ordered[third:2 * third],
        "high": ordered[2 * third:],
    }

    return {
        "n_transitions": len(flat),
        "mean_segment_reward": _mean(rewards),
        "mean_terminal_reward": _mean(terminal_rewards),
        "mean_shaping_reward": _mean(shaping_rewards),
        "phi_before_correct_corr": _pearson(phi_before, final_correct),
        "phi_after_correct_corr": _pearson(phi_after, final_correct),
        "phi_delta_correct_corr": _pearson(phi_delta, final_correct),
        "shaping_reward_correct_corr": _pearson(shaping_rewards, final_correct),
        "positive_phi_delta": {
            "count": len(positive_delta),
            "correct_rate": _mean([float(t.final_correct) for t in positive_delta]),
        },
        "nonpositive_phi_delta": {
            "count": len(nonpositive_delta),
            "correct_rate": _mean([float(t.final_correct) for t in nonpositive_delta]),
        },
        "temperature_by_rollout_stage": {
            stage: _bucket_temperature_stats(items)
            for stage, items in sorted(by_stage.items())
        },
        "temperature_by_phi_before_tertile": {
            name: _bucket_temperature_stats(items)
            for name, items in phi_buckets.items()
        },
    }


def train(config_path: str, parallel_size: int | None = None,
          run_name: str | None = None, log_dir: str = "logs") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger, _, final_run_name = setup_experiment_logger(
        component="prefix_ppo_training", run_name=run_name, log_dir=log_dir, config=cfg,
    )
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    training_cfg = cfg["ppo"]["training"]
    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    train_prompts = load_prompts(cfg["paths"]["train_dataset"])
    val_prompts = load_prompts(cfg["paths"]["val_dataset"])
    val_size = int(training_cfg.get("val_size", 0) or 0)
    if val_size > 0 and val_size < len(val_prompts):
        val_fixed = random.Random(seed).sample(val_prompts, val_size)
        val_selection = "sampled"
    else:
        val_fixed = list(val_prompts)
        val_selection = "full"

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise RuntimeError("Prefix PPO requires GPUs for online vLLM rollout")
    device = torch.device(f"cuda:{n_gpu - 1}")
    value_model, calibration_temperature = _load_value_model(cfg, device)
    policy = PrefixPolicyValueNet(value_model.hidden_dim, len(temp_bins)).to(device)

    # Initialize policy toward validation-selected best fixed temperature.
    best_idx = len(temp_bins) // 2
    labels = load_temperature_labels(cfg["paths"]["val_dataset"])
    if labels:
        accuracy = {temp: sum(values) / len(values) for temp, values in labels.items() if values}
        if accuracy:
            best_temp = max(accuracy, key=accuracy.get)
            best_idx = min(range(len(temp_bins)), key=lambda idx: abs(temp_bins[idx] - best_temp))
    fixed_temp_bias = float(training_cfg.get("fixed_temp_bias", 1.0))
    nonfixed_temp_bias = float(training_cfg.get("nonfixed_temp_bias", 0.0))
    with torch.no_grad():
        policy.pi.bias.fill_(nonfixed_temp_bias)
        policy.pi.bias[best_idx] = fixed_temp_bias
    logger.info(
        "train_prompts=%d val_prompts_total=%d val_prompts_used=%d "
        "val_selection=%s best_fixed=%.1f fixed_temp_bias=%.3f nonfixed_temp_bias=%.3f",
        len(train_prompts), len(val_prompts), len(val_fixed), val_selection,
        temp_bins[best_idx], fixed_temp_bias, nonfixed_temp_bias,
    )

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

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(training_cfg["lr"]))
    max_iterations = int(training_cfg["max_iterations"])
    rollout_size = int(training_cfg["online_rollout_size"])
    ppo_epochs = int(training_cfg["ppo_epochs"])
    mini_batch_size = int(training_cfg["mini_batch_size"])
    gamma = float(training_cfg["gamma"])
    gae_lambda = float(training_cfg["gae_lambda"])
    clip_eps = float(training_cfg["clip_eps"])
    value_coef = float(training_cfg["value_coef"])
    entropy_coef = float(training_cfg["entropy_coef"])
    patience_limit = int(training_cfg["early_stop_patience"])

    output = Path(cfg["paths"]["ppo_ckpt"])
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_output, best_output = _checkpoint_variants(output)
    best_val = -1.0
    best_state = None
    best_checkpoint = None
    patience = 0
    for iteration in range(max_iterations):
        rng = random.Random(seed + iteration * 1000)
        batch_prompts = rng.sample(train_prompts, min(rollout_size, len(train_prompts)))
        policy.eval()
        rollout = engine.rollout(
            batch_prompts, policy, stochastic=True, rng=rng,
            generation_seed=seed + iteration * 100000,
        )
        transitions = [
            transition
            for prompt_chains in rollout.transitions
            for chain in prompt_chains
            for transition in chain
        ]
        if len(transitions) < mini_batch_size:
            logger.info("iter=%d too_few_transitions=%d", iteration + 1, len(transitions))
            continue
        observations = torch.stack([t.observation for t in transitions]).to(device)
        actions = torch.stack([t.action for t in transitions]).long().to(device)
        old_logprobs = torch.stack([t.logprob for t in transitions]).to(device)
        old_values = torch.stack([t.value for t in transitions]).to(device)
        rewards = torch.tensor([t.reward for t in transitions], device=device)
        dones = torch.tensor([float(t.done) for t in transitions], device=device)
        advantages, returns = compute_gae(rewards, dones, old_values, gamma, gae_lambda)
        diagnostics = _rollout_diagnostics(rollout)

        policy.train()
        totals = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(ppo_epochs):
            permutation = torch.randperm(len(transitions), device=device)
            for start in range(0, len(transitions), mini_batch_size):
                idx = permutation[start:start + mini_batch_size]
                logits, values = policy(observations[idx])
                distribution = torch.distributions.Categorical(logits=logits)
                new_logprobs = distribution.log_prob(actions[idx])
                entropy = distribution.entropy().mean()
                ratio = torch.exp(new_logprobs - old_logprobs[idx])
                surrogate_a = ratio * advantages[idx]
                surrogate_b = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[idx]
                policy_loss = -torch.minimum(surrogate_a, surrogate_b).mean()
                value_loss = F.mse_loss(values, returns[idx])
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                totals["policy"] += float(policy_loss.item())
                totals["value"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["updates"] += 1

        policy.eval()
        train_accuracy = sum(rollout.majority_correct) / max(1, len(rollout.majority_correct))
        current_state = _cpu_state_dict(policy)
        latest_checkpoint = _build_checkpoint(
            cfg=cfg,
            policy_state=current_state,
            iteration=iteration + 1,
            best_validation_accuracy=best_val,
            current_validation_accuracy=None,
            current_train_accuracy=train_accuracy,
            patience=patience,
            run_name=final_run_name,
        )
        torch.save(latest_checkpoint, latest_output)
        logger.info("saved_latest=%s iter=%d stage=pre_validation train_acc=%.4f",
                    latest_output, iteration + 1, train_accuracy)
        if bool(cfg["inference"].get("reset_prefix_cache_before_validation", True)):
            cache_reset = runner.reset_prefix_cache(reset_connector=True)
            logger.info("reset_prefix_cache_before_validation=%s", cache_reset)
        val_rollout = engine.rollout(
            val_fixed, policy, stochastic=False, rng=random.Random(seed),
            collect_transitions=False,
            generation_seed=seed,
        ) if val_fixed else None
        val_accuracy = (
            sum(val_rollout.majority_correct) / len(val_rollout.majority_correct)
            if val_rollout is not None else 0.0
        )
        updates = max(1, int(totals["updates"]))
        logger.info(
            "iter=%d train_acc=%.4f val_acc=%.4f reward=%.5f transitions=%d "
            "terminal_reward=%.5f shaping_reward=%.5f policy=%.5f value=%.5f entropy=%.5f",
            iteration + 1, train_accuracy, val_accuracy, float(rewards.mean().item()),
            len(transitions),
            float(diagnostics.get("mean_terminal_reward", 0.0)),
            float(diagnostics.get("mean_shaping_reward", 0.0)),
            totals["policy"] / updates,
            totals["value"] / updates, totals["entropy"] / updates,
        )
        logger.info("iter=%d rollout_diagnostics=%s",
                    iteration + 1, json.dumps(diagnostics, sort_keys=True))
        if val_accuracy > best_val:
            best_val = val_accuracy
            patience = 0
            best_state = current_state
            best_checkpoint = _build_checkpoint(
                cfg=cfg,
                policy_state=best_state,
                iteration=iteration + 1,
                best_validation_accuracy=best_val,
                current_validation_accuracy=val_accuracy,
                current_train_accuracy=train_accuracy,
                patience=patience,
                run_name=final_run_name,
            )
            torch.save(best_checkpoint, best_output)
            logger.info("saved_best=%s iter=%d best_val=%.4f",
                        best_output, iteration + 1, best_val)
        else:
            patience += 1

        latest_checkpoint = _build_checkpoint(
            cfg=cfg,
            policy_state=current_state,
            iteration=iteration + 1,
            best_validation_accuracy=best_val,
            current_validation_accuracy=val_accuracy,
            current_train_accuracy=train_accuracy,
            patience=patience,
            run_name=final_run_name,
        )
        torch.save(latest_checkpoint, latest_output)
        logger.info("saved_latest=%s iter=%d patience=%d best_val=%.4f",
                    latest_output, iteration + 1, patience, best_val)
        if patience >= patience_limit:
            break

    if best_state is None:
        best_state = _cpu_state_dict(policy)
        best_checkpoint = _build_checkpoint(
            cfg=cfg,
            policy_state=best_state,
            iteration=max_iterations,
            best_validation_accuracy=best_val,
            current_validation_accuracy=None,
            current_train_accuracy=None,
            patience=patience,
            run_name=final_run_name,
        )
    checkpoint = best_checkpoint or _build_checkpoint(
        cfg=cfg,
        policy_state=best_state,
        iteration=max_iterations,
        best_validation_accuracy=best_val,
        current_validation_accuracy=None,
        current_train_accuracy=None,
        patience=patience,
        run_name=final_run_name,
    )
    torch.save(checkpoint, output)
    logger.info("saved=%s best_val=%.4f run_name=%s", output, best_val, final_run_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()
    try:
        train(args.config, args.parallel_size, args.run_name, args.log_dir)
    except Exception as exc:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        logger, _, _ = setup_experiment_logger(
            component="prefix_ppo_training", run_name=args.run_name,
            log_dir=args.log_dir, config=cfg,
        )
        log_exception(logger, exc)
        raise


if __name__ == "__main__":
    main()
