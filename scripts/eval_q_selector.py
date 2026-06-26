#!/usr/bin/env python3
"""Online evaluation for a direct Prefix-Q argmax temperature selector."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.segmenter import batch_build_masked_concat_segment_obs_from_lp
from inference.vllm_runner import VLLMFeatureExporter
from mil.prefix_value import PrefixRecurrentState, PrefixValueModel, calibrated_probability
from ppo.prefix_training import load_prompts
from utils.answer_verifier import extract_answer, verify_answer, verify_answer_by_value
from utils.calibration import (
    answer_entropy,
    binary_nll,
    brier_score,
    expected_calibration_error,
)


NO_ANSWER = "<NO_ANSWER>"


def allowed_temperature_indices(temp_bins: Sequence[float],
                                allowed_temperatures: Sequence[float] | None) -> List[int]:
    if allowed_temperatures is None:
        return list(range(len(temp_bins)))
    allowed = {round(float(temp), 8) for temp in allowed_temperatures}
    return [
        idx for idx, temp in enumerate(temp_bins)
        if round(float(temp), 8) in allowed
    ]


def select_q_temperature(q_values: Sequence[float],
                         temp_bins: Sequence[float],
                         allowed_indices: Sequence[int],
                         tie_margin: float = 0.02) -> Dict[str, Any]:
    if not allowed_indices:
        raise ValueError("allowed_indices cannot be empty")
    candidates = [
        (int(idx), float(temp_bins[int(idx)]), float(q_values[int(idx)]))
        for idx in allowed_indices
    ]
    candidates.sort(key=lambda item: (-item[2], item[1]))
    best_idx, best_temp, best_q = candidates[0]
    near_best = [
        item for item in candidates
        if best_q - item[2] <= float(tie_margin)
    ]
    selected_idx, selected_temp, selected_q = min(near_best, key=lambda item: item[1])
    second_q = candidates[1][2] if len(candidates) > 1 else best_q
    return {
        "temperature_index": selected_idx,
        "temperature": selected_temp,
        "selected_q": selected_q,
        "best_temperature_index": best_idx,
        "best_temperature": best_temp,
        "best_q": best_q,
        "second_q": second_q,
        "margin_to_second": best_q - second_q,
        "q_values": {str(temp): q for _idx, temp, q in candidates},
    }


def _stage(segment_idx: int, max_rounds: int) -> str:
    q = (segment_idx + 1) / max(1, max_rounds)
    if q < 0.30:
        return "early"
    if q < 0.65:
        return "middle"
    return "late"


def _render(runner: VLLMFeatureExporter, question: str, system_prompt: str,
            use_math_chat: bool) -> str:
    if not use_math_chat:
        return question
    return runner.render_messages(
        runner.build_math_messages(question, system_prompt=system_prompt)
    )


def _load_q_model(cfg: Dict[str, Any], device: torch.device) -> tuple[PrefixValueModel, float]:
    checkpoint = torch.load(
        cfg["paths"]["prefix_value_ckpt"], map_location=device, weights_only=False,
    )
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
    model.load_state_dict(checkpoint["prefix_value"])
    if model.q_head is None:
        raise RuntimeError("q selector requires a PrefixValueModel checkpoint with q_head")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, float(checkpoint.get("calibration_temperature", 1.0))


def load_eval_prompts(cfg: Dict[str, Any], input_path: str | None = None,
                      max_prompts: int | None = None) -> tuple[List[Dict[str, str]], str]:
    data_path = str(input_path or cfg["paths"]["test_dataset"])
    prompts = load_prompts(data_path)
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    return prompts, data_path


@torch.no_grad()
def evaluate_q_selector(config_path: str, seed: int = 42,
                        parallel_size: int | None = None,
                        max_prompts: int | None = None,
                        input_path: str | None = None,
                        gpu_memory_utilization: float | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts, data_path = load_eval_prompts(cfg, input_path=input_path, max_prompts=max_prompts)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise RuntimeError("Q selector online evaluation requires GPUs")
    device = torch.device(f"cuda:{n_gpu - 1}")
    value_model, calibration_temperature = _load_q_model(cfg, device)
    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    selector_cfg = cfg.get("q_selector", {})
    allowed_indices = allowed_temperature_indices(
        temp_bins,
        selector_cfg.get("allowed_temperatures",
                         cfg.get("prefix_value", {}).get("continuations", {}).get("temperatures")),
    )
    tie_margin = float(selector_cfg.get("tie_margin", 0.02))
    first_temperature = float(selector_cfg.get("first_segment_temperature", 0.7))

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
    rng = random.Random(seed)
    votes = int(cfg["inference"]["num_votes"])
    segment_size = int(cfg["data"]["segment_size"])
    use_prompt_hidden = int(getattr(value_model, "prompt_dim", 0)) > 0
    max_new_tokens = int(cfg["inference"]["max_new_tokens"])
    max_rounds = max(1, (max_new_tokens + segment_size - 1) // segment_size)
    rendered = [
        _render(
            runner,
            str(prompt.get("question", prompt.get("prompt", ""))),
            str(cfg["inference"].get("system_prompt", "")),
            bool(cfg["inference"].get("use_math_chat_prompt", True)),
        )
        for prompt in prompts
    ]
    gold = [str(prompt.get("answer", "")) for prompt in prompts]

    generated = [[""] * votes for _ in prompts]
    active = [[True] * votes for _ in prompts]
    states: List[List[PrefixRecurrentState | None]] = [[None] * votes for _ in prompts]
    hidden_obs: List[List[torch.Tensor | None]] = [[None] * votes for _ in prompts]
    phi_values: List[List[float | None]] = [[None] * votes for _ in prompts]
    token_counts = [[0] * votes for _ in prompts]
    segment_counts = [[0] * votes for _ in prompts]
    temperatures: List[List[float]] = [[] for _ in prompts]
    q_decisions: List[List[Dict[str, Any]]] = [[] for _ in prompts]
    stage_temperatures: Dict[str, List[float]] = defaultdict(list)

    started = time.perf_counter()
    for segment_idx in range(max_rounds):
        round_prompts: List[str] = []
        round_temps: List[float] = []
        round_map: List[Tuple[int, int]] = []
        for i in range(len(prompts)):
            for vote in range(votes):
                if not active[i][vote]:
                    continue
                if hidden_obs[i][vote] is None:
                    temperature = first_temperature
                    decision = {
                        "vote": vote,
                        "segment_index": segment_idx,
                        "stage": _stage(segment_idx, max_rounds),
                        "temperature": temperature,
                        "source": "first_segment",
                    }
                else:
                    hidden = hidden_obs[i][vote].unsqueeze(0).to(device)
                    q_logits = value_model.q_from_hidden(hidden).squeeze(0)
                    q_probs = torch.sigmoid(q_logits).detach().cpu().tolist()
                    selected = select_q_temperature(
                        q_probs, temp_bins, allowed_indices, tie_margin=tie_margin,
                    )
                    temperature = float(selected["temperature"])
                    decision = {
                        "vote": vote,
                        "segment_index": segment_idx,
                        "stage": _stage(segment_idx, max_rounds),
                        "prefix_value": phi_values[i][vote],
                        **selected,
                    }
                temperatures[i].append(temperature)
                q_decisions[i].append(decision)
                stage_temperatures[decision["stage"]].append(temperature)
                round_prompts.append(rendered[i] + generated[i][vote])
                round_temps.append(temperature)
                round_map.append((i, vote))

        if not round_map:
            break

        features = runner.generate_with_features(
            round_prompts,
            round_temps,
            segment_size,
            top_k=int(cfg["inference"]["top_k_logprobs"]),
            return_logprobs=True,
            return_hidden=False,
            return_prompt_hidden=use_prompt_hidden,
            device=device,
            seeds=[
                seed + segment_idx * len(prompts) * votes + i * votes + vote
                for i, vote in round_map
            ],
        )

        valid_positions: List[int] = []
        lp_tensors: List[torch.Tensor] = []
        token_lists: List[List[str]] = []
        texts: List[str] = []
        prompt_hiddens: List[torch.Tensor] = []
        done_flags: List[bool] = []
        for pos, ((i, vote), item) in enumerate(zip(round_map, features)):
            generated[i][vote] += item["text"]
            n_tokens = len(item["token_ids"])
            token_counts[i][vote] += n_tokens
            segment_counts[i][vote] += 1
            done = (
                not item["token_ids"] or item["finish_reason"] == "stop" or
                (runner.tokenizer.eos_token_id is not None and
                 runner.tokenizer.eos_token_id in item["token_ids"])
            )
            done_flags.append(done)
            if item["logprobs"] is not None and n_tokens > 0:
                valid_positions.append(pos)
                lp_tensors.append(item["logprobs"])
                token_lists.append(item["tokens"])
                texts.append(item["text"])
                if use_prompt_hidden:
                    prompt_hidden = item.get("prompt_hidden")
                    if prompt_hidden is None:
                        raise RuntimeError("prompt-aware q selector requires prompt_hidden")
                    prompt_hiddens.append(prompt_hidden)

        masked_list = batch_build_masked_concat_segment_obs_from_lp(
            lp_tensors, token_lists, texts,
            segment_size=segment_size,
            token_dim=int(cfg["data"]["instance_dim"]),
            device=device,
            segment_mode="fixed_window",
        ) if lp_tensors else []
        position_to_masked = dict(zip(valid_positions, masked_list))
        position_to_prompt_hidden = (
            dict(zip(valid_positions, prompt_hiddens)) if use_prompt_hidden else {}
        )

        if valid_positions:
            step_features = torch.stack([
                position_to_masked[pos].features[0] for pos in valid_positions
            ]).to(device)
            step_masks = torch.stack([
                position_to_masked[pos].token_mask[0] for pos in valid_positions
            ]).to(device)
            hidden_parts = []
            for pos in valid_positions:
                i, vote = round_map[pos]
                state = states[i][vote]
                if state is None:
                    if use_prompt_hidden:
                        hidden_parts.append(value_model.initial_hidden(
                            position_to_prompt_hidden[pos].unsqueeze(0).to(device)
                        ))
                    else:
                        hidden_parts.append(torch.zeros(1, 1, value_model.hidden_dim, device=device))
                else:
                    hidden_parts.append(state.hidden.to(device))
            hidden = torch.cat(hidden_parts, dim=1)
            logits, encoded, next_hidden = value_model.step_batch(
                step_features, step_masks, hidden=hidden, position=segment_idx,
            )
            probs = calibrated_probability(logits, calibration_temperature)
            for batch_idx, pos in enumerate(valid_positions):
                i, vote = round_map[pos]
                states[i][vote] = PrefixRecurrentState(
                    next_hidden[:, batch_idx:batch_idx + 1].detach(), segment_idx + 1,
                )
                hidden_obs[i][vote] = encoded[batch_idx].detach()
                phi_values[i][vote] = float(probs[batch_idx].item())

        for (i, vote), done in zip(round_map, done_flags):
            if done:
                active[i][vote] = False

    elapsed = time.perf_counter() - started
    predictions: List[Dict[str, Any]] = []
    majority_correct: List[int] = []
    individual_correct_all: List[List[int]] = []
    confidences: List[float] = []
    entropies: List[float] = []
    for i, prompt in enumerate(prompts):
        extracted = [
            answer if (answer := extract_answer(text)) is not None else NO_ANSWER
            for text in generated[i]
        ]
        counts = Counter(extracted)
        majority_answer, majority_count = counts.most_common(1)[0] if counts else (NO_ANSWER, 0)
        correct = int(
            majority_answer != NO_ANSWER and verify_answer_by_value(majority_answer, gold[i])
        )
        individual = [int(verify_answer(text, gold[i])) for text in generated[i]]
        confidence = majority_count / max(1, votes)
        entropy = answer_entropy(extracted)
        majority_correct.append(correct)
        individual_correct_all.append(individual)
        confidences.append(confidence)
        entropies.append(entropy)
        predictions.append({
            "problem_id": prompt["problem_id"],
            "majority_correct": correct,
            "individual_correct": individual,
            "extracted_answers": extracted,
            "majority_answer": majority_answer,
            "majority_count": int(majority_count),
            "sc_confidence": confidence,
            "answer_entropy": entropy,
            "temperatures": temperatures[i],
            "segment_counts": segment_counts[i],
            "token_counts": token_counts[i],
            "q_decisions": q_decisions[i],
        })

    correctness = majority_correct
    total_tokens = sum(sum(row) for row in token_counts)
    individual_total = sum(sum(row) for row in individual_correct_all)
    flat_temps = [temp for row in temperatures for temp in row]
    temp_distribution = {
        str(temp): count for temp, count in sorted(Counter(flat_temps).items())
    }
    stage_distribution = {
        stage: {str(temp): count for temp, count in sorted(Counter(values).items())}
        for stage, values in sorted(stage_temperatures.items())
    }
    return {
        "method": "prefix_q_argmax_selector",
        "seed": seed,
        "config": config_path,
        "input_path": data_path,
        "n_prompts": len(prompts),
        "num_votes": votes,
        "allowed_temperatures": [temp_bins[idx] for idx in allowed_indices],
        "first_segment_temperature": first_temperature,
        "tie_margin": tie_margin,
        "majority_accuracy": sum(majority_correct) / max(1, len(majority_correct)),
        "pass_at_1_accuracy": sum(row[0] for row in individual_correct_all) / max(1, len(individual_correct_all)),
        "individual_accuracy": individual_total / max(1, len(prompts) * votes),
        "ece": expected_calibration_error(confidences, correctness),
        "brier": brier_score(confidences, correctness),
        "nll": binary_nll(confidences, correctness),
        "mean_confidence": sum(confidences) / max(1, len(confidences)),
        "mean_answer_entropy": sum(entropies) / max(1, len(entropies)),
        "average_tokens": total_tokens / max(1, len(prompts) * votes),
        "total_tokens": total_tokens,
        "selected_temperature_distribution": temp_distribution,
        "stage_selected_temperature_distribution": stage_distribution,
        "wall_seconds": elapsed,
        "predictions": predictions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--input", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--output", default="results/q_selector_eval.json")
    args = parser.parse_args()
    metrics = evaluate_q_selector(
        args.config, seed=args.seed,
        parallel_size=args.parallel_size,
        max_prompts=args.max_prompts,
        input_path=args.input,
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
