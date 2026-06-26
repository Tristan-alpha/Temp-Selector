#!/usr/bin/env python3
"""Generate Prefix-Q trajectories and render text-level segment coloring."""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.segmenter import batch_build_masked_concat_segment_obs_from_lp
from inference.vllm_runner import VLLMFeatureExporter
from mil.prefix_value import PrefixRecurrentState, calibrated_probability
from scripts.eval_q_selector import (
    NO_ANSWER,
    _load_q_model,
    _render,
    _stage,
    allowed_temperature_indices,
    load_eval_prompts,
    select_q_temperature,
)
from utils.answer_verifier import extract_answer, verify_answer, verify_answer_by_value
from utils.calibration import answer_entropy


def _temperature_color(temp: float, min_temp: float = 0.1, max_temp: float = 1.5,
                       lightness: float = 90.0) -> str:
    span = max(max_temp - min_temp, 1e-9)
    ratio = min(1.0, max(0.0, (float(temp) - min_temp) / span))
    hue = 120.0 * (1.0 - ratio)
    return f"hsl({hue:.1f} 74% {lightness:.1f}%)"


def _temperature_accent(temp: float) -> str:
    return _temperature_color(temp, lightness=38.0)


def _decision_tooltip(segment: Dict[str, Any]) -> str:
    parts = [
        f"segment={segment.get('segment_index')}",
        f"temperature={float(segment.get('temperature', 0.0)):.1f}",
        f"stage={segment.get('stage', 'unknown')}",
        f"tokens={segment.get('n_tokens', 0)}",
        f"finish={segment.get('finish_reason')}",
    ]
    if segment.get("source") == "first_segment":
        parts.append("source=first_segment")
    if segment.get("prefix_value") is not None:
        parts.append(f"phi_before={float(segment['prefix_value']):.3f}")
    if segment.get("phi_after") is not None:
        parts.append(f"phi_after={float(segment['phi_after']):.3f}")
    if segment.get("selected_q") is not None:
        parts.append(f"selected_q={float(segment['selected_q']):.3f}")
    if segment.get("margin_to_second") is not None:
        parts.append(f"q_margin={float(segment['margin_to_second']):.3f}")
    return " | ".join(parts)


@torch.no_grad()
def generate_q_selector_text_segments(config_path: str,
                                      seed: int = 42,
                                      parallel_size: int | None = None,
                                      max_prompts: int = 5,
                                      input_path: str | None = None,
                                      gpu_memory_utilization: float | None = None,
                                      num_votes: int = 1,
                                      max_new_tokens: int | None = None) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompts, data_path = load_eval_prompts(cfg, input_path=input_path, max_prompts=max_prompts)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu == 0:
        raise RuntimeError("text segment generation requires GPUs")
    device = torch.device(f"cuda:{n_gpu - 1}")

    value_model, calibration_temperature = _load_q_model(cfg, device)
    temp_bins = [float(x) for x in cfg["data"]["temp_bins"]]
    selector_cfg = cfg.get("q_selector", {})
    allowed_indices = allowed_temperature_indices(
        temp_bins,
        selector_cfg.get(
            "allowed_temperatures",
            cfg.get("prefix_value", {}).get("continuations", {}).get("temperatures"),
        ),
    )
    tie_margin = float(selector_cfg.get("tie_margin", 0.02))
    first_temperature = float(selector_cfg.get("first_segment_temperature", 0.7))
    effective_max_new_tokens = int(max_new_tokens or cfg["inference"]["max_new_tokens"])

    runner = VLLMFeatureExporter(
        model_name_or_path=cfg["inference"]["model_name_or_path"],
        max_new_tokens=effective_max_new_tokens,
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

    votes = int(num_votes)
    if votes <= 0:
        raise ValueError("--num-votes must be positive")
    segment_size = int(cfg["data"]["segment_size"])
    use_prompt_hidden = int(getattr(value_model, "prompt_dim", 0)) > 0
    max_rounds = max(1, (effective_max_new_tokens + segment_size - 1) // segment_size)
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
    segments: List[List[List[Dict[str, Any]]]] = [[[] for _ in range(votes)] for _ in prompts]

    started = time.perf_counter()
    for segment_idx in range(max_rounds):
        round_prompts: List[str] = []
        round_temps: List[float] = []
        round_map: List[Tuple[int, int]] = []
        pending: List[Dict[str, Any]] = []
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
                        "prefix_value": None,
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
                round_prompts.append(rendered[i] + generated[i][vote])
                round_temps.append(float(temperature))
                round_map.append((i, vote))
                pending.append(decision)
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
        for pos, ((i, vote), decision, item) in enumerate(zip(round_map, pending, features)):
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
            segments[i][vote].append({
                **decision,
                "text": item["text"],
                "n_tokens": int(n_tokens),
                "finish_reason": item["finish_reason"],
                "done": bool(done),
                "phi_after": None,
            })
            if item["logprobs"] is not None and n_tokens > 0:
                valid_positions.append(pos)
                lp_tensors.append(item["logprobs"])
                token_lists.append(item["tokens"])
                texts.append(item["text"])
                if use_prompt_hidden:
                    prompt_hidden = item.get("prompt_hidden")
                    if prompt_hidden is None:
                        raise RuntimeError("prompt-aware q selector text export requires prompt_hidden")
                    prompt_hiddens.append(prompt_hidden)

        masked_list = batch_build_masked_concat_segment_obs_from_lp(
            lp_tensors,
            token_lists,
            texts,
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
                segments[i][vote][-1]["phi_after"] = float(probs[batch_idx].item())

        for (i, vote), done in zip(round_map, done_flags):
            if done:
                active[i][vote] = False

    elapsed = time.perf_counter() - started
    predictions: List[Dict[str, Any]] = []
    majority_correct: List[int] = []
    individual_correct_all: List[List[int]] = []
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
        majority_correct.append(correct)
        individual_correct_all.append(individual)
        predictions.append({
            "problem_id": prompt["problem_id"],
            "gold_answer": gold[i],
            "majority_correct": correct,
            "individual_correct": individual,
            "extracted_answers": extracted,
            "majority_answer": majority_answer,
            "majority_count": int(majority_count),
            "sc_confidence": majority_count / max(1, votes),
            "answer_entropy": answer_entropy(extracted),
            "segment_counts": segment_counts[i],
            "token_counts": token_counts[i],
            "trajectories": [
                {
                    "vote": vote,
                    "correct": individual[vote],
                    "extracted_answer": extracted[vote],
                    "text": generated[i][vote],
                    "segments": segments[i][vote],
                }
                for vote in range(votes)
            ],
        })

    flat_temps = [
        float(segment["temperature"])
        for prediction in predictions
        for trajectory in prediction["trajectories"]
        for segment in trajectory["segments"]
    ]
    return {
        "method": "prefix_q_argmax_selector_text_segments",
        "seed": seed,
        "config": config_path,
        "input_path": data_path,
        "n_prompts": len(prompts),
        "num_votes": votes,
        "allowed_temperatures": [temp_bins[idx] for idx in allowed_indices],
        "first_segment_temperature": first_temperature,
        "tie_margin": tie_margin,
        "max_new_tokens": effective_max_new_tokens,
        "segment_size": segment_size,
        "majority_accuracy": sum(majority_correct) / max(1, len(majority_correct)),
        "individual_accuracy": (
            sum(sum(row) for row in individual_correct_all) /
            max(1, len(prompts) * votes)
        ),
        "selected_temperature_distribution": {
            str(temp): count for temp, count in sorted(Counter(flat_temps).items())
        },
        "wall_seconds": elapsed,
        "predictions": predictions,
    }


def _trajectory_html(prediction: Dict[str, Any], trajectory: Dict[str, Any],
                     rank: int) -> str:
    segments_html = []
    for segment in trajectory["segments"]:
        temp = float(segment["temperature"])
        title = html.escape(_decision_tooltip(segment), quote=True)
        text = html.escape(segment.get("text", ""))
        first = " first" if segment.get("source") == "first_segment" else ""
        q_bits = []
        if segment.get("prefix_value") is not None:
            q_bits.append(f"phi {float(segment['prefix_value']):.3f}")
        if segment.get("selected_q") is not None:
            q_bits.append(f"q {float(segment['selected_q']):.3f}")
        if segment.get("margin_to_second") is not None:
            q_bits.append(f"margin {float(segment['margin_to_second']):.3f}")
        details = " · ".join(q_bits)
        segments_html.append(
            f'<section class="text-segment{first}" title="{title}" '
            f'style="background:{_temperature_color(temp)}; border-left-color:{_temperature_accent(temp)}">'
            f'<div class="seg-head"><b>segment {int(segment["segment_index"])}</b>'
            f'<span class="temp" style="background:{_temperature_accent(temp)}">{temp:.1f}</span>'
            f'<small>{html.escape(details)}</small></div>'
            f'<div class="seg-text">{text}</div>'
            '</section>'
        )
    status = "correct" if int(trajectory.get("correct", 0)) else "wrong"
    return f"""
    <article class="trajectory-card">
      <div class="card-title">
        <div>
          <h2>{rank}. {html.escape(str(prediction["problem_id"]))} · vote {trajectory["vote"]}</h2>
          <p>answer <code>{html.escape(str(trajectory.get("extracted_answer", "")))}</code>,
             gold <code>{html.escape(str(prediction.get("gold_answer", "")))}</code>,
             segments {len(trajectory["segments"])},
             tokens {sum(int(s.get("n_tokens", 0)) for s in trajectory["segments"])}</p>
        </div>
        <span class="badge {status}">{status}</span>
      </div>
      <div class="segment-flow">{''.join(segments_html)}</div>
    </article>
    """


def render_text_segment_html(data: Dict[str, Any], source_json: str) -> str:
    trajectories = []
    for prediction in data["predictions"]:
        for trajectory in prediction["trajectories"]:
            trajectories.append((prediction, trajectory))
    cards = "\n".join(
        _trajectory_html(prediction, trajectory, idx + 1)
        for idx, (prediction, trajectory) in enumerate(trajectories)
    )
    legend = "\n".join(
        f'<span class="legend-chip"><span style="background:{_temperature_accent(temp)}"></span>{temp:.1f}</span>'
        for temp in data.get("allowed_temperatures", [])
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Prefix-Q Text-Level Segment Coloring</title>
  <style>
    :root {{
      --ink: #172026;
      --muted: #5d6972;
      --line: #d8dee4;
      --panel: #ffffff;
      --bg: #f6f8fa;
      --ok: #17694b;
      --bad: #a3372a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 22px 46px;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 26px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 16px;
      line-height: 1.35;
      letter-spacing: 0;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: #edf2f5;
      border-radius: 4px;
      padding: 1px 5px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 18px 0;
    }}
    .metric, .legend, .trajectory-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 10px 12px; }}
    .metric b {{
      display: block;
      font-size: 18px;
      margin-bottom: 2px;
    }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .legend {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      padding: 10px 12px;
      margin-bottom: 16px;
    }}
    .legend-chip {{
      display: inline-flex;
      gap: 5px;
      align-items: center;
      font-size: 13px;
    }}
    .legend-chip span {{
      width: 18px;
      height: 12px;
      border-radius: 2px;
      border: 1px solid rgb(0 0 0 / 0.18);
    }}
    .trajectory-card {{
      padding: 14px;
      margin: 14px 0;
    }}
    .card-title {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      margin-bottom: 12px;
    }}
    .badge {{
      flex: 0 0 auto;
      min-width: 72px;
      text-align: center;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .badge.correct {{ color: var(--ok); background: #e5f3ed; }}
    .badge.wrong {{ color: var(--bad); background: #fae9e5; }}
    .segment-flow {{
      display: grid;
      gap: 8px;
    }}
    .text-segment {{
      border: 1px solid rgb(0 0 0 / 0.12);
      border-left: 8px solid;
      border-radius: 7px;
      padding: 8px 10px 9px;
    }}
    .text-segment.first {{
      outline: 2px solid rgb(0 0 0 / 0.18);
      outline-offset: 1px;
    }}
    .seg-head {{
      display: flex;
      flex-wrap: wrap;
      gap: 7px;
      align-items: center;
      margin-bottom: 5px;
    }}
    .seg-head b {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    .seg-head small {{
      color: #46525a;
      font-size: 12px;
    }}
    .temp {{
      color: white;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 12px;
      font-weight: 800;
      text-shadow: 0 1px 1px rgb(0 0 0 / 0.22);
    }}
    .seg-text {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: ui-serif, Georgia, Cambria, "Times New Roman", serif;
      font-size: 15px;
      line-height: 1.55;
    }}
    @media (max-width: 760px) {{
      main {{ padding: 20px 14px 34px; }}
      .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .card-title {{ display: grid; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Prefix-Q Text-Level Segment Coloring</h1>
      <p>Source JSON: <code>{html.escape(source_json)}</code>. Each card is one generated trajectory. Low temperatures are green; high temperatures are red. The outlined first segment is the fixed first-segment temperature.</p>
    </header>
    <section class="summary">
      <div class="metric"><b>{int(data.get("n_prompts", 0))}</b><span>prompts</span></div>
      <div class="metric"><b>{int(data.get("num_votes", 0))}</b><span>vote per prompt</span></div>
      <div class="metric"><b>{float(data.get("individual_accuracy", 0.0)):.3f}</b><span>individual accuracy</span></div>
      <div class="metric"><b>{float(data.get("wall_seconds", 0.0)):.1f}s</b><span>generation time</span></div>
    </section>
    <section class="legend">{legend}</section>
    {cards}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/min_pvm_q_500_seed42.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--max-prompts", type=int, default=5)
    parser.add_argument("--input", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--num-votes", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-json", default="results/q_selector_text_segments_seed42.json")
    parser.add_argument("--output-html", default="results/q_selector_text_segments_seed42.html")
    args = parser.parse_args()

    result = generate_q_selector_text_segments(
        config_path=args.config,
        seed=args.seed,
        parallel_size=args.parallel_size,
        max_prompts=args.max_prompts,
        input_path=args.input,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_votes=args.num_votes,
        max_new_tokens=args.max_new_tokens,
    )
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    output_html = Path(args.output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(render_text_segment_html(result, str(output_json)), encoding="utf-8")
    compact = {key: value for key, value in result.items() if key != "predictions"}
    compact["output_json"] = str(output_json)
    compact["output_html"] = str(output_html)
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
