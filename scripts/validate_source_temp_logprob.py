#!/usr/bin/env python3
"""Validate whether logprob-prefix phenomena survive source-temperature control.

The script builds a small balanced source-temperature prefix set, extracts the
same top-k logprob features used by the PVM, scores each prefix with the frozen
PVM, optionally generates continuation labels, and writes a compact summary.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.segmenter import build_masked_concat_segment_obs_from_lp
from inference.vllm_runner import VLLMFeatureExporter
from mil.prefix_value import PrefixValueModel, calibrated_probability
from utils.answer_verifier import extract_final_answer, verify_answer


DEFAULT_CONFIG = Path("configs/training/min_pvm_ppo_500_seed42.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--source-temperatures", default="0.1,0.7,1.1,1.5")
    parser.add_argument("--sources-per-label", type=int, default=2)
    parser.add_argument("--prefix-quantiles", default="0.25,0.50,0.75")
    parser.add_argument("--continuation-temperatures", default="0.1,0.7,1.1,1.5")
    parser.add_argument("--seeds-per-temperature", type=int, default=2)
    parser.add_argument("--continuation-max-tokens", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-micro-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Only select source rows/prefix specs; do not load vLLM.")
    parser.add_argument("--skip-continuations", action="store_true",
                        help="Extract logprobs/PVM phi but skip continuation generation.")
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(str(key))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def unpack_generation_output(out: Any) -> Tuple[str, List[int], Any]:
    if isinstance(out, Mapping):
        return (
            str(out.get("text", "")),
            list(out.get("token_ids", []) or []),
            out.get("finish_reason"),
        )
    completions = getattr(out, "outputs", None)
    if completions:
        completion = completions[0]
        return (
            str(getattr(completion, "text", "")),
            list(getattr(completion, "token_ids", []) or []),
            getattr(completion, "finish_reason", None),
        )
    return str(getattr(out, "text", "")), list(getattr(out, "token_ids", []) or []), getattr(out, "finish_reason", None)


def sample_prefix(sample_id: str) -> str:
    return sample_id.rsplit("_t", 1)[0] if "_t" in sample_id else sample_id


def prefix_stage(prefix_segments: int, n_segments: int) -> str:
    q = prefix_segments / max(1, n_segments)
    if q < 0.30:
        return "early"
    if q < 0.65:
        return "middle"
    return "late"


def prefix_count_for_quantile(q: float, n_segments: int) -> int:
    return min(n_segments - 1, max(1, math.ceil(float(q) * n_segments)))


def select_source_rows(
    rows: Sequence[Mapping[str, Any]],
    source_temperatures: Sequence[float],
    sources_per_label: int,
) -> List[Dict[str, Any]]:
    by_group: Dict[Tuple[float, int], List[Mapping[str, Any]]] = defaultdict(list)
    wanted = {float(t) for t in source_temperatures}
    for row in rows:
        temp = float(row.get("temperature", 0.0))
        if temp not in wanted:
            continue
        label = int(row.get("individual_label", 1))
        if label not in (0, 1):
            continue
        by_group[(temp, label)].append(row)

    selected: List[Dict[str, Any]] = []
    for temp in source_temperatures:
        for label in (0, 1):
            bucket = sorted(
                by_group.get((float(temp), label), []),
                key=lambda row: (sample_prefix(str(row.get("sample_id", ""))), str(row.get("sample_id", ""))),
            )
            if len(bucket) < sources_per_label:
                raise RuntimeError(
                    f"not enough rows for source_temperature={temp} label={label}: "
                    f"needed {sources_per_label}, found {len(bucket)}"
                )
            selected.extend(dict(row) for row in bucket[:sources_per_label])
    return selected


def build_prefix_specs(
    source_rows: Sequence[Mapping[str, Any]],
    prefix_quantiles: Sequence[float],
    segment_size: int,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for row in source_rows:
        token_ids = list(row.get("token_ids", []))
        n_segments = max(1, math.ceil(len(token_ids) / max(1, segment_size)))
        if n_segments < 2:
            continue
        seen_counts = set()
        for q in prefix_quantiles:
            count = prefix_count_for_quantile(float(q), n_segments)
            if count in seen_counts:
                continue
            seen_counts.add(count)
            specs.append({
                "problem_id": sample_prefix(str(row.get("sample_id", ""))),
                "source_sample_id": str(row.get("sample_id", "")),
                "source_temperature": float(row.get("temperature", 0.0)),
                "source_individual_label": int(row.get("individual_label", 1)),
                "source_correct": 1 - int(row.get("individual_label", 1)),
                "prefix_segments": count,
                "prefix_quantile": float(q),
                "prefix_stage": prefix_stage(count, n_segments),
                "n_segments": n_segments,
                "prefix_token_end": min(len(token_ids), count * segment_size),
            })
    return specs


def load_pvm(cfg: Mapping[str, Any], checkpoint_path: Path, device: torch.device) -> Tuple[PrefixValueModel, float]:
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
    model.load_state_dict(checkpoint["prefix_value"])
    model.eval()
    return model, float(checkpoint.get("calibration_temperature", 1.0))


def render_prompt(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping) and metadata.get("rendered_prompt"):
        return str(metadata["rendered_prompt"])
    return str(row.get("prompt", ""))


def token_stats(lp: torch.Tensor, prefix_token_end: int) -> Dict[str, float]:
    sub = lp[:prefix_token_end].float()
    if int(sub.shape[0]) == 0:
        return {
            "mean_sampled_logprob": float("nan"),
            "mean_topk_entropy": float("nan"),
            "mean_top1_logprob": float("nan"),
            "mean_top1_top2_margin": float("nan"),
            "tail_mean_sampled_logprob": float("nan"),
            "tail_mean_topk_entropy": float("nan"),
        }
    topk = sub[:, 1:]
    entropy = -(torch.exp(topk) * topk).sum(dim=1)
    top1 = topk[:, 0]
    if topk.shape[1] > 1:
        margin = topk[:, 0] - topk[:, 1]
    else:
        margin = torch.zeros_like(top1)
    tail_n = min(32, int(sub.shape[0]))
    return {
        "mean_sampled_logprob": float(sub[:, 0].mean().item()),
        "mean_topk_entropy": float(entropy.mean().item()),
        "mean_top1_logprob": float(top1.mean().item()),
        "mean_top1_top2_margin": float(margin.mean().item()),
        "tail_mean_sampled_logprob": float(sub[-tail_n:, 0].mean().item()),
        "tail_mean_topk_entropy": float(entropy[-tail_n:].mean().item()),
    }


def score_prefixes(
    cfg: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    specs: Sequence[Mapping[str, Any]],
    runner: VLLMFeatureExporter,
    checkpoint_path: Path,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, torch.Tensor]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pvm, calibration_temperature = load_pvm(cfg, checkpoint_path, device)
    segment_size = int(cfg["data"]["segment_size"])
    token_dim = int(cfg["data"]["instance_dim"])
    by_id = {str(row["sample_id"]): row for row in source_rows}
    source_ids = [str(row["sample_id"]) for row in source_rows]

    prompts = []
    full_ids = []
    prompt_lens = []
    source_temps = []
    tokenizer = runner.tokenizer
    for sid in source_ids:
        row = by_id[sid]
        prompt_ids = tokenizer(render_prompt(row), add_special_tokens=False).input_ids
        response_ids = list(row.get("token_ids", []))
        prompts.append(prompt_ids)
        full_ids.append(list(prompt_ids) + response_ids)
        prompt_lens.append(len(prompt_ids))
        source_temps.append(float(row.get("temperature", 1.0)))

    extracted = runner.extract_from_ids(
        full_ids,
        prompt_lens,
        temperatures=source_temps,
        top_k=top_k,
        return_logprobs=True,
        return_hidden=False,
        return_prompt_hidden=False,
        device=device,
    )
    logprobs_by_source = {
        sid: lp.detach().cpu()
        for sid, lp in zip(source_ids, extracted["logprobs"])
    }

    metrics: List[Dict[str, Any]] = []
    with torch.no_grad():
        for spec in specs:
            sid = str(spec["source_sample_id"])
            row = by_id[sid]
            lp = logprobs_by_source[sid]
            masked = build_masked_concat_segment_obs_from_lp(
                lp.to(device),
                list(row.get("tokens", [])),
                str(row.get("response", "")),
                segment_size=segment_size,
                token_dim=token_dim,
                device=device,
                segment_mode=str(cfg["data"].get("segment_mode", "fixed_window")),
            )
            prefix_segments = min(int(spec["prefix_segments"]), int(masked.features.shape[0]))
            features = masked.features[:prefix_segments].unsqueeze(0).float()
            token_mask = masked.token_mask[:prefix_segments].unsqueeze(0).float()
            segment_mask = torch.ones(1, prefix_segments, device=device)
            out = pvm(features, token_mask, segment_mask)
            phi = calibrated_probability(out["terminal_logits"], calibration_temperature)[0].item()
            metrics.append({
                **dict(spec),
                "pvm_phi": float(phi),
                **token_stats(lp, int(spec["prefix_token_end"])),
            })
    return metrics, logprobs_by_source


def continuation_seed(base_seed: int, prefix_idx: int, temp_idx: int, seed_idx: int,
                      n_temps: int, seeds_per_temperature: int) -> int:
    return int(base_seed) + prefix_idx * n_temps * seeds_per_temperature + temp_idx * seeds_per_temperature + seed_idx


def generate_continuations(
    rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    temperatures: Sequence[float],
    seeds_per_temperature: int,
    continuation_max_tokens: int,
    seed: int,
    parallel_size: int,
    gpu_memory_utilization: float,
    max_batch_size: int,
) -> List[Dict[str, Any]]:
    from vllm import LLM, SamplingParams

    by_id = {str(row["sample_id"]): row for row in source_rows}
    llm = LLM(
        model=str(cfg["inference"]["model_name_or_path"]),
        tensor_parallel_size=int(parallel_size),
        max_model_len=int(cfg["inference"].get("max_new_tokens", 8192)) + 2048,
        gpu_memory_utilization=float(gpu_memory_utilization),
    )
    tokenizer = llm.get_tokenizer()
    prompts: List[str] = []
    params: List[Any] = []
    request_map: List[Dict[str, Any]] = []
    for prefix_idx, row in enumerate(rows):
        source = by_id[str(row["source_sample_id"])]
        prefix_ids = list(source.get("token_ids", []))[:int(row["prefix_token_end"])]
        prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
        rendered = render_prompt(source)
        for temp_idx, temp in enumerate(temperatures):
            for seed_idx in range(seeds_per_temperature):
                prompts.append(rendered + prefix_text)
                generation_seed = continuation_seed(
                    seed, prefix_idx, temp_idx, seed_idx,
                    len(temperatures), seeds_per_temperature,
                )
                params.append(SamplingParams(
                    n=1,
                    temperature=float(temp),
                    max_tokens=int(continuation_max_tokens),
                    top_p=1.0,
                    top_k=-1,
                    seed=int(generation_seed),
                ))
                request_map.append({
                    "prefix_idx": prefix_idx,
                    "temperature": float(temp),
                    "temperature_index": temp_idx,
                    "seed_index": seed_idx,
                    "generation_seed": int(generation_seed),
                    "prefix_text": prefix_text,
                })

    outputs = []
    step = max(1, int(max_batch_size))
    for start in range(0, len(prompts), step):
        end = min(len(prompts), start + step)
        outputs.extend(llm.generate(prompts[start:end], params[start:end], use_tqdm=True))

    grouped: List[List[Dict[str, Any]]] = [[] for _ in rows]
    for out, req in zip(outputs, request_map):
        prefix_idx = int(req["prefix_idx"])
        source = by_id[str(rows[prefix_idx]["source_sample_id"])]
        generated_text, generated_token_ids, finish_reason = unpack_generation_output(out)
        full_response = str(req["prefix_text"]) + generated_text
        gold = str(source.get("metadata", {}).get("gold_answer", ""))
        grouped[prefix_idx].append({
            "temperature": float(req["temperature"]),
            "temperature_index": int(req["temperature_index"]),
            "seed_index": int(req["seed_index"]),
            "generation_seed": int(req["generation_seed"]),
            "correct": bool(verify_answer(full_response, gold)),
            "extracted_answer": extract_final_answer(full_response),
            "generated_tokens": len(generated_token_ids),
            "finish_reason": finish_reason,
        })

    result: List[Dict[str, Any]] = []
    for row, continuations in zip(rows, grouped):
        by_temp: Dict[float, List[int]] = defaultdict(list)
        for item in continuations:
            by_temp[float(item["temperature"])].append(1 if item["correct"] else 0)
        per_temp: Dict[str, Dict[str, float | int]] = {}
        rates = []
        for temp in sorted(by_temp):
            vals = by_temp[temp]
            rate = sum(vals) / max(1, len(vals))
            rates.append(rate)
            per_temp[str(float(temp))] = {
                "n_correct": int(sum(vals)),
                "n_total": int(len(vals)),
                "success_rate": float(rate),
            }
        max_rate = max(rates) if rates else 0.0
        min_rate = min(rates) if rates else 0.0
        result.append({
            **dict(row),
            "n_correct": int(sum(1 for item in continuations if item["correct"])),
            "n_total": int(len(continuations)),
            "observed_success_rate": (
                sum(1 for item in continuations if item["correct"]) / max(1, len(continuations))
            ),
            "temperature_sensitivity": float(max_rate - min_rate),
            "per_temperature_stats": per_temp,
            "continuations": continuations,
        })
    return result


def release_runner(runner: VLLMFeatureExporter) -> None:
    try:
        runner._cleanup_hs_tmpdir()
    except Exception:
        pass
    try:
        runner._llm = None
        runner._tokenizer = None
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def classify_temperature_response(row: Mapping[str, Any], weak_threshold: float = 0.25) -> str:
    stats = row.get("per_temperature_stats", {})
    if not isinstance(stats, Mapping) or not stats:
        return "not_generated"
    rates = {float(temp): float(value["success_rate"]) for temp, value in stats.items()}
    sensitivity = max(rates.values()) - min(rates.values())
    if sensitivity <= weak_threshold:
        return "weak_or_none"
    low = [rate for temp, rate in rates.items() if temp <= 0.7]
    high = [rate for temp, rate in rates.items() if temp >= 0.9]
    low_mean = statistics.fmean(low) if low else float("nan")
    high_mean = statistics.fmean(high) if high else float("nan")
    if low_mean == low_mean and high_mean == high_mean:
        if low_mean - high_mean >= weak_threshold:
            return "low_temp_better"
        if high_mean - low_mean >= weak_threshold:
            return "high_temp_better"
    return "mixed_sensitive"


def mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if float(v) == float(v)]
    return statistics.fmean(vals) if vals else float("nan")


def summarize(rows: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_source_temp: Dict[float, List[Mapping[str, Any]]] = defaultdict(list)
    by_source_temp_response: Dict[Tuple[float, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        source_temp = float(row["source_temperature"])
        by_source_temp[source_temp].append(row)
        by_source_temp_response[(source_temp, str(row.get("temperature_response", "not_generated")))].append(row)

    temp_rows: List[Dict[str, Any]] = []
    for temp, items in sorted(by_source_temp.items()):
        ordered = sorted(items, key=lambda item: float(item["pvm_phi"]))
        k = max(1, len(ordered) // 4)
        low = ordered[:k]
        high = ordered[-k:]
        temp_rows.append({
            "source_temperature": temp,
            "n_prefixes": len(items),
            "mean_phi": mean(row["pvm_phi"] for row in items),
            "mean_source_correct": mean(row["source_correct"] for row in items),
            "mean_observed_success_rate": mean(row.get("observed_success_rate", float("nan")) for row in items),
            "low_phi_mean_success": mean(row.get("observed_success_rate", float("nan")) for row in low),
            "high_phi_mean_success": mean(row.get("observed_success_rate", float("nan")) for row in high),
            "high_minus_low_success": (
                mean(row.get("observed_success_rate", float("nan")) for row in high) -
                mean(row.get("observed_success_rate", float("nan")) for row in low)
            ),
            "low_phi_mean_topk_entropy": mean(row["mean_topk_entropy"] for row in low),
            "high_phi_mean_topk_entropy": mean(row["mean_topk_entropy"] for row in high),
            "low_phi_mean_sampled_logprob": mean(row["mean_sampled_logprob"] for row in low),
            "high_phi_mean_sampled_logprob": mean(row["mean_sampled_logprob"] for row in high),
            "mean_temperature_sensitivity": mean(row.get("temperature_sensitivity", float("nan")) for row in items),
        })

    response_rows: List[Dict[str, Any]] = []
    for (temp, response), items in sorted(by_source_temp_response.items()):
        response_rows.append({
            "source_temperature": temp,
            "temperature_response": response,
            "n_prefixes": len(items),
            "mean_phi": mean(row["pvm_phi"] for row in items),
            "mean_topk_entropy": mean(row["mean_topk_entropy"] for row in items),
            "mean_sampled_logprob": mean(row["mean_sampled_logprob"] for row in items),
            "mean_temperature_sensitivity": mean(row.get("temperature_sensitivity", float("nan")) for row in items),
            "mean_observed_success_rate": mean(row.get("observed_success_rate", float("nan")) for row in items),
        })
    return temp_rows, response_rows


def plot_summary(path: Path, summary_rows: Sequence[Mapping[str, Any]]) -> None:
    if not summary_rows:
        return
    temps = [float(row["source_temperature"]) for row in summary_rows]
    deltas = [float(row["high_minus_low_success"]) for row in summary_rows]
    low_entropy = [float(row["low_phi_mean_topk_entropy"]) for row in summary_rows]
    high_entropy = [float(row["high_phi_mean_topk_entropy"]) for row in summary_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)
    fig.suptitle("Source-temperature balanced logprob validation", fontsize=13, fontweight="bold")
    axes[0].bar([str(t) for t in temps], deltas, color="#5477C4", alpha=0.8)
    axes[0].axhline(0, color="#444444", linewidth=1)
    axes[0].set_xlabel("Source temperature")
    axes[0].set_ylabel("High-phi minus low-phi success")
    axes[0].grid(True, axis="y", alpha=0.3)
    x = range(len(temps))
    axes[1].plot(x, low_entropy, marker="o", label="Low phi", color="#CC6F47")
    axes[1].plot(x, high_entropy, marker="o", label="High phi", color="#5477C4")
    axes[1].set_xticks(list(x), [str(t) for t in temps])
    axes[1].set_xlabel("Source temperature")
    axes[1].set_ylabel("Mean top-k entropy")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend(frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path)
    plt.close(fig)


def write_summary_md(path: Path, summary_rows: Sequence[Mapping[str, Any]],
                     response_rows: Sequence[Mapping[str, Any]],
                     manifest: Mapping[str, Any]) -> None:
    lines = [
        "# Source Temperature Logprob Validation",
        "",
        "This is a small source-temperature balanced validation. It controls the temperature used to generate the prefix source trajectory, then checks whether PVM/logprob phenomena still appear.",
        "",
        "## Source Temperature Summary",
        "",
        "| source_T | n | high-low success | low entropy | high entropy | low sampled lp | high sampled lp | mean sensitivity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            "| {source_temperature:g} | {n_prefixes} | {high_minus_low_success:.3f} | "
            "{low_phi_mean_topk_entropy:.3f} | {high_phi_mean_topk_entropy:.3f} | "
            "{low_phi_mean_sampled_logprob:.3f} | {high_phi_mean_sampled_logprob:.3f} | "
            "{mean_temperature_sensitivity:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "## Temperature Response Summary",
        "",
        "| source_T | response | n | mean phi | mean entropy | mean sampled lp | mean sensitivity | mean success |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in response_rows:
        lines.append(
            "| {source_temperature:g} | {temperature_response} | {n_prefixes} | {mean_phi:.3f} | "
            "{mean_topk_entropy:.3f} | {mean_sampled_logprob:.3f} | "
            "{mean_temperature_sensitivity:.3f} | {mean_observed_success_rate:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "## Data Quality",
        "",
        f"- Selected source rows: {manifest['counts']['source_rows']}",
        f"- Selected prefixes: {manifest['counts']['prefixes']}",
        f"- Continuation labels generated: {manifest['counts'].get('continuations', 0)}",
        f"- Dry run: {manifest['parameters']['dry_run']}",
        f"- Skip continuations: {manifest['parameters']['skip_continuations']}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or Path("results") / f"source_temp_logprob_validation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    source_temperatures = parse_float_list(args.source_temperatures)
    prefix_quantiles = parse_float_list(args.prefix_quantiles)
    continuation_temperatures = parse_float_list(args.continuation_temperatures)
    input_path = Path(cfg["paths"][f"{args.split}_dataset"])
    checkpoint_path = Path(cfg["paths"]["prefix_value_ckpt"])
    top_k = int(args.top_k or cfg["inference"].get("top_k_logprobs", 4096))
    gpu_memory = (
        float(args.gpu_memory_utilization)
        if args.gpu_memory_utilization is not None
        else float(cfg["inference"].get("gpu_memory_utilization", 0.80))
    )

    dataset_rows = read_jsonl(input_path)
    source_rows = select_source_rows(dataset_rows, source_temperatures, args.sources_per_label)
    specs = build_prefix_specs(
        source_rows,
        prefix_quantiles,
        segment_size=int(cfg["data"]["segment_size"]),
    )
    write_jsonl(out_dir / "selected_sources.jsonl", source_rows)
    write_jsonl(out_dir / "selected_prefix_specs.jsonl", specs)

    manifest: Dict[str, Any] = {
        "inputs": {
            "config": str(args.config),
            "dataset": str(input_path),
            "checkpoint": str(checkpoint_path),
        },
        "parameters": {
            "split": args.split,
            "source_temperatures": source_temperatures,
            "sources_per_label": int(args.sources_per_label),
            "prefix_quantiles": prefix_quantiles,
            "continuation_temperatures": continuation_temperatures,
            "seeds_per_temperature": int(args.seeds_per_temperature),
            "continuation_max_tokens": int(args.continuation_max_tokens),
            "top_k": top_k,
            "parallel_size": int(args.parallel_size),
            "vllm_micro_batch_size": int(args.vllm_micro_batch_size),
            "dry_run": bool(args.dry_run),
            "skip_continuations": bool(args.skip_continuations),
        },
        "counts": {
            "dataset_rows": len(dataset_rows),
            "source_rows": len(source_rows),
            "prefixes": len(specs),
            "continuations": 0,
        },
        "outputs": {
            "selected_sources": str(out_dir / "selected_sources.jsonl"),
            "selected_prefix_specs": str(out_dir / "selected_prefix_specs.jsonl"),
        },
    }
    if args.dry_run:
        write_json(out_dir / "run_manifest.json", manifest)
        print(json.dumps(manifest, indent=2))
        return

    runner = VLLMFeatureExporter(
        model_name_or_path=str(cfg["inference"]["model_name_or_path"]),
        max_new_tokens=int(cfg["inference"].get("max_new_tokens", 8192)),
        parallel_size=int(args.parallel_size),
        gpu_memory_utilization=gpu_memory,
        reserve_training_gpu=False,
        max_batch_size=int(args.vllm_micro_batch_size),
        enforce_eager=bool(cfg["inference"].get("vllm_enforce_eager", False)),
        enable_prefix_caching=False,
    )
    scored, _ = score_prefixes(cfg, source_rows, specs, runner, checkpoint_path, top_k)
    write_jsonl(out_dir / "prefix_metrics_scored.jsonl", scored)
    write_csv(out_dir / "prefix_metrics_scored.csv", scored)
    manifest["outputs"].update({
        "prefix_metrics_scored": str(out_dir / "prefix_metrics_scored.csv"),
    })

    final_rows = scored
    if not args.skip_continuations:
        release_runner(runner)
        final_rows = generate_continuations(
            scored,
            source_rows,
            cfg,
            continuation_temperatures,
            int(args.seeds_per_temperature),
            int(args.continuation_max_tokens),
            int(args.seed),
            int(args.parallel_size),
            gpu_memory,
            int(args.vllm_micro_batch_size),
        )
        for row in final_rows:
            row["temperature_response"] = classify_temperature_response(row)
        manifest["counts"]["continuations"] = sum(int(row.get("n_total", 0)) for row in final_rows)
    else:
        for row in final_rows:
            row["temperature_response"] = "not_generated"

    summary_rows, response_rows = summarize(final_rows)
    write_jsonl(out_dir / "prefix_metrics.jsonl", final_rows)
    write_csv(out_dir / "prefix_metrics.csv", final_rows)
    write_csv(out_dir / "source_temperature_summary.csv", summary_rows)
    write_csv(out_dir / "temperature_response_summary.csv", response_rows)
    plot_summary(out_dir / "fig_source_temp_validation.png", summary_rows)
    manifest["outputs"].update({
        "prefix_metrics": str(out_dir / "prefix_metrics.csv"),
        "source_temperature_summary": str(out_dir / "source_temperature_summary.csv"),
        "temperature_response_summary": str(out_dir / "temperature_response_summary.csv"),
        "summary": str(out_dir / "summary.md"),
        "figure": str(out_dir / "fig_source_temp_validation.png"),
    })
    manifest["elapsed_seconds"] = time.perf_counter() - started
    write_json(out_dir / "run_manifest.json", manifest)
    write_summary_md(out_dir / "summary.md", summary_rows, response_rows, manifest)
    print(json.dumps({
        "output_dir": str(out_dir),
        "prefixes": len(final_rows),
        "continuations": manifest["counts"]["continuations"],
        "elapsed_seconds": manifest["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
