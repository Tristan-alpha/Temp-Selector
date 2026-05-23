"""Online PPO policy evaluation with vLLM per-segment temperature control.

Uses vLLM Automatic Prefix Caching (APC).

Entry point:
    python -m ppo.eval --data data/prompts.jsonl --config configs/base.yaml --ppo-ckpt data/ppo_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from features.segmenter import build_segment_obs_from_lp
from features.dataset_eval import load_temperature_labels
from ppo.model import PolicyValueNet
from inference.vllm_runner import VLLMFeatureExporter
from utils.jsonl import sample_prefix
from utils.exp_logger import setup_experiment_logger
from utils.math import safe_div


def load_test_prompts(dataset_path: str) -> List[Dict[str, Any]]:
    """Extract unique (question, answer) pairs from a labeled test dataset JSONL.

    Each row is a BagSample with sample_id, prompt, and metadata.gold_answer.
    Deduplicates by sample_prefix so each unique prompt appears once.
    """
    seen: set = set()
    prompts: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id", ""))
            prefix = sample_prefix(sid)
            if prefix in seen:
                continue
            seen.add(prefix)
            prompts.append({
                "question": row.get("prompt", ""),
                "answer": row.get("metadata", {}).get("gold_answer", ""),
            })
    return prompts


@dataclass
class OnlineResult:
    accuracy: float = 0.0
    n_correct: int = 0
    n_total: int = 0
    temperatures: List[float] = field(default_factory=list)
    segment_counts: List[int] = field(default_factory=list)


class OnlineTemperatureEvaluator:
    def __init__(
        self,
        model_name_or_path: str,
        ppo_ckpt: str,
        config: Dict[str, Any],
        parallel_size: int | None = None,
    ):
        self.segment_size = int(config["data"]["segment_size"])
        self.segment_mode = config["data"].get("segment_mode", "fixed_window")
        self.max_new_tokens = int(config["inference"]["max_new_tokens"])
        self.instance_dim = int(config["data"]["instance_dim"])
        self.pooling_mode = config["data"].get("segment_pooling", "mean")
        self.model_obs_dim = self.instance_dim * self.segment_size if self.pooling_mode == "concat" else self.instance_dim
        self.hidden_dim = int(config["ppo"]["model"]["hidden_dim"])
        self.temp_bins = [float(x) for x in config["data"]["temp_bins"]]
        self.n_actions = len(self.temp_bins)
        self.top_k_logprobs = int(config["inference"]["top_k_logprobs"])
        self.num_votes = int(config["inference"].get("num_votes", 1))
        self.system_prompt = config["inference"].get("system_prompt", "")
        self.use_math_chat = bool(config["inference"].get("use_math_chat_prompt", True))
        self.feature_mode = config["inference"].get("feature_mode", "topk_logprobs")
        self.hs_needed = self.feature_mode == "hidden_states"

        gpu_mem = float(config.get("inference", {}).get("gpu_memory_utilization", 0.90))
        self.runner = VLLMFeatureExporter(
            model_name_or_path=model_name_or_path,
            max_new_tokens=self.max_new_tokens,
            parallel_size=parallel_size,
            gpu_memory_utilization=gpu_mem,
            reserve_training_gpu=True,
        )
        self.tokenizer = self.runner.tokenizer

        ckpt = torch.load(ppo_ckpt, map_location="cpu", weights_only=False)
        policy_state = ckpt.get("policy_value", ckpt)
        self.policy = PolicyValueNet(obs_dim=self.model_obs_dim, n_actions=self.n_actions, hidden=self.hidden_dim)
        self.policy.load_state_dict(policy_state, strict=False)
        self.policy.eval()

        self._timing: Dict[str, float] = {}  # accumulated seconds per phase

    def _render_prompt(self, question: str) -> str:
        if not self.use_math_chat:
            return question
        return self.runner.render_messages(
            self.runner.build_math_messages(question, system_prompt=self.system_prompt))

    def _evaluate_strategy_batch(
        self,
        prompts_data: List[Dict[str, Any]],
        strategy: str,
        best_fixed_temp: float,
        rng: random.Random,
    ) -> OnlineResult:
        V = self.num_votes
        N = len(prompts_data)
        rendered = [self._render_prompt(
            p.get("question") or p.get("problem") or p.get("prompt", ""))
            for p in prompts_data]
        gold_answers = [p.get("answer", "") for p in prompts_data]

        generated: List[List[str]] = [[""] * V for _ in range(N)]
        active: List[List[bool]] = [[True] * V for _ in range(N)]
        segment_obs: List[List[Optional[List[float]]]] = [[None] * V for _ in range(N)]
        all_temps: List[List[float]] = [[] for _ in range(N)]
        n_segments: List[int] = [0] * N

        from utils.answer_verifier import self_consistency_correct

        max_rounds = self.max_new_tokens // self.segment_size

        t_total0 = time.perf_counter()
        self._timing.setdefault("policy_decision", 0.0)
        self._timing.setdefault("vllm_generate", 0.0)
        self._timing.setdefault("feature_postproc", 0.0)
        self._timing.setdefault("pp_text", 0.0)
        self._timing.setdefault("pp_eos", 0.0)
        self._timing.setdefault("pp_build_obs", 0.0)
        self._timing.setdefault("pp_tolist", 0.0)
        for _seg_idx in range(max_rounds):
            round_prompts: List[str] = []
            round_temps: List[float] = []
            round_map: List[Tuple[int, int]] = []  # (prompt_idx, chain_idx)

            t0 = time.perf_counter()
            for i in range(N):
                for v in range(V):
                    if not active[i][v]:
                        continue
                    round_prompts.append(rendered[i] + generated[i][v])

                    if strategy == "ppo":
                        if _seg_idx == 0 or segment_obs[i][v] is None:
                            temp = 0.7
                        else:
                            obs_t = torch.tensor(segment_obs[i][v][-1], dtype=torch.float32).unsqueeze(0)
                            with torch.no_grad():
                                logits, _ = self.policy(obs_t)
                                action = logits.argmax(dim=-1).item()
                            temp = self.temp_bins[action]
                    elif strategy == "best-fixed":
                        temp = best_fixed_temp
                    else:
                        temp = rng.choice(self.temp_bins)

                    if v == 0:
                        all_temps[i].append(temp)
                    round_temps.append(temp)
                    round_map.append((i, v))

            if not round_map:
                break

            t1 = time.perf_counter()
            feats = self.runner.generate_with_features(
                round_prompts, round_temps, self.segment_size,
                top_k=self.top_k_logprobs,
                return_logprobs=True,
                return_hidden=self.hs_needed,
            )
            t2 = time.perf_counter()

            pp_text_r = pp_eos_r = pp_build_r = pp_tolist_r = 0.0
            for j, (i, v) in enumerate(round_map):
                f = feats[j]

                ta = time.perf_counter()
                generated[i][v] += f["text"]

                if v == 0:
                    n_segments[i] += 1
                tb = time.perf_counter()

                if (self.tokenizer.eos_token_id is not None and
                    self.tokenizer.eos_token_id in f["token_ids"]) or \
                   f["finish_reason"] == 'stop' or not f["token_ids"]:
                    active[i][v] = False
                    pp_text_r += tb - ta
                    pp_eos_r += time.perf_counter() - tb
                    continue

                tc = time.perf_counter()
                if f["logprobs"] is not None:
                    extra = [f["hidden_states"]] if f["hidden_states"] is not None else None
                    obs = build_segment_obs_from_lp(
                        f["logprobs"], f["tokens"], f["text"],
                        self.segment_size, self.instance_dim,
                        segment_mode=self.segment_mode,
                        include_topk=(not self.hs_needed),
                        extra_parts=extra,
                        pooling_mode=self.pooling_mode,
                    )
                    td = time.perf_counter()
                    segment_obs[i][v] = obs.tolist()
                    te = time.perf_counter()
                    pp_text_r += tb - ta
                    pp_eos_r += tc - tb
                    pp_build_r += td - tc
                    pp_tolist_r += te - td

            t3 = time.perf_counter()
            n_active = len(round_map)
            self._timing["policy_decision"] += t1 - t0
            self._timing["vllm_generate"] += t2 - t1
            self._timing["feature_postproc"] += t3 - t2
            self._timing["pp_text"] += pp_text_r
            self._timing["pp_eos"] += pp_eos_r
            self._timing["pp_build_obs"] += pp_build_r
            self._timing["pp_tolist"] += pp_tolist_r
            pp_other_r = (t3 - t2) - (pp_text_r + pp_eos_r + pp_build_r + pp_tolist_r)
            print(f"  [timing] round={_seg_idx:3d}  active={n_active:4d}  "
                  f"decision={t1 - t0:.3f}s  generate={t2 - t1:.2f}s  "
                  f"postproc={t3 - t2:.1f}s "
                  f"(text={pp_text_r:.1f}s eos={pp_eos_r:.1f}s "
                  f"build_obs={pp_build_r:.1f}s tolist={pp_tolist_r:.1f}s "
                  f"other={pp_other_r:.1f}s)  total={t3 - t0:.1f}s")

        t_total1 = time.perf_counter()
        result = OnlineResult()
        for i in range(N):
            majority_correct = 1 if self_consistency_correct(generated[i], gold_answers[i]) else 0
            result.n_total += 1
            if majority_correct:
                result.n_correct += 1
            result.temperatures.extend(all_temps[i])
            result.segment_counts.append(n_segments[i])

        t_total2 = time.perf_counter()
        result.accuracy = safe_div(result.n_correct, result.n_total)

        d = self._timing
        wall = t_total2 - t_total0
        accounted = (d.get("policy_decision", 0) + d.get("vllm_generate", 0) +
                     d.get("feature_postproc", 0) + (t_total2 - t_total1))
        print(f"  [timing] strategy={strategy:12s}  "
              f"decision={d.get('policy_decision', 0):.1f}s  "
              f"generate={d.get('vllm_generate', 0):.1f}s  "
              f"postproc={d.get('feature_postproc', 0):.1f}s  "
              f"(text={d.get('pp_text', 0):.1f}s eos={d.get('pp_eos', 0):.1f}s "
              f"build_obs={d.get('pp_build_obs', 0):.1f}s tolist={d.get('pp_tolist', 0):.1f}s)  "
              f"scoring={t_total2 - t_total1:.1f}s  "
              f"other={wall - accounted:.1f}s  "
              f"wall={wall:.1f}s")
        return result

    def evaluate(
        self,
        data_path: str,
        prompts_data: List[Dict[str, Any]],
        seed: int = 42,
    ) -> Dict[str, Any]:
        if not prompts_data:
            return {"error": "No prompts found."}

        temp_labels = load_temperature_labels(data_path)
        best_fixed_temp = self.temp_bins[len(self.temp_bins) // 2]
        if temp_labels:
            per_temp_acc = {t: sum(lbls) / len(lbls) for t, lbls in temp_labels.items() if lbls}
            if per_temp_acc:
                best_fixed_temp = max(per_temp_acc, key=per_temp_acc.get)

        rng = random.Random(seed)
        torch.manual_seed(seed)

        results: Dict[str, Any] = {
            "n_prompts": len(prompts_data),
            "num_votes": self.num_votes,
            "segment_size": self.segment_size,
            "temperature_bins": self.temp_bins,
            "best_fixed_temperature": best_fixed_temp,
        }

        strategies = [
            ("ppo", "PPO dynamic temperature"),
            ("best-fixed", f"Best fixed temperature (T={best_fixed_temp:.1f})"),
            ("random", "Random temperature"),
        ]

        for strategy_key, strategy_label in strategies:
            strat_rng = random.Random(seed)
            result = self._evaluate_strategy_batch(
                prompts_data, strategy_key, best_fixed_temp, strat_rng,
            )

            import numpy as np
            temp_arr = np.array(result.temperatures) if result.temperatures else np.array([])

            results[strategy_key] = {
                "label": strategy_label,
                "accuracy": result.accuracy,
                "n_correct": result.n_correct,
                "n_total": result.n_total,
                "mean_temperature": float(temp_arr.mean()) if len(temp_arr) > 0 else 0.0,
                "std_temperature": float(temp_arr.std()) if len(temp_arr) > 0 else 0.0,
                "mean_segments": float(np.mean(result.segment_counts)) if result.segment_counts else 0.0,
                "total_segments": sum(result.segment_counts),
            }

        ppo_acc = results["ppo"]["accuracy"]
        best_acc = results["best-fixed"]["accuracy"]
        rand_acc = results["random"]["accuracy"]
        results["improvement_over_random"] = ppo_acc - rand_acc
        results["improvement_over_best_fixed"] = ppo_acc - best_acc
        results["_note"] = (
            "Online evaluation: the PPO policy truly controls generation temperature "
            "segment-by-segment via vLLM with APC."
        )

        return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Online PPO evaluation — vLLM generation with per-segment temperature control."
    )
    parser.add_argument("--test-data", default=None, help="Override paths.test_dataset from config (prompts + temp statistics)")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--ppo-ckpt", default=None, help="Override paths.ppo_ckpt from config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--parallel-size", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    test_data_path = args.test_data or config["paths"]["test_dataset"]
    ppo_ckpt = args.ppo_ckpt or config["paths"]["ppo_ckpt"]

    logger, log_path, final_run_name = setup_experiment_logger(
        component="online_evaluate",
        run_name=args.run_name,
        log_dir=args.log_dir,
        config={"test_data": test_data_path, "config": args.config, "ppo_ckpt": ppo_ckpt},
    )

    # Load test prompts directly from the labeled test dataset — guarantees
    # no data leakage from train/val prompts.
    prompts_data = load_test_prompts(test_data_path)
    logger.info("test_prompts=%d", len(prompts_data))

    logger.info("Initializing vLLM + PPO online evaluator ...")
    evaluator = OnlineTemperatureEvaluator(
        model_name_or_path=config["inference"]["model_name_or_path"],
        ppo_ckpt=ppo_ckpt,
        config=config,
        parallel_size=args.parallel_size,
    )
    logger.info("Evaluator ready.  Running online evaluation ...")
    results = evaluator.evaluate(data_path=test_data_path, prompts_data=prompts_data, seed=args.seed)

    logger.info("online_results=%s", json.dumps(results, indent=2, default=str))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("metrics_saved=%s", args.output)

    print("\n" + "=" * 70)
    print("ONLINE EVALUATION RESULTS  (vLLM + APC, per-segment temperature)")
    print("=" * 70)
    print(f"Prompts evaluated: {results['n_prompts']}")
    print(f"Majority voting:   {results['num_votes']} votes per prompt")
    print(f"Segment size: {results['segment_size']} tokens")
    print(f"Best fixed temp (from dataset): T={results['best_fixed_temperature']:.1f}")
    print()

    for key in ["ppo", "best-fixed", "random"]:
        if key in results:
            r = results[key]
            print(f"  {r['label']}:")
            print(f"    accuracy={r['accuracy']:.4f}  correct={r['n_correct']}/{r['n_total']}")
            print(f"    mean_temp={r['mean_temperature']:.2f} ± {r['std_temperature']:.2f}")
            print(f"    avg_segments={r['mean_segments']:.1f}  total_segments={r['total_segments']}")

    print(f"\n  Improvement over random:     {results['improvement_over_random']:+.4f}")
    print(f"  Improvement over best fixed:  {results['improvement_over_best_fixed']:+.4f}")
    print("\n  " + results["_note"])
    print("=" * 70 + "\n")

    logger.info("online_evaluation_complete run_name=%s log_path=%s", final_run_name, log_path)


if __name__ == "__main__":
    main()
