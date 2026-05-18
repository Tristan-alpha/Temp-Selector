"""Online PPO policy evaluation with vLLM per-segment temperature control.

Uses vLLM Automatic Prefix Caching (APC).

Entry point:
    python -m ppo.eval --data data/prompts.jsonl --config configs/base.yaml --ppo-ckpt data/ppo_ckpt.pt
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import yaml

from features.segmenter import build_segments, segment_pooling
from features.dataset_eval import load_temperature_labels
from ppo.model import PolicyValueNet
from inference.vllm_runner import VLLMFeatureExporter
from utils.jsonl import sample_prefix
from utils.exp_logger import setup_experiment_logger
from utils.math import safe_div


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
    errors: int = 0


class OnlineTemperatureEvaluator:
    def __init__(
        self,
        model_name_or_path: str,
        ppo_ckpt: str,
        config: Dict[str, Any],
        parallel_size: int | None = None,
    ):
        self.segment_size = int(config["data"]["segment_size"])
        self.max_new_tokens = int(config["inference"]["max_new_tokens"])
        self.obs_dim = int(config["data"]["instance_dim"])
        self.hidden_dim = int(config["ppo"]["model"]["hidden_dim"])
        self.temp_bins = [float(x) for x in config["data"]["temp_bins"]]
        self.n_actions = len(self.temp_bins)
        self.top_k_logprobs = int(config["inference"]["top_k_logprobs"])
        self.num_votes = int(config["inference"].get("num_votes", 1))
        self.system_prompt = config["inference"].get("system_prompt", "")
        self.use_math_chat = bool(config["inference"].get("use_math_chat_prompt", True))
        self.feature_mode = config["inference"].get("feature_mode", "basic")

        gpu_mem = float(config.get("inference", {}).get("gpu_memory_utilization", 0.90))
        self.runner = VLLMFeatureExporter(
            model_name_or_path=model_name_or_path,
            max_new_tokens=self.max_new_tokens,
            parallel_size=parallel_size,
            gpu_memory_utilization=gpu_mem,
            feature_mode=self.feature_mode,
            reserve_training_gpu=True,
        )
        self.tokenizer = self.runner.tokenizer

        ckpt = torch.load(ppo_ckpt, map_location="cpu", weights_only=False)
        policy_state = ckpt.get("policy_value", ckpt)
        self.policy = PolicyValueNet(obs_dim=self.obs_dim, n_actions=self.n_actions, hidden=self.hidden_dim)
        self.policy.load_state_dict(policy_state, strict=False)
        self.policy.eval()

    def _render_prompt(self, question: str) -> str:
        if not self.use_math_chat:
            return question
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            parts = [f"[SYSTEM]\n{self.system_prompt}", f"[USER]\n{question}", "[ASSISTANT]\n"]
            return "\n\n".join(parts)

    def _evaluate_strategy_batch(
        self,
        prompts_data: List[Dict[str, Any]],
        strategy: str,
        best_fixed_temp: float,
        rng: random.Random,
    ) -> OnlineResult:
        V = self.num_votes
        N = len(prompts_data)
        rendered = [self._render_prompt(self._get_question(p)) for p in prompts_data]
        gold_answers = [p.get("answer", "") for p in prompts_data]

        generated: List[List[str]] = [[""] * V for _ in range(N)]
        active = [True] * N
        segment_obs: List[Optional[List[float]]] = [None] * N
        all_temps: List[List[float]] = [[] for _ in range(N)]
        n_segments: List[int] = [0] * N

        from utils.answer_verifier import self_consistency_correct

        max_rounds = self.max_new_tokens // self.segment_size + 1

        for _seg_idx in range(max_rounds):
            round_prompts: List[str] = []
            round_temps: List[float] = []
            round_indices: List[int] = []

            for i in range(N):
                if not active[i]:
                    continue
                round_prompts.append(rendered[i] + generated[i][0])

                if strategy == "ppo":
                    if _seg_idx == 0 or segment_obs[i] is None:
                        temp = 0.7
                    else:
                        obs_t = torch.tensor(segment_obs[i], dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad():
                            logits, _ = self.policy(obs_t)
                            action = logits.argmax(dim=-1).item()
                        temp = self.temp_bins[action]
                elif strategy == "best-fixed":
                    temp = best_fixed_temp
                else:
                    temp = rng.choice(self.temp_bins)

                all_temps[i].append(temp)
                round_temps.append(temp)
                round_indices.append(i)

            if not round_indices:
                break

            feats = self.runner.generate_with_features(
                round_prompts, round_temps, self.segment_size,
                top_k=self.top_k_logprobs, n=V,
            )

            for j, i in enumerate(round_indices):
                f = feats[j]
                for v in range(min(V, len(f["all_texts"]))):
                    generated[i][v] += f["all_texts"][v]

                n_segments[i] += 1

                if (self.tokenizer.eos_token_id is not None and
                    self.tokenizer.eos_token_id in f["token_ids"]) or \
                   f["finish_reason"] == 'stop' or not f["token_ids"]:
                    active[i] = False
                    continue

                if f["logprobs"] is not None:
                    lp_t = f["logprobs"]
                    n_tok = len(f["token_ids"])
                    parts = [torch.tensor(
                        [[float(lp_t[k, 0]), 0.0] for k in range(n_tok)],
                        dtype=torch.float32)]
                    for k in range(n_tok):
                        probs = torch.softmax(lp_t[k, 1:].float(), dim=0)
                        ent = -(probs * torch.log(probs + 1e-12)).sum()
                        parts[0][k, 1] = ent.item()
                    tok_vecs = torch.cat(parts, dim=1)
                    if tok_vecs.shape[1] < self.obs_dim:
                        tok_vecs = torch.cat([
                            tok_vecs,
                            torch.zeros(n_tok, self.obs_dim - tok_vecs.shape[1]),
                        ], dim=1)
                    else:
                        tok_vecs = tok_vecs[:, :self.obs_dim]
                    spans = build_segments(tokens=f["tokens"], mode="step",
                                           segment_size=self.segment_size,
                                           response=f["text"])
                    obs = segment_pooling(tok_vecs, spans, self.obs_dim,
                                          mode="mean",
                                          segment_size=self.segment_size)
                    segment_obs[i] = obs.tolist()

        result = OnlineResult()
        for i in range(N):
            majority_correct = 1 if self_consistency_correct(generated[i], gold_answers[i]) else 0
            result.n_total += 1
            if majority_correct:
                result.n_correct += 1
            result.temperatures.extend(all_temps[i])
            result.segment_counts.append(n_segments[i])

        result.accuracy = safe_div(result.n_correct, result.n_total)
        return result

    @staticmethod
    def _get_question(p: Dict[str, Any]) -> str:
        return p.get("question") or p.get("problem") or p.get("prompt", "")

    def evaluate(
        self,
        data_path: str,
        prompts_data: List[Dict[str, Any]] | None = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        if prompts_data is None:
            prompts_data = load_prompts(data_path)
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
                "errors": result.errors,
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
    parser.add_argument("--parallel-size", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--output", default=None, help="Optional path to save metrics JSON")
    args = parser.parse_args()

    config = load_config(args.config)
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
