"""Dataset-level statistics and majority-voting analysis."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import yaml

from utils.math import safe_div
from utils.exp_logger import setup_experiment_logger
from utils.jsonl import strip_vote_suffix


def evaluate_dataset(data_path: str) -> Dict[str, Any]:
    total = 0
    voting_correct = 0
    individual_correct = 0
    temp_counts: Dict[float, int] = defaultdict(int)
    temp_correct: Dict[float, int] = defaultdict(int)
    temp_individual_correct: Dict[float, int] = defaultdict(int)

    # For majority voting: group by (base_id, temperature) → list of individual_correct
    vote_groups: Dict[Tuple[str, float], List[int]] = defaultdict(list)
    has_voting = False

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            voting_label = int(row.get("voting_label", 0))     # 0=correct, 1=error
            voting_correct_bool = 1 - voting_label             # flip: 1=correct
            temp = float(row.get("temperature", 0.0))
            voting_correct += voting_correct_bool
            temp_counts[temp] = temp_counts.get(temp, 0) + 1
            if voting_correct_bool:
                temp_correct[temp] = temp_correct.get(temp, 0) + 1

            individual_label = row.get("individual_label")
            if individual_label is not None:
                has_voting = True
                indiv_correct = 1 - int(individual_label)  # flip: 1=correct
                individual_correct += indiv_correct
                temp_individual_correct[temp] = temp_individual_correct.get(temp, 0) + indiv_correct
                base_id = strip_vote_suffix(row.get("sample_id", ""))
                vote_groups[(base_id, temp)].append(indiv_correct)

    per_temp_acc = {}
    for t, cnt in sorted(temp_counts.items()):
        entry: Dict[str, Any] = {
            "count": cnt,
            "majority_correct": temp_correct.get(t, 0),
            "majority_accuracy": safe_div(temp_correct.get(t, 0), cnt),
        }
        if has_voting:
            entry["individual_correct"] = temp_individual_correct.get(t, 0)
            entry["individual_accuracy"] = safe_div(temp_individual_correct.get(t, 0), cnt)
        per_temp_acc[f"t={t:.1f}"] = entry

    result: Dict[str, Any] = {
        "n_samples": total,
        "voting_accuracy": safe_div(voting_correct, total),
        "voting_error_ratio": safe_div(total - voting_correct, total),
        "per_temperature_breakdown": per_temp_acc,
    }

    if has_voting and vote_groups:
        group_labels = []
        group_indiv_accs = []
        for (_base_id, _temp), votes in sorted(vote_groups.items()):
            n_correct = sum(votes)
            majority = 1 if n_correct >= (len(votes) + 1) // 2 else 0
            group_labels.append(majority)
            group_indiv_accs.append(n_correct / len(votes))

        import numpy as np
        num_votes_per_group = len(next(iter(vote_groups.values())))
        result["majority_voting"] = {
            "num_votes": num_votes_per_group,
            "n_groups": len(vote_groups),
            "majority_accuracy": safe_div(sum(group_labels), len(group_labels)),
            "mean_individual_accuracy": float(np.mean(group_indiv_accs)),
            "individual_sample_accuracy": safe_div(individual_correct, total),
        }

        best_temp = max(per_temp_acc, key=lambda k: per_temp_acc[k]["majority_accuracy"])
        result["majority_voting"]["best_temperature_majority"] = {
            "temperature": best_temp,
            "accuracy": per_temp_acc[best_temp]["majority_accuracy"],
        }

    return result


def load_temperature_labels(data_path: str) -> Dict[float, List[int]]:
    """Return per-temperature *correctness* lists from a dataset JSONL file.

    Reads voting_label (0=correct, 1=error — majority vote result).
    Flips so that returned values are 1=correct, 0=error (for PPO accuracy).
    """
    temp_labels: Dict[float, List[int]] = defaultdict(list)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            temp = float(row.get("temperature", 0.0))
            label = int(row.get("voting_label", 0))
            temp_labels[temp].append(1 - label)  # flip: 0→1 (correct), 1→0 (error)
    return dict(temp_labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset statistics and majority-voting analysis.")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", default="datasets", help="Directory to save per-split JSON results")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = [
        ("train", cfg["paths"]["train_dataset"]),
        ("val", cfg["paths"]["val_dataset"]),
        ("test", cfg["paths"]["test_dataset"]),
    ]

    logger, _log_path, final_run_name = setup_experiment_logger(
        component="dataset_eval",
        run_name=args.run_name,
        log_dir=args.log_dir,
        config={"splits": [s[0] for s in splits]},
    )

    from pathlib import Path as _Path
    out_dir = _Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, data_path in splits:
        result = evaluate_dataset(data_path)
        logger.info("dataset_stats split=%s %s", split_name, json.dumps(result, indent=2))

        out_path = out_dir / f"eval_stats_{split_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        logger.info("eval_stats_saved=%s", str(out_path))

        print(f"\n=== {split_name} ({data_path}) ===")
        print(f"  Samples: {result['n_samples']}, "
              f"voting accuracy={result.get('voting_accuracy', 0):.3f}")
        mv = result.get("majority_voting")
        if mv:
            print(f"  Majority Voting: {mv['num_votes']} votes x {mv['n_groups']} groups, "
                  f"accuracy={mv['majority_accuracy']:.4f}")
            bt = mv.get("best_temperature_majority", {})
            print(f"  Best temp: {bt.get('temperature', '?')} → accuracy={bt.get('accuracy', 0):.4f}")

    logger.info("dataset_eval_complete run_name=%s", final_run_name)
    print()


if __name__ == "__main__":
    main()
