from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_pvm_phase3_tokens import (
    SourceEntropy,
    assign_length_tertiles,
    bootstrap_mean_diff,
    build_extraction_jobs,
    load_prefix_rows,
    permutation_p_value,
    phase3_for_prefixes,
    summarize_results,
    token_decile_summary,
)


class _TinyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False

        class _Encoded:
            input_ids = [ord(ch) % 17 for ch in text]

        return _Encoded()


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_load_prefix_rows_joins_high_low_without_dropping(tmp_path: Path):
    pvm = tmp_path / "pvm.csv"
    cont = tmp_path / "continuations.jsonl"
    _write_csv(pvm, [
        {
            "record_index": 0,
            "problem_id": "p0",
            "source_sample_id": "s0",
            "prefix_segments": 1,
            "prefix_stage": "early",
            "source_individual_label": 0,
            "pvm_phi": 0.1,
            "pvm_bucket": "low",
            "observed_success_rate": 0.0,
            "n_correct": 0,
            "n_total": 4,
        },
        {
            "record_index": 1,
            "problem_id": "p1",
            "source_sample_id": "s1",
            "prefix_segments": 2,
            "prefix_stage": "middle",
            "source_individual_label": 0,
            "pvm_phi": 0.9,
            "pvm_bucket": "high",
            "observed_success_rate": 1.0,
            "n_correct": 4,
            "n_total": 4,
        },
        {
            "record_index": 2,
            "problem_id": "p2",
            "source_sample_id": "s2",
            "prefix_segments": 2,
            "prefix_stage": "middle",
            "source_individual_label": 0,
            "pvm_phi": 0.5,
            "pvm_bucket": "mid",
            "observed_success_rate": 0.5,
            "n_correct": 2,
            "n_total": 4,
        },
    ])
    _write_jsonl(cont, [
        {"source_sample_id": "s0", "prefix_segments": 1, "prefix_token_end": 3},
        {"source_sample_id": "s1", "prefix_segments": 2, "prefix_token_end": 5},
        {"source_sample_id": "s2", "prefix_segments": 2, "prefix_token_end": 7},
    ])

    rows = load_prefix_rows(pvm, cont)

    assert [row.pvm_bucket for row in rows] == ["low", "high"]
    assert [row.prefix_token_end for row in rows] == [3, 5]


def test_build_extraction_jobs_deduplicates_by_source_max_prefix():
    prefixes = load_prefix_rows(
        pvm_scores_path=_fixture_pvm_rows(),
        continuations_path=_fixture_continuations(),
    )
    source_rows = {
        "s0": {"metadata": {"rendered_prompt": "abc"}, "token_ids": list(range(10))},
        "s1": {"metadata": {"rendered_prompt": "de"}, "token_ids": list(range(9))},
    }

    jobs = build_extraction_jobs(prefixes, source_rows, _TinyTokenizer())

    by_source = {job.source_sample_id: job for job in jobs}
    assert set(by_source) == {"s0", "s1"}
    assert by_source["s0"].max_prefix_token_end == 5
    assert by_source["s0"].response_ids == list(range(5))
    assert by_source["s1"].max_prefix_token_end == 4


def test_phase3_for_prefixes_slices_overlapping_prefixes():
    prefixes = load_prefix_rows(_fixture_pvm_rows(), _fixture_continuations())
    entropy = {
        "s0": SourceEntropy(
            prev_entropy=np.array([0.1, 0.4, 0.3, 0.9, 0.2]),
            last_entropy=np.array([0.2, 0.3, 0.8, 0.8, 0.7]),
            delta_entropy=np.array([0.1, -0.1, 0.5, -0.1, 0.5]),
            prev_sampled_logprob=np.zeros(5),
            last_sampled_logprob=np.ones(5),
        ),
        "s1": SourceEntropy(
            prev_entropy=np.array([0.5, 0.5, 0.5, 0.5]),
            last_entropy=np.array([0.4, 0.6, 0.7, 0.3]),
            delta_entropy=np.array([-0.1, 0.1, 0.2, -0.2]),
            prev_sampled_logprob=np.zeros(4),
            last_sampled_logprob=np.ones(4),
        ),
    }

    rows = phase3_for_prefixes(prefixes, entropy)
    by_key = {(row["source_sample_id"], row["prefix_segments"]): row for row in rows}

    assert by_key[("s0", 1)]["phase3_tokens"] == 2
    assert by_key[("s0", 1)]["phase3_rate"] == pytest.approx(2 / 3)
    assert by_key[("s0", 2)]["phase3_tokens"] == 3
    assert by_key[("s0", 2)]["phase3_rate"] == pytest.approx(3 / 5)
    assert by_key[("s1", 1)]["phase3_tokens"] == 2


def test_statistics_are_reproducible_for_extreme_split():
    high = [1.0, 1.0]
    low = [0.0, 0.0]

    diff = bootstrap_mean_diff(high, low, n_bootstrap=100, seed=123)
    p_value = permutation_p_value(high, low, n_permutations=100, seed=123)

    assert diff == {"observed": 1.0, "ci_low": 1.0, "ci_high": 1.0}
    assert p_value == pytest.approx(1 / 3)


def test_summary_and_deciles_keep_expected_denominators():
    prefixes = load_prefix_rows(_fixture_pvm_rows(), _fixture_continuations())
    entropy = {
        "s0": SourceEntropy(
            prev_entropy=np.zeros(5),
            last_entropy=np.array([1, 0, 1, 0, 1], dtype=float),
            delta_entropy=np.array([1, 0, 1, 0, 1], dtype=float),
            prev_sampled_logprob=np.zeros(5),
            last_sampled_logprob=np.zeros(5),
        ),
        "s1": SourceEntropy(
            prev_entropy=np.zeros(4),
            last_entropy=np.array([0, 1, 1, 0], dtype=float),
            delta_entropy=np.array([0, 1, 1, 0], dtype=float),
            prev_sampled_logprob=np.zeros(4),
            last_sampled_logprob=np.zeros(4),
        ),
    }
    rows = phase3_for_prefixes(prefixes, entropy)
    summary = summarize_results(rows, n_bootstrap=10, n_permutations=10, seed=1)
    deciles = token_decile_summary(prefixes, entropy)

    assert summary["buckets"]["low"]["n_prefixes"] == 2
    assert summary["buckets"]["high"]["n_prefixes"] == 1
    assert sum(row["total_tokens"] for row in deciles if row["pvm_bucket"] == "low") == 8
    assert assign_length_tertiles([5, 1, 3]) == ["long", "short", "mid"]


def _fixture_pvm_rows() -> Path:
    path = Path("/tmp/test_phase3_pvm.csv")
    _write_csv(path, [
        {
            "record_index": 0,
            "problem_id": "p0",
            "source_sample_id": "s0",
            "prefix_segments": 1,
            "prefix_stage": "early",
            "source_individual_label": 0,
            "pvm_phi": 0.1,
            "pvm_bucket": "low",
            "observed_success_rate": 0.25,
            "n_correct": 1,
            "n_total": 4,
        },
        {
            "record_index": 1,
            "problem_id": "p0",
            "source_sample_id": "s0",
            "prefix_segments": 2,
            "prefix_stage": "late",
            "source_individual_label": 0,
            "pvm_phi": 0.2,
            "pvm_bucket": "low",
            "observed_success_rate": 0.50,
            "n_correct": 2,
            "n_total": 4,
        },
        {
            "record_index": 2,
            "problem_id": "p1",
            "source_sample_id": "s1",
            "prefix_segments": 1,
            "prefix_stage": "middle",
            "source_individual_label": 0,
            "pvm_phi": 0.9,
            "pvm_bucket": "high",
            "observed_success_rate": 1.00,
            "n_correct": 4,
            "n_total": 4,
        },
    ])
    return path


def _fixture_continuations() -> Path:
    path = Path("/tmp/test_phase3_continuations.jsonl")
    _write_jsonl(path, [
        {"source_sample_id": "s0", "prefix_segments": 1, "prefix_token_end": 3},
        {"source_sample_id": "s0", "prefix_segments": 2, "prefix_token_end": 5},
        {"source_sample_id": "s1", "prefix_segments": 1, "prefix_token_end": 4},
    ])
    return path
