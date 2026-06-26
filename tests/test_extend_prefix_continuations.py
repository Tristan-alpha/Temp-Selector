"""Tests for extending prefix continuation labels with additional seeds."""

from __future__ import annotations

import json

from scripts.extend_prefix_continuations import (
    append_generation_seed,
    default_checkpoint_dir,
    extend_records_with_results,
    extend_labels,
    progress_path,
    missing_request_plan,
    validate_extended_records,
    write_batch_checkpoint,
    write_progress,
)


TEMPERATURES = [0.1, 0.3]


def _record() -> dict:
    return {
        "problem_id": "p0",
        "source_sample_id": "p0_t0.1_v0",
        "prefix_segments": 1,
        "prefix_token_end": 64,
        "n_correct": 2,
        "n_total": 4,
        "continuations": [
            {
                "temperature": 0.1,
                "temperature_index": 0,
                "seed_index": 0,
                "generation_seed": 42,
                "correct": True,
            },
            {
                "temperature": 0.1,
                "temperature_index": 0,
                "seed_index": 1,
                "generation_seed": 43,
                "correct": False,
            },
            {
                "temperature": 0.3,
                "temperature_index": 1,
                "seed_index": 0,
                "generation_seed": 44,
                "correct": True,
            },
            {
                "temperature": 0.3,
                "temperature_index": 1,
                "seed_index": 1,
                "generation_seed": 45,
                "correct": False,
            },
        ],
        "generation": {
            "seed": 42,
            "seeds_per_temperature": 2,
            "temperatures": TEMPERATURES,
        },
    }


def test_missing_request_plan_only_adds_absent_seed_indices():
    plan = missing_request_plan(
        [_record()],
        TEMPERATURES,
        target_seeds_per_temperature=4,
        base_seed=42,
        append_seed_offset=10_000,
    )
    assert [(item["temperature_index"], item["seed_index"]) for item in plan] == [
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    ]
    assert len({item["generation_seed"] for item in plan}) == 4
    assert min(item["generation_seed"] for item in plan) >= 10_000


def test_missing_request_plan_uses_global_record_offset_for_seed():
    plan = missing_request_plan(
        [_record()],
        TEMPERATURES,
        target_seeds_per_temperature=4,
        base_seed=42,
        append_seed_offset=10_000,
        record_index_offset=10,
    )
    first = plan[0]
    assert first["record_idx"] == 0
    assert first["global_record_idx"] == 10
    assert first["generation_seed"] == 10_000 + 42 + 10 * len(TEMPERATURES) * 4 + 2


def test_append_generation_seed_uses_target_stride():
    seed = append_generation_seed(
        base_seed=42,
        record_idx=3,
        temp_idx=1,
        seed_index=7,
        n_temperatures=8,
        target_seeds_per_temperature=32,
        append_seed_offset=10_000_000,
    )
    assert seed == 10_000_000 + 42 + 3 * 8 * 32 + 1 * 32 + 7


def test_extend_records_preserves_existing_and_recomputes_stats():
    record = _record()
    plan = missing_request_plan(
        [record],
        TEMPERATURES,
        target_seeds_per_temperature=4,
        base_seed=42,
        append_seed_offset=10_000,
    )
    generated = [
        {
            "temperature": item["temperature"],
            "temperature_index": item["temperature_index"],
            "seed_index": item["seed_index"],
            "generation_seed": item["generation_seed"],
            "correct": item["seed_index"] == 2,
        }
        for item in plan
    ]
    extended = extend_records_with_results(
        [record],
        plan,
        generated,
        target_seeds_per_temperature=4,
        temperatures=TEMPERATURES,
        generation_update={
            "append_from_existing": "old.jsonl",
            "append_seed_offset": 10_000,
            "generated_missing_continuations": len(generated),
        },
    )
    out = extended[0]
    old_keys = {
        (item["temperature_index"], item["seed_index"], item["generation_seed"])
        for item in record["continuations"]
    }
    out_keys = {
        (item["temperature_index"], item["seed_index"], item["generation_seed"])
        for item in out["continuations"]
    }
    assert old_keys <= out_keys
    assert out["n_total"] == 8
    assert out["n_correct"] == 4
    assert out["per_temperature_stats"]["0.1"] == {
        "n_correct": 2,
        "n_total": 4,
        "success_rate": 0.5,
    }
    assert out["generation"]["seeds_per_temperature"] == 4
    assert out["generation"]["append_from_existing"] == "old.jsonl"


def test_validate_extended_records_checks_seed_coverage():
    record = _record()
    plan = missing_request_plan(
        [record],
        TEMPERATURES,
        target_seeds_per_temperature=4,
        base_seed=42,
        append_seed_offset=10_000,
    )
    generated = [
        {
            "temperature": item["temperature"],
            "temperature_index": item["temperature_index"],
            "seed_index": item["seed_index"],
            "generation_seed": item["generation_seed"],
            "correct": True,
        }
        for item in plan
    ]
    extended = extend_records_with_results(
        [record],
        plan,
        generated,
        target_seeds_per_temperature=4,
        temperatures=TEMPERATURES,
        generation_update={},
    )
    validation = validate_extended_records(
        extended,
        TEMPERATURES,
        target_seeds_per_temperature=4,
    )
    assert validation["passed"]
    assert validation["n_total_distribution"] == {"8": 1}


def test_validate_extended_records_reports_missing_seed():
    validation = validate_extended_records(
        [_record()],
        TEMPERATURES,
        target_seeds_per_temperature=4,
    )
    assert not validation["passed"]
    assert validation["n_errors"] > 0


def test_batch_checkpoint_and_progress_are_written_atomically(tmp_path):
    record = _record()
    plan = missing_request_plan(
        [record],
        TEMPERATURES,
        target_seeds_per_temperature=4,
        base_seed=42,
        append_seed_offset=10_000,
    )
    generated = [
        {
            "temperature": item["temperature"],
            "temperature_index": item["temperature_index"],
            "seed_index": item["seed_index"],
            "generation_seed": item["generation_seed"],
            "correct": True,
            "generated_text": " continuation",
            "full_response_text": "prefix continuation",
        }
        for item in plan
    ]
    extended = extend_records_with_results(
        [record],
        plan,
        generated,
        target_seeds_per_temperature=4,
        temperatures=TEMPERATURES,
        generation_update={"save_generated_text": True},
    )
    validation = validate_extended_records(
        extended,
        TEMPERATURES,
        target_seeds_per_temperature=4,
    )
    meta = write_batch_checkpoint(
        tmp_path / "batches",
        10,
        11,
        extended,
        validation,
        generated_count=len(generated),
        elapsed_seconds=1.25,
    )
    assert validation["passed"]
    assert (tmp_path / "batches" / "batch_10_11.jsonl").exists()
    assert (tmp_path / "batches" / "batch_10_11.meta.json").exists()
    assert meta["record_start"] == 10
    assert meta["record_end"] == 11

    output = tmp_path / "labels.jsonl"
    progress = write_progress(
        output,
        tmp_path / "batches",
        "partial",
        [
            {
                "record_start": 10,
                "record_end": 11,
                "generated_count": len(generated),
            }
        ],
        total_records=1,
        total_missing_continuations=len(generated),
        started_at=0.0,
    )
    assert progress_path(output).exists()
    saved_progress = json.loads(progress_path(output).read_text())
    assert saved_progress["status"] == "partial"
    assert saved_progress["n_completed_records"] == 1
    assert saved_progress["n_completed_generated_continuations"] == len(generated)


def test_resume_from_completed_batches_assembles_final_without_regeneration(tmp_path):
    data_path = tmp_path / "train.jsonl"
    existing_path = tmp_path / "existing.jsonl"
    output_path = tmp_path / "extended.jsonl"
    checkpoint_dir = default_checkpoint_dir(output_path)
    config_path = tmp_path / "config.yaml"

    row = {
        "sample_id": "p0_t0.1_v0",
        "prompt": "What is 1+1?",
        "token_ids": [1, 2, 3],
        "metadata": {"gold_answer": "2"},
    }
    data_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    existing_record = {
        "problem_id": "p0",
        "source_sample_id": "p0_t0.1_v0",
        "prefix_token_end": 1,
        "n_correct": 1,
        "n_total": 1,
        "continuations": [
            {
                "temperature": 0.1,
                "temperature_index": 0,
                "seed_index": 0,
                "generation_seed": 42,
                "correct": True,
            }
        ],
    }
    existing_path.write_text(json.dumps(existing_record) + "\n", encoding="utf-8")
    config_path.write_text(
        "\n".join(
            [
                "seed: 42",
                "paths:",
                f"  train_dataset: {data_path}",
                "inference:",
                "  model_name_or_path: unused",
                "  max_new_tokens: 8",
                "prefix_value:",
                "  continuations:",
                "    temperatures: [0.1]",
                "    prefix_sampling_seed: 42",
                "    max_new_tokens: 8",
            ]
        ),
        encoding="utf-8",
    )
    completed_record = extend_records_with_results(
        [existing_record],
        [
            {
                "record_idx": 0,
                "temperature": 0.1,
                "temperature_index": 0,
                "seed_index": 1,
                "generation_seed": 10_000_043,
            }
        ],
        [
            {
                "temperature": 0.1,
                "temperature_index": 0,
                "seed_index": 1,
                "generation_seed": 10_000_043,
                "correct": False,
                "generated_text": " because 2",
                "full_response_text": "prefix because 2",
            }
        ],
        target_seeds_per_temperature=2,
        temperatures=[0.1],
        generation_update={"save_generated_text": True},
    )
    validation = validate_extended_records(
        completed_record,
        [0.1],
        target_seeds_per_temperature=2,
    )
    write_batch_checkpoint(
        checkpoint_dir,
        0,
        1,
        completed_record,
        validation,
        generated_count=1,
        elapsed_seconds=0.1,
    )

    metadata = extend_labels(
        config_path=str(config_path),
        split="train",
        existing_path=str(existing_path),
        output_path=str(output_path),
        target_seeds_per_temperature=2,
        append_seed_offset=10_000_000,
        records_per_batch=1,
        resume=True,
        save_generated_text=True,
    )
    assert metadata["validation"]["passed"]
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["n_total"] == 2
    new_item = [
        item for item in rows[0]["continuations"]
        if item["seed_index"] == 1
    ][0]
    old_item = [
        item for item in rows[0]["continuations"]
        if item["seed_index"] == 0
    ][0]
    assert new_item["generated_text"] == " because 2"
    assert new_item["full_response_text"] == "prefix because 2"
    assert "generated_text" not in old_item
    progress = json.loads(progress_path(output_path).read_text())
    assert progress["status"] == "complete"
