from scripts.compare_legacy_full import paired_bootstrap
from ppo.prefix_eval import select_best_fixed_temperature


def test_paired_bootstrap_positive_difference():
    legacy = [{"predictions": [
        {"problem_id": "a", "majority_correct": 0},
        {"problem_id": "b", "majority_correct": 0},
    ]}]
    full = [{"predictions": [
        {"problem_id": "a", "majority_correct": 1},
        {"problem_id": "b", "majority_correct": 0},
    ]}]
    mean, low, high = paired_bootstrap(legacy, full, iterations=1000, seed=42)
    assert mean == 0.5
    assert low >= 0.0
    assert high <= 1.0


def test_best_fixed_temperature_is_selected_from_validation(tmp_path):
    val_path = tmp_path / "val.jsonl"
    val_path.write_text(
        "\n".join([
            '{"sample_id":"a_t0.1_v0","temperature":0.1,"voting_label":0}',
            '{"sample_id":"b_t0.1_v0","temperature":0.1,"voting_label":0}',
            '{"sample_id":"a_t0.3_v0","temperature":0.3,"voting_label":1}',
            '{"sample_id":"b_t0.3_v0","temperature":0.3,"voting_label":1}',
        ]) + "\n",
        encoding="utf-8",
    )
    assert select_best_fixed_temperature(str(val_path), [0.1, 0.3]) == 0.1
