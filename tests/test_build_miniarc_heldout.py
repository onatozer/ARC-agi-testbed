"""build_miniarc_dataset.py's held-out protocol: heldout_tasks are never in training, only their test pairs are evaluated."""
import json

import numpy as np

from dataset.build_miniarc_dataset import MiniARCDataProcessConfig, load_puzzles_miniarc


def _task(k: int) -> dict:
    grid = [[k, 0], [0, k]]
    return {"train": [{"input": grid, "output": grid[::-1]}, {"input": grid[::-1], "output": grid}], "test": [{"input": grid, "output": grid[::-1]}]}


def _write_tasks(tmp_path, n=4):
    for k in range(1, n + 1):
        (tmp_path / f"task{k}.json").write_text(json.dumps(_task(k)))


def test_heldout_tasks_leave_training_and_form_the_test_split(tmp_path):
    _write_tasks(tmp_path)
    out = tmp_path / "out"
    config = MiniARCDataProcessConfig(input_dir=str(tmp_path), output_dir=str(out), num_aug=2, grid_size=2, heldout_tasks=1, heldout_seed=0)
    np.random.seed(0)
    results, test_puzzles = load_puzzles_miniarc(config)

    heldout = json.loads((out / "heldout_tasks.json").read_text())
    assert len(heldout) == 1 and set(test_puzzles) == set(heldout)
    train_ids = {copy.id.split("|")[0] for group in results["train"]["all"] for copy in group}
    test_ids = {copy.id.split("|")[0] for group in results["test"]["all"] for copy in group}
    assert heldout[0] not in train_ids and test_ids == set(heldout)
    # In-train tasks contribute their test pair to training too, as the train split's "validation" set, under the same
    # puzzle ids (augmented copies included) as their demos in "all"; the held-out task contributes only its test pair
    assert all(len(copy.examples) == 2 for group in results["train"]["all"] for copy in group)
    assert all(len(copy.examples) == 1 for group in results["train"]["validation"] for copy in group)
    assert all(len(copy.examples) == 1 for group in results["test"]["all"] for copy in group)
    assert [[c.id for c in g] for g in results["train"]["validation"]] == [[c.id for c in g] for g in results["train"]["all"]]
    assert len(results["train"]["all"]) == 3 and len(results["test"]["all"]) == 1


def test_default_build_keeps_every_task_in_both_splits(tmp_path):
    _write_tasks(tmp_path)
    config = MiniARCDataProcessConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"), num_aug=2, grid_size=2)
    np.random.seed(0)
    results, test_puzzles = load_puzzles_miniarc(config)
    assert len(test_puzzles) == 4 and len(results["train"]["all"]) == 4 and len(results["test"]["all"]) == 4
    assert not (tmp_path / "out" / "heldout_tasks.json").exists()
