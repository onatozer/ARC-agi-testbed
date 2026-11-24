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
    # In-train tasks contribute their test pair to training too (3 examples per copy, the test pair flagged as validation);
    # the held-out task only its test pair
    assert all(len(copy.examples) == 3 and copy.validation == [False, False, True] for group in results["train"]["all"] for copy in group)
    assert all(len(copy.examples) == 1 and copy.validation == [False] for group in results["test"]["all"] for copy in group)
    assert len(results["train"]["all"]) == 3 and len(results["test"]["all"]) == 1


def test_heldout_build_writes_the_validation_flag_and_the_loader_serves_it(tmp_path):
    from dataset.build_arc_dataset import convert_dataset
    from dataset.puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig

    _write_tasks(tmp_path)
    out = tmp_path / "out"
    convert_dataset(MiniARCDataProcessConfig(input_dir=str(tmp_path), output_dir=str(out), num_aug=2, grid_size=2, heldout_tasks=1, seed=0), load_puzzles=load_puzzles_miniarc)
    flags = np.load(out / "train" / "all__validation.npy")
    assert flags.shape == (3 * 3 * 3,) and flags.sum() == 3 * 3  # 3 tasks x 3 copies x 3 examples, one test pair per copy
    assert not (out / "test" / "all__validation.npy").exists()
    cfg = PuzzleDatasetConfig(seed=0, dataset_path=str(out), global_batch_size=9, test_set_mode=False, epochs_per_iter=1, rank=0, num_replicas=1)
    _, batch, _ = next(iter(PuzzleDataset(cfg, split="train")))
    assert batch["validation"].shape == (9,) and batch["validation"].sum() == 3


def test_default_build_keeps_every_task_in_both_splits(tmp_path):
    _write_tasks(tmp_path)
    config = MiniARCDataProcessConfig(input_dir=str(tmp_path), output_dir=str(tmp_path / "out"), num_aug=2, grid_size=2)
    np.random.seed(0)
    results, test_puzzles = load_puzzles_miniarc(config)
    assert len(test_puzzles) == 4 and len(results["train"]["all"]) == 4 and len(results["test"]["all"]) == 4
    assert not (tmp_path / "out" / "heldout_tasks.json").exists()


def test_arc_heldout_test_set_keeps_its_demos_out_of_training(tmp_path):
    from dataset.build_arc_dataset import DataProcessConfig, load_puzzles_arcagi

    for subset in ("training", "evaluation"):
        tasks = {f"{subset}{k}": _task(k) for k in (1, 2)}
        (tmp_path / f"arc_{subset}_challenges.json").write_text(json.dumps({n: {**t, "test": [{"input": p["input"]} for p in t["test"]]} for n, t in tasks.items()}))
        (tmp_path / f"arc_{subset}_solutions.json").write_text(json.dumps({n: [p["output"] for p in t["test"]] for n, t in tasks.items()}))

    for heldout in (False, True):
        config = DataProcessConfig(input_file_prefix=str(tmp_path / "arc"), output_dir=str(tmp_path / "out"), subsets=["training", "evaluation"],
                                   test_set_name="evaluation", num_aug=0, heldout_test_set=heldout)
        np.random.seed(0)
        results, test_puzzles = load_puzzles_arcagi(config)
        train_ids = {copy.id for group in results["train"]["all"] for copy in group}
        assert set(test_puzzles) == {"evaluation1", "evaluation2"}
        assert ({"evaluation1", "evaluation2"} <= train_ids) != heldout  # the eval demos train only without the held-out protocol
        assert all(copy.validation == [False, False, True] for group in results["train"]["all"] for copy in group if copy.id.startswith("training"))
