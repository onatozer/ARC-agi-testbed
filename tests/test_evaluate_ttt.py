import json
from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from evaluate_ttt import aggregate, build_task_dataset, load_tasks
from evaluators.eval_config import TTTConfig

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")

TASK = {
    "train": [
        {"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]},
        {"input": [[5, 0, 6]], "output": [[6, 0, 5]]},
    ],
    "test": [{"input": [[7, 8], [0, 9]], "output": [[9, 0], [8, 7]]}],
}

# The ARC build config a checkpoint carries in config.dataset (runtime keys are ignored by the builder)
BUILDER = dict(
    name="Arc-AGI", data_path="unused", input_file_prefix="unused", subsets=["training", "evaluation"], test_set_name="evaluation",
    grid_size=30, scale_aug=True, border_tokens=False, object_ids=False, num_aug=1000, test_aug=False,
)


def _ttt_cfg(*overrides: str) -> TTTConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="cfg_eval_ttt", overrides=list(overrides))
    return TTTConfig(**OmegaConf.to_container(cfg, resolve=True))


def test_default_recipe_is_varc_and_cli_overrides_reach_it():
    cfg = _ttt_cfg("checkpoint=ckpt.pth", "epochs=3", "train_overrides.lr=1e-3")
    assert cfg.checkpoint == "ckpt.pth" and cfg.suffix == "ttt_varc"
    assert (cfg.num_aux_tasks, cfg.epochs, cfg.warmup_epochs, cfg.ttt_runs_per_task, cfg.num_views) == (51, 3, 10, 2, 10)
    assert cfg.train_overrides["lr"] == 1e-3 and cfg.train_overrides["global_batch_size"] == 8


def test_task_dataset_gives_each_aux_task_its_own_group_and_identifier(tmp_path):
    build_task_dataset(BUILDER, "task", TASK, str(tmp_path), num_aux_tasks=6, seed=0)

    def split(name):
        meta = json.loads((tmp_path / name / "dataset.json").read_text())
        ids = np.load(tmp_path / name / "all__puzzle_identifiers.npy")
        return meta, ids

    train_meta, train_ids = split("train")
    test_meta, test_ids = split("test")
    # One group per aux task, so a sampler epoch covers every aux task's 2 demos
    assert train_meta["total_groups"] == 6 and train_meta["mean_puzzle_examples"] == 2
    # The test split holds the test pair under every aux task, with the identifiers trained on
    assert test_meta["total_groups"] == 1 and sorted(test_ids) == sorted(train_ids) and len(set(train_ids)) == 6
    names = json.loads((tmp_path / "identifiers.json").read_text())
    assert names[0] == "<blank>" and "task" in names and len(names) == 7
    assert json.loads((tmp_path / "test_puzzles.json").read_text()) == {"task": TASK}


MINIARC_BUILDER = dict(name="Mini-ARC", data_path="unused", input_dir="unused", grid_size=5, scale_aug=False, border_tokens=False, object_ids=False, num_aug=31, test_aug=True)


def test_task_dataset_accepts_a_miniarc_build(tmp_path):
    build_task_dataset(MINIARC_BUILDER, "task", TASK, str(tmp_path), num_aux_tasks=4, seed=0)
    train_meta = json.loads((tmp_path / "train" / "dataset.json").read_text())
    assert train_meta["total_groups"] == 4 and train_meta["seq_len"] == 25
    assert json.loads((tmp_path / "test_puzzles.json").read_text()) == {"task": TASK}


def test_load_tasks_reads_a_builds_test_puzzles_without_arc_agi_files(tmp_path):
    (tmp_path / "test_puzzles.json").write_text(json.dumps({"a": TASK, "b": TASK}))
    assert load_tasks(MINIARC_BUILDER, str(tmp_path)) == {"a": TASK, "b": TASK}


def test_task_ids_file_is_a_cli_option():
    cfg = _ttt_cfg("checkpoint=ckpt.pth", "task_ids_file=heldout_tasks.json", "train_overrides.freeze_weights=true")
    assert cfg.task_ids_file == "heldout_tasks.json" and cfg.train_overrides["freeze_weights"] is True


def test_aggregate_waits_for_every_task_then_averages_and_merges(tmp_path):
    (tmp_path / "tasks").mkdir()
    for name, score in (("a", 1.0), ("b", 0.0)):
        (tmp_path / "tasks" / f"{name}.json").write_text(json.dumps({"metrics": {"ARC/pass@2": score}}))
        sub = tmp_path / "tasks" / name / "evaluator_ARC"
        sub.mkdir(parents=True)
        (sub / "submission.json").write_text(json.dumps({name: [{"attempt_1": [[score]]}]}))

    assert aggregate(["a", "b", "c"], tmp_path) is None
    assert aggregate(["a", "b"], tmp_path) == {"ARC/pass@2": 0.5}
    assert set(json.loads((tmp_path / "submission.json").read_text())) == {"a", "b"}
