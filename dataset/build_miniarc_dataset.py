import os
import json
import glob

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from .build_arc_dataset import convert_single_arc_puzzle, convert_dataset


class MiniARCDataProcessConfig(BaseModel):
    input_dir: str = "data/MiniARC"
    output_dir: str = "data/miniarc-aug-1000"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1 # start > 1 to handle multiple datasets
    grid_size: int = 10
    scale_aug: bool = False # random integer upscaling of train examples (arXiv:2511.14761)
    test_aug: bool = True # False: the test split keeps only each puzzle's original test pairs; augmented copies are for training only
    border_tokens: bool = False # ViTARC border tokens + supervised 2D pad (arXiv 2410.06405)
    object_ids: bool = False # also save a per-cell object index of each input (ViTARC OPE, arXiv 2410.06405)
    # Held-out protocol: this many tasks (drawn with heldout_seed over the sorted names) are left out of training entirely;
    # only their test pairs form the test split, and the other tasks' test pairs join their demos in training, flagged in
    # the per-row all__validation.npy (batch["validation"]) so training can treat them differently if it wants. 0 = every
    # task's demos in training, its test pairs in the test split (the TRM-on-ARC protocol)
    heldout_tasks: int = 0
    heldout_seed: int = 0
    heldout_demos_in_train: bool = False # transductive held-out protocol: the held-out tasks' demos also join training (their test pairs stay the test split)
    val_tasks: int = 0 # this many more tasks held out the same way, listed in val_tasks.json (checkpoint selection); the heldout_tasks draw is unchanged


def load_puzzles_miniarc(config: MiniARCDataProcessConfig):
    # MiniARC has one task per file, with test outputs included.
    # Mirror ARC's evaluation set: demo pairs -> train, test pairs -> test.
    puzzles = []
    for filename in sorted(glob.glob(os.path.join(config.input_dir, "*.json"))):
        with open(filename, "r") as f:
            puzzles.append((os.path.splitext(os.path.basename(filename))[0], json.load(f)))

    names = sorted(name for name, _ in puzzles)
    heldout = set()
    if config.heldout_tasks > 0:
        heldout = set(np.random.default_rng(config.heldout_seed).choice(names, size=config.heldout_tasks, replace=False).tolist())
    assert config.val_tasks == 0 or config.heldout_tasks > 0, "val_tasks needs the held-out protocol (heldout_tasks > 0)"
    val = set()
    if config.val_tasks > 0:
        rest = [n for n in names if n not in heldout]
        val = set(np.random.default_rng(config.heldout_seed + 1).choice(rest, size=config.val_tasks, replace=False).tolist())

    # Shuffle puzzles
    np.random.shuffle(puzzles)

    test_puzzles = {}  # the tasks the evaluators (and evaluate_ttt.py) see: demos plus test pairs with outputs
    results = {}
    for name, puzzle in puzzles:
        if config.heldout_tasks == 0:
            test_puzzles[name] = puzzle
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, {"train": ("train", "all"), "test": ("test", "all")}, test_aug=config.test_aug)
        elif name in heldout or name in val:
            test_puzzles[name] = puzzle
            splits = {"train": ("train", "all"), "test": ("test", "all")} if config.heldout_demos_in_train else {"test": ("test", "all")}
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, splits, test_aug=config.test_aug)
        else:
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, {"train": ("train", "all"), "test": ("train", "all")}, test_aug=config.test_aug)

    if config.heldout_tasks > 0:
        os.makedirs(config.output_dir, exist_ok=True)
        with open(os.path.join(config.output_dir, "heldout_tasks.json"), "w") as f:
            json.dump(sorted(heldout), f, indent=2)
    if val:
        with open(os.path.join(config.output_dir, "val_tasks.json"), "w") as f:
            json.dump(sorted(val), f, indent=2)
    print (f"Total puzzles: {len(puzzles)}, held out: {len(heldout)}")
    return results, test_puzzles


@hydra.main(config_path="../config/dataset", config_name="miniarc", version_base=None)
def launch(config: DictConfig):
    """
    Run:
        python -m dataset.build_miniarc_dataset
    Override on the CLI, e.g.:
        python -m dataset.build_miniarc_dataset grid_size=5 num_aug=31 data_path=data/miniarc5-aug-31
        python -m dataset.build_miniarc_dataset grid_size=5 num_aug=31 heldout_tasks=30 data_path=data/miniarc5-aug-31-ho30
    """
    if config.seed is None:
        config.seed = int(np.random.default_rng().integers(low=0, high=1e8))

    convert_dataset(MiniARCDataProcessConfig(**OmegaConf.to_container(config, resolve=True)), load_puzzles=load_puzzles_miniarc)


if __name__ == "__main__":
    launch()
