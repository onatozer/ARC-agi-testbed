from typing import List
import os
import json
import glob

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict

from .build_arc_dataset import convert_single_arc_puzzle, convert_dataset


class SmallARCDataProcessConfig(BaseModel):
    model_config = ConfigDict(extra='allow')

    input_file_prefix: str = "kaggle/combined/arc-agi"
    miniarc_dir: str = "data/MiniARC"
    train_subsets: List[str] = ["training", "concept"] # all pairs -> train
    test_subsets: List[str] = ["evaluation"] # demo pairs -> train, test pairs -> test
    max_grid: int = 10 # keep only puzzles whose grids all fit in max_grid x max_grid
    output_dir: str = "data/smallarc-aug-1000"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1 # start > 1 to handle multiple datasets
    grid_size: int = 10
    scale_aug: bool = False # random integer upscaling of train examples (arXiv:2511.14761)
    test_aug: bool = True # False: the test split keeps only each puzzle's original test pairs; augmented copies are for training only
    border_tokens: bool = False # ViTARC border tokens + supervised 2D pad (arXiv 2410.06405)
    object_ids: bool = False # also save a per-cell object index of each input (ViTARC OPE, arXiv 2410.06405)


# Source prefixes for puzzle names, used by evaluators.small_arc to report per-source metrics
SourceSeparator = "/"
SourceNames = {"training": "arc1_training", "evaluation": "arc1_eval", "concept": "concept", "miniarc": "miniarc"}


def _fits(puzzle: dict, max_grid: int):
    return all(len(grid) <= max_grid and all(len(row) <= max_grid for row in grid)
               for examples in puzzle.values() for example in examples for grid in (example["input"], example["output"]))


def _load_kaggle_subset(config: SmallARCDataProcessConfig, subset_name: str):
    with open(f"{config.input_file_prefix}_{subset_name}_challenges.json", "r") as f:
        puzzles = json.load(f)
    # Solutions are needed both to filter by output size and to score the test pairs
    with open(f"{config.input_file_prefix}_{subset_name}_solutions.json", "r") as f:
        sols = json.load(f)

    for puzzle_id, puzzle in puzzles.items():
        for idx, sol_grid in enumerate(sols[puzzle_id]):
            puzzle["test"][idx]["output"] = sol_grid
    return list(puzzles.items())


def _load_miniarc(config: SmallARCDataProcessConfig):
    puzzles = []
    for filename in sorted(glob.glob(os.path.join(config.miniarc_dir, "*.json"))):
        with open(filename, "r") as f:
            puzzles.append((os.path.splitext(os.path.basename(filename))[0], json.load(f)))
    return puzzles


def load_puzzles_smallarc(config: SmallARCDataProcessConfig):
    # MiniARC + ARC-AGI-1 (training, evaluation) + ConceptARC, restricted to puzzles whose grids are all <= max_grid.
    # Mirrors the ARC / MiniARC builders: train subsets put every pair in train;
    # test subsets (ARC eval, MiniARC) put demo pairs in train and test pairs in test.
    train_mapping = {"train": ("train", "all"), "test": ("train", "all")}
    test_mapping = {"train": ("train", "all"), "test": ("test", "all")}

    sources = [(name, _load_kaggle_subset(config, name), train_mapping) for name in config.train_subsets]
    sources += [(name, _load_kaggle_subset(config, name), test_mapping) for name in config.test_subsets]
    sources.append(("miniarc", _load_miniarc(config), test_mapping))

    test_puzzles = {}
    results = {}
    total_puzzles = 0
    for subset_name, puzzles, dest_mapping in sources:
        kept = [(f"{SourceNames[subset_name]}{SourceSeparator}{name}", puzzle) for name, puzzle in puzzles if _fits(puzzle, config.max_grid)]
        print (f"[{subset_name}] kept {len(kept)} / {len(puzzles)} puzzles")

        # Shuffle puzzles
        np.random.shuffle(kept)

        for name, puzzle in kept:
            if dest_mapping["test"][0] == "test":
                test_puzzles[name] = puzzle
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, dest_mapping, test_aug=config.test_aug)
            total_puzzles += 1

    print (f"Total puzzles: {total_puzzles}, scored: {len(test_puzzles)}")
    return results, test_puzzles


@hydra.main(config_path="../config/dataset", config_name="smallarc", version_base=None)
def launch(config: DictConfig):
    """
    Run:
        python -m dataset.build_smallarc_dataset
    Override on the CLI, e.g.:
        python -m dataset.build_smallarc_dataset num_aug=100 data_path=data/smallarc-aug-100
    """
    if config.seed is None:
        config.seed = int(np.random.default_rng().integers(low=0, high=1e8))

    convert_dataset(SmallARCDataProcessConfig(**OmegaConf.to_container(config, resolve=True)), load_puzzles=load_puzzles_smallarc)


if __name__ == "__main__":
    launch()
