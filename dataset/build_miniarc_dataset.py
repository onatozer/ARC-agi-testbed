import os
import json
import glob

import numpy as np
from argdantic import ArgParser
from pydantic import BaseModel

from dataset.build_arc_dataset import convert_single_arc_puzzle, convert_dataset


cli = ArgParser()


class MiniARCDataProcessConfig(BaseModel):
    input_dir: str = "data/MiniARC"
    output_dir: str = "data/miniarc-aug-1000"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1 # start > 1 to handle multiple datasets
    max_grid_size: int = 30 # MiniARC grids are all 5x5; 30 keeps seq_len identical to ARC
    scale_aug: bool = False # random integer upscaling of train examples (arXiv:2511.14761)


def load_puzzles_miniarc(config: MiniARCDataProcessConfig):
    # MiniARC has one task per file, with test outputs included.
    # Mirror ARC's evaluation set: demo pairs -> train, test pairs -> test.
    dest_mapping = {"train": ("train", "all"), "test": ("test", "all")}

    puzzles = []
    for filename in sorted(glob.glob(os.path.join(config.input_dir, "*.json"))):
        with open(filename, "r") as f:
            puzzles.append((os.path.splitext(os.path.basename(filename))[0], json.load(f)))

    # Shuffle puzzles
    np.random.shuffle(puzzles)

    test_puzzles = {}
    results = {}
    for name, puzzle in puzzles:
        test_puzzles[name] = puzzle
        convert_single_arc_puzzle(results, name, puzzle, config.num_aug, dest_mapping)

    print (f"Total puzzles: {len(puzzles)}")
    return results, test_puzzles


@cli.command(singleton=True)
def main(config: MiniARCDataProcessConfig):
    convert_dataset(config, load_puzzles=load_puzzles_miniarc)


if __name__ == "__main__":
    cli()
