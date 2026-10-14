from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dataset.build_sudoku_dataset import convert_subset
from evaluators.sudoku import Sudoku
from models.losses import IGNORE_LABEL_ID

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the evaluator reduces its stats on CUDA")

# Two different valid 4x4 grids that both keep the clue grid CLUES
GRID_A = [1, 2, 3, 4, 3, 4, 1, 2, 2, 1, 4, 3, 4, 3, 2, 1]
GRID_B = [1, 2, 3, 4, 3, 4, 1, 2, 2, 3, 4, 1, 4, 1, 2, 3]
CLUES = [1, 0, 0, 0, 0, 0, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0]


def _tokens(grid):
    return np.array(grid) + 1  # dataset encoding: 1 = blank, d + 1 = digit d


def test_counts_any_valid_solution_that_keeps_the_clues():
    evaluator = Sudoku(data_path="", eval_metadata=SimpleNamespace(seq_len=16))
    evaluator.begin_eval()
    broken = list(GRID_A)
    broken[0], broken[1] = broken[1], broken[0]  # rows still permutations, columns / boxes not
    clue_dropped = [2, 1, 4, 3, 4, 3, 2, 1, 1, 2, 3, 4, 3, 4, 1, 2]  # a valid grid that ignores the clues
    # Rows: the label; another valid solution of the same puzzle; an invalid grid; a valid grid that breaks the clues
    inputs = np.stack([_tokens(CLUES)] * 4 + [np.zeros(16, dtype=np.int64)])
    labels = np.stack([_tokens(GRID_A)] * 4 + [np.full(16, IGNORE_LABEL_ID)])  # last row: batch padding
    preds = np.stack([_tokens(GRID_A), _tokens(GRID_B), _tokens(broken), _tokens(clue_dropped), _tokens(GRID_A)])
    evaluator.update_batch(
        {"inputs": torch.tensor(inputs), "labels": torch.tensor(labels), "puzzle_identifiers": torch.zeros(5, dtype=torch.int32)},
        {"preds": torch.tensor(preds)},
    )
    results = evaluator.result(None, rank=0, world_size=1)

    assert results["Sudoku/valid_solution_rate"] == pytest.approx(2 / 4)  # the padding row is not counted
    assert results["Sudoku/exact_match_accuracy"] == pytest.approx(1 / 4)


def test_generated_test_split_is_held_out_from_train(tmp_path):
    config = OmegaConf.create({
        "grid_size": 4, "source_repo": None, "num_train": 200, "num_test": 50, "num_aug": 2, "clue_ratio": 0.3,
        "seed": 0, "output_dir": str(tmp_path),
    })
    train_inputs = convert_subset("train", config)
    convert_subset("test", config, exclude=train_inputs)
    train = {bytes(r) for r in np.load(tmp_path / "train" / "all__inputs.npy")}
    test = np.load(tmp_path / "test" / "all__inputs.npy")
    assert len(test) == 50 and not any(bytes(r) in train for r in test)
