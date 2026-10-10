import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from dataset.build_nqueens_dataset import convert_subset
from evaluators.nqueens import NQueens

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the evaluator reduces its stats on CUDA")

N = 4
# The two 4-queens solutions, as column per row
SOLUTION_A = [1, 3, 0, 2]
SOLUTION_B = [2, 0, 3, 1]


def _board(cols):
    board = np.zeros((N, N), dtype=np.int64)
    for row, col in enumerate(cols):
        if col is not None:
            board[row, col] = 1
    return board.reshape(-1) + 1  # dataset encoding: 1 = empty, 2 = queen


def test_counts_real_puzzles_and_checks_the_given_queens():
    evaluator = NQueens(data_path="", eval_metadata=SimpleNamespace(blank_identifier_id=0), board_size=N)
    evaluator.begin_eval()
    # Row 0: matches its label. Row 1: the other valid solution, which keeps its given queen.
    # Row 2: valid, but drops the queen given in its input. Row 3: batch padding (blank identifier).
    inputs = np.stack([_board([1, None, None, None]), _board([2, None, None, None]), _board([1, None, None, None]), _board([None] * N)])
    labels = np.stack([_board(SOLUTION_A), _board(SOLUTION_A), _board(SOLUTION_A), _board([None] * N)])
    preds = np.stack([_board(SOLUTION_A), _board(SOLUTION_B), _board(SOLUTION_B), _board(SOLUTION_A)])
    evaluator.update_batch(
        {"inputs": torch.tensor(inputs), "labels": torch.tensor(labels), "puzzle_identifiers": torch.tensor([1, 1, 1, 0])},
        {"preds": torch.tensor(preds)},
    )
    results = evaluator.result(None, rank=0, world_size=1)

    assert results["NQueens/valid_solution_rate"] == 1.0  # the padding row is not counted
    assert results["NQueens/accuracy"] == pytest.approx(2 / 3)
    assert results["NQueens/exact_match_accuracy"] == pytest.approx(1 / 3)


def test_builder_does_not_give_real_puzzles_the_blank_identifier(tmp_path):
    config = OmegaConf.create({
        "board_size": 6, "source_repo": None, "num_train": 5, "num_test": 5, "num_aug": 2,
        "num_given_queens": None, "seed": 0, "output_dir": str(tmp_path),
    })
    convert_subset("test", config)
    metadata = json.loads((tmp_path / "test" / "dataset.json").read_text())
    identifiers = np.load(tmp_path / "test" / "all__puzzle_identifiers.npy")
    assert metadata["blank_identifier_id"] == 0 and metadata["num_puzzle_identifiers"] == 2
    assert (identifiers != metadata["blank_identifier_id"]).all()
    assert json.loads((tmp_path / "identifiers.json").read_text()) == ["<blank>", "nqueens"]
