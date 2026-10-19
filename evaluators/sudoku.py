"""
Sudoku evaluator: scores a prediction as a valid solution when it is a complete grid satisfying every row, column and
box constraint and keeps all of the puzzle's clues. Puzzles with few clues can have several solutions, so this counts
any of them, not only the labelled one.

Metrics computed:
- Exact match accuracy: prediction == label
- Valid solution rate: prediction is a valid solution of the puzzle (consistent with its clues)
- Unit satisfaction: fraction of rows, columns and boxes that hold each digit exactly once
"""

import math
from typing import Dict, Optional
import numpy as np
import torch
import torch.distributed as dist

from dataset.build_sudoku_dataset import get_box_dims
from dataset.common import PuzzleDatasetMetadata
from models.losses import IGNORE_LABEL_ID


class Sudoku:
    required_outputs = {"preds"}

    def __init__(self, data_path: str, eval_metadata: PuzzleDatasetMetadata, grid_size: Optional[int] = None):
        """grid_size: None derives it from the dataset's seq_len = grid_size ** 2."""
        super().__init__()
        self.grid_size = grid_size if grid_size is not None else math.isqrt(eval_metadata.seq_len)
        self.box_rows, self.box_cols = get_box_dims(self.grid_size)
        self.begin_eval()

    def begin_eval(self):
        self._local_stats = {"total": 0, "exact_match": 0, "valid_solutions": 0, "units_satisfied": 0}

    def _units_satisfied(self, grid: np.ndarray) -> int:
        """Number of the grid's 3N rows, columns and boxes that hold each digit 1..N exactly once."""
        N, digits = self.grid_size, set(range(1, self.grid_size + 1))
        units = [grid[i] for i in range(N)] + [grid[:, i] for i in range(N)]
        units += [grid[r:r + self.box_rows, c:c + self.box_cols].flatten()
                  for r in range(0, N, self.box_rows) for c in range(0, N, self.box_cols)]
        return sum(set(u.tolist()) == digits for u in units)

    def valid_rows(self, inputs: torch.Tensor, preds: torch.Tensor) -> torch.Tensor:
        """[B] bool: each prediction is a valid solution of its puzzle (every unit a permutation, every clue kept), for
        the N-copy vote / best-of-N of evaluators.eval_fn on puzzles with several solutions."""
        N = self.grid_size
        out = []
        for inp, pred in zip(inputs.cpu().numpy() - 1, preds.cpu().numpy() - 1):
            out.append(self._units_satisfied(pred.reshape(N, N)) == 3 * N and not np.any((inp != 0) & (pred != inp)))
        return torch.tensor(out, dtype=torch.bool, device=preds.device)

    def update_batch(self, batch: Dict[str, torch.Tensor], preds: Dict[str, torch.Tensor]):
        labels = batch["labels"].cpu().numpy()
        # Batch padding rows have labels that are all IGNORE_LABEL_ID
        mask = (labels != IGNORE_LABEL_ID).any(-1)
        # Tokens: 0 PAD, 1 blank, d + 1 for digit d
        predictions = preds["preds"].cpu().numpy()[mask] - 1
        inputs = batch["inputs"].cpu().numpy()[mask] - 1
        labels = labels[mask] - 1

        N = self.grid_size
        for pred, inp, label in zip(predictions, inputs, labels):
            self._local_stats["total"] += 1
            self._local_stats["exact_match"] += int(np.array_equal(pred, label))
            units = self._units_satisfied(pred.reshape(N, N))
            keeps_clues = not np.any((inp != 0) & (pred != inp))
            self._local_stats["units_satisfied"] += units
            self._local_stats["valid_solutions"] += int(units == 3 * N and keeps_clues)

    def result(self, save_path: Optional[str], rank: int, world_size: int,
               group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        stats = torch.tensor([self._local_stats[k] for k in ("total", "exact_match", "valid_solutions", "units_satisfied")],
                             dtype=torch.float32, device="cuda")
        if world_size > 1:
            dist.reduce(stats, dst=0, group=group)
        if rank != 0:
            return None

        total = max(int(stats[0].item()), 1)
        return {
            "Sudoku/exact_match_accuracy": stats[1].item() / total,
            "Sudoku/valid_solution_rate": stats[2].item() / total,
            "Sudoku/unit_satisfaction": stats[3].item() / (total * 3 * self.grid_size),
        }
