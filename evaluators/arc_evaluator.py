"""ARC-AGI evaluator with voting and pass@K metrics.

Ported from TinyRecursiveModels/evaluators/arc.py. Collects per-batch
predictions across distributed workers, de-augments them, votes, and
computes pass@K accuracy against the ground-truth test puzzles.

Usage from the training loop::

    evaluator = ARCEvaluator(data_path, eval_metadata)
    evaluator.begin_eval()
    for batch in eval_loader:
        preds, q_halt = run_model(batch)
        evaluator.update_batch(batch, preds, q_halt)
    results = evaluator.result(save_path, rank, world)
"""
from typing import Dict, Optional, Sequence
import hashlib
import json
import os

import numpy as np
import torch
import torch.distributed as dist

from ..data.common import dihedral_transform, inverse_dihedral_transform, PuzzleDatasetMetadata

ARC_GRID_SIZE = 30
PuzzleIdSeparator = "|||"


def _arc_grid_to_np(grid):
    arr = np.array(grid, dtype=np.uint8)
    assert arr.ndim == 2
    return arr


def _grid_hash(grid: np.ndarray) -> str:
    assert grid.ndim == 2 and grid.dtype == np.uint8
    buf = [x.to_bytes(1, byteorder="big") for x in grid.shape]
    buf.append(grid.tobytes())
    return hashlib.sha256(b"".join(buf)).hexdigest()


def _crop(grid_flat: np.ndarray) -> np.ndarray:
    """Find the content rectangle (tokens >= 2) in a 30x30 flat grid."""
    grid = grid_flat.reshape(ARC_GRID_SIZE, ARC_GRID_SIZE)
    nr, nc = grid.shape
    max_area, best = 0, (0, 0)
    num_c = nc
    for num_r in range(1, nr + 1):
        for c in range(1, num_c + 1):
            x = grid[num_r - 1, c - 1]
            if x < 2 or x > 11:
                num_c = c - 1
                break
        area = num_r * num_c
        if area > max_area:
            max_area = area
            best = (num_r, num_c)
    return (grid[: best[0], : best[1]] - 2).astype(np.uint8)


def _inverse_aug(name: str):
    if PuzzleIdSeparator not in name:
        return name, lambda x: x
    trans_id_str, perm_str = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id_str[1:])
    inv_perm = np.argsort([int(c) for c in perm_str]).astype(np.uint8)

    def _map(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map


class ARCEvaluator:
    required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}

    def __init__(
        self,
        data_path: str,
        eval_metadata: PuzzleDatasetMetadata,
        submission_K: int = 2,
        pass_Ks: Sequence[int] = (1, 2, 5, 10, 100, 1000),
        aggregated_voting: bool = True,
    ):
        self.pass_Ks = list(pass_Ks)
        self.submission_K = submission_K
        self.aggregated_voting = aggregated_voting
        self.blank_identifier_id = eval_metadata.blank_identifier_id

        with open(os.path.join(data_path, "identifiers.json"), "r") as f:
            self.identifier_map = json.load(f)
        with open(os.path.join(data_path, "test_puzzles.json"), "r") as f:
            self.test_puzzles = json.load(f)

        self._local_hmap: Dict[str, np.ndarray] = {}
        self._local_preds: dict = {}

    def begin_eval(self):
        if not self.aggregated_voting:
            self._local_hmap = {}
            self._local_preds = {}

    def update_batch(
        self,
        batch: Dict[str, torch.Tensor],
        preds: Dict[str, torch.Tensor],
    ):
        outputs: dict = {}
        q_values = None
        for collection in (batch, preds):
            for k, v in collection.items():
                if k in self.required_outputs:
                    if k == "q_halt_logits":
                        q_values = v.to(torch.float64).sigmoid().cpu()
                    else:
                        outputs[k] = v.cpu()
        assert q_values is not None

        mask = outputs["puzzle_identifiers"] != self.blank_identifier_id
        outputs = {k: v[mask] for k, v in outputs.items()}

        for identifier, inp, pred, q in zip(
            outputs["puzzle_identifiers"].numpy(),
            outputs["inputs"].numpy(),
            outputs["preds"].numpy(),
            q_values.numpy(),
        ):
            name = self.identifier_map[int(identifier)]
            orig_name, inv_fn = _inverse_aug(name)
            input_hash = _grid_hash(inv_fn(_crop(inp)))
            pred_grid = inv_fn(_crop(pred))
            assert np.all((pred_grid >= 0) & (pred_grid <= 9))
            pred_hash = _grid_hash(pred_grid)

            self._local_hmap[pred_hash] = pred_grid
            self._local_preds.setdefault(orig_name, {})
            self._local_preds[orig_name].setdefault(input_hash, [])
            self._local_preds[orig_name][input_hash].append((pred_hash, float(q)))

    def result(
        self,
        save_path: Optional[str],
        rank: int,
        world_size: int,
        group=None,
    ) -> Optional[Dict[str, float]]:
        global_hmap_preds = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(
            (self._local_hmap, self._local_preds), global_hmap_preds, dst=0, group=group
        )

        if rank != 0:
            return None

        submission = {}
        correct = [0.0 for _ in range(len(self.pass_Ks))]

        for name, puzzle in self.test_puzzles.items():
            submission[name] = []
            num_test_correct = [0 for _ in range(len(self.pass_Ks))]
            for pair in puzzle["test"]:
                input_hash = _grid_hash(_arc_grid_to_np(pair["input"]))
                label_hash = _grid_hash(_arc_grid_to_np(pair["output"]))

                p_map: dict = {}
                for hmap, preds in global_hmap_preds:
                    for h, q in preds.get(name, {}).get(input_hash, []):
                        p_map.setdefault(h, [0, 0])
                        p_map[h][0] += 1
                        p_map[h][1] += q

                if not p_map:
                    continue

                for h, stats in p_map.items():
                    stats[1] /= stats[0]
                p_map_sorted = sorted(p_map.items(), key=lambda kv: kv[1], reverse=True)

                for i, k in enumerate(self.pass_Ks):
                    ok = any(h == label_hash for h, _ in p_map_sorted[:k])
                    num_test_correct[i] += ok

                pred_grids = []
                for h, _ in p_map_sorted[: self.submission_K]:
                    for hmap, _ in global_hmap_preds:
                        if h in hmap:
                            pred_grids.append(hmap[h])
                            break
                while len(pred_grids) < self.submission_K:
                    pred_grids.append(pred_grids[0] if pred_grids else np.zeros((1, 1), dtype=np.uint8))
                submission[name].append(
                    {f"attempt_{i+1}": g.tolist() for i, g in enumerate(pred_grids)}
                )

            for i in range(len(self.pass_Ks)):
                correct[i] += num_test_correct[i] / max(len(puzzle["test"]), 1)

        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, "submission.json"), "w") as f:
                json.dump(submission, f)

        n_puzzles = max(len(self.test_puzzles), 1)
        return {f"ARC/pass@{k}": correct[i] / n_puzzles for i, k in enumerate(self.pass_Ks)}
