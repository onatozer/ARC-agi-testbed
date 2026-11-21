"""
Sudoku Dataset Builder for Tiny Recursive Models (TRM)

Builds a Sudoku puzzle dataset compatible with the TRM framework.

- grid_size == 9: puzzles are pulled from a HuggingFace source
  (``sapientinc/sudoku-extreme`` by default), with optional difficulty
  filtering and subsampling.
- grid_size in {4, 6}: puzzles are generated locally via backtracking.

Configuration is handled with Hydra (see config/dataset/build_sudoku.yaml).

Run:
    python dataset/build_sudoku_dataset.py
Override on the CLI, e.g.:
    python dataset/build_sudoku_dataset.py grid_size=4 num_train=5000 \\
        output_dir=data/sudoku-4x4
"""

from typing import Optional, Tuple, List
import os
import csv
import json
import numpy as np

from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import hydra
from omegaconf import DictConfig

from .common import PuzzleDatasetMetadata


# (box_rows, box_cols) for each supported grid size
BOX_DIMS: dict[int, Tuple[int, int]] = {
    4: (2, 2),
    6: (2, 3),
    9: (3, 3),
}


def get_box_dims(grid_size: int) -> Tuple[int, int]:
    if grid_size not in BOX_DIMS:
        raise ValueError(f"Unsupported grid_size {grid_size}. Supported: {sorted(BOX_DIMS.keys())}")
    return BOX_DIMS[grid_size]


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray, grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
    box_rows, box_cols = get_box_dims(grid_size)
    num_bands = grid_size // box_rows
    num_stacks = grid_size // box_cols

    digit_map = np.pad(np.random.permutation(np.arange(1, grid_size + 1)), (1, 0))

    # Transpose only valid when box is square (preserves box constraints)
    transpose_flag = (box_rows == box_cols) and (np.random.rand() < 0.5)

    bands = np.random.permutation(num_bands)
    row_perm = np.concatenate([b * box_rows + np.random.permutation(box_rows) for b in bands])

    stacks = np.random.permutation(num_stacks)
    col_perm = np.concatenate([s * box_cols + np.random.permutation(box_cols) for s in stacks])

    n2 = grid_size ** 2
    mapping = np.array([row_perm[i // grid_size] * grid_size + col_perm[i % grid_size] for i in range(n2)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        if transpose_flag:
            x = x.T
        return digit_map[x.flatten()[mapping].reshape(grid_size, grid_size).copy()]

    return apply_transformation(board), apply_transformation(solution)


# ---------------------------------------------------------------------------
# Sudoku generator (backtracking) - used for grid_size < 9
# ---------------------------------------------------------------------------

def _is_valid(board: np.ndarray, row: int, col: int, num: int,
              box_rows: int, box_cols: int) -> bool:
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    r0 = (row // box_rows) * box_rows
    c0 = (col // box_cols) * box_cols
    if num in board[r0:r0 + box_rows, c0:c0 + box_cols]:
        return False
    return True


def _solve(board: np.ndarray, grid_size: int, box_rows: int, box_cols: int) -> bool:
    for i in range(grid_size):
        for j in range(grid_size):
            if board[i, j] == 0:
                for num in np.random.permutation(np.arange(1, grid_size + 1)):
                    if _is_valid(board, i, j, num, box_rows, box_cols):
                        board[i, j] = num
                        if _solve(board, grid_size, box_rows, box_cols):
                            return True
                        board[i, j] = 0
                return False
    return True


def _count_solutions(board: np.ndarray, grid_size: int, box_rows: int, box_cols: int, limit: int = 2) -> int:
    """Counts the board's solutions by backtracking, stopping at limit."""
    for i in range(grid_size):
        for j in range(grid_size):
            if board[i, j] == 0:
                count = 0
                for num in range(1, grid_size + 1):
                    if _is_valid(board, i, j, num, box_rows, box_cols):
                        board[i, j] = num
                        count += _count_solutions(board, grid_size, box_rows, box_cols, limit - count)
                        board[i, j] = 0
                        if count >= limit:
                            return count
                return count
    return 1


def generate_puzzle(grid_size: int, clue_ratio: float = 0.3, unique_solution: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    box_rows, box_cols = get_box_dims(grid_size)

    solution = np.zeros((grid_size, grid_size), dtype=np.int64)
    _solve(solution, grid_size, box_rows, box_cols)

    num_clues = max(grid_size, int(grid_size ** 2 * clue_ratio))

    puzzle = solution.copy()
    cells = np.random.permutation(grid_size ** 2)
    num_remove = grid_size ** 2 - num_clues
    for cell in cells[:num_remove]:
        puzzle[cell // grid_size, cell % grid_size] = 0

    # Reveal removed cells (in the reverse removal order) until the puzzle has exactly one solution
    if unique_solution:
        for cell in reversed(cells[:num_remove]):
            if _count_solutions(puzzle.copy(), grid_size, box_rows, box_cols) == 1:
                break
            puzzle[cell // grid_size, cell % grid_size] = solution[cell // grid_size, cell % grid_size]

    return puzzle, solution


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_from_hf(set_name: str, config: DictConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    inputs, labels = [], []
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for _source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    if set_name == "train" and config.subsample_size is not None:
        total = len(inputs)
        if config.subsample_size < total:
            idx = np.random.choice(total, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in idx]
            labels = [labels[i] for i in idx]

    return inputs, labels


def _generate_subset(set_name: str, config: DictConfig, exclude: frozenset = frozenset()) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generates puzzles whose inputs are not in exclude (the training inputs, when building the test set)."""
    n = config.num_train if set_name == "train" else config.num_test
    inputs, labels = [], []
    progress = tqdm(total=n, desc=f"Generating {config.grid_size}x{config.grid_size} {set_name}")
    while len(inputs) < n:
        puzzle, solution = generate_puzzle(config.grid_size, config.clue_ratio, bool(config.get("unique_solution", False)))
        if puzzle.tobytes() in exclude:
            continue
        inputs.append(puzzle)
        labels.append(solution)
        progress.update()
    return inputs, labels


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_subset(set_name: str, config: DictConfig, exclude: frozenset = frozenset()) -> frozenset:
    """Builds and saves one split; returns its (augmented) inputs, so the test split can be generated held out from them."""
    grid_size = config.grid_size

    if config.source_repo is not None:
        inputs, labels = _load_from_hf(set_name, config)
    else:
        inputs, labels = _generate_subset(set_name, config, exclude)
    
    num_aug = config.num_aug if set_name == "train" else 0

    results: dict = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for orig_inp, orig_out in zip(tqdm(inputs, desc=f"Processing {set_name}"), labels):
        for aug_idx in range(1 + num_aug):
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out, grid_size)

            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        results["group_indices"].append(puzzle_id)

    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        assert np.all((arr >= 0) & (arr <= grid_size))
        # Shift: blank(0)->1, digit d->d+1; PAD token stays 0 (added during collation)
        return arr + 1

    seen_inputs = frozenset(x.astype(np.int64).tobytes() for x in results["inputs"])
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=grid_size ** 2,
        # PAD(0) + blank(1) + digits 1..grid_size -> tokens 2..grid_size+1
        vocab_size=grid_size + 2,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_samples=example_id,
        sets=["all"]
    )

    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)

    return seen_inputs


@hydra.main(config_path="../config/dataset", config_name="sudoku", version_base=None)
def launch(config: DictConfig):
    if config.seed is not None:
        np.random.seed(config.seed)
    else:
        rng = np.random.default_rng()
        config.seed = int(rng.integers(low=0 , high=1e8))

    print("=" * 80)
    print("Sudoku Dataset Builder")
    print("=" * 80)
    print("Configuration:")
    print(f"  - Grid size: {config.grid_size}x{config.grid_size}")
    if config.grid_size == 9:
        print(f"  - Source repo: {config.source_repo}")
        print(f"  - Subsample size: {config.subsample_size}")
        print(f"  - Min difficulty: {config.min_difficulty}")
    else:
        print(f"  - Training puzzles: {config.num_train}")
        print(f"  - Test puzzles: {config.num_test}")
        print(f"  - Clue ratio: {config.clue_ratio}")
    print(f"  - Augmentation factor: {config.num_aug + 1}")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Random seed: {config.seed}")
    print("=" * 80)

    # Generated test puzzles are held out: none has the same input as a training puzzle (or augmentation)
    train_inputs = convert_subset("train", config)
    convert_subset("test", config, exclude=train_inputs)

    print("=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    launch()
