<<<<<<< HEAD
from typing import Optional, Tuple, List
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata


cli = ArgParser()

# (box_rows, box_cols) for each supported grid size
BOX_DIMS: dict[int, Tuple[int, int]] = {
    4: (2, 2),
    6: (2, 3),
    9: (3, 3),
}


class DataProcessConfig(BaseModel):
    # 9x9 HuggingFace source (ignored for smaller grids)
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    grid_size: int = 9  # 4, 6, or 9

    # 9x9 dataset filters
    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None

    # Generated puzzle counts (used when grid_size != 9)
    num_train: int = 10000
    num_test: int = 1000
    # Fraction of cells to reveal as clues (0.0–1.0); defaults to 0.3
    clue_ratio: float = 0.3

    num_aug: int = 0


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
# Sudoku generator (backtracking) — used for grid_size < 9
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


def generate_puzzle(grid_size: int, clue_ratio: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
    box_rows, box_cols = get_box_dims(grid_size)

    solution = np.zeros((grid_size, grid_size), dtype=np.int64)
    _solve(solution, grid_size, box_rows, box_cols)

    num_clues = max(grid_size, int(grid_size ** 2 * clue_ratio))

    puzzle = solution.copy()
    cells = np.random.permutation(grid_size ** 2)
    num_remove = grid_size ** 2 - num_clues
    for cell in cells[:num_remove]:
        puzzle[cell // grid_size, cell % grid_size] = 0

    return puzzle, solution


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _load_from_hf(set_name: str, config: DataProcessConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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


def _generate_subset(set_name: str, config: DataProcessConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    n = config.num_train if set_name == "train" else config.num_test
    inputs, labels = [], []
    for _ in tqdm(range(n), desc=f"Generating {config.grid_size}x{config.grid_size} {set_name}"):
        puzzle, solution = generate_puzzle(config.grid_size, config.clue_ratio)
        inputs.append(puzzle)
        labels.append(solution)
    return inputs, labels


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_subset(set_name: str, config: DataProcessConfig):
    grid_size = config.grid_size

    if grid_size == 9:
        inputs, labels = _load_from_hf(set_name, config)
    else:
        inputs, labels = _generate_subset(set_name, config)

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
        total_puzzles=len(results["group_indices"]) - 1,
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


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
||||||| empty tree
=======
from typing import Optional
import os
import csv
import json
import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from dataset.common import PuzzleDatasetMetadata


cli = ArgParser()


class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/sudoku-extreme"
    output_dir: str = "data/sudoku-extreme-full"

    subsample_size: Optional[int] = None
    min_difficulty: Optional[int] = None
    num_aug: int = 0
    
    seed: int = 42


def shuffle_sudoku(board: np.ndarray, solution: np.ndarray):
    # Create a random digit mapping: a permutation of 1..9, with zero (blank) unchanged
    digit_map = np.pad(np.random.permutation(np.arange(1, 10)), (1, 0))
    
    # Randomly decide whether to transpose.
    transpose_flag = np.random.rand() < 0.5

    # Generate a valid row permutation:
    # - Shuffle the 3 bands (each band = 3 rows) and for each band, shuffle its 3 rows.
    bands = np.random.permutation(3)
    row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])

    # Similarly for columns (stacks).
    stacks = np.random.permutation(3)
    col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])

    # Build an 81->81 mapping. For each new cell at (i, j)
    # (row index = i // 9, col index = i % 9),
    # its value comes from old row = row_perm[i//9] and old col = col_perm[i%9].
    mapping = np.array([row_perm[i // 9] * 9 + col_perm[i % 9] for i in range(81)])

    def apply_transformation(x: np.ndarray) -> np.ndarray:
        # Apply transpose flag
        if transpose_flag:
            x = x.T
        # Apply the position mapping.
        new_board = x.flatten()[mapping].reshape(9, 9).copy()
        # Apply digit mapping
        return digit_map[new_board]

    return apply_transformation(board), apply_transformation(solution)


def convert_subset(set_name: str, config: DataProcessConfig):
    # Read CSV
    inputs = []
    labels = []
    
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            if (config.min_difficulty is None) or (int(rating) >= config.min_difficulty):
                assert len(q) == 81 and len(a) == 81
                
                inputs.append(np.frombuffer(q.replace('.', '0').encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))
                labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(9, 9) - ord('0'))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    num_augments = config.num_aug if set_name == "train" else 0

    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for orig_inp, orig_out in zip(tqdm(inputs), labels):
        for aug_idx in range(1 + num_augments):
            # First index is not augmented
            if aug_idx == 0:
                inp, out = orig_inp, orig_out
            else:
                inp, out = shuffle_sudoku(orig_inp, orig_out)

            # Push puzzle (only single example)
            results["inputs"].append(inp)
            results["labels"].append(out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
        
    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.concatenate(seq).reshape(len(seq), -1)
        
        assert np.all((arr >= 0) & (arr <= 9))
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=81,
        vocab_size=10 + 1,  # PAD + "0" ... "9"
        
        pad_id=0,
        ignore_label_id=0,
        
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_samples=example_id,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    np.random.seed(config.seed)

    convert_subset("train", config)
    convert_subset("test", config)


if __name__ == "__main__":
    cli()
>>>>>>> eqr/main
