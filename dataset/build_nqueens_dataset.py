"""
N-Queens Dataset Builder for Tiny Recursive Models (TRM)

This script generates N-Queens puzzle datasets compatible with the TRM framework.
The N-Queens problem involves placing N chess queens on an NxN chessboard such that
no two queens threaten each other (no two queens share the same row, column, or diagonal).

Dataset Format:
- Input: NxN board with some queens placed (partial solution or empty board)
- Label: NxN board with complete valid solution
- Vocab: 0 (PAD), 1 (empty cell), 2 (queen)
"""

from typing import List, Tuple
import os
import csv
import json
import numpy as np
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata, dihedral_transform


def is_safe(board: np.ndarray, row: int, col: int) -> bool:
    """
    Check if it's safe to place a queen at position (row, col)
    
    Args:
        board: NxN numpy array (0 = empty, 1 = queen)
        row: Row index
        col: Column index
    
    Returns:
        True if position is safe, False otherwise
    """
    N = len(board)
    
    # Check row
    if np.any(board[row, :]):
        return False
    
    # Check column
    if np.any(board[:, col]):
        return False
    
    # Check main diagonal (top-left to bottom-right)
    for i in range(N):
        j = col - row + i
        if 0 <= j < N and board[i, j]:
            return False
    
    # Check anti-diagonal (top-right to bottom-left)
    for i in range(N):
        j = col + row - i
        if 0 <= j < N and board[i, j]:
            return False
    
    return True


def solve_nqueens_backtrack(board: np.ndarray, row: int, solutions: List[np.ndarray], max_solutions: int = 1):
    """
    Solve N-Queens using backtracking
    
    Args:
        board: Current board state
        row: Current row to place queen
        solutions: List to store found solutions
        max_solutions: Maximum number of solutions to find
    """
    N = len(board)
    
    # Base case: all queens placed
    if row == N:
        solutions.append(board.copy())
        return
    
    # If we have enough solutions, stop
    if len(solutions) >= max_solutions:
        return
    
    # Try placing queen in each column of current row
    for col in range(N):
        if is_safe(board, row, col):
            board[row, col] = 1
            solve_nqueens_backtrack(board, row + 1, solutions, max_solutions)
            board[row, col] = 0  # Backtrack


def generate_nqueens_solution(N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random valid N-Queens solution
    
    Args:
        N: Board size
        rng: Random number generator
    
    Returns:
        NxN numpy array with a valid solution
    """
    # Start with empty board
    board = np.zeros((N, N), dtype=np.int32)
    
    # Use backtracking to find all solutions, then pick one randomly
    # For efficiency, we'll use a randomized backtracking approach
    def random_solve(row: int) -> bool:
        if row == N:
            return True
        
        # Try columns in random order
        cols = rng.permutation(N)
        for col in cols:
            if is_safe(board, row, col):
                board[row, col] = 1
                if random_solve(row + 1):
                    return True
                board[row, col] = 0
        
        return False
    
    random_solve(0)
    return board


def create_partial_input(solution: np.ndarray, num_given: int, rng: np.random.Generator) -> np.ndarray:
    """
    Create a partial N-Queens puzzle by removing some queens from the solution
    
    Args:
        solution: Complete valid solution
        num_given: Number of queens to keep in the input
        rng: Random number generator
    
    Returns:
        Partial board with only num_given queens
    """
    N = len(solution)
    
    # Find all queen positions
    queen_positions = np.argwhere(solution == 1)
    
    if num_given == 0:
        # Empty board
        return np.zeros((N, N), dtype=np.int32)
    elif num_given >= N:
        # Full solution
        return solution.copy()
    else:
        # Randomly select num_given queens to keep
        selected_indices = rng.choice(len(queen_positions), size=num_given, replace=False)
        selected_positions = queen_positions[selected_indices]
        
        partial = np.zeros((N, N), dtype=np.int32)
        for row, col in selected_positions:
            partial[row, col] = 1
        
        return partial


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _generate_subset(set_name: str, config: DictConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate N-Queens puzzles locally via randomized backtracking."""
    rng = np.random.default_rng(config.seed)

    N = config.board_size
    num_puzzles = config.num_train if set_name == "train" else config.num_test

    inputs, labels = [], []
    for _ in tqdm(range(num_puzzles), desc=f"Generating {N}x{N} {set_name} puzzles"):
        # Generate a valid solution
        solution = generate_nqueens_solution(N, rng)

        # Determine how many queens to give in the input
        if config.num_given_queens is not None:
            num_given = config.num_given_queens
        else:
            # Randomly choose between 0 and N-1 queens
            num_given = rng.integers(0, N)

        # Create partial input
        partial = create_partial_input(solution, num_given, rng)

        inputs.append(partial)
        labels.append(solution)

    return inputs, labels


def _load_from_hf(set_name: str, config: DictConfig) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load pre-built N-Queens puzzles from a HuggingFace dataset repo.

    Expects a ``{set_name}.csv`` with a header row followed by rows whose first
    two columns are the flattened input board and the flattened solution board,
    each a length-(N*N) string of '0' (empty) / '1' (queen).
    """
    N = config.board_size
    inputs, labels = [], []
    path = hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset")
    with open(path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            q, a = row[0], row[1]
            assert len(q) == N * N and len(a) == N * N
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(N, N) - ord('0'))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(N, N) - ord('0'))

    return inputs, labels


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_subset(set_name: str, config: DictConfig):
    """
    Build the N-Queens dataset for the train or test set.

    Args:
        set_name: "train" or "test"
        config: Hydra config
    """
    N = config.board_size

    # source_repo set -> load from HuggingFace; otherwise generate on the fly
    if config.source_repo is not None:
        inputs, labels = _load_from_hf(set_name, config)
    else:
        inputs, labels = _generate_subset(set_name, config)

    # Generate dataset with augmentation
    num_augments = config.num_aug if set_name == "train" else 0
    
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for inp, out in zip(tqdm(inputs, desc=f"Processing {set_name} with augmentation"), labels):
        # Apply dihedral transformations for augmentation
        # tid=0 is identity (no transformation)
        for aug_idx in range(1 + num_augments):
            if aug_idx == 0:
                # No augmentation
                aug_inp = inp
                aug_out = out
            else:
                # Apply dihedral transformation (rotation/reflection)
                # Use (aug_idx - 1) to get transformations 0-7
                tid = (aug_idx - 1) % 8
                aug_inp = dihedral_transform(inp, tid)
                aug_out = dihedral_transform(out, tid)
            
            results["inputs"].append(aug_inp)
            results["labels"].append(aug_out)
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
        
        # Push group (all augmentations of same puzzle belong to one group)
        results["group_indices"].append(puzzle_id)
    
    # Convert to numpy arrays with proper encoding
    # Vocab: 0 (PAD), 1 (empty cell), 2 (queen)
    def _seq_to_numpy(seq):
        arr = np.vstack([s.reshape(-1) for s in seq])
        # Add 1 to shift: 0 (empty) -> 1, 1 (queen) -> 2
        return arr + 1
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }
    
    # Create metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=N * N,  # Flattened board
        vocab_size=3,  # PAD (0), empty cell (1), queen (2)
        pad_id=0,
        ignore_label_id=0,  # We want to predict all cells
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )
    
    # Save dataset
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metadata as JSON
    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f, indent=2)
    
    # Save data arrays
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
    
    # Save identifiers mapping (for visualization)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)
    
    print(f"Saved {set_name} dataset to {save_dir}")
    print(f"  - Total examples: {len(results['inputs'])}")
    print(f"  - Total groups: {metadata.total_groups}")
    print(f"  - Sequence length: {metadata.seq_len}")
    print(f"  - Vocabulary size: {metadata.vocab_size}")


# NOTE: I'm pretty sure we're doing the seeding here incorrectly, don't think we're using the rng. On the other mind might not really matter
# If we're just loading from hugging face anyways
@hydra.main(config_path = "../config/dataset", config_name = "nqueens", version_base = None)
def launch(config: DictConfig):
    """
    Main entry point for dataset generation

    Run:
        python dataset/build_nqueens_dataset.py
    Override on the CLI, e.g.:
        python dataset/build_nqueens_dataset.py board_size=6 num_train=5000
    """
    # Use the provided seed, or draw a random one when unspecified
    if config.seed is None:
        rng = np.random.default_rng()
        config.seed = int(rng.integers(low=0, high=1e8))

    print("=" * 80)
    print("N-Queens Dataset Builder")
    print("=" * 80)
    print("Configuration:")
    print(f"  - Board size: {config.board_size}x{config.board_size}")
    if config.source_repo is not None:
        print(f"  - Source repo: {config.source_repo}")
    else:
        print(f"  - Training puzzles: {config.num_train}")
        print(f"  - Test puzzles: {config.num_test}")
        print(f"  - Num given queens: {config.num_given_queens}")
    print(f"  - Augmentation factor: {config.num_aug + 1}")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Random seed: {config.seed}")
    print("=" * 80)
    
    # Generate train and test sets
    convert_subset("train", config)
    convert_subset("test", config)
    
    print("=" * 80)
    print("Dataset generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    launch()
