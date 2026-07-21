from __future__ import annotations

from typing import List, Optional, Tuple
from collections import deque
import json
import csv
import os

import numpy as np
from pydantic import BaseModel
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _bfs(open_mask: np.ndarray, start: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    n = open_mask.shape[0]
    dist = np.full((n, n), -1, dtype=np.int32)
    parent = np.full((n, n, 2), -1, dtype=np.int32)
    q: deque[Tuple[int, int]] = deque([start])
    dist[start] = 0

    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and open_mask[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = d + 1
                parent[ny, nx] = (y, x)
                q.append((ny, nx))

    return dist, parent


def _bfs_dist(open_mask: np.ndarray, start: Tuple[int, int]) -> np.ndarray:
    n = open_mask.shape[0]
    dist = np.full((n, n), -1, dtype=np.int32)
    q: deque[Tuple[int, int]] = deque([start])
    dist[start] = 0
    while q:
        y, x = q.popleft()
        d = dist[y, x]
        for dy, dx in DIRS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < n and 0 <= nx < n and open_mask[ny, nx] and dist[ny, nx] == -1:
                dist[ny, nx] = d + 1
                q.append((ny, nx))
    return dist


def _shortest_path_count_cap(
    open_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    cap: int = 2,
) -> int:
    dist_s = _bfs_dist(open_mask, start)
    if dist_s[goal] == -1:
        return 0
    dist_g = _bfs_dist(open_mask, goal)
    shortest = int(dist_s[goal])
    n = open_mask.shape[0]

    layers: List[List[Tuple[int, int]]] = [[] for _ in range(shortest + 1)]
    for y in range(n):
        for x in range(n):
            if dist_s[y, x] >= 0 and dist_g[y, x] >= 0 and dist_s[y, x] + dist_g[y, x] == shortest:
                layers[int(dist_s[y, x])].append((y, x))

    counts: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]
    counts[start[0]][start[1]] = 1
    for d in range(shortest):
        for y, x in layers[d]:
            curr = counts[y][x]
            if curr == 0:
                continue
            for dy, dx in DIRS:
                ny, nx = y + dy, x + dx
                if 0 <= ny < n and 0 <= nx < n and dist_s[ny, nx] == d + 1:
                    if dist_g[ny, nx] >= 0 and dist_s[ny, nx] + dist_g[ny, nx] == shortest:
                        counts[ny][nx] = min(cap, counts[ny][nx] + curr)
        if counts[goal[0]][goal[1]] >= cap:
            return cap

    return counts[goal[0]][goal[1]]


def _generate_perfect_maze(n: int, rng: np.random.Generator) -> np.ndarray:
    if n < 3:
        return np.ones((n, n), dtype=bool)
    if n % 2 == 0:
        raise ValueError("Perfect maze requires an odd grid size")

    open_mask = np.zeros((n, n), dtype=bool)
    cells_h = (n - 1) // 2
    cells_w = (n - 1) // 2
    visited = np.zeros((cells_h, cells_w), dtype=bool)

    def cell_to_grid(r: int, c: int) -> Tuple[int, int]:
        return 2 * r + 1, 2 * c + 1

    start = (int(rng.integers(cells_h)), int(rng.integers(cells_w)))
    stack = [start]
    visited[start] = True
    sr, sc = cell_to_grid(*start)
    open_mask[sr, sc] = True

    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < cells_h and 0 <= nc < cells_w and not visited[nr, nc]:
                neighbors.append((nr, nc))
        if not neighbors:
            stack.pop()
            continue
        nr, nc = neighbors[int(rng.integers(len(neighbors)))]
        visited[nr, nc] = True
        gr, gc = cell_to_grid(r, c)
        ngr, ngc = cell_to_grid(nr, nc)
        open_mask[(gr + ngr) // 2, (gc + ngc) // 2] = True
        open_mask[ngr, ngc] = True
        stack.append((nr, nc))

    return open_mask


def _reconstruct_path(
    parent: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
) -> Optional[List[Tuple[int, int]]]:
    path = [goal]
    cur = goal
    while cur != start:
        py, px = parent[cur]
        if py < 0:
            return None
        cur = (int(py), int(px))
        path.append(cur)
    path.reverse()
    return path


def _sample_length(rng: np.random.Generator, config: DictConfig) -> int:
    max_possible = config.grid_size * config.grid_size - 1
    dist = config.length_distribution.lower()

    if dist == "fixed":
        if not 1 <= config.length_value <= max_possible:
            raise ValueError(f"length_value must be in [1, {max_possible}]")
        return int(config.length_value)

    if dist == "uniform":
        low = max(1, min(config.min_path_length, max_possible))
        high = max(1, min(config.max_path_length, max_possible))
        if low > high:
            raise ValueError("min_path_length must be <= max_path_length")
        return int(rng.integers(low, high + 1))

    if dist == "normal":
        low = max(1, min(config.min_path_length, max_possible))
        high = max(1, min(config.max_path_length, max_possible))
        if low > high:
            raise ValueError("min_path_length must be <= max_path_length")
        if config.length_std <= 0:
            raise ValueError("length_std must be > 0 for normal distribution")
        sample = int(round(rng.normal(config.length_mean, config.length_std)))
        return int(min(high, max(low, sample)))

    if dist == "list":
        if not config.length_values:
            raise ValueError("length_values must be provided for list distribution")
        values = [int(v) for v in config.length_values]
        if any(v < 1 or v > max_possible for v in values):
            raise ValueError(f"length_values must be in [1, {max_possible}]")
        weights = None
        if config.length_weights is not None:
            if len(config.length_weights) != len(values):
                raise ValueError("length_weights must be the same length as length_values")
            weights = np.array(config.length_weights, dtype=np.float64)
            if np.any(weights < 0):
                raise ValueError("length_weights must be non-negative")
            total = float(weights.sum())
            if total <= 0:
                raise ValueError("length_weights must sum to a positive value")
            weights = weights / total
        return int(rng.choice(values, p=weights))

    raise ValueError(f"Unknown length_distribution: {config.length_distribution}")


def _generate_single_sample(
    n: int,
    length: int,
    wall_prob: float,
    rng: np.random.Generator,
    max_grid_attempts: int,
    max_start_attempts: int,
    require_unique: bool,
    maze_mode: str,
) -> Optional[Tuple[np.ndarray, Tuple[int, int], Tuple[int, int], List[Tuple[int, int]]]]:
    mode = maze_mode.lower()
    for _ in range(max_grid_attempts):
        if mode == "perfect":
            maze_n = n if n % 2 == 1 else n - 1
            if maze_n < 3:
                return None
            open_mask = np.zeros((n, n), dtype=bool)
            open_mask[:maze_n, :maze_n] = _generate_perfect_maze(maze_n, rng)
        else:
            open_mask = rng.random((n, n)) >= wall_prob
        if int(open_mask.sum()) < length + 1:
            continue
        open_positions = np.argwhere(open_mask)
        if open_positions.shape[0] < 2:
            continue
        for _ in range(max_start_attempts):
            start = tuple(open_positions[rng.integers(open_positions.shape[0])])
            dist, parent = _bfs(open_mask, start)
            candidates = np.argwhere(dist == length)
            if candidates.shape[0] == 0:
                continue
            goal = tuple(candidates[rng.integers(candidates.shape[0])])
            path = _reconstruct_path(parent, start, goal)
            if path is None or len(path) != length + 1:
                continue
            if require_unique and mode != "perfect":
                count = _shortest_path_count_cap(open_mask, start, goal, cap=2)
                if count != 1:
                    continue
            return open_mask, start, goal, path
    return None


def _maze_to_arrays(
    open_mask: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    path: List[Tuple[int, int]],
    char2id: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    n = open_mask.shape[0]
    grid = np.full((n, n), ord(" "), dtype=np.uint8)
    grid[~open_mask] = ord("#")
    grid[start] = ord("S")
    grid[goal] = ord("G")

    label = grid.copy()
    for y, x in path[1:-1]:
        label[y, x] = ord("o")

    return char2id[grid].reshape(-1), char2id[label].reshape(-1)


def _build_split(
    split_name: str,
    num_samples: int,
    config: DictConfig,
    rng: np.random.Generator,
    char2id: np.ndarray,
    seen_hashes: Optional[set[bytes]],
):
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    lengths: List[int] = []

    with tqdm(total=num_samples, desc=f"{split_name} samples") as bar:
        while example_id < num_samples:
            for _ in range(config.max_length_resamples):
                length = _sample_length(rng, config)
                sample = _generate_single_sample(
                    config.grid_size,
                    length,
                    config.wall_prob,
                    rng,
                    config.max_grid_attempts,
                    config.max_start_attempts,
                    config.require_unique,
                    config.maze_mode,
                )
                if sample is None:
                    if config.strict_length:
                        raise RuntimeError(
                            f"Failed to generate a maze with shortest path length {length} after "
                            f"{config.max_grid_attempts}x{config.max_start_attempts} attempts. "
                            "Try adjusting wall_prob or the length range."
                        )
                    continue

                open_mask, start, goal, path = sample
                inp, out = _maze_to_arrays(open_mask, start, goal, path, char2id)
                if seen_hashes is not None:
                    key = inp.tobytes()
                    if key in seen_hashes:
                        continue
                    seen_hashes.add(key)

                results["inputs"].append(inp)
                results["labels"].append(out)
                example_id += 1
                puzzle_id += 1

                results["puzzle_indices"].append(example_id)
                results["puzzle_identifiers"].append(0)
                results["group_indices"].append(puzzle_id)
                lengths.append(length)
                bar.update(1)
                break
            else:
                raise RuntimeError(
                    "Failed to generate a maze after resampling lengths. "
                    "Try adjusting wall_prob or the length range."
                )

    if not results["inputs"]:
        raise RuntimeError(f"No samples generated for split {split_name}")

    results_np = {
        "inputs": np.stack(results["inputs"]).astype(np.uint8),
        "labels": np.stack(results["labels"]).astype(np.uint8),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    metadata = PuzzleDatasetMetadata(
        seq_len=config.grid_size * config.grid_size,
        vocab_size=len(CHARSET) + 1,
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results_np["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_samples=example_id,
        sets=["all"],
    )

    save_dir = os.path.join(config.output_dir, split_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    for k, v in results_np.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    if lengths:
        lengths_np = np.array(lengths, dtype=np.int32)
        print(
            f"{split_name} length stats: min={int(lengths_np.min())}, "
            f"max={int(lengths_np.max())}, mean={lengths_np.mean():.2f}"
        )

def convert_subset(set_name: str, config: DictConfig):
    # Read CSV
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []
    
    with open(hf_hub_download(config.source_repo, f"{set_name}.csv", repo_type="dataset"), newline="") as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)
                
            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)
    
    for inp, out in zip(tqdm(inputs), labels):
        # Dihedral transformations for augmentation
        for aug_idx in range(8 if (set_name == "train" and config.aug) else 1):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)
            
    # Char mappings
    assert len(all_chars - set(CHARSET)) == 0
    
    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])
        
        return arr
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=int(math.prod(grid_size)),  # type: ignore
        vocab_size=len(CHARSET) + 1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
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

    
@hydra.main(config_path = "../config/dataset", config_name = "maze_unique", version_base = None)
def launch(config: DictConfig):

    # Use the provided seed, or draw a random one when unspecified
    if config.seed is None:
        rng = np.random.default_rng()
        config.seed = int(rng.integers(low=0, high=1e8))
        
    else:
        rng = np.random.default_rng(config.seed)
    
    char2id = np.zeros(256, dtype=np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET), dtype=np.uint8) + 1

    os.makedirs(config.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("Maze unique dataset builder")
    print("=" * 80)
    print("Configuration:")
    print(f"  - Board size: {config.board_size}x{config.board_size}")
    if config.source_repo is not None:
        print(f"  - Source repo: {config.source_repo}")
    else:
        print(f"  - Training puzzles: {config.num_train}")
        print(f"  - Test puzzles: {config.num_test}")
    print(f"  - Augmentation factor: {config.num_aug + 1}")
    print(f"  - Output directory: {config.output_dir}")
    print(f"  - Random seed: {config.seed}")
    print("=" * 80)
    
    # Generate the maze dataset if a hf path isnt' specified, otherwise load in the hf data
    if config.source_repo is None:
        seen_hashes = set() if config.dedupe else None
        _build_split("train", config.train_samples, config, rng, char2id, seen_hashes)
        _build_split("test", config.test_samples, config, rng, char2id, seen_hashes)

        with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
            json.dump(["<blank>"], f)

    else:
        convert_subset("train", config)
        convert_subset("test", config)
