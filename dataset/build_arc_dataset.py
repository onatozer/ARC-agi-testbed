from typing import List, Tuple, Dict
from dataclasses import dataclass
import os
import json
import hashlib
import numpy as np

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

from .common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


class DataProcessConfig(BaseModel):
    input_file_prefix: str
    output_dir: str
    subsets: List[str]
    test_set_name: str
    test_set_name2: str = "your_test_set"
    seed: int = 42
    num_aug: int = 1000
    puzzle_identifiers_start: int = 1 # start > 1 to handle multiple datasets
    grid_size: int = 30 # canvas side length; seq_len = max_grid_size ** 2
    scale_aug: bool = False # random integer upscaling of train examples (arXiv:2511.14761)
    border_tokens: bool = False # ViTARC border tokens + supervised 2D pad (arXiv 2410.06405); canvas becomes grid_size x (grid_size + 1)
    
ARCMaxGridSize = 30
ARCAugmentRetriesFactor = 5

PuzzleIdSeparator = "|||"

# ViTARC border-token ids (border_tokens=True). PAD (0) is only batch fill and stays unsupervised; colors stay at 2..11
EndXToken, EndYToken, EndXYToken, NewlineToken, Pad2DToken = 1, 12, 13, 14, 15
    

@dataclass
class ARCPuzzle:
    id: str
    examples: List[Tuple[np.ndarray, np.ndarray]]

    
def arc_grid_to_np(grid: List[List[int]]):
    arr = np.array(grid)

    # Shape check
    assert arr.ndim == 2
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # Element check
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_seq_translational_augment(inp: np.ndarray, out: np.ndarray, do_translation: bool, max_grid_size: int = ARCMaxGridSize, do_scale: bool = False, border_tokens: bool = False):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # Random integer upscale (each cell -> s x s block), shared by input and output, s.t. both still fit the canvas
    if do_scale:
        max_scale = max_grid_size // max(*inp.shape, *out.shape)
        scale = np.random.randint(1, max_scale + 1)
        inp, out = [grid.repeat(scale, axis=0).repeat(scale, axis=1) for grid in (inp, out)]

    # Compute random top-left pad
    if do_translation:
        pad_r = np.random.randint(0, max_grid_size - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, max_grid_size - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # Pad grid
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        if border_tokens:
            result.append(_border_token_canvas(grid, pad_r, pad_c, max_grid_size).flatten())
            continue
        grid = np.pad(grid + 2, ((pad_r, max_grid_size - pad_r - nrow), (pad_c, max_grid_size - pad_c - ncol)), constant_values=0)

        # Add <eos>
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < max_grid_size:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < max_grid_size:
            grid[pad_r:eos_row, eos_col] = 1

        result.append(grid.flatten())

    return result


def _border_token_canvas(grid: np.ndarray, pad_r: int, pad_c: int, max_grid_size: int) -> np.ndarray:
    """ViTARC visual tokens (arXiv 2410.06405, Sec. 4): the grid on a max_grid_size x (max_grid_size + 1) canvas, where
    the extra last column is <2d_nl>, the true grid's right/bottom edges are <2d_endxgrid>/<2d_endygrid> with
    <2d_endxygrid> at the corner, and every other canvas cell is a supervised <2d_pad>."""
    nrow, ncol = grid.shape
    canvas = np.full((max_grid_size, max_grid_size + 1), Pad2DToken, dtype=np.uint8)
    canvas[:, max_grid_size] = NewlineToken
    canvas[pad_r:pad_r + nrow, pad_c:pad_c + ncol] = grid + 2

    end_row, end_col = pad_r + nrow, pad_c + ncol
    if end_col < max_grid_size:
        canvas[pad_r:end_row, end_col] = EndXToken
    if end_row < max_grid_size:
        canvas[end_row, pad_c:end_col] = EndYToken
    if end_row < max_grid_size and end_col < max_grid_size:
        canvas[end_row, end_col] = EndXYToken
    return canvas


def grid_hash(grid: np.ndarray):
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, byteorder='big') for x in grid.shape]
    buffer.append(grid.tobytes())
    
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: dict):
    # Hash the puzzle for checking equivalence
    hashes = []
    for example_type, example in puzzle.items():
        for input, label in example.examples:
            hashes.append(f"{grid_hash(input)}|{grid_hash(label)}")
            
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()


def aug(name: str):
    # Augment plan
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))])  # Permute colors, Excluding "0" (black)
    
    name_with_aug_repr = f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)
    
    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # Inverse the "aug" function
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)
    
    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]
    
    return name.split(PuzzleIdSeparator)[0], _map_grid


def convert_single_arc_puzzle(results: dict, name: str, puzzle: dict, aug_count: int, dest_mapping: Dict[str, Tuple[str, str]]):
    # Convert
    dests = set(dest_mapping.values())
    converted = {dest: ARCPuzzle(name, []) for dest in dests}
    for example_type, examples in puzzle.items():
        # Map to target split
        dest = dest_mapping[example_type]
        converted[dest].examples.extend([(arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"])) for example in examples])

    group = [converted]
    
    # Augment
    if aug_count > 0:
        hashes = {puzzle_hash(converted)}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, _map_grid = aug(name)

            # Check duplicate
            augmented = {dest: ARCPuzzle(aug_name, [(_map_grid(input), _map_grid(label)) for (input, label) in puzzle.examples]) for dest, puzzle in converted.items()}
            h = puzzle_hash(augmented)
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)
                
            if len(group) >= aug_count + 1:
                break
            
        if len(group) < aug_count + 1:
            print (f"[Puzzle {name}] augmentation not full, only {len(group)}")

    # Append
    for dest in dests:
        # Convert the examples
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted[dest] for converted in group])


def load_puzzles_arcagi(config: DataProcessConfig):
    train_examples_dest = ("train", "all")
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        config.test_set_name2: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))]
    }
    
    test_puzzles = {}
    results = {}

    total_puzzles = 0
    for subset_name in config.subsets:
        # Load all puzzles in this subset
        with open(f"{config.input_file_prefix}_{subset_name}_challenges.json", "r") as f:
            puzzles = json.load(f)

        sols_filename = f"{config.input_file_prefix}_{subset_name}_solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # Fill with dummy
            print (f"{subset_name} solutions not found, filling with dummy")

            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # Shuffle puzzles
        puzzles = list(puzzles.items())
        np.random.shuffle(puzzles)
        
        # Assign by fraction
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break
                    
            assert test_examples_dest is not None
            
            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle
                
            convert_single_arc_puzzle(results, name, puzzle, config.num_aug, {"train": train_examples_dest, "test": test_examples_dest})
            total_puzzles += 1

    print (f"Total puzzles: {total_puzzles}")
    return results, test_puzzles


def convert_dataset(config: DataProcessConfig, load_puzzles=load_puzzles_arcagi):
    np.random.seed(config.seed)
    
    # Read dataset
    data, test_puzzles = load_puzzles(config)
    
    # Map global puzzle identifiers
    num_identifiers = config.puzzle_identifiers_start  # 0 is blank, start at 1
    identifier_map = {}
    for split_name, split in data.items():
        for subset_name, subset in split.items():
            for group in subset:
                for puzzle in group:
                    if puzzle.id not in identifier_map:
                        identifier_map[puzzle.id] = num_identifiers
                        num_identifiers += 1
    print (f"Total puzzle IDs (including <blank>): {num_identifiers}")

    # Save
    for split_name, split in data.items():
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)
        
        # Translational augmentations
        enable_translational_augment = split_name == "train"

        # Statistics
        total_examples = 0
        total_puzzles = 0
        total_groups = 0
        
        for subset_name, subset in split.items(): # "all" is the only subset
            # Construct subset
            results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
            results["puzzle_indices"].append(0)
            results["group_indices"].append(0)
            
            example_id = 0
            puzzle_id = 0
            
            for group in subset:
                for puzzle in group:
                    # Push puzzle
                    no_aug_id = np.random.randint(0, len(puzzle.examples))
                    for _idx_ex, (inp, out) in enumerate(puzzle.examples):
                        do_aug = enable_translational_augment and _idx_ex != no_aug_id
                        inp, out = np_grid_to_seq_translational_augment(inp, out, do_translation=do_aug, max_grid_size=config.grid_size, do_scale=do_aug and config.scale_aug, border_tokens=config.border_tokens)
                            
                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        example_id += 1
                        
                        total_examples += 1

                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(identifier_map[puzzle.id])
                    
                    puzzle_id += 1
                    total_puzzles += 1
                    
                # Push group
                results["group_indices"].append(puzzle_id)
                total_groups += 1
            
            for k, v in results.items():
                if k in {"inputs", "labels"}:
                    v = np.stack(v, 0)
                else:
                    v = np.array(v, dtype=np.int32)
                
                np.save(os.path.join(config.output_dir, split_name, f"{subset_name}__{k}.npy"), v)
        
        # Metadata
        metadata = PuzzleDatasetMetadata(
            seq_len=config.grid_size * (config.grid_size + 1 if config.border_tokens else config.grid_size),
            vocab_size=Pad2DToken + 1 if config.border_tokens else 10 + 2,  # PAD + EOS + "0" ... "9" (+ ViTARC border tokens)
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / total_puzzles,
            total_puzzles=total_puzzles,
            sets=list(split.keys())
        )

        # Save metadata as JSON.
        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            
    # Save IDs mapping
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        ids_mapping = {v: k for k, v in identifier_map.items()}
        json.dump([ids_mapping.get(i, "<blank>") for i in range(num_identifiers)], f)
    
    # Save Test Puzzles
    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@hydra.main(config_path="../config/dataset", config_name="arc", version_base=None)
def launch(config: DictConfig):
    """
    Run:
        python -m dataset.build_arc_dataset
    Override on the CLI, e.g.:
        python -m dataset.build_arc_dataset num_aug=100 scale_aug=true
    """
    if config.seed is None:
        config.seed = int(np.random.default_rng().integers(low=0, high=1e8))

    config = DataProcessConfig(**OmegaConf.to_container(config, resolve=True))
    if config.border_tokens and not config.output_dir.endswith("-border"):
        # Never overwrite a plain dataset with the border-token format
        config.output_dir += "-border"
        print(f"border_tokens: writing to {config.output_dir}")
    convert_dataset(config)


if __name__ == "__main__":
    launch()












