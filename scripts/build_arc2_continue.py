"""Held-out ARC-AGI-2 build for continuing a TRM pretrained on ARC-AGI-1 (the official arc_v1_public checkpoint):
  - ARC-AGI-2 training tasks the checkpoint never saw (not in ARC-AGI-1 training / evaluation / ConceptARC): one group each;
  - replay: every other ARC-AGI-2 training task and ConceptARC, packed into as many groups as make REPLAY_FRACTION of the
    groups, so the sampler (one puzzle per group per epoch) mixes in ~that fraction of replay rows while still rotating
    through every replay task (each row gets its masked copy, so the old tasks keep the reconstruction objective too);
  - ARC-AGI-2 evaluation: its test pairs only (the held-out protocol: its demos never reach training).
Same format as dataset.build_arc_dataset otherwise (1000 augmentations, no scale augmentation, as the official TRM build).
    PYTHONPATH=. python scripts/build_arc2_continue.py <output_dir> [num_aug 1000]
"""
import json
import os
import sys

import numpy as np

from dataset.build_arc_dataset import DataProcessConfig, PuzzleIdSeparator, convert_dataset, load_puzzles_arcagi

PREFIX = "kaggle/combined/arc-agi"
REPLAY_FRACTION = float(os.environ.get("REPLAY_FRACTION", 0.10))  # 0: the new tasks only

seen = set()
for subset in ("training", "evaluation", "concept"):
    with open(f"{PREFIX}_{subset}_challenges.json") as f:
        seen |= set(json.load(f))


def load_puzzles(config: DataProcessConfig):
    results, test_puzzles = load_puzzles_arcagi(config)

    base = lambda group: group[0].id.split(PuzzleIdSeparator)[0]
    groups = results["train"]["all"]
    new = [g for g in groups if base(g) not in seen]
    replay = [g for g in groups if base(g) in seen]

    n_groups = round(len(new) * REPLAY_FRACTION / (1 - REPLAY_FRACTION))
    if n_groups == 0:
        results["train"]["all"] = new
        print(f"new tasks: {len(new)}, no replay")
        return results, test_puzzles
    order = np.random.default_rng(0).permutation(len(replay))
    packed = [sum((replay[i] for i in part), []) for part in np.array_split(order, n_groups)]
    results["train"]["all"] = new + packed

    print(f"new tasks: {len(new)}, replay tasks: {len(replay)} in {len(packed)} groups")
    return results, test_puzzles


if __name__ == "__main__":
    config = DataProcessConfig(
        input_file_prefix=PREFIX, output_dir=sys.argv[1], subsets=["training2", "concept", "evaluation2"], test_set_name="evaluation2",
        heldout_test_set=True, num_aug=int(sys.argv[2]) if len(sys.argv) > 2 else 1000, grid_size=30, scale_aug=False, test_aug=False, seed=42,
    )
    convert_dataset(config, load_puzzles=load_puzzles)
