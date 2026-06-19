## Building the dataset
### Build Dataset args:
| Argument | Default | What it does |
|---|---|---|
| `--grid-size` | `9` | Grid dimensions: 4 (4×4), 6 (6×6), or 9 (9×9). Controls puzzle size, seq_len, and vocab_size. |
| `--output-dir` | `data/sudoku-extreme-full` | Where the built dataset is saved. Point data_paths here when training. |
| `--source-repo` | `sapientinc/sudoku-extreme` | HuggingFace repo to pull from. Only used when grid-size=9. |
| `--subsample-size` | `None` | For 9×9 only: randomly keep this many training puzzles instead of the full dataset. |
| `--min-difficulty` | `None` | For 9×9 only: skip puzzles with a difficulty rating below this threshold. |
| `--num-train` | `10000` | For 4×4 / 6×6: how many puzzles to generate for the train split. |
| `--num-test` | `1000` | For 4×4 / 6×6: how many puzzles to generate for the test split. |
| `--clue-ratio` | `None` | For 4×4 / 6×6: how many filled cells to leave in each puzzle. None defaults to ~30% of cells (e.g. ~4-5 clues for 4×4, ~10-11 for 6×6). Lower = harder. |
| `--num-aug` | `0` | Number of extra shuffled copies to generate per training puzzle (digit permutation + row/band/column shuffling). Augments the training set without new solves. |

### Secret Command for Building the datatset:
```bash
# For simple 4x4 sudoku problem
python3 TRM/dataset/build_sudoku_dataset.py --output-dir data/4x4-sudoku  --num-train 100 --num-test 10 --num-aug 0
```

## Running the dataset

### Magic command for running the dataset:
```bash
# Simple sudoku
run_name="pretrain_mlp_t_sudoku"
python TRM/pretrain.py \
arch=trm \
data_paths="[data/4x4-sudoku]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```