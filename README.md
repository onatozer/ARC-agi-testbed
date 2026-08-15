# Installing and running the code
Before you can train or run the model, first you must build the dataset using the corresponding builder from the [dataset directory](#building-the-dataset). As an example, in order to build the ARC-AGI datasets, run:
```bash
# ARC-AGI-1
python3 -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --num-aug 1000 \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2
```

The entire codebase is designed so that a training run can be initiated through a simple call of pretrain.py from main. 
```python
python3 pretrain.py --config-name cfg_pretrain
```
All of the numerous hyperparameter, architectural configurations, or dataset changes are managed through config files stored in the config directory and managed through the hydra package. It is recommended that you control these values by altering and managing which config file you pass into pretain.py. More details on how the config directory is setup can be found [here](#config-directory). 


The configs are nested so that you can specify configs for each section of the 
## Requirements

Running the code requires the following python and Cuda versions. If any version error occurs during installation, specific_requirements.txt is also provided. 

- Python 3.10 (or similar)
- Cuda 12.6.0 (or similar)

Run the following commands after setting up your python virtual environment:

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # install torch based on your cuda version
pip install -r requirements.txt # install requirements
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # login if you want the logger to sync results to your Weights & Biases (https://wandb.ai/)
```

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

## Training the Model


# Notes on codebase

Models is structured so that the base architecture is wrapped around by the loss head. This allows for us to experiment with different types of loss functions on the same architecture. The loss head also defines the metrics that are output for logging. General configs, ie for the dataset and training are defined in the config directory, whereas the model specific configs are defined in their respective files. Specific model configs can still be set my the yaml files in the config directory


A PuzzleDataset can be initialized with 'shared_sampler_state', which is just a dictionary (created by instantiating a 'Manager' from the mp library and calling .dict() ), that contains the following entries:
```python
# From the PuzzleDataset class in puzzle_dataset.py
def _build_state_snapshot(self) -> Dict[str, Any]:
    return {
        "split": self.split,
        "_iters": self._iters,
        "train_state": self._normalize_train_state(self._train_state),
    }
```
This gives the necessary information for distributed training to be resumed from any state.

## Directory structure
ATM, the directory structure looks like this. 
```bash
├── DIS
├── assets
├── config # Hydra configs for any possible setting you would want to tweak in the model run
│   ├── arch
│   ├── dataset
│   ├── eval
│   └── train
├── data
│   ├── 4x4-sudoku
│   ├── MiniARC
│   ├── MiniARC_traces
│   ├── arc1concept-aug-1000
│   ├── nqueens
│   └── sudoku-extreme-1k-aug-0
├── dataset
├── evaluators
├── kaggle
│   └── combined
# This is where the actual model architecture itself gets defined. recursive_reasoning contains the recursive architectures, which are wrapped around by the various loss heads defined in losses, and outside the directory are the other miscellaneous stuff used throughout, ie the layers
├── models
│   ├── losses
│   └── recursive_reasoning
├── outputs
│   └── 2026-06-21
├── queens
├── runs
├── scripts
├── train
├── utils
├── venv
│   ├── bin
│   ├── lib
│   └── share
├── visualization
└── wandb
    ├── latest-run -> run-20260617_134706-cpaz0iss
    └── run-20260617_134706-cpaz0iss
├── pretrain.py
├── evaluate.py
```
## Config Directory





## Scripts Directory
Basically just wrappers that make it more convinient to train, eval and setup the dataset/model weights of the model. 

Basic usage for 'train.sh' is:
```bash
train.sh <pretrain_cfg> any_additional_cfg_params=1
```
The train.sh is written so that pretrain cfg can be passed in as the exact path, or just the config file name itself, and the script will still work. If you want to alter any of the config arguments, you can simply pass them in through the terminal, and hydra manages them so that they properly override the cfg value. You can also specify the torchrun kwargs, such as:

```bash
train.sh <pretrain_cfg> --standalone # Use to tell Pytorch that you're running the code on single device 
--nproc-per-node=1 # Use to tell pytorch how many GPUs you're using per device  
```

Similar design process goes into the usage of eval.sh
```bash
scripts/eval.sh <checkpoint-path> <eval_cfg> [evaluate.py overrides...]
``` 

downlaod_arifacts.sh downloads the model & dataset paths from hugging-face, containing the following args:
| Flag                  | Effect                                            |
| --------------------- | ------------------------------------------------- |
| `--with-ckpts`        | Download checkpoints **in addition to** data      |
| `--ckpts-only`        | Download **only** checkpoints                     |
| `--skip-data`         | Skip data download                                |
| `--ckpt-pattern GLOB` | Regex that allows you to specify which checkpoints to accept |
| `--data-repo REPO`    | Override HF dataset repo                          |
| `--ckpt-repo REPO`    | Override HF model repo                            |
| `--cache-dir DIR`     | Override cache location                           |
| `--data-dir DIR`      | Override data destination                         |
| `--ckpt-dir DIR`      | Override checkpoint destination                   |


## Things to clean up:
Not all of the files in the utils directory are being used atm. Train directory, I'm pretty sure doesn't need to be used at all, some of the scripts are likely no longer relevant, maybe we want to move the ARC and mini-ARC datasets into the same directory. 

Losses directory can likely just be consolidated into a single file, don't clearly see a reason atm for why it should be structured the way it is rn. I also think that we can streamline the model-running process even further by putting some type of catch in pretrain.py that builds the dataset according to the hydra config rather rather than forcing the user to have to manually call the build_dataset function and track what they named it to. 
