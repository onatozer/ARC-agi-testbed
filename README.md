# Meta-training recursive reasoning models for test-time training on ARC

Code for the meta-training + test-time training (TTT) experiments on ARC-style visual reasoning tasks. It trains small
recursive reasoning models (TRM, GRAM, HRM) and a ViT baseline (VARC), meta-trains them with Meta-SGD so they adapt
quickly to a new task's demo pairs, and evaluates them on held-out tasks after per-task test-time training.

Every run is a call to `pretrain.py` (training and meta-training), `evaluate_ttt.py` (test-time training + scoring) or
`evaluate.py` (plain evaluation). All settings live in [Hydra](https://hydra.cc) configs under `config/` and any field can
be overridden on the command line.

## Requirements

- Python 3.10, CUDA 12.6 (or similar)
- One GPU is enough for the MiniARC experiments.

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126  # match your CUDA version
pip install -r requirements.txt   # specific_requirements.txt pins the exact versions we used
```

Logging uses Weights & Biases. To run without an account, add `+wandb_mode=disabled` to `pretrain.py` and
`wandb_mode=disabled` to `evaluate.py` / `evaluate_ttt.py`.

## Repository layout

| Path | Contents |
|---|---|
| `pretrain.py` | Training loop; also the Meta-SGD meta-training step (`meta_sgd_batch`, enabled by `meta_sgd.enabled=true`) and masked-reconstruction rows (`recon.*`) |
| `evaluate_ttt.py` | Per-task test-time training from a checkpoint (ordinary Adam TTT, or Meta-SGD steps with the learned step sizes), then scoring |
| `evaluate.py` | Evaluation of a checkpoint without TTT |
| `config/` | Hydra configs: `cfg_pretrain*.yaml` (training entry points), `arch/` (one file per model), `dataset/` (build + load settings), `eval/` (TTT and eval recipes), `schema.py` (typed config, incl. `MetaSGDConfig` and `ReconConfig`) |
| `models/recursive_reasoning/` | TRM / HRM (`merged_arch.py`), GRAM (`gram.py`), TreeRM, TRM-root |
| `models/varc.py` | ViT baseline (VARC) |
| `models/losses/` | Loss heads wrapping each model (ACT, GRAM, and the `*RowWeight` heads used with reconstruction rows) |
| `dataset/` | Dataset builders (`build_miniarc_dataset.py`, `build_arc_dataset.py`, Sudoku, N-Queens, Maze) and the `PuzzleDataset` loader |
| `evaluators/` | Task-specific scoring (ARC pass@k with augmentation voting, Sudoku, N-Queens) |
| `data/MiniARC/` | The MiniARC tasks (5x5 grids) |
| `kaggle/combined/` | ARC-AGI-1, ARC-AGI-2 and ConceptARC task files |
| `scripts/` | Launchers for the ARC-AGI runs (`run_arc2.sh`, `run_arc1.sh` and their queues) |
| `tests/` | Unit tests (`pytest tests/`) |

## Main experiment: held-out MiniARC

The pipeline has three stages: (1) pretraining with masked-input reconstruction, (2) Meta-SGD meta-training, (3)
test-time training on each held-out task with the learned step sizes. Commands below are for TRM; see the end of the
section for VARC and GRAM.

### 0. Build the dataset

```bash
python -m dataset.build_miniarc_dataset grid_size=5 num_aug=31 heldout_tasks=30 data_path=data/miniarc5-aug-31-ho30
```

`heldout_tasks=30` keeps 30 tasks out of training entirely (only their test pairs are in the test split; their names are
written to `heldout_tasks.json`). The training tasks' test pairs become `validation` rows, which Meta-SGD uses as query
examples. `heldout_demos_in_train=true` gives the transductive variant where the held-out demos also join pretraining.

### 1. Pretraining with reconstruction rows

```bash
DATA=$PWD/data/miniarc5-aug-31-ho30
TRM="dataset.grid_size=5 arch.halt_max_steps=8 arch.L_cycles=4 arch.num_heads=4 arch.activation_function=conv_swiglu arch.puzzle_emb_len=1"
HEAD=arch.loss.name=losses.loss_heads@ACTLossHeadRowWeight

python pretrain.py --config-name cfg_pretrain_miniarc dataset.data_path=$DATA $TRM $HEAD \
  recon.copies=5 global_batch_size=128 max_steps=4000 +run_name=trm_pretrain
```

`recon.copies=5` adds 5 masked copies of every training input (`recon.fraction`, 24% of its color cells, set to a new
mask token; labelled with the full input, weighted 1/5) on one shared puzzle-embedding row. Checkpoints are written under the run's output directory
(`.../checkpoints/*.pth`).

### 2. Meta-SGD meta-training

```bash
python pretrain.py --config-name cfg_pretrain_miniarc dataset.data_path=$DATA $TRM $HEAD \
  +load_checkpoint=<stage-1 checkpoint> +load_weights_only=true \
  global_batch_size=256 lr=1e-4 lr_warmup_steps=50 max_steps=500 ema=false \
  recon.copies=1 meta_sgd.enabled=true meta_sgd.segment_inner_steps=true +run_name=trm_meta
```

Each task in a batch (16 augmented copies, `meta_sgd.copies_per_group`) is one episode: its puzzle-embedding row starts
from the table mean, it takes `meta_sgd.inner_steps` RMS-normalised inner steps on its demos (plus one masked copy of each
demo input) with a learned step size per tensor and per step, and the query loss on its validation pair updates the
weights and step sizes (first order). Optionally, `meta_sgd.recon_steps=K` in a second meta-training run from this
checkpoint learns the sizes of K reconstruction steps (used at test time in place of the fixed 0.01x phase below). Defaults of `MetaSGDConfig` in `config/schema.py` are the values we used.
`segment_inner_steps=true` is for the multi-segment models (TRM, GRAM): one supervision segment per inner step.

### 3. Test-time training on the held-out tasks

```bash
python evaluate_ttt.py checkpoint=<stage-2 checkpoint> eval=ttt_mini \
  task_ids_file=data/miniarc5-aug-31-ho30/heldout_tasks.json meta_sgd_steps=1000 recon_steps=20
```

`meta_sgd_steps=1000` takes 1000 TTT steps with the learned step sizes stretched over the loop (step size `k` of `K` is
used for steps `k*1000/K` to `(k+1)*1000/K`), followed by 20 reconstruction steps on the masked demo and test inputs at
0.01x those step sizes (`recon_alpha_scale`). Results (pass@k, per-task predictions) are written to
`<run_dir>/eval_preds/step_<N>_<suffix>/`, one file per task, so an interrupted run resumes.

### Baselines

- **Pretraining only**: stage 1 without `recon.*`/`$HEAD`, scored with `evaluate.py checkpoint=<ckpt>`.
- **Ordinary fine-tuning TTT** (Adam, as in VARC): `evaluate_ttt.py checkpoint=<stage-1 checkpoint> eval=ttt_mini task_ids_file=...`
  (`meta_sgd_steps=0`, the default). `eval=ttt_varc` is the full VARC recipe (51 augmentations, 100 epochs, 2 runs, 10 views).

### Other architectures

- **VARC (ViT)**: `--config-name cfg_pretrain_miniarc_varc` with
  `dataset.grid_size=5 arch.image_size=10 arch.hidden_size=128 arch.num_layers=4 arch.mlp_dim=256 arch.num_heads=8`,
  the same `$HEAD`, and in stage 2 `meta_sgd.inner_steps=5 meta_sgd.msl_anneal_fraction=0.3` instead of
  `meta_sgd.segment_inner_steps=true`.
- **GRAM**: `--config-name cfg_pretrain_miniarc arch=gram` with
  `arch.loss.name=losses.loss_heads@LossHeadGRAMRowWeight` and `meta_sgd.segment_inner_steps=true`.

## Other datasets

```bash
python -m dataset.build_arc_dataset                                   # ARC-AGI-1 (settings in config/dataset/arc.yaml)
python -m dataset.build_arc_dataset data_path=data/arc2concept-aug-1000 \
  'subsets=[training2,evaluation2,concept]' test_set_name=evaluation2  # ARC-AGI-2
python -m dataset.build_miniarc_dataset                               # MiniARC, all tasks (config/dataset/miniarc.yaml)
python dataset/build_sudoku_dataset.py grid_size=4 num_train=5000 output_dir=data/sudoku-4x4  # Sudoku (config/dataset/sudoku.yaml)
```

`heldout_test_set=true` keeps the test set's demos out of training (the held-out protocol of the MiniARC runs).
`scripts/run_arc2.sh` runs the full recipe on ARC-AGI-2 from the official ARC-AGI-1 TRM checkpoint (build, pretraining,
meta-training and test time; see the header of the script).

The ARC builders read the same dataset yaml used for training, so `python pretrain.py --config-name cfg_pretrain_arc`
(or `cfg_pretrain_miniarc`) finds the built data without extra flags. `scale_aug=true` randomly upscales training
examples by an integer factor, following VARC; the test split is never scaled.

## Notes

- **Config structure.** `cfg_pretrain.yaml` holds the training settings and picks one file from `config/arch/` and one from
  `config/dataset/`; the `cfg_pretrain_<dataset>.yaml` files override these per dataset. Switch model with `arch=<name>`
  and override nested fields with dots (`arch.hidden_size=256`). Fields not declared in the yaml need a leading `+`.
- **Model / loss head.** Each model is wrapped by a loss head (`arch.loss.name`), which computes the loss and the logged
  metrics, so the same architecture can be trained with different objectives.
- **Multi-GPU.** `torchrun --nproc_per_node=<gpus> pretrain.py ...`; `evaluate_ttt.py` splits tasks across GPUs the same way.
- **Resuming.** `+load_checkpoint=<path>` resumes a run; add `+load_weights_only=true` to start a new run from its weights.
  With wandb >= 0.30, resuming a run crashes on `wandb.run._step`; use `wandb==0.27.2` (pinned in `specific_requirements.txt`).
- **Tests.** `pytest tests/`. The Meta-SGD, reconstruction and held-out-build tests are `tests/test_meta_sgd.py`,
  `tests/test_build_miniarc_heldout.py` and `tests/test_evaluate_ttt.py`.
