#!/usr/bin/env bash
# Shared definitions of the Meta-SGD / VARC / MiniARC-ho30 queues (msgd2_varc.sh, msgd3_varc.sh): paths, the previous
# loop's VARC model and meta-training recipe, and the pretrain / ttt launchers (one GPU job at a time; both skip finished work).
WT=${WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
RUNS=${MINIARC5_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5}
DATA_HO=$WT/data/miniarc5-aug-31-ho30
HO=$DATA_HO/heldout_tasks.json
PY=$WT/venv/bin/python
ENV="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_SILENT=true"

# The previous loop's VARC: 10x10 canvas, hidden 128, 4 layers (cfg_pretrain_miniarc_varc.yaml has since grown)
VARC_ARCH="dataset.grid_size=5 arch.image_size=10 arch.hidden_size=128 arch.num_layers=4 arch.mlp_dim=256 arch.num_heads=8"
# The previous loop's TRM (trm_ho2): 8 segments, H 3 / L 4, 2 layers, hidden 128, 4 heads (resolved configs diffed)
TRM_ARCH="dataset.grid_size=5 arch.halt_max_steps=8 arch.L_cycles=4 arch.num_heads=4 arch.activation_function=conv_swiglu arch.puzzle_emb_len=1"
# The previous loop's meta-training recipe: batch 16 (= the meta-batch: every puzzle in it is a task), 2000 steps
COMMON="global_batch_size=16 optimizer=adam_atan2 lr=1e-3 lr_warmup_steps=100 lr_min_ratio=1.0 epochs=100000 train_epochs_per_iter=100000 \
max_steps=2000 eval_interval_steps=1000 max_eval_steps=null seed=0 puzzle_emb_lr=3e-2 checkpoint_interval_steps=null \
heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000 +project_name=ARC-agi-testbed +entity=ozeronatwanb"
MSGD="meta_sgd.enabled=true meta_sgd.alpha_lr=1e-3"
# Round 3's recipe (REPORTS_SGD.md post-mortem fixes 1-4), from the plain checkpoint: its batch size, a tenth of its lr, 1000
# steps; the task with 16 augmented copies as the episode, fresh rows, alpha initialised per group (rows 1, weights 0.01)
FT3="+load_weights_only=true global_batch_size=256 lr=1e-4 lr_warmup_steps=50 max_steps=1000 eval_interval_steps=500"
EPISODE3="meta_sgd.enabled=true meta_sgd.copies_per_group=16 meta_sgd.fresh_rows=true meta_sgd.inner_lr=0.01 meta_sgd.inner_lr_rows=1 meta_sgd.alpha_lr=1e-2"

ckpt_of() { ls -t "$RUNS/$1"/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }

pretrain() {  # name, overrides... (later overrides of the same key win)
  local name=$1; shift
  local dir=$RUNS/$name
  if [ -n "$(ckpt_of "$name")" ]; then echo "[skip] $name has a checkpoint"; return; fi
  mkdir -p "$dir"; echo "[train] $name $(date +%H:%M:%S)"
  # PRETRAIN_CFG / ARCH: another model (e.g. TRM: cfg_pretrain_miniarc and TRM_ARCH); default the VARC above
  (cd "$WT" && env $ENV OUTPUT_ROOT="$dir" /usr/bin/time -v "$PY" pretrain.py --config-name "${PRETRAIN_CFG:-cfg_pretrain_miniarc_varc}" \
    dataset.data_path="$DATA_HO" ${ARCH-$VARC_ARCH} $COMMON +run_name="$name" "$@") > "$dir/train.log" 2>&1
}

ttt() {  # run, suffix, evaluate_ttt overrides...  (evaluate_ttt.py skips tasks that already have a result)
  local run=$1 suffix=$2; shift 2
  local ckpt; ckpt=$(ckpt_of "$run")
  local rundir; rundir=$(dirname "$(dirname "$ckpt")")
  if [ -f "$rundir/eval_preds/$(basename "$ckpt" | sed 's/^\(step_[0-9]*\)_.*/\1/')_$suffix/ttt_metrics.json" ]; then echo "[skip] $run/$suffix"; return; fi
  echo "[ttt] $run/$suffix $(date +%H:%M:%S)"
  (cd "$WT" && env $ENV /usr/bin/time -v "$PY" evaluate_ttt.py checkpoint="$ckpt" eval=ttt_mini suffix="$suffix" task_ids_file="$HO" "$@") > "$rundir/ttt_$suffix.log" 2>&1
}

grid() {  # run, prefix: the learned inner loop for 1/5/10 steps, and the ordinary ttt_mini fine-tune from the same checkpoint
  local run=$1 pfx=$2
  for n in 1 5 10; do ttt "$run" "${pfx}_s$n" meta_sgd_steps=$n; done
  ttt "$run" "${pfx}_adam"
}
