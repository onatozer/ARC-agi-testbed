#!/usr/bin/env bash
# Second-order Meta-SGD on VARC / MiniARC with 30 held-out tasks: the VARC grid of l-reinit-ttt's REPORTS_MAML.md, re-run with
# the meta-sgd-step implementation (pretrain.py meta_sgd_batch, evaluate_ttt.py meta_sgd_steps). One GPU job at a time.
#   bash scripts/miniarc5/msgd2_varc.sh            # runs everything that has no result yet
# Baseline checkpoint varc_ho2 (plain VARC, 4000 steps on the same build) comes from the previous loop and is reused.
set -euo pipefail
WT=$(cd "$(dirname "$0")/../.." && pwd)
RUNS=${MINIARC5_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5}
DATA_HO=$WT/data/miniarc5-aug-31-ho30
HO=$DATA_HO/heldout_tasks.json
PY=$WT/venv/bin/python
ENV="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_SILENT=true"

# The previous loop's VARC: 10x10 canvas, hidden 128, 4 layers (cfg_pretrain_miniarc_varc.yaml has since grown)
VARC_ARCH="dataset.grid_size=5 arch.image_size=10 arch.hidden_size=128 arch.num_layers=4 arch.mlp_dim=256 arch.num_heads=8"
# The previous loop's meta-training recipe: batch 16 (= the meta-batch here: every puzzle in it is a task), 2000 steps
COMMON="global_batch_size=16 optimizer=adam_atan2 lr=1e-3 lr_warmup_steps=100 lr_min_ratio=1.0 epochs=100000 train_epochs_per_iter=100000 \
max_steps=2000 eval_interval_steps=1000 max_eval_steps=null seed=0 puzzle_emb_lr=3e-2 checkpoint_interval_steps=null \
heavy_metrics_log_interval=100000 steps_hist_log_interval_steps=100000 flops_log_interval_steps=100000 +project_name=ARC-agi-testbed +entity=ozeronatwanb"
MSGD="meta_sgd.enabled=true meta_sgd.alpha_lr=1e-3"

ckpt_of() { ls -t "$RUNS/$1"/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }

pretrain() {  # name, overrides...
  local name=$1; shift
  local dir=$RUNS/$name
  if [ -n "$(ckpt_of "$name")" ]; then echo "[skip] $name has a checkpoint"; return; fi
  mkdir -p "$dir"; echo "[train] $name $(date +%H:%M:%S)"
  (cd "$WT" && env $ENV OUTPUT_ROOT="$dir" /usr/bin/time -v "$PY" pretrain.py --config-name cfg_pretrain_miniarc_varc \
    dataset.data_path="$DATA_HO" $VARC_ARCH $COMMON +run_name="$name" "$@") > "$dir/train.log" 2>&1
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

# Baseline through the new eval path: plain ttt_mini, and fixed-alpha inner steps on the rows / the layers (the previous picks)
ttt varc_ho2 ms4v_adam
for n in 5 10; do
  ttt varc_ho2 ms4ve_s$n meta_sgd_steps=$n +train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.inner_lr=1 "+train_overrides.meta_sgd.adapt_substrings=[puzzle_emb]"
  ttt varc_ho2 ms4vl_s$n meta_sgd_steps=$n +train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.inner_lr=0.01 "+train_overrides.meta_sgd.adapt_substrings=[inner.layers.]"
done

# Meta-SGD: rows only (alpha0 1), layers only (alpha0 0.01), and the paper's default of everything (alpha0 0.01, learned per element)
pretrain msgd2_varc_emb $MSGD meta_sgd.inner_lr=1 "meta_sgd.adapt_substrings=[puzzle_emb]"
grid msgd2_varc_emb ms4ve
pretrain msgd2_varc_lh $MSGD meta_sgd.inner_lr=0.01 "meta_sgd.adapt_substrings=[inner.layers.]"
grid msgd2_varc_lh ms4vl
pretrain msgd2_varc_all $MSGD meta_sgd.inner_lr=0.01
grid msgd2_varc_all ms4va
# Follow-ups (user, 2026-10-23): the data-budget confound, on the all-parameters variant only.
# (a) Meta-SGD from the plain checkpoint for 2000 meta-steps, against the plain checkpoint continued for the same steps
PLAIN=$(ckpt_of varc_ho2)
pretrain msgd2_varc_all_ft $MSGD meta_sgd.inner_lr=0.01 +load_checkpoint="$PLAIN" +load_weights_only=true
pretrain varc_ho2_cont +load_checkpoint="$PLAIN" +load_weights_only=true
ttt msgd2_varc_all_ft ms4vaf_adam
ttt msgd2_varc_all_ft ms4vaf_s10 meta_sgd_steps=10
ttt varc_ho2_cont ms4vc_adam
# (b) Meta-SGD from scratch at the baseline's budget: 4000 steps x batch 128 (~30 tasks per meta-step)
pretrain msgd2_varc_all_full $MSGD meta_sgd.inner_lr=0.01 global_batch_size=128 max_steps=4000 eval_interval_steps=2000
ttt msgd2_varc_all_full ms4vafull_adam
ttt msgd2_varc_all_full ms4vafull_s10 meta_sgd_steps=10
echo "[done] $(date +%H:%M:%S)"
