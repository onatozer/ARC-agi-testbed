#!/usr/bin/env bash
# ARC-AGI-1, TRM at TRM's ARC size, the PR #70 recipe with 1 reconstruction copy, one seed, on one 7-GPU node.
#   MODE=pre  bash scripts/run_arc1.sh [overrides...]          reconstruction pretraining (TRM's ARC recipe: batch ~768, 100k epochs)
#   MODE=meta bash scripts/run_arc1.sh <pretrain ckpt> [...]   stage-1 Meta-SGD meta-training
#   MODE=eval bash scripts/run_arc1.sh <meta ckpt> [...]       test time: 1000 stretched steps + 20 reconstruction steps, 400 eval tasks
set -euo pipefail
cd "$(dirname "$0")/.."

D=$PWD/data/arc1concept-aug-1000-ho  # python -m dataset.build_arc_dataset heldout_test_set=true data_path=<D>
R=$PWD/runs
RUN=${RUN:-arc1_recon1}
export OMP_NUM_THREADS=4 WANDB_SILENT=true
LAUNCH="python -m torch.distributed.run --standalone --nproc_per_node ${GPUS:-7}"

TRM="arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.loss.name=losses.loss_heads@ACTLossHeadRowWeight"
# 770 = 7 x 110, the nearest multiple of 7 to TRM's 768
COMMON="--config-name cfg_pretrain_arc dataset.data_path=$D $TRM recon.copies=1 seed=0 +wandb_mode=offline \
  global_batch_size=770 weight_decay=0.1 puzzle_emb_weight_decay=0.1 puzzle_emb_lr=1e-2"
mkdir -p $R

case ${MODE:-pre} in
  pre)
    OUTPUT_ROOT=$R/${RUN}_pre $LAUNCH pretrain.py $COMMON +run_name=${RUN}_pre lr=1e-4 lr_warmup_steps=2000 \
      epochs=100000 max_steps=null eval_interval_steps=null checkpoint_interval_steps=20000 ema=true "$@" ;;
  meta)
    OUTPUT_ROOT=$R/${RUN}_meta $LAUNCH pretrain.py $COMMON +run_name=${RUN}_meta +load_checkpoint=$1 +load_weights_only=true \
      lr=1e-4 lr_warmup_steps=50 epochs=100000 max_steps=${META_STEPS:-550} eval_interval_steps=null ema=false \
      meta_sgd.enabled=true meta_sgd.segment_inner_steps=true "${@:2}" ;;
  eval)
    $LAUNCH evaluate_ttt.py checkpoint=$1 eval=ttt_mini output_dir=$R/${RUN}_eval meta_sgd_steps=1000 recon_steps=20 wandb_mode=disabled "${@:2}" ;;
esac
