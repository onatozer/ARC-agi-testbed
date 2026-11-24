#!/usr/bin/env bash
# ARC-AGI-2, TRM continued from the official ARC-AGI-1 checkpoint (arcprize/trm_arc_prize_verification arc_v1_public), the
# PR #70 recipe with 1 reconstruction copy, one seed, on one 7-GPU node:
#   MODE=build bash scripts/run_arc2.sh                     both held-out ARC-AGI-2 builds
#   MODE=pre   bash scripts/run_arc2.sh [overrides...]       continued pretraining: 233 new tasks + ~10% replay, 100k epochs
#   MODE=meta  bash scripts/run_arc2.sh <pretrain ckpt> [...] stage-1 Meta-SGD meta-training on all 1,160 training tasks
#   MODE=eval  bash scripts/run_arc2.sh <meta ckpt> [...]     test time on the 120 ARC-AGI-2 eval tasks
set -euo pipefail
set -f  # the Hydra list overrides below are not globs
cd "$(dirname "$0")/.."

V1=${V1:-/workspace/hf/arc_v1_public/step_518071}
D_PRE=${D_PRE:-$PWD/data/arc2-continue-aug-1000-ho}  # scripts/build_arc2_continue.py
D_META=${D_META:-$PWD/data/arc2concept-aug-1000-ho}   # dataset.build_arc_dataset, held out
NUM_AUG=${NUM_AUG:-1000}
R=${R:-$PWD/runs}
RUN=${RUN:-arc2_recon1}
export OMP_NUM_THREADS=4 WANDB_SILENT=true
LAUNCH="python -m torch.distributed.run --standalone --nproc_per_node ${GPUS:-7}"

# The checkpoint's dataset fields: evaluate_ttt.py builds each eval task from them (ARC-AGI-2's eval set, no scale augmentation)
DATA_FIELDS="dataset.test_set_name=evaluation2 dataset.subsets=[training2,concept,evaluation2] dataset.heldout_test_set=true dataset.scale_aug=false"
# Official TRM's ARC model (its update rule: lambda_ 1) with the row-weighting head
TRM="arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.lambda_=1.0 arch.loss.name=losses.loss_heads@ACTLossHeadRowWeight"
# 770 = 7 x 110, the nearest multiple of 7 to TRM's 768
COMMON="--config-name cfg_pretrain_arc $DATA_FIELDS $TRM recon.copies=1 seed=0 +wandb_mode=offline \
  global_batch_size=770 weight_decay=0.1 puzzle_emb_weight_decay=0.1 puzzle_emb_lr=1e-2 eval_interval_steps=null"
mkdir -p $R logs

case ${MODE:-pre} in
  build)
    PYTHONPATH=. python scripts/build_arc2_continue.py $D_PRE $NUM_AUG
    python -m dataset.build_arc_dataset "subsets=[training2,concept,evaluation2]" test_set_name=evaluation2 heldout_test_set=true \
      scale_aug=false seed=42 num_aug=$NUM_AUG data_path=$D_META ;;
  pre)
    OUTPUT_ROOT=$R/${RUN}_pre $LAUNCH pretrain.py $COMMON dataset.data_path=$D_PRE +run_name=${RUN}_pre +load_checkpoint=$V1 +load_weights_only=true \
      lr=1e-4 lr_warmup_steps=2000 epochs=1000000 max_steps=${PRE_STEPS:-146000} checkpoint_interval_steps=20000 ema=true "$@" ;;
  meta)
    OUTPUT_ROOT=$R/${RUN}_meta $LAUNCH pretrain.py $COMMON dataset.data_path=$D_META +run_name=${RUN}_meta +load_checkpoint=$1 +load_weights_only=true \
      lr=1e-4 lr_warmup_steps=50 epochs=1000000 max_steps=${META_STEPS:-1600} ema=false \
      meta_sgd.enabled=true meta_sgd.segment_inner_steps=true "${@:2}" ;;
  eval)
    $LAUNCH evaluate_ttt.py checkpoint=$1 eval=ttt_mini output_dir=$R/${RUN}_eval meta_sgd_steps=1000 recon_steps=20 wandb_mode=disabled "${@:2}" ;;
esac
