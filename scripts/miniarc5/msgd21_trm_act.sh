#!/usr/bin/env bash
# Round 21 (user): TRM with ACT halting on throughout (pretraining, meta-fine-tuning and the adaptation loop). Round 18's
# recipe from trm_ho3 (8 segment inner steps, alpha per tensor and per step, MSL, first order, normalised, alpha init
# 1e-3 / 3e-2) without arch.disable_act_halt. Evaluated with the trajectory-indexed (segment) schedule at 1000 steps,
# against plain trm_ho3 with ACT on (round 19: 12/14/14, .835).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
BASE=$(ckpt_of trm_ho3)
NORM="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true"
MULTI="meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3 meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000"
pretrain msgd21_trm_act +load_checkpoint=$BASE $FT3 $EPISODE3 $NORM $MULTI meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2
ttt msgd21_trm_act ms15m_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
echo "[done21] $(date +%H:%M:%S)"
