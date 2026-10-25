#!/usr/bin/env bash
# Round 22 (user): round 21 (TRM, ACT on throughout) with meta_sgd.halt_wait: a halted row sits out until its trajectory
# ends (every 8 inner steps), then all rows restart together, in meta-fine-tuning and in the adaptation loop, so the
# trajectory-indexed alphas see exactly their segment position. Control: plain trm_ho3 under the same rule (ACT on,
# halt_wait, hand-set alphas). 1000 steps each. Run after round 21, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
BASE=$(ckpt_of trm_ho3)
NORM="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true meta_sgd.halt_wait=true"
MULTI="meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3 meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2 +train_overrides.meta_sgd.segment_inner_steps=true +train_overrides.meta_sgd.halt_wait=true"
ttt trm_ho3 ms16twait_s1000 meta_sgd_steps=1000 $NORM_EVAL
pretrain msgd22_trm_wait +load_checkpoint=$BASE $FT3 $EPISODE3 $NORM $MULTI meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2
ttt msgd22_trm_wait ms16m_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
echo "[done22] $(date +%H:%M:%S)"
