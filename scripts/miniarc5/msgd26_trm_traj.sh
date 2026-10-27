#!/usr/bin/env bash
# Round 26 (user): the trajectory query loss with the fixed target budget (lr 0.1, K 20, lambda 10: every training query
# row's target path decodes to the correct output; REPORTS_SGD.md "the target budget"), targets cached from the current
# unadapted weights with each puzzle's own embedding row. From trm_ho3 on round 21's recipe (ACT on, 8 segment steps,
# alpha per tensor and per step, first order, normalised, init 1e-3 / 3e-2), MSL off.
#   msgd26_traj_once  targets built once, before meta-step 1 (traj_refresh=1)
#   msgd26_traj_q25   rebuilt every 25% of meta-training
# Each evaluated with ACT on and the segment schedule at 1000 steps.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
BASE=$(ckpt_of trm_ho3)
R="+load_checkpoint=$BASE $FT3 $EPISODE3 meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=null meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000 meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.query_loss=trajectory meta_sgd.traj_lambda=10 meta_sgd.traj_lr=0.1 meta_sgd.traj_iters=20"
pretrain msgd26_traj_once $R meta_sgd.traj_refresh=1
ttt msgd26_traj_once ms20o_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
pretrain msgd26_traj_q25 $R meta_sgd.traj_refresh=0.25
ttt msgd26_traj_q25 ms20q_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
echo "[done26] $(date +%H:%M:%S)"
