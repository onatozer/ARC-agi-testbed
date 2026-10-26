#!/usr/bin/env bash
# Round 25 (user): learned latent trajectory targets as the TRM Meta-SGD query loss (meta_sgd.query_loss=trajectory;
# REPORTS_SGD.md "Learned trajectory targets"). From trm_ho3 with round 21's recipe (ACT on, 8 segment steps, alpha per
# tensor and per step, first order, normalised, init 1e-3 / 3e-2), MSL off, lambda 10 (diagnostic).
#   msgd25_traj_k10    targets re-optimised every episode at the adapted weights, K = 10 (smoke: ~60-100 s / meta-step)
#   msgd25_traj_cache  targets cached from the current unadapted weights (each puzzle's own embedding row), K = 20,
#                      rebuilt every 25% of training (~75 min per rebuild)
# Each evaluated with ACT on and the segment schedule at 1000 steps, against plain trm_ho3 (12/14/14, .835) and round 21's
# CE-meta checkpoint (11/11/13, .832). The per-step run first (user).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
BASE=$(ckpt_of trm_ho3)
R="+load_checkpoint=$BASE $FT3 $EPISODE3 meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=null meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000 meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.query_loss=trajectory meta_sgd.traj_lambda=10"
pretrain msgd25_traj_k10 $R meta_sgd.traj_iters=10
ttt msgd25_traj_k10 ms19k_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
pretrain msgd25_traj_cache $R meta_sgd.traj_iters=20 meta_sgd.traj_refresh=0.25
ttt msgd25_traj_cache ms19c_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
echo "[done25] $(date +%H:%M:%S)"
