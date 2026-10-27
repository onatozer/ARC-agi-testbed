#!/usr/bin/env bash
# Round 29 (user): round 26's built-once trajectory recipe with the three post-mortem fixes together:
#   1. targets cached with the table-mean row (derived from meta_sgd.fresh_rows), the episode's own starting point
#   2. traj_lambda_growth: stiffer late consistency links, chosen by traj_diag_fixes.py (largest Delta_1 / Delta_N with
#      >= 95% of target paths decoding correctly)
#   3. traj_ce_weight=1: the query CE joins the matching loss, so lm_head / q_head get a meta-gradient
# From trm_ho3 (this branch), evaluated with ACT on and the segment schedule at 1000 steps.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
LOG=autoresearch_runs/miniarc5/traj_diag_fixes.log
venv/bin/python scripts/miniarc5/traj_diag_fixes.py > "$LOG" 2>&1
G=$(sed -n 's/^chosen growth: //p' "$LOG")
echo "[growth] $G"
BASE=$(ckpt_of trm_ho3)
R="+load_checkpoint=$BASE $FT3 $EPISODE3 meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=null meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000 meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.query_loss=trajectory meta_sgd.traj_lambda=10 meta_sgd.traj_lr=0.1 meta_sgd.traj_iters=20 meta_sgd.traj_refresh=1"
pretrain msgd29_traj_fixes $R meta_sgd.traj_lambda_growth=$G meta_sgd.traj_ce_weight=1
ttt msgd29_traj_fixes ms23f_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
echo "[done29] $(date +%H:%M:%S)"
