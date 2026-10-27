#!/usr/bin/env bash
# Round 27 (user): random starts. trm_ho3r = the plain TRM retrained with arch.random_initialization=true (otherwise
# trm_ho3's recipe, this branch and build). Meta-fine-tune with the trajectory query loss and M = 4 random starts per query
# row (meta_sgd.traj_starts=4: one cached target path per start, outer loss over all four), round 26's budget (lr 0.1, K 20,
# lambda 10) and round 26's better refresh (pick_traj_refresh.py). Every checkpoint evaluated twice at 1000 steps (ACT on):
# one start, no vote (num_views=1), and 4 restarts pooled into a majority vote (num_views=4), so a voting gain is not
# mistaken for a training gain. Run after round 26, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH arch.random_initialization=true"
pretrain trm_ho3r global_batch_size=128 max_steps=4000 eval_interval_steps=2000
BASE=$(ckpt_of trm_ho3r)
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2 +train_overrides.meta_sgd.segment_inner_steps=true"
for v in 1 4; do ttt trm_ho3r ms21t_v$v meta_sgd_steps=1000 num_views=$v $NORM_EVAL; done
REFRESH=$("$PY" "$WT/scripts/miniarc5/pick_traj_refresh.py" "$RUNS")
echo "[traj_refresh] $REFRESH"
R="+load_checkpoint=$BASE $FT3 $EPISODE3 meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=null meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000 meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.query_loss=trajectory meta_sgd.traj_lambda=10 meta_sgd.traj_lr=0.1 meta_sgd.traj_iters=20"
pretrain msgd27_traj_ms4 $R meta_sgd.traj_starts=4 meta_sgd.traj_refresh=$REFRESH
for v in 1 4; do ttt msgd27_traj_ms4 ms21m_v${v}_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment num_views=$v; done
echo "[done27] $(date +%H:%M:%S)"
