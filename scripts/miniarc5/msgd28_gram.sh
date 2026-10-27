#!/usr/bin/env bash
# Round 28 (user): GRAM in TRM's place, architecture swap only (no change to its posterior). gram_ho3 = GRAM pretrained on
# this branch and build with trm_ho3's architecture and recipe (4000 x 128). GRAM's code reads seven fields
# config/arch/gram.yaml does not define (a pre-existing gap on this branch and main); they are passed here with the earlier
# GRAM runs' values. GRAM has no halt head (always 8 segments) and samples its latent from the posterior (which reads the
# labels) in training mode and from the prior in eval mode, as it normally trains. Meta-fine-tune: round 26's trajectory
# recipe and better refresh, one start. Controls on gram_ho3: ttt_mini and the normalised segment rule with the hand-set
# alphas. 1000 steps. Run after round 27, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
GRAM_FIELDS="+arch.kl_reduction=sum +arch.lprm_prior_rollouts=0 +arch.lprm_rollout_cycles=grad +arch.posterior_nograd_cycles=false +arch.prior_sample_frac=0.0 +arch.prior_sample_per_puzzle=false +arch.value_bootstrap_weight=0.5"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="arch=gram $TRM_ARCH $GRAM_FIELDS"
pretrain gram_ho3 global_batch_size=128 max_steps=4000 eval_interval_steps=2000
BASE=$(ckpt_of gram_ho3)
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2 +train_overrides.meta_sgd.segment_inner_steps=true"
ttt gram_ho3 ms22g_adam
ttt gram_ho3 ms22gnorm_s1000 meta_sgd_steps=1000 $NORM_EVAL
REFRESH=$("$PY" "$WT/scripts/miniarc5/pick_traj_refresh.py" "$RUNS")
echo "[traj_refresh] $REFRESH"
R="+load_checkpoint=$BASE $FT3 $EPISODE3 meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=null meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000 meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.query_loss=trajectory meta_sgd.traj_lambda=10 meta_sgd.traj_lr=0.1 meta_sgd.traj_iters=20"
pretrain msgd28_gram_traj $R meta_sgd.traj_refresh=$REFRESH
ttt msgd28_gram_traj ms22m_sg1000 meta_sgd_steps=1000 meta_sgd_schedule=segment
echo "[done28] $(date +%H:%M:%S)"
