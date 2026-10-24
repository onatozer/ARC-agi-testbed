#!/usr/bin/env bash
# Round 18 (user): TRM with segment-indexed alphas and ACT halting off, from trm_ho3 (plain TRM retrained on this branch and
# build with trm_ho2's recipe, resolved configs diffed). Meta-fine-tune: 8 segment inner steps (one whole 8-segment
# trajectory), alpha per tensor and per step, MSL, first order, normalised step, arch.disable_act_halt (rows run all 8
# segments in lockstep, also at test time). Alpha init 1e-3 / 3e-2 (control), 0.1 and 1 (user). Evaluated with
# meta_sgd_schedule=segment at 120 / 300 / 1000. Controls on trm_ho3: ttt_mini, and the same segment-step rule with ACT off
# and the hand-set alphas.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
pretrain trm_ho3 global_batch_size=128 max_steps=4000 eval_interval_steps=2000
BASE=$(ckpt_of trm_ho3)
NORM="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10 meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true +arch.disable_act_halt=true"
MULTI="meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3 meta_sgd.inner_steps=8 meta_sgd.first_order_steps=100000"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2 +train_overrides.meta_sgd.segment_inner_steps=true +train_overrides.arch.disable_act_halt=true"
ttt trm_ho3 ms14t_adam
for n in 120 300 1000; do ttt trm_ho3 ms14tnorm_s$n meta_sgd_steps=$n $NORM_EVAL; done
for spec in "msgd18_t_a1e-3 1e-3 3e-2 ms14a3" "msgd18_t_a0.1 0.1 0.1 ms14a01" "msgd18_t_a1 1 1 ms14a1"; do
  set -- $spec
  pretrain $1 +load_checkpoint=$BASE $FT3 $EPISODE3 $NORM $MULTI meta_sgd.inner_lr=$2 meta_sgd.inner_lr_rows=$3
  for n in 120 300 1000; do ttt $1 ${4}_sg$n meta_sgd_steps=$n meta_sgd_schedule=segment; done
done
echo "[done18] $(date +%H:%M:%S)"
