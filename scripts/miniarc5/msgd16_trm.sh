#!/usr/bin/env bash
# Round 16 (user): Meta-SGD on TRM, tractable through meta_sgd.segment_inner_steps (an inner step is one ACT segment
# continuing the previous step's carry, as train_batch does) and meta_sgd.last_segment_grad (the query trajectory
# backpropagates only through each row's final segment). Meta-fine-tuning of the plain TRM checkpoint trm_ho2 (previous
# loop; 8 segments, H 3 / L 4, 4000 x 128) on the VARC headline recipe (msgd11_ms5: round 3's episodes, normalised step,
# alpha per tensor and per step, 5 inner steps, MSL, first order for 300 meta-steps). Smoke: ~10 s / meta-step first
# order, ~22 s second order with the GPU shared, so ~4-5 h. Controls: trm_ho2 under ttt_mini and under the same
# segment-step normalised rule with the hand-set alphas.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
export PRETRAIN_CFG=cfg_pretrain_miniarc ARCH="$TRM_ARCH"
PLAIN=$(ckpt_of trm_ho2)
NORM_RECIPE="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.alpha_range=10"
SEG="meta_sgd.segment_inner_steps=true meta_sgd.last_segment_grad=true"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2 +train_overrides.meta_sgd.segment_inner_steps=true"

# Plain TRM baselines on this branch
ttt trm_ho2 ms12t_adam
for n in 120 300; do ttt trm_ho2 ms12tnorm_s$n meta_sgd_steps=$n $NORM_EVAL; done
# Meta-fine-tune
pretrain msgd16_trm_ms5 +load_checkpoint=$PLAIN $FT3 $EPISODE3 $NORM_RECIPE $SEG \
  meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3 meta_sgd.first_order_steps=300 meta_sgd.inner_steps=5
for n in 120 300 1000; do ttt msgd16_trm_ms5 ms12m5_st$n meta_sgd_steps=$n meta_sgd_schedule=stretch; done
ttt msgd16_trm_ms5 ms12m5_s120 meta_sgd_steps=120
ttt msgd16_trm_ms5 ms12m5_adam
ttt trm_ho2 ms12tnorm_s1000 meta_sgd_steps=1000 $NORM_EVAL
echo "[done16] $(date +%H:%M:%S)"
