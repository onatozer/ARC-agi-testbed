#!/usr/bin/env bash
# Round 15 (user): second order off, on VARC. The two normalised recipes that worked, meta-fine-tuned from varc_ho2 entirely
# first order (first_order_steps beyond max_steps), evaluated as their second-order counterparts were:
#   msgd15_n1_fo  vs msgd7_norm  (one inner step):   normalised loop 120 / 300 / 1000 steps
#   msgd15_ms5_fo vs msgd11_ms5  (5 inner steps, per-step alpha, MSL): stretched 120 / 300 / 1000, held 120
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
PLAIN=$(ckpt_of varc_ho2)
R3="+load_checkpoint=$PLAIN $FT3 $EPISODE3"
NORM_RECIPE="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.alpha_range=10"
MULTI="meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3"
FO="meta_sgd.first_order_steps=100000"

pretrain msgd15_n1_fo $R3 $NORM_RECIPE $FO
for n in 120 300 1000; do ttt msgd15_n1_fo ms11n_s$n meta_sgd_steps=$n; done
pretrain msgd15_ms5_fo $R3 $NORM_RECIPE $MULTI meta_sgd.inner_steps=5 $FO
for n in 120 300 1000; do ttt msgd15_ms5_fo ms11m5_st$n meta_sgd_steps=$n meta_sgd_schedule=stretch; done
ttt msgd15_ms5_fo ms11m5_s120 meta_sgd_steps=120
echo "[done15] $(date +%H:%M:%S)"
