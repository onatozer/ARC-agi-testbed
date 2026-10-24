#!/usr/bin/env bash
# Round 17 (user): very large alpha inits on VARC, from varc_ho3 (plain VARC trained on this branch and build; no foreign
# checkpoints). The 5-step normalised first-order recipe (msgd15_ms5_fo's) with alpha initialised at 1e-3 / 3e-2 (the
# usual weights / rows values, the control), 0.1 and 1 (weights and rows alike). Evaluated stretched at 120 / 300 / 1000;
# plain varc_ho3 under the same rule with the hand-set alphas at 300 / 1000 (120 exists: ms10vnorm_s120).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
BASE=$(ckpt_of varc_ho3)
R3="+load_checkpoint=$BASE $FT3 $EPISODE3"
NORM="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.alpha_range=10"
MULTI="meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3 meta_sgd.inner_steps=5 meta_sgd.first_order_steps=100000"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
for n in 300 1000; do ttt varc_ho3 ms10vnorm_s$n meta_sgd_steps=$n $NORM_EVAL; done
for spec in "msgd17_v_a1e-3 1e-3 3e-2 ms13a3" "msgd17_v_a0.1 0.1 0.1 ms13a01" "msgd17_v_a1 1 1 ms13a1"; do
  set -- $spec
  pretrain $1 $R3 $NORM $MULTI meta_sgd.inner_lr=$2 meta_sgd.inner_lr_rows=$3
  for n in 120 300 1000; do ttt $1 ${4}_st$n meta_sgd_steps=$n meta_sgd_schedule=stretch; done
done
echo "[done17] $(date +%H:%M:%S)"
