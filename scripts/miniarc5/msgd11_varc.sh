#!/usr/bin/env bash
# Round 11 (user: next steps 1-3 of REPORTS_SGD.md round 10).
#  (2) Longer test loops: the normalised rule for 300 and 1000 steps on the meta-trained (msgd7_norm) and plain checkpoints.
#  (1) Meta-train the loop that is used: 5 and 10 inner steps under the normalised rule, alpha per tensor and per step,
#      multi-step loss annealed over the first 30%, first order for the first 300 meta-steps (as the round-5 MAML++ run).
#  (3) From scratch at the baseline budget (4000 x batch 128, lr 1e-3) with the normalised recipe; the plain from-scratch
#      control is the baseline varc_ho2 itself (same build, budget and seed).
# One job at a time; evaluations at 10 / 30 / 120 learned steps and ttt_mini.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
PLAIN=$(ckpt_of varc_ho2)
NORM_RECIPE="meta_sgd.normalize_grad=true meta_sgd.alpha_per_tensor=true meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.alpha_range=10"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
MULTI="meta_sgd.alpha_per_step=true meta_sgd.msl_anneal_fraction=0.3 meta_sgd.first_order_steps=300"
loops() { local run=$1 pfx=$2; for n in 10 30 120; do ttt "$run" "${pfx}_s$n" meta_sgd_steps=$n; done; ttt "$run" "${pfx}_adam"; }

# (2)
for n in 300 1000; do
  ttt msgd7_norm ms8n_s$n meta_sgd_steps=$n
  ttt varc_ho2 ms4vnorm_s$n meta_sgd_steps=$n $NORM_EVAL
done
# (1)
pretrain msgd11_ms5 +load_checkpoint=$PLAIN $FT3 $EPISODE3 $NORM_RECIPE $MULTI meta_sgd.inner_steps=5;   loops msgd11_ms5 ms9m5
# (3)
pretrain msgd11_scratch $EPISODE3 $NORM_RECIPE global_batch_size=128 lr=1e-3 lr_warmup_steps=100 max_steps=4000 eval_interval_steps=2000
loops msgd11_scratch ms9sc
# (1), 10 steps
pretrain msgd11_ms10 +load_checkpoint=$PLAIN $FT3 $EPISODE3 $NORM_RECIPE $MULTI meta_sgd.inner_steps=10; loops msgd11_ms10 ms9m10
echo "[done11] $(date +%H:%M:%S)"
