#!/usr/bin/env bash
# Round 5: "How to train your MAML" (arXiv 1810.09502) on round 3's recipe, from the plain checkpoint, plus the meta-parameter
# lrs round 3 never varied. Cheap single-step runs first (~10 min + 4 evals each), the 5-step MSL run last (~2 h).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
PLAIN=$(ckpt_of varc_ho2)
R3="+load_checkpoint=$PLAIN $FT3 $EPISODE3"

pretrain msgd5_pt $R3 meta_sgd.alpha_per_tensor=true;            grid msgd5_pt ms7pt
pretrain msgd5_alr1e-3 $R3 meta_sgd.alpha_lr=1e-3;               grid msgd5_alr1e-3 ms7a3
pretrain msgd5_alr1e-1 $R3 meta_sgd.alpha_lr=1e-1;               grid msgd5_alr1e-1 ms7a1
pretrain msgd5_lr3e-4 $R3 lr=3e-4;                               grid msgd5_lr3e-4 ms7lr
pretrain msgd5_cos $R3 lr_min_ratio=0.1;                         grid msgd5_cos ms7cos
# MAML++: 5 inner steps, multi-step loss annealed over the first 30%, alpha per tensor and per step, first order for the first 300 steps
pretrain msgd5_pp $R3 meta_sgd.inner_steps=5 meta_sgd.msl_anneal_fraction=0.3 meta_sgd.alpha_per_tensor=true meta_sgd.alpha_per_step=true meta_sgd.first_order_steps=300
grid msgd5_pp ms7pp
echo "[done5] $(date +%H:%M:%S)"
