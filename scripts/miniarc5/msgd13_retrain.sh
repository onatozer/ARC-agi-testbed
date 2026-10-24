#!/usr/bin/env bash
# Round 13 (user): retrain the plain baseline on this branch and this session's build (varc_ho2 came from l-reinit-ttt on
# another worktree's build with other augmentation draws), with varc_ho2's exact recipe (resolved configs diffed: only the
# data path, unresolved interpolations, the disabled meta-learning block and W&B metadata differ). Then the two baseline
# evaluations every comparison rests on: ttt_mini (varc_ho2: 4/4/4, .811) and the 120-step RMS-normalised loop with
# hand-set alpha (varc_ho2: 4.0/6.7/8.0, .827 over 3 seeds). Run it chained after the running queue, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
NORM="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
pretrain varc_ho3 global_batch_size=128 max_steps=4000 eval_interval_steps=2000
ttt varc_ho3 ms10v_adam
ttt varc_ho3 ms10vnorm_s120 meta_sgd_steps=120 $NORM
echo "[done13] $(date +%H:%M:%S)"
