#!/usr/bin/env bash
# Round 14 (user): separate round 13's gap (retrained baseline .015-.018 below varc_ho2).
#  varc_ho4: this branch's code on varc_ho2's own build (the treerm-candidates worktree's): code replication alone.
#  varc_ho5: this session's build, training seed 1: training-run variance.
# Same recipe and the same two baseline evaluations as round 13.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
OLD_BUILD=$HOME/ARC-agi-testbed/.claude/worktrees/treerm-candidates/data/miniarc5-aug-31-ho30
NORM="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
pretrain varc_ho4 global_batch_size=128 max_steps=4000 eval_interval_steps=2000 dataset.data_path=$OLD_BUILD
pretrain varc_ho5 global_batch_size=128 max_steps=4000 eval_interval_steps=2000 seed=1
for run in varc_ho4 varc_ho5; do
  ttt $run ms10v_adam
  ttt $run ms10vnorm_s120 meta_sgd_steps=120 $NORM
done
echo "[done14] $(date +%H:%M:%S)"
