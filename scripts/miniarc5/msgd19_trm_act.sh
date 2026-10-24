#!/usr/bin/env bash
# Round 19 (user): separate build from ACT-off in round 18's plain-TRM drop. trm_ho3 (this branch and build) under the
# normalised segment-step rule with the hand-set alphas and ACT halting ON, 1000 steps, against the same with ACT off
# (ms14tnorm_s1000: 8/9/12, .815) and trm_ho2 with ACT on (other build: 14/16/17, .825). Run after round 18, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2 +train_overrides.meta_sgd.segment_inner_steps=true"
ttt trm_ho3 ms14tnormact_s1000 meta_sgd_steps=1000 $NORM_EVAL
echo "[done19] $(date +%H:%M:%S)"
