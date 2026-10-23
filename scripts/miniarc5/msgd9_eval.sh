#!/usr/bin/env bash
# Round 9: controls for round 8's 7/7/8 (.833) from the normalised learned loop at 120 steps. (a) The plain checkpoint under
# the same RMS-normalised rule with the hand-set alphas the meta-run started from (1e-3 weights / 3e-2 rows): is the gain
# the rule or the meta-training? (b) A second eval seed of the 120-step result. Evaluations only.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
NORM="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
for n in 10 30 120; do ttt varc_ho2 ms4vnorm_s$n meta_sgd_steps=$n $NORM; done
ttt msgd7_norm ms8n_s120_seed1 meta_sgd_steps=120 seed=1
echo "[done9] $(date +%H:%M:%S)"
