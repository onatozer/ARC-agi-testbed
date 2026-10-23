#!/usr/bin/env bash
# Round 10: more eval seeds of the two 120-step normalised loops (meta-trained vs plain, same rule) to size their
# difference against the 2-4 pass@1 seed spread. Evaluations only.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
NORM="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
ttt varc_ho2 ms4vnorm_s120_seed1 meta_sgd_steps=120 seed=1 $NORM
ttt varc_ho2 ms4vnorm_s120_seed2 meta_sgd_steps=120 seed=2 $NORM
ttt msgd7_norm ms8n_s120_seed2 meta_sgd_steps=120 seed=2
echo "[done10] $(date +%H:%M:%S)"
