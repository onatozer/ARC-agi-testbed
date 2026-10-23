#!/usr/bin/env bash
# Round 12 (user): the per-step alphas of the 5- and 10-step meta-trained checkpoints as a learned schedule shape at eval
# (meta_sgd_schedule=stretch: the k-th trained alpha for the k-th fraction of the steps) against holding the last one,
# at 120 / 300 / 1000 steps. Run it chained after the round-11 script in one shell, not behind a pgrep wait (that matches the chain itself). Evaluations only.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
for spec in "msgd11_ms5 ms9m5" "msgd11_ms10 ms9m10"; do
  set -- $spec
  [ -n "$(ckpt_of $1)" ] || { echo "[skip] $1 has no checkpoint yet"; continue; }
  for n in 120 300 1000; do ttt $1 ${2}_st$n meta_sgd_steps=$n meta_sgd_schedule=stretch; done
  for n in 300 1000; do ttt $1 ${2}_s$n meta_sgd_steps=$n; done
done
echo "[done12] $(date +%H:%M:%S)"
