#!/usr/bin/env bash
# Round 12 (user): the per-step alphas of the 5- and 10-step meta-trained checkpoints as a learned schedule shape at eval
# (meta_sgd_schedule=stretch: the k-th trained alpha for the k-th fraction of the steps) against holding the last one,
# at 120 / 300 / 1000 steps. Chained after round 11 (bracketed pgrep: no self-match). Evaluations only.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
while pgrep -f "[m]sgd11_varc.sh" > /dev/null; do sleep 60; done
for spec in "msgd11_ms5 ms9m5" "msgd11_ms10 ms9m10"; do
  set -- $spec
  for n in 120 300 1000; do ttt $1 ${2}_st$n meta_sgd_steps=$n meta_sgd_schedule=stretch; done
  for n in 300 1000; do ttt $1 ${2}_s$n meta_sgd_steps=$n; done
done
echo "[done12] $(date +%H:%M:%S)"
