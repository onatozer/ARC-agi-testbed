#!/usr/bin/env bash
# Round 6 (user): the one-to-one comparison with ttt_mini's 120 optimiser steps. The round-3 checkpoint's learned loop run
# for 30 and 120 full-batch steps, and the plain checkpoint under the same rule with hand-set alpha (0.01 weights, 1 rows).
# Chained after round 5: the bracket in the pgrep pattern keeps it from matching this script's own command line.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
while pgrep -f "[m]sgd5_varc.sh" > /dev/null; do sleep 60; done

FIXED="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.inner_lr=0.01 +train_overrides.meta_sgd.inner_lr_rows=1"
for n in 30 120; do
  ttt msgd3_varc_ft ms5va_s$n meta_sgd_steps=$n
  ttt varc_ho2 ms4vfix_s$n meta_sgd_steps=$n $FIXED
done
echo "[done6] $(date +%H:%M:%S)"
