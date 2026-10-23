#!/usr/bin/env bash
# Round 8: the normalised-step checkpoint (round 7, the first learned loop to solve a task at pass@1 in 10 steps, still
# improving with steps) at round 6's step counts, for the equal-procedure comparison with the plain checkpoint's 120-step
# loop (2/3/4, .801). Chained after round 7 (bracketed pgrep: no self-match).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
while pgrep -f "[m]sgd7_varc.sh" > /dev/null; do sleep 60; done
for n in 30 120; do ttt msgd7_norm ms8n_s$n meta_sgd_steps=$n; done
echo "[done8] $(date +%H:%M:%S)"
