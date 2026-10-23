#!/usr/bin/env bash
# Round 4b: ttt_mini scaled up on the plain checkpoint. Round 4a: 40 epochs (5/5/5, tok .831) and 32 aux tasks (4/6/7, .823)
# each beat the eval-seed spread (2-4 / 4-5 / 4-6, tok .805-.812); views and runs did nothing. Combine and go further.
# Waits for round 3 (the re-run wrapper's command line names msgd3_varc.sh). One job at a time.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"

while pgrep -f "scripts/miniarc5/msgd3_varc.sh" > /dev/null; do sleep 60; done

ttt varc_ho2 ms6v_e40_aux32 epochs=40 warmup_epochs=4 num_aux_tasks=32
ttt varc_ho2 ms6v_e80 epochs=80 warmup_epochs=8
ttt varc_ho2 ms6v_aux48 num_aux_tasks=48
ttt varc_ho2 ms6v_e80_aux48 epochs=80 warmup_epochs=8 num_aux_tasks=48
ttt varc_ho2 ms6v_e40_aux32_s1 epochs=40 warmup_epochs=4 num_aux_tasks=32 seed=1
echo "[done4b] $(date +%H:%M:%S)"
