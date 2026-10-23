#!/usr/bin/env bash
# Round 4a: the plain checkpoint's test-time recipe, which is where the held-out accuracy actually comes from (4/30 with
# ttt_mini). Controls the table lacked (no adaptation at all; the eval's seed noise) and the cheap levers ttt_mini scaled
# down from ttt_varc (VARC canvas views, epochs, aux tasks, runs). Each eval is 1-5 min. Waits for round 3 to leave the GPU.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"

while pgrep -f "scripts/miniarc5/msgd3_varc.sh" > /dev/null; do sleep 60; done

ttt varc_ho2 ms6v_none epochs=0 warmup_epochs=0
ttt varc_ho2 ms6v_seed1 seed=1
ttt varc_ho2 ms6v_seed2 seed=2
ttt varc_ho2 ms6v_views4 num_views=4
ttt varc_ho2 ms6v_runs2 ttt_runs_per_task=2
ttt varc_ho2 ms6v_e40 epochs=40 warmup_epochs=4
ttt varc_ho2 ms6v_aux32 num_aux_tasks=32
echo "[done4] $(date +%H:%M:%S)"
