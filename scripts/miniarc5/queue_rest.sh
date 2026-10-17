#!/bin/bash
# After queue_p12.sh: rerun the standard TreeRM baseline (its first attempt crashed at the step-2000 eval, fixed), wait
# for the held-out TreeRM checkpoint, then phases 3 (TTT), 4 (head-freeze continuations) and 5 (seeds, sensitivity).
source "$(dirname "$0")/recipes.sh"
rm -rf "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_std"
echo "== treerm standard (rerun)"; run_treerm treerm_std "$DATA_STD"
"$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_std"
until [ -n "$(ckpt_of treerm_ho)" ] && [ -n "$(ckpt_of varc_ho)" ]; do sleep 60; done
echo "== phase 3"; bash "$HERE/queue_p3.sh"
echo "== phase 4"; bash "$HERE/queue_p4.sh"
echo "== phase 5"; bash "$HERE/queue_p5.sh"
