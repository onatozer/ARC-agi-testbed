#!/bin/bash
# Confirmation-tier queue (8x8 N-Queens, 3000 steps x batch 128). usage: queuenq.sh <file with lines "name overrides...">
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=${AUTORESEARCH_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/nqueens8}
mkdir -p "$RUNS"
COMMON="arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=8 global_batch_size=128 max_steps=3000 epochs=100000 train_epochs_per_iter=100000 eval_interval_steps=500 lr_warmup_steps=100"
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  { echo "=== $NAME: $ARGS"; AUTORESEARCH_RUNS="$RUNS" bash "$HERE/runnq.sh" "$NAME" $COMMON $ARGS; } >> "$RUNS/results.txt" 2>&1
done < "$1"
