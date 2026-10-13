#!/bin/bash
# Fast-tier queue. usage: queue6.sh <file with lines "name overrides...">; runs them in order, appends summaries to $RUNS/results.txt
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=${AUTORESEARCH_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/sudoku6}
mkdir -p "$RUNS"
COMMON="arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=8 global_batch_size=64 max_steps=2000 epochs=100000 train_epochs_per_iter=100000 eval_interval_steps=250 max_eval_steps=8 lr_warmup_steps=50"
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  { echo "=== $NAME: $ARGS"; bash "$HERE/run6.sh" "$NAME" $COMMON $ARGS; } >> "$RUNS/results.txt" 2>&1
done < "$1"
