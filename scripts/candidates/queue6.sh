#!/bin/bash
# Runs the lines "name overrides..." of a queue file in order (one GPU job at a time via run6.sh's lock) and appends
# each run's summary to $RUNS/results.txt. usage: queue6.sh <queue file>
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=${CANDIDATES_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/candidates6}
mkdir -p "$RUNS"
# The fast tier of REPORTS.md: hidden 128 backbone, batch 64 x 2000 steps, eval on 512 test puzzles every 500 steps
COMMON="arch.hidden_size=128 arch.num_heads=4 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=8 \
global_batch_size=64 max_steps=2000 epochs=100000 train_epochs_per_iter=100000 eval_interval_steps=500 max_eval_steps=8 lr_warmup_steps=50 lr=1e-3 seed=0"
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  case "$NAME" in \#*) continue;; esac
  { echo "=== $NAME: $ARGS"; bash "$HERE/run6.sh" "$NAME" $COMMON $ARGS; } >> "$RUNS/results.txt" 2>&1
done < "$1"
