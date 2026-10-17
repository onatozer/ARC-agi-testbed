#!/bin/bash
# Runs the lines "name overrides..." of a queue file in order on a tier (one GPU job at a time via run.sh's lock) and
# appends each run's summary to $RUNS/<tier>/results.txt. usage: queue.sh <sudoku6|nqueens8> <queue file>
TIER=$1; QUEUE=$2
HERE=$(cd "$(dirname "$0")" && pwd)
RUNS=${CANDIDATES_RUNS:-$HOME/ARC-agi-testbed/autoresearch_runs/candidates}/$TIER
mkdir -p "$RUNS"
BACKBONE="arch.hidden_size=128 arch.num_heads=4 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=4 arch.halt_max_steps=8 epochs=100000 train_epochs_per_iter=100000 seed=0"
case "$TIER" in
  # REPORTS.md's fast tier: batch 64 x 2000 steps, lr 1e-3, eval on 512 test puzzles every 500 steps
  sudoku6)  COMMON="$BACKBONE global_batch_size=64 max_steps=2000 eval_interval_steps=500 max_eval_steps=8 lr_warmup_steps=50 lr=1e-3";;
  # REPORTS.md's confirmation tier: batch 128 x 3000 steps, lr 3e-4, eval on the full 1000-puzzle test set every 500 steps
  nqueens8) COMMON="$BACKBONE global_batch_size=128 max_steps=3000 eval_interval_steps=500 max_eval_steps=null lr_warmup_steps=100 lr=3e-4";;
  *) echo "unknown tier $TIER" >&2; exit 2;;
esac
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  case "$NAME" in \#*) continue;; esac
  { echo "=== $NAME: $ARGS"; bash "$HERE/run.sh" "$TIER" "$NAME" $COMMON $ARGS; } >> "$RUNS/results.txt" 2>&1
done < "$QUEUE"
