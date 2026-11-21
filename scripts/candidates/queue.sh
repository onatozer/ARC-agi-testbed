#!/bin/bash
# Runs the lines "name overrides..." of a queue file in order on a tier (one GPU job at a time via run.sh's lock) and
# appends each run's summary to $RUNS/<tier>/results.txt. usage: queue.sh <sudoku6|nqueens8|nqueens8ho|sudoku4|sudoku4u> <queue file>
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
  # The GRAM paper's 8x8 benchmark as rebuilt here (REPORTS_GRAM_TTS.md section 11): 1-3 givens, held-out test inputs
  nqueens8ho) COMMON="$BACKBONE global_batch_size=128 max_steps=3000 eval_interval_steps=500 max_eval_steps=null lr_warmup_steps=100 lr=3e-4";;
  # The same at 10k steps: at 3000 neither TRM (23.7%, rising) nor GRAM (0%) has converged on the held-out build
  nqueens8ho10k) COMMON="$BACKBONE global_batch_size=128 max_steps=10000 eval_interval_steps=2000 max_eval_steps=null lr_warmup_steps=100 lr=3e-4";;
  # The paper's enumerated build (data/nqueens-8off) at our size, 10k steps: the reference for the replication
  nqueens8off10k) COMMON="$BACKBONE global_batch_size=128 max_steps=10000 eval_interval_steps=2000 max_eval_steps=null lr_warmup_steps=100 lr=3e-4";;
  # The same at 30k steps: the crossover with budget alone (GRAM's curve had not flattened at 10k, TRM's had)
  nqueens8off30k) COMMON="$BACKBONE global_batch_size=128 max_steps=30000 eval_interval_steps=5000 max_eval_steps=null lr_warmup_steps=100 lr=3e-4";;
  # The official gram-release recipe (REPORTS_GRAM_TTS.md section 18): hidden 512, 8 heads, H 3 / L 6, 16 supervision
  # steps, batch 768, lr 1e-4 with 2000 warm-up, 45k steps (the user's 2026-10-10 W&B run matched it); eval every 5000
  nqueens8off) COMMON="arch.hidden_size=512 arch.num_heads=8 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 arch.halt_max_steps=16 epochs=100000 train_epochs_per_iter=100000 seed=0 global_batch_size=768 max_steps=45000 eval_interval_steps=5000 max_eval_steps=null lr_warmup_steps=2000 lr=1e-4";;
  # The gram-posterior-fixes branch's held-out 4x4 Sudoku testbed (REPORTS_GRAM_TTS.md section 4): hidden 128, 2 heads,
  # H 3 / L 3, 4 supervision steps; batch 128 x 5000 steps, lr 3e-4, eval on the 1000 held-out puzzles every 1000 steps
  sudoku4|sudoku4u)  COMMON="arch.hidden_size=128 arch.num_heads=2 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=3 arch.halt_max_steps=4 epochs=100000 train_epochs_per_iter=100000 seed=0 global_batch_size=128 max_steps=5000 eval_interval_steps=1000 max_eval_steps=null lr_warmup_steps=100 lr=3e-4";;
  *) echo "unknown tier $TIER" >&2; exit 2;;
esac
while read -r NAME ARGS; do
  [ -z "$NAME" ] && continue
  case "$NAME" in \#*) continue;; esac
  { echo "=== $NAME: $ARGS"; bash "$HERE/run.sh" "$TIER" "$NAME" $COMMON $ARGS; } >> "$RUNS/results.txt" 2>&1
done < "$QUEUE"
