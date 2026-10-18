#!/bin/bash
# The test-time scaling ladder of REPORTS_GRAM_TTS.md from one GRAM checkpoint, under the shared GPU lock:
#   ladder.sh <gram.pth> <rung> [seed]
#   rung 0: harness checks (TreeGRAM K=1 / 1 sim at depth 8 must equal the mean inference; width 1 sampled)
#   rung s: sampled guidance at depth 8, width 1 and width 8 with vote / LPRM best-of-N (the re-tune's stop check)
#   rung 1: depth {8,16} x width {1,4,8,16} with vote and LPRM best-of-N (one process per width)
#   rung 2: TreeGRAM K {4,8} x sims {K,2K,4K} x depth {8,16}, argmax pick
# Results: the checkpoint's run dir eval_preds/*/eval_metrics_*.json; print with scripts/candidates/ladder_table.py
set -u
CKPT=$1; RUNG=$2; SEED=${3:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../.." && pwd)
LOCK=${GPU_LOCK:-$HOME/ARC-agi-testbed/autoresearch_runs/mcts_bench/gpu.lock}
run() { flock -o "$LOCK" env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$WT" "$WT/venv/bin/python" "$WT/evaluate.py" --config-path "$WT/config" --config-name cfg_eval ckpt="$CKPT" seed="$SEED" "$@"; }
case "$RUNG" in
  0) run eval=treegram_from_gram k=1 'sims=[1]' 'depth=[8]'
     run eval=gram_width width=1 'depth=[8]';;
  s) run eval=gram_width width=1 'depth=[8]'
     run eval=gram_width width=8 'depth=[8]';;
  1) for w in 1 4 8 16; do b=250; [ "$w" -ge 16 ] && b=125; run eval=gram_width width=$w 'depth=[8,16]' global_batch_size=$b; done;;
  2) run eval=treegram_from_gram k=4 'sims=[4,8,16]' 'depth=[8,16]'
     run eval=treegram_from_gram k=8 'sims=[8,16,32]' 'depth=[8,16]' global_batch_size=125;;
  *) echo "unknown rung $RUNG" >&2; exit 2;;
esac
