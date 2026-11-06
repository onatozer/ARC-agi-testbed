#!/usr/bin/env bash
# ARC-AGI-2 meta-training rerun with the weight step size starting at 1e-4 (user; the 1e-5 run's learned weight steps had
# risen 2-5.6x, to 1.7e-5-5.6e-5), then eval; RUN=arc2_lr1e4 keeps the 1e-5 run (arc2_recon1_*) intact. Waits for the
# previous queue first.   nohup bash scripts/queue_arc2_meta2.sh > logs/queue_arc2_meta2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
ckpt() { ls -t runs/$1/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }
while pgrep -f "[q]ueue_arc2_meta.sh" >/dev/null; do sleep 60; done
PRE=$(ls runs/arc2_recon1_pre/outputs/*/*/*/checkpoints/step_146000_*.pth)
export RUN=arc2_lr1e4
echo "[queue] meta-training (inner_lr 1e-4) from $PRE $(date -u)"

MODE=meta bash scripts/run_arc2.sh "$PRE" meta_sgd.inner_lr=1e-4 > logs/arc2_lr1e4_meta.log 2>&1 || { echo "[queue] meta-training failed $(date -u)"; exit 1; }
META=$(ckpt ${RUN}_meta)
echo "[queue] meta-training done $(date -u): $META"

MODE=eval bash scripts/run_arc2.sh "$META" > logs/arc2_lr1e4_eval.log 2>&1 || { echo "[queue] evaluation failed $(date -u)"; exit 1; }
echo "[queue] done $(date -u)"
