#!/usr/bin/env bash
# Chain the ARC-AGI-1 stages on the pod: wait for pretraining, then meta-training, then the 400-task evaluation.
#   nohup bash scripts/queue_arc1.sh > logs/queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
ckpt() { ls -t runs/$1/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }

while pgrep -f "[p]retrain.py.*arc1_recon1_pre" >/dev/null; do sleep 300; done
if grep -aq Traceback logs/pre.log; then echo "[queue] pretraining crashed, stopping $(date -u)"; exit 1; fi
PRE=$(ckpt arc1_recon1_pre)
echo "[queue] pretraining done $(date -u): $PRE"

# Meta-training at TRM's batch; if eager mode does not fit 110 rows per GPU, half the batch
if ! MODE=meta bash scripts/run_arc1.sh "$PRE" > logs/meta.log 2>&1; then
  echo "[queue] meta-training failed at batch 770, retrying at 385 $(date -u)"
  rm -rf runs/arc1_recon1_meta
  MODE=meta bash scripts/run_arc1.sh "$PRE" global_batch_size=385 > logs/meta.log 2>&1 || { echo "[queue] meta-training failed again $(date -u)"; exit 1; }
fi
META=$(ckpt arc1_recon1_meta)
echo "[queue] meta-training done $(date -u): $META"

MODE=eval bash scripts/run_arc1.sh "$META" > logs/eval.log 2>&1 || { echo "[queue] evaluation failed $(date -u)"; exit 1; }
echo "[queue] done $(date -u)"
