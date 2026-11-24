#!/usr/bin/env bash
# Chain the ARC-AGI-2 stages on the pod: wait for the builds, then continued pretraining, meta-training and the 120-task eval.
#   nohup bash scripts/queue_arc2.sh > logs/queue_arc2.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
ckpt() { ls -t runs/$1/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }

while pgrep -f "[r]un_arc2.sh" >/dev/null; do sleep 60; done  # MODE=build
for d in data/arc2-continue-aug-1000-ho data/arc2concept-aug-1000-ho; do
  [ -f $d/train/dataset.json ] || { echo "[queue] build missing: $d $(date -u)"; exit 1; }
done
echo "[queue] builds done $(date -u)"

MODE=pre bash scripts/run_arc2.sh > logs/arc2_pre.log 2>&1 || { echo "[queue] pretraining failed $(date -u)"; exit 1; }
PRE=$(ckpt arc2_recon1_pre)
echo "[queue] pretraining done $(date -u): $PRE"

# Meta-training at TRM's batch; if eager mode does not fit 110 rows per GPU, half the batch
if ! MODE=meta bash scripts/run_arc2.sh "$PRE" > logs/arc2_meta.log 2>&1; then
  echo "[queue] meta-training failed at batch 770, retrying at 385 $(date -u)"
  rm -rf runs/arc2_recon1_meta
  MODE=meta bash scripts/run_arc2.sh "$PRE" global_batch_size=385 > logs/arc2_meta.log 2>&1 || { echo "[queue] meta-training failed again $(date -u)"; exit 1; }
fi
META=$(ckpt arc2_recon1_meta)
echo "[queue] meta-training done $(date -u): $META"

MODE=eval bash scripts/run_arc2.sh "$META" > logs/arc2_eval.log 2>&1 || { echo "[queue] evaluation failed $(date -u)"; exit 1; }
echo "[queue] done $(date -u)"
