#!/usr/bin/env bash
# ARC-AGI-2 meta-training restart + eval, from the finished continued pretraining. The weight step size starts at 1e-5:
# the MiniARC recipe's 1e-3 destroys TRM's 512-wide model (scripts/inner_step_probe.py: query CE after the 8 inner steps
# 3.01 at 1e-3, 0.46 with no steps, 0.052 at 1e-5).   nohup bash scripts/queue_arc2_meta.sh > logs/queue_arc2_meta.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
ckpt() { ls -t runs/$1/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }
PRE=$(ls runs/arc2_recon1_pre/outputs/*/*/*/checkpoints/step_146000_*.pth)
echo "[queue] meta-training from $PRE $(date -u)"

rm -rf runs/arc2_recon1_meta  # the diverged run (no checkpoint)
MODE=meta bash scripts/run_arc2.sh "$PRE" meta_sgd.inner_lr=1e-5 > logs/arc2_meta.log 2>&1 || { echo "[queue] meta-training failed $(date -u)"; exit 1; }
META=$(ckpt arc2_recon1_meta)
echo "[queue] meta-training done $(date -u): $META"

MODE=eval bash scripts/run_arc2.sh "$META" > logs/arc2_eval.log 2>&1 || { echo "[queue] evaluation failed $(date -u)"; exit 1; }
echo "[queue] done $(date -u)"
