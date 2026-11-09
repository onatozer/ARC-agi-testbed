#!/usr/bin/env bash
# ARC-AGI-2 second-order meta-training, take 5: take 3 (queue_arc2_so3.sh: 8 rows per GPU, ~768 rows per update, alpha_lr
# 3e-2) with the meta lr decaying 1e-4 -> 1e-5 (lr_min_ratio 0.1), the gradient noise scale logged, and K backbone inner
# steps of which only the last 8 are second order (meta_sgd.second_order_steps: memory as take 3's 8), ROWS (4) rows-only
# steps (user: fixed, not K/2, to spend the time on backbone steps). TIME_KS: time a 2-update smoke per K first, then K =
# the largest whose 400 updates fit MAX_HOURS (or set K). Test as trained (ROWS rows-only + K steps), scored along the way.
#   nohup bash scripts/queue_arc2_so5.sh > logs/queue_arc2_so5.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
say() { echo "[queue5] $* $(date -u)"; }
export GPUS=${GPUS:-7} D_META=$PWD/data/arc2-new233-ho V1=${V1:-/workspace/hf/arc_v1_public/step_518071}
ROWS=${ROWS:-4}; B=$((8 * GPUS)); A=$(( (768 + B / 2) / B ))
take5() { echo "recon.copies=0 meta_sgd.inner_lr=1e-5 puzzle_emb_lr=0 meta_sgd.second_order=true meta_sgd.alpha_range=2 \
  meta_sgd.inner_steps=$1 meta_sgd.second_order_steps=8 meta_sgd.rows_steps=$ROWS lr_min_ratio=0.1 meta_sgd.noise_scale=true \
  global_batch_size=$B meta_sgd.accum_steps=$A"; }

K=${K:-}
if [ -z "$K" ]; then
  for k in ${TIME_KS:-32 64 128}; do
    rm -rf runs/arc2_so5t_meta
    RUN=arc2_so5t META_STEPS=3 MODE=meta bash scripts/run_arc2.sh $V1 $(take5 $k) > logs/arc2_so5t_$k.log 2>&1 || { say "K=$k smoke failed"; break; }
    s=$(tr '\r' '\n' < logs/arc2_so5t_$k.log | grep -oE '[0-9.]+s/it' | tail -1 | tr -d 's/it')
    say "K=$k: ${s}s per update"
    python3 -c "import sys; sys.exit(0 if 400 * $s / 3600 <= ${MAX_HOURS:-10} else 1)" && K=$k || break
  done
fi
[ -n "$K" ] || { say "no K fits"; exit 1; }
say "meta-training K=$K, batch $B x accum_steps $A"
RUN=arc2_so5 META_STEPS=400 MODE=meta bash scripts/run_arc2.sh $V1 $(take5 $K) checkpoint_interval_steps=50 > logs/arc2_so5_meta.log 2>&1 || say "meta-training exited non-zero"
CK=$(ls -t runs/arc2_so5_meta/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1)
[ -n "$CK" ] || { say "no checkpoint"; exit 1; }
say "testing $CK"
SC=$(python3 -c "print(str([s for s in (0, 8, 16, 32, 64, 128) if s < $K]).replace(' ', ''))")
RUN=arc2_so5 MODE=eval bash scripts/run_arc2.sh $CK meta_sgd_steps=$K meta_sgd_rows_steps=$ROWS score_steps=$SC > logs/arc2_so5_eval.log 2>&1 || { say "eval failed"; exit 1; }
say "done"
