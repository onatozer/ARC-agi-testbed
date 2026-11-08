#!/usr/bin/env bash
# ARC-AGI-2 second-order meta-training, take 4: take 3 (below) with whole-task episodes. Take 3's 8 rows per GPU cut a task's
# 16 copies (~70 rows) into fragments; here each episode is as many augmented copies as fit one GPU (largest of 6..2 that
# passes 5 second-order steps, gradient checkpointing on), the step sizes learn 10x slower (alpha_lr 3e-3: take 3's largest
# one reached its cap as the query loss turned). Take 3's description: from the official
# ARC-AGI-1 checkpoint on the 233 never-seen ARC-AGI-2 training tasks, the largest per-GPU batch that fits second order x
# accumulation = ~768 rows (~48 tasks) per update (the first-order run's), 400 updates; the step sizes kept within e^+-2 of
# their start (take 1 went NaN when one ran 0.03 -> 22), a checkpoint every 50 updates, the newest finite one tested on
# the 120 eval tasks with score_steps (then re-tested at a shorter length if the curve peaks early). The no-meta
# baseline (official weights + normalised TTT) is done: 4.44% pass@2. With the embedding first (meta_sgd.rows_steps=4, user).
#   nohup bash scripts/queue_arc2_so4.sh > logs/queue_arc2_so4.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
say() { echo "[queue4] $* $(date -u)"; }
export GPUS=${GPUS:-7} D_META=$PWD/data/arc2-new233-ho V1=${V1:-/workspace/hf/arc_v1_public/step_518071}
NOREC="recon.copies=0 meta_sgd.inner_lr=1e-5 puzzle_emb_lr=0 meta_sgd.rows_steps=4"  # rows_steps 4: the embedding first (user), 4 + 8
SCORE="score_steps=[0,25,50,100,250,500,1000] meta_sgd_rows_steps=200"  # 200 rows-only steps, then 1000 (MiniARC round 37/65)

[ -f $D_META/train/dataset.json ] || REPLAY_FRACTION=0 PYTHONPATH=. python scripts/build_arc2_continue.py $D_META 1000 > logs/build_new233.log 2>&1 || { say "build failed"; exit 1; }
say "build done"

# Rows per copy from the build's metadata; the largest copies-per-episode whose whole task fits one GPU under second order
RPC=$(python3 -c "import json; print(json.load(open('$D_META/train/dataset.json'))['mean_puzzle_examples'])")
TAKE4="meta_sgd.second_order=true meta_sgd.alpha_lr=3e-3 +gradient_checkpoint=true"
C= B=
for copies in 6 5 4 3 2; do
  rows=$(python3 -c "import math; print(math.ceil($copies * $RPC))")
  rm -rf runs/arc2_so4smoke_meta
  if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True RUN=arc2_so4smoke META_STEPS=5 MODE=meta bash scripts/run_arc2.sh $V1 $NOREC $TAKE4 meta_sgd.copies_per_group=$copies \
       global_batch_size=$((rows * GPUS)) > logs/arc2_so4smoke_$copies.log 2>&1 && ls runs/arc2_so4smoke_meta/outputs/*/*/*/checkpoints/*.pth >/dev/null 2>&1; then
    C=$copies B=$((rows * GPUS)); break
  fi
done
[ -n "$C" ] || { say "no copies-per-episode fits second order"; exit 1; }
A=$(( (768 + B / 2) / B ))
say "$RPC rows per copy: $C copies per episode, batch $B ($((B / GPUS)) rows per GPU) x accum_steps $A"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RUN=arc2_so4 META_STEPS=400 MODE=meta bash scripts/run_arc2.sh $V1 $NOREC $TAKE4 meta_sgd.copies_per_group=$C global_batch_size=$B \
  meta_sgd.accum_steps=$A meta_sgd.alpha_range=2 checkpoint_interval_steps=50 > logs/arc2_so4_meta.log 2>&1 || say "meta-training exited non-zero"
CK=$(python3 - <<PY
import glob, os, torch
for p in sorted(glob.glob("runs/arc2_so4_meta/outputs/*/*/*/checkpoints/*.pth"), key=os.path.getmtime, reverse=True):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    if all(torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()):
        print(p); break
PY
)
[ -n "$CK" ] || { say "no finite checkpoint"; exit 1; }
say "testing $CK"
RUN=arc2_so4 MODE=eval bash scripts/run_arc2.sh $CK $SCORE > logs/arc2_so4_eval.log 2>&1 || { say "eval failed"; exit 1; }

# If an intermediate step beats the 1000-step result (pass@2, then token accuracy +0.005), re-test at that length
K=$(python3 - <<PY
import json
m = json.load(open("runs/arc2_so4_eval/ttt_metrics.json"))["metrics"]
best, key = None, (m["ARC/pass@2"], m["all/accuracy"])
for k in (25, 50, 100, 250, 500):
    v = (m.get(f"step{k}/ARC/pass@2", -1), m.get(f"step{k}/all/accuracy", -1))
    if v[0] > key[0] or (v[0] == key[0] and v[1] > key[1] + 0.005):
        best, key = k, v
print(best or "")
PY
)
say "best intermediate step: ${K:-none (1000 is best)}"
if [ -n "$K" ]; then
  steps=$(python3 -c "print(sorted({0, *[s for s in (25, 50, 100, 250, 500) if s < $K]}))" | tr -d " ")
  RUN=arc2_so4_k$K MODE=eval bash scripts/run_arc2.sh $CK meta_sgd_steps=$K score_steps=$steps meta_sgd_rows_steps=200 > logs/arc2_so4_k${K}_eval.log 2>&1 || say "shorter re-test failed"
fi
say "done"
