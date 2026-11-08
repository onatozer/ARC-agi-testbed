#!/usr/bin/env bash
# ARC-AGI-2, second-order meta-training from the official ARC-AGI-1 TRM checkpoint, on the 233 ARC-AGI-2 training tasks it
# never saw (no continued pretraining, no reconstruction), then test time on the 120 eval tasks, against the same
# checkpoint with hand-set normalised TTT (no meta-training). Both scored along the 1000 steps (score_steps).
#   nohup bash scripts/queue_arc2_so.sh > logs/queue_arc2_so.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
ckpt() { ls -t runs/$1/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }
say() { echo "[queue] $* $(date -u)"; }
export GPUS=${GPUS:-2} D_META=$PWD/data/arc2-new233-ho V1=${V1:-/workspace/hf/arc_v1_public/step_518071}
NOREC="recon.copies=0 meta_sgd.inner_lr=1e-5 puzzle_emb_lr=0"  # 1e-5: the first ARC-AGI-2 run's probe
SCORE="score_steps=[0,25,50,100,250,500,1000]"

# 1. The meta build: the 233 new ARC-AGI-2 training tasks only, the eval tasks' test pairs held out
[ -f $D_META/train/dataset.json ] || REPLAY_FRACTION=0 PYTHONPATH=. python scripts/build_arc2_continue.py $D_META 1000 > logs/build_new233.log 2>&1 || { say "build failed"; exit 1; }
say "build done"

# 2. The official weights as one of our checkpoints (one meta-step at lr 0: nothing moves), for the no-meta baseline
[ -n "$(ckpt arc2_base_meta)" ] || RUN=arc2_base META_STEPS=1 MODE=meta bash scripts/run_arc2.sh $V1 $NOREC lr=0 meta_sgd.alpha_lr=0 > logs/arc2_base_meta.log 2>&1 || { say "conversion failed"; exit 1; }
say "baseline checkpoint: $(ckpt arc2_base_meta)"

# 3. The largest batch that fits second order (3 meta-steps each)
B=
for b in 128 96 64 48 32; do
  rm -rf runs/arc2_sosmoke_meta
  if RUN=arc2_sosmoke META_STEPS=3 MODE=meta bash scripts/run_arc2.sh $V1 $NOREC meta_sgd.second_order=true global_batch_size=$b > logs/arc2_sosmoke_$b.log 2>&1 && [ -n "$(ckpt arc2_sosmoke_meta)" ]; then B=$b; break; fi
done
[ -n "$B" ] || { say "no batch fits second order"; exit 1; }
say "second-order batch $B: $(tail -c 400 logs/arc2_sosmoke_$B.log | tr '\r' '\n' | grep -oE '[0-9.]+s/it' | tail -1)"

# 4. Second-order meta-training
[ -n "$(ckpt arc2_so_meta)" ] || RUN=arc2_so MODE=meta bash scripts/run_arc2.sh $V1 $NOREC meta_sgd.second_order=true global_batch_size=$B > logs/arc2_so_meta.log 2>&1 || { say "meta-training failed"; exit 1; }
say "meta-training done: $(ckpt arc2_so_meta)"

# 5. Test time: the meta-trained checkpoint, then the no-meta baseline
RUN=arc2_so MODE=eval bash scripts/run_arc2.sh $(ckpt arc2_so_meta) $SCORE > logs/arc2_so_eval.log 2>&1 || say "meta eval failed"
say "meta eval done"
RUN=arc2_base MODE=eval bash scripts/run_arc2.sh $(ckpt arc2_base_meta) $SCORE > logs/arc2_base_eval.log 2>&1 || say "baseline eval failed"
say "done"
