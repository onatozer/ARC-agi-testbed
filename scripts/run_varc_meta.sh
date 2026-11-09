#!/usr/bin/env bash
# VARC (the official VARC-ViT-18M, trained on ARC-AGI-1 training + RE-ARC only) meta-trained on a held-out benchmark with
# the ARC-AGI-2 TRM run's recipe (queue_arc2_so3.sh): second order, the embedding first (4 rows-only + 8 steps),
# accumulation to ~768 rows per update, 400 updates, step sizes kept within e^+-2 of their start, then test time on the
# benchmark's held-out tasks with score_steps. The weight step size comes from scripts/inner_step_probe.py.
#   DATA=data/miniarc5-aug-31-ho30 HO=data/miniarc5-aug-31-ho30/heldout_tasks.json RUN=varc_miniarc GRID=5 \
#     nohup bash scripts/run_varc_meta.sh > logs/varc_miniarc.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
say() { echo "[varc] $* $(date -u)"; }
GPUS=${GPUS:-2}; R=${R:-$PWD/runs}; OFFICIAL=${OFFICIAL:-/workspace/varc/checkpoint_final.pt}
LAUNCH="python -m torch.distributed.run --standalone --nproc_per_node $GPUS"
ckpt() { ls -t $R/$1/outputs/*/*/*/checkpoints/*.pth 2>/dev/null | head -1; }
export OMP_NUM_THREADS=4 WANDB_SILENT=true

ARCH="arch=varc arch.image_size=64 arch.hidden_size=512 arch.num_layers=10 arch.num_heads=8 arch.mlp_dim=512 arch.puzzle_emb_ndim=512 \
  arch.board_height=$GRID arch.board_width=$GRID dataset.grid_size=$GRID"
META="meta_sgd.enabled=true meta_sgd.second_order=true meta_sgd.rows_steps=4 meta_sgd.inner_steps=8 meta_sgd.inner_lr_rows=3e-2 \
  meta_sgd.alpha_range=2 recon.copies=0 puzzle_emb_lr=0 ema=false lr=1e-4 lr_warmup_steps=50 epochs=1000000 eval_interval_steps=null +wandb_mode=offline"
CONF="--config-name ${CFG:-cfg_pretrain_miniarc_varc} dataset.data_path=$PWD/$DATA $ARCH $META ${EXTRA:-}"  # EXTRA: e.g. meta_sgd.alpha_lr=3e-3

# 1. The official weights in our VARC's layout, then as one of our checkpoints (one meta-step at lr 0) for the probe
[ -f $R/${RUN}_official.pt ] || PYTHONPATH=. python scripts/convert_varc_checkpoint.py $OFFICIAL $R/${RUN}_official.pt || { say "conversion failed"; exit 1; }
[ -n "$(ckpt ${RUN}_base)" ] || OUTPUT_ROOT=$R/${RUN}_base $LAUNCH pretrain.py $CONF +run_name=${RUN}_base +load_checkpoint=$R/${RUN}_official.pt \
  +load_weights_only=true global_batch_size=$((16 * GPUS)) max_steps=1 lr=0 meta_sgd.alpha_lr=0 meta_sgd.second_order=false > logs/${RUN}_base.log 2>&1 || { say "base checkpoint failed"; exit 1; }

# 2. The weight step size: the probe's lowest query loss after the 8 steps (rows at 3e-2)
PYTHONPATH=. python scripts/inner_step_probe.py $(ckpt ${RUN}_base) $PWD/$DATA 4 32 > logs/${RUN}_probe.log 2>&1 || { say "probe failed"; exit 1; }
LR=$(python3 - <<PY
import re
best = min((float(m.group(2)), float(m.group(1))) for m in re.finditer(r"weights alpha ([0-9.e-]+), rows alpha 0.03: .*query CE after the steps ([0-9.]+)", open("logs/${RUN}_probe.log").read()) if float(m.group(1)) > 0)
print(best[1])
PY
)
say "probe: $(grep -c 'query CE' logs/${RUN}_probe.log) settings, weight step size $LR"

# 3. The largest per-GPU batch that fits second order, then TARGET_ROWS rows per update
B=
for rows in ${FIT_ROWS:-32 16 8 4}; do  # FIT_ROWS: per-GPU sizes to try
  rm -rf $R/${RUN}_smoke
  if OUTPUT_ROOT=$R/${RUN}_smoke $LAUNCH pretrain.py $CONF +run_name=${RUN}_smoke +load_checkpoint=$R/${RUN}_official.pt +load_weights_only=true \
       meta_sgd.inner_lr=$LR global_batch_size=$((rows * GPUS)) max_steps=3 > logs/${RUN}_smoke_$rows.log 2>&1 && [ -n "$(ckpt ${RUN}_smoke)" ]; then B=$((rows * GPUS)); break; fi
done
[ -n "$B" ] || { say "no batch fits second order"; exit 1; }
A=$(( (${TARGET_ROWS:-768} + B / 2) / B ))  # rows per update: 768 (the ARC-AGI-2 runs'); MiniARC 256 (its TRM rounds 60-64)
say "batch $B x accum_steps $A"

# 4. Meta-training
OUTPUT_ROOT=$R/${RUN}_meta $LAUNCH pretrain.py $CONF +run_name=${RUN}_meta +load_checkpoint=$R/${RUN}_official.pt +load_weights_only=true \
  meta_sgd.inner_lr=$LR global_batch_size=$B meta_sgd.accum_steps=$A max_steps=${META_STEPS:-400} checkpoint_interval_steps=${CKPT_EVERY:-50} \
  > logs/${RUN}_meta.log 2>&1 || say "meta-training exited non-zero"
FINITE=$(python3 - <<PY
import glob, os, torch
for p in sorted(glob.glob("$R/${RUN}_meta/outputs/*/*/*/checkpoints/*.pth"), key=os.path.getmtime, reverse=True):
    sd = torch.load(p, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    if all(torch.isfinite(v).all() for v in sd.values() if torch.is_tensor(v) and v.is_floating_point()):
        print(p)
PY
)
[ -n "$FINITE" ] || { say "no finite checkpoint"; exit 1; }
CK=$(echo "$FINITE" | head -1)
# VAL (a build's val_tasks.json): test time on those tasks for the un-meta-trained base and every finite checkpoint; the
# best pass@2 (then token accuracy) is the one tested. The base winning means meta-training did not help on them
if [ -n "${VAL:-}" ]; then
  BASE=$(ckpt ${RUN}_base); rm -f logs/${RUN}_val_table.txt
  for c in $BASE $FINITE; do
    n=$(basename $c .pth | cut -d_ -f2); [ "$c" = "$BASE" ] && n=base
    [ -f $R/${RUN}_val_$n/ttt_metrics.json ] || $LAUNCH evaluate_ttt.py checkpoint=$c eval=ttt_mini output_dir=$R/${RUN}_val_$n task_ids_file=$PWD/$VAL \
      meta_sgd_steps=${VAL_STEPS:-250} meta_sgd_rows_steps=200 score_steps=[0,25,50,100] wandb_mode=disabled > logs/${RUN}_val_$n.log 2>&1 || say "validation of $n failed"
    [ -f $R/${RUN}_val_$n/ttt_metrics.json ] && python3 -c "import json; m = json.load(open('$R/${RUN}_val_$n/ttt_metrics.json'))['metrics']; print(m['ARC/pass@2'], m['all/accuracy'], '$n', '$c')" >> logs/${RUN}_val_table.txt
  done
  cat logs/${RUN}_val_table.txt
  CK=$(sort -k1,1g -k2,2g logs/${RUN}_val_table.txt | tail -1 | cut -d" " -f4)
  [ -n "$CK" ] || { say "no validation result"; exit 1; }
fi
say "testing $CK"

# 5. Test time on the held-out tasks
$LAUNCH evaluate_ttt.py checkpoint=$CK eval=ttt_mini output_dir=$R/${RUN}_eval task_ids_file=$PWD/$HO meta_sgd_steps=1000 meta_sgd_rows_steps=200 \
  score_steps=[0,25,50,100,250,500,1000] wandb_mode=disabled > logs/${RUN}_eval.log 2>&1 || { say "eval failed"; exit 1; }
say "done"
