#!/usr/bin/env bash
# Meta-SGD on VARC / MiniARC ho30, round 3 (REPORTS_SGD.md post-mortem fixes 1-4): meta-fine-tune the plain checkpoint at its
# own batch size and a tenth of its lr; the episode is the task with 16 augmented copies (support = all their demos, query =
# all their test pairs, as TTT adapts); embedding rows start fresh from the table mean; alpha initialised per group (rows 1,
# weights 0.01) with a 10x larger lr. Control: the plain checkpoint continued with the same recipe and the ordinary step.
# Waits for the round-2 queue (msgd2_varc.sh) to leave the GPU. One job at a time.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"

while pgrep -f "scripts/miniarc5/msgd2_varc.sh" > /dev/null; do sleep 60; done

PLAIN=$(ckpt_of varc_ho2)
# The baseline's batch size (256 holds ~3.5 tasks of 16 copies; a pass has 16x the rows so the MiniARC cap no longer binds),
# a tenth of its lr, 1000 steps (~30 passes over the 119 tasks)
FT="+load_checkpoint=$PLAIN +load_weights_only=true global_batch_size=256 lr=1e-4 lr_warmup_steps=50 max_steps=1000 eval_interval_steps=500"
EPISODE="meta_sgd.copies_per_group=16 meta_sgd.fresh_rows=true meta_sgd.inner_lr=0.01 meta_sgd.inner_lr_rows=1 meta_sgd.alpha_lr=1e-2"

pretrain varc_ho2_cont2 $FT
pretrain msgd3_varc_ft $FT meta_sgd.enabled=true $EPISODE
ttt varc_ho2_cont2 ms5vc_adam
for n in 1 5 10; do ttt msgd3_varc_ft ms5va_s$n meta_sgd_steps=$n; done
ttt msgd3_varc_ft ms5va_adam
echo "[done3] $(date +%H:%M:%S)"
