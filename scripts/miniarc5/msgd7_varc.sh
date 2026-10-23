#!/usr/bin/env bash
# Round 7 (user): the learned loop cannot move the weights (REPORTS_SGD.md, round-5 section: alpha hit the clamp, weights
# moved 30x less than under Adam). Three ways to let it, on round 3's recipe from the plain checkpoint, alpha per tensor:
#   1. a wider clamp (alpha_range 10: alpha up to 0.01*e^10 = 220), pure Meta-SGD;
#   2. the weights' alpha starting at 1 instead of 0.01;
#   3. the inner step on the gradient / its RMS per tensor (meta_sgd.normalize_grad), so a step moves a tensor by ~alpha as
#      Adam does; alpha then has Adam's units: init 1e-3 on the weights (ttt_mini's lr 3e-4 x a few), 3e-2 on the rows
#      (their sign-SGD lr), with the wide clamp so it can find its scale.
# Chained after round 6 (bracketed pgrep: no self-match). One job at a time.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
while pgrep -f "[m]sgd6_eval.sh" > /dev/null; do sleep 60; done
PLAIN=$(ckpt_of varc_ho2)
R3="+load_checkpoint=$PLAIN $FT3 $EPISODE3 meta_sgd.alpha_per_tensor=true"

pretrain msgd7_range10 $R3 meta_sgd.alpha_range=10;                                                           grid msgd7_range10 ms8r
pretrain msgd7_init1 $R3 meta_sgd.inner_lr=1;                                                                 grid msgd7_init1 ms8i
pretrain msgd7_norm $R3 meta_sgd.normalize_grad=true meta_sgd.inner_lr=1e-3 meta_sgd.inner_lr_rows=3e-2 meta_sgd.alpha_range=10; grid msgd7_norm ms8n
echo "[done7] $(date +%H:%M:%S)"
