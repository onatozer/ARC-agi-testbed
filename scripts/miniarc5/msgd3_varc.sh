#!/usr/bin/env bash
# Meta-SGD on VARC / MiniARC ho30, round 3 (REPORTS_SGD.md post-mortem fixes 1-4): meta-fine-tune the plain checkpoint at its
# own batch size and a tenth of its lr; the episode is the task with 16 augmented copies (support = all their demos, query =
# all their test pairs, as TTT adapts); embedding rows start fresh from the table mean; alpha initialised per group (rows 1,
# weights 0.01) with a 10x larger lr. Control: the plain checkpoint continued with the same recipe and the ordinary step.
# One job at a time: chain the queues in one shell instead of pgrep waits (a wrapper whose command line names the script matches itself).
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"


PLAIN=$(ckpt_of varc_ho2)
# The baseline's batch size (256 holds ~3.5 tasks of 16 copies; a pass has 16x the rows so the MiniARC cap no longer binds),
# a tenth of its lr, 1000 steps (~30 passes over the 119 tasks)
FT="+load_checkpoint=$PLAIN $FT3"
EPISODE="$EPISODE3"

pretrain varc_ho2_cont2 $FT
pretrain msgd3_varc_ft $FT meta_sgd.enabled=true $EPISODE
ttt varc_ho2_cont2 ms5vc_adam
for n in 1 5 10; do ttt msgd3_varc_ft ms5va_s$n meta_sgd_steps=$n; done
ttt msgd3_varc_ft ms5va_adam
echo "[done3] $(date +%H:%M:%S)"
