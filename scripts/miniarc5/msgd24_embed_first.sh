#!/usr/bin/env bash
# Round 24 (user): arXiv 2609.21181's idea (embedding first, then the backbone with the embedding frozen) inside the best
# recipes' test-time loop: meta_sgd_embed_first=0.5, i.e. 500 normalised steps on the puzzle-embedding rows only, then 500
# on the weights only, everything else as the single-phase runs it is compared with (same alphas and schedule):
#   varc_ho3, hand-set alpha                     (single phase: 11/13/13, .878)
#   msgd17_v_a1e-3, learned alpha, stretched     (12/13/14, .885)
#   trm_ho3, ACT on, hand-set alpha              (12/14/14, .835)
#   msgd18_trm, ACT off, learned alpha, segment  (11/13/15, .829)
# Run after round 23, in one shell.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"
NORM_EVAL="+train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.normalize_grad=true +train_overrides.meta_sgd.alpha_per_tensor=true +train_overrides.meta_sgd.inner_lr=1e-3 +train_overrides.meta_sgd.inner_lr_rows=3e-2"
EF="meta_sgd_steps=1000 meta_sgd_embed_first=0.5"
ttt varc_ho3 ms18v_ef $EF $NORM_EVAL
ttt msgd17_v_a1e-3 ms18va_ef $EF meta_sgd_schedule=stretch
ttt trm_ho3 ms18t_ef $EF $NORM_EVAL +train_overrides.meta_sgd.segment_inner_steps=true
ttt msgd18_trm ms18tm_ef $EF meta_sgd_schedule=segment
echo "[done24] $(date +%H:%M:%S)"
