#!/usr/bin/env bash
# Second-order Meta-SGD on VARC / MiniARC with 30 held-out tasks: the VARC grid of l-reinit-ttt's REPORTS_MAML.md, re-run with
# the meta-sgd-step implementation (pretrain.py meta_sgd_batch, evaluate_ttt.py meta_sgd_steps). One GPU job at a time.
#   bash scripts/miniarc5/msgd2_varc.sh            # runs everything that has no result yet
# Baseline checkpoint varc_ho2 (plain VARC, 4000 steps on the same build) comes from the previous loop and is reused.
set -euo pipefail
source "$(dirname "$0")/msgd_common.sh"

# Baseline through the new eval path: plain ttt_mini, and fixed-alpha inner steps on the rows / the layers (the previous picks)
ttt varc_ho2 ms4v_adam
for n in 5 10; do
  ttt varc_ho2 ms4ve_s$n meta_sgd_steps=$n +train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.inner_lr=1 "+train_overrides.meta_sgd.adapt_substrings=[puzzle_emb]"
  ttt varc_ho2 ms4vl_s$n meta_sgd_steps=$n +train_overrides.meta_sgd.enabled=true +train_overrides.meta_sgd.inner_lr=0.01 "+train_overrides.meta_sgd.adapt_substrings=[inner.layers.]"
done

# Meta-SGD: rows only (alpha0 1), layers only (alpha0 0.01), and the paper's default of everything (alpha0 0.01, learned per element)
pretrain msgd2_varc_emb $MSGD meta_sgd.inner_lr=1 "meta_sgd.adapt_substrings=[puzzle_emb]"
grid msgd2_varc_emb ms4ve
pretrain msgd2_varc_lh $MSGD meta_sgd.inner_lr=0.01 "meta_sgd.adapt_substrings=[inner.layers.]"
grid msgd2_varc_lh ms4vl
pretrain msgd2_varc_all $MSGD meta_sgd.inner_lr=0.01
grid msgd2_varc_all ms4va
# Follow-ups (user, 2026-10-23): the data-budget confound, on the all-parameters variant only.
# (a) Meta-SGD from the plain checkpoint for 2000 meta-steps, against the plain checkpoint continued for the same steps
PLAIN=$(ckpt_of varc_ho2)
pretrain msgd2_varc_all_ft $MSGD meta_sgd.inner_lr=0.01 +load_checkpoint="$PLAIN" +load_weights_only=true
pretrain varc_ho2_cont +load_checkpoint="$PLAIN" +load_weights_only=true
ttt msgd2_varc_all_ft ms4vaf_adam
ttt msgd2_varc_all_ft ms4vaf_s10 meta_sgd_steps=10
ttt varc_ho2_cont ms4vc_adam
# (b) Meta-SGD from scratch at the baseline's budget: 4000 steps x batch 128 (~30 tasks per meta-step)
pretrain msgd2_varc_all_full $MSGD meta_sgd.inner_lr=0.01 global_batch_size=128 max_steps=4000 eval_interval_steps=2000
ttt msgd2_varc_all_full ms4vafull_adam
ttt msgd2_varc_all_full ms4vafull_s10 meta_sgd_steps=10
echo "[done] $(date +%H:%M:%S)"
