#!/bin/bash
# Meta-SGD ablations (REPORTS_MAML.md "Partial Meta-SGD"): the partial-MAML queues with learned per-element step
# sizes (maml.learn_alpha, init at the probed alpha, Adam at 1e-3, clamped to +-5 nats). The inner loop adapts one
# part, the outer loop trains everything plus the step sizes. Base TRM / base GRAM on the held-out build; the plain
# checkpoints (trm_ho2, gram_ho) under the same subset adaptation at the fixed probed alpha are the controls.
# One GPU: embedding rows -> the block -> GRAM's heads.
source "$(dirname "$0")/maml_parts.sh"
MSGD="maml.learn_alpha=true maml.alpha_lr=1e-3"

maml_pretrain msgd_emb 10 "[puzzle_emb]" $MSGD
grid msgd_emb trm_ho2 ms3e 10 "[puzzle_emb]"

maml_pretrain msgd_lh 0.1 "[inner.L_level.]" $MSGD
grid msgd_lh trm_ho2 ms3l 0.1 "[inner.L_level.]"

PARTS="[prior_head,posterior_head,v_head]"
maml_pretrain msgd_gramheads 0.01 "$PARTS" arch=gram $MSGD
grid msgd_gramheads gram_ho ms3g 0.01 "$PARTS"
ttt_adam gram_ho ms3g_adam  # the plain GRAM under the ordinary loop (its full-TTT reference)
