#!/bin/bash
# Meta-SGD ablations, VARC first (REPORTS_MAML.md "Partial Meta-SGD"): the plain VARC control on the held-out build
# (the MiniARC VARC recipe: hidden 128, 4 layers, mlp 256, canvas 10, 606,768 params; puzzle_emb_lr 3e-2 as the
# TRM runs), step-size probes on it (embedding rows at 1 / 10 / 100, the transformer layers at 0.01 / 0.1 / 1), then
# Meta-SGD of the embedding rows and of the layers at the picked alphas with their TTT grids; then the TRM block and
# GRAM-heads ablations of queue_msgd_parts.sh. One GPU, in sequence.
source "$(dirname "$0")/maml_parts.sh"
MSGD="maml.learn_alpha=true maml.alpha_lr=1e-3"

run_varc varc_ho2 "$DATA_HO" puzzle_emb_lr=3e-2
for A in 1 10 100; do probe varc_ho2 pve_a$A "$A" "[puzzle_emb]"; done
for A in 0.01 0.1 1; do probe varc_ho2 pvl_a${A//./p} "$A" "[inner.layers.]"; done
AE=$(pick_alpha varc_ho2 pve 1 10 100); AL=$(pick_alpha varc_ho2 pvl 0.01 0.1 1)
echo "picked alphas: embedding rows $AE, layers $AL"

maml_pretrain_varc msgd_varc_emb "$AE" "[puzzle_emb]" $MSGD
grid msgd_varc_emb varc_ho2 ms3ve "$AE" "[puzzle_emb]"
maml_pretrain_varc msgd_varc_lh "$AL" "[inner.layers.]" $MSGD
grid msgd_varc_lh varc_ho2 ms3vl "$AL" "[inner.layers.]"
ttt_adam varc_ho2 ms3v_adam  # the plain VARC under the ordinary loop on the rebuilt build (its full-TTT reference)

maml_pretrain msgd_lh 0.1 "[inner.L_level.]" $MSGD
grid msgd_lh trm_ho2 ms3l 0.1 "[inner.L_level.]"

PARTS="[prior_head,posterior_head,v_head]"
maml_pretrain msgd_gramheads 0.01 "$PARTS" arch=gram $MSGD
grid msgd_gramheads gram_ho ms3g 0.01 "$PARTS"
ttt_adam gram_ho ms3g_adam
