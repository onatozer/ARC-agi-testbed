#!/bin/bash
# Partial MAML, GRAM's guidance heads only: the inner loop adapts the prior, posterior and LPRM value heads
# (alpha ALPHA, set from the probe on the plain gram_ho checkpoint; the 40-step smoke diverged at 0.1), the backbone
# and embeddings train in the outer loop only. Base GRAM (shared block), held-out build; gram_ho (plain) is the control.
# usage: ALPHA=<alpha> queue_maml_gramheads.sh
source "$(dirname "$0")/maml_parts.sh"
: "${ALPHA:?set ALPHA from the heads probe}"
PARTS="[prior_head,posterior_head,v_head]"
maml_pretrain maml_gramheads "$ALPHA" "$PARTS" arch=gram
grid maml_gramheads gram_ho mm3g "$ALPHA" "$PARTS"
ttt_adam gram_ho mm3g_adam  # the plain GRAM under the ordinary loop (its full-TTT reference)
