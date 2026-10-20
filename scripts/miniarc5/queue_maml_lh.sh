#!/bin/bash
# Partial MAML, the reasoning block only: the inner loop adapts L_level (the base TRM's single block serving both the
# L and H updates; alpha 0.1), the embeddings and heads train in the outer loop only. Base TRM, held-out build.
source "$(dirname "$0")/maml_parts.sh"
maml_pretrain maml_lh 0.1 "[inner.L_level.]"
grid maml_lh trm_ho2 mm3l 0.1 "[inner.L_level.]"
