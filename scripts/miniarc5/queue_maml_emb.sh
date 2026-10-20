#!/bin/bash
# Partial MAML, embedding rows only: the inner loop adapts the puzzle-embedding table (alpha 10, the probe of
# REPORTS_MAML.md; the rows' gradient is small), everything else trains in the outer loop. Base TRM, held-out build.
source "$(dirname "$0")/maml_parts.sh"
maml_pretrain maml_emb 10 "[puzzle_emb]"
grid maml_emb trm_ho2 mm3e 10 "[puzzle_emb]"
