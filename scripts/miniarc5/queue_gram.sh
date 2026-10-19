#!/bin/bash
# REPORTS_GRAM_TTS.md on MiniARC 5x5 (standard build): GRAM with the pooled LPRM head and the paper's final target,
# prior sample fraction 0.25 and the draw bootstrap, sampled guidance at eval; the recipes.sh recipe plus
# puzzle_emb_lr=3e-2. One GPU job at a time; the checkpoint is kept for the ladder.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
run_trm sc_gram_pf "$DATA_STD" arch=gram +arch.eval_prior_mean=false arch.prior_sample_frac=0.25 arch.lprm_prior_rollouts=4 arch.lprm_pool=mean arch.lprm_target=final puzzle_emb_lr=3e-2
