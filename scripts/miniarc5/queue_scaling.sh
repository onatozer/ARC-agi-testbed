#!/bin/bash
# REPORTS_SCALING.md on MiniARC 5x5 (standard build): TreeRM with the identity candidate at K=4 and K=1, TreeGRAM with
# the visit-weighted prior target; the recipe of recipes.sh plus puzzle_emb_lr=3e-2 (REPORTS_ARC.md's recipe fix),
# the arch defaults otherwise (wta candidate loss, argmax eval pick). One GPU job at a time; checkpoints kept.
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
EMB="puzzle_emb_lr=3e-2"
run_treerm sc_id_k4 "$DATA_STD" arch.candidate_loss=wta $EMB
run_treerm sc_id_k1 "$DATA_STD" arch.candidate_loss=wta arch.num_candidates=1 arch.mcts_simulations=2 arch.eval_mcts_simulations=1 $EMB
run_treerm sc_tgv_k4 "$DATA_STD" arch=treegram arch.candidate_loss=wta $EMB
