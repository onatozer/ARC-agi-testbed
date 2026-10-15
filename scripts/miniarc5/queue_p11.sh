#!/bin/bash
# Phase 11: seed replicates at the 8000-step budget for the headline pair (TreeRM K=8 + emb lr 3e-2 vs TRM emb lr 3e-2).
source "$(dirname "$0")/recipes.sh"
run_treerm treerm_k8_emblr3_8k_s1 "$DATA_STD" puzzle_emb_lr=3e-2 arch.num_candidates=8 max_steps=8000 seed=1
run_trm    trm_emblr3_8k_s1       "$DATA_STD" puzzle_emb_lr=3e-2 max_steps=8000 seed=1
for name in treerm_k8_emblr3_8k_s1 trm_emblr3_8k_s1; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
