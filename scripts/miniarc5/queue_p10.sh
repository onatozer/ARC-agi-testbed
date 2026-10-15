#!/bin/bash
# Phase 10: the best TreeRM configuration so far (K=8, puzzle_emb_lr=3e-2) at the 8000-step budget and as a seed replicate.
source "$(dirname "$0")/recipes.sh"
run_treerm treerm_k8_emblr3_8k "$DATA_STD" puzzle_emb_lr=3e-2 arch.num_candidates=8 max_steps=8000
run_treerm treerm_k8_emblr3_s1 "$DATA_STD" puzzle_emb_lr=3e-2 arch.num_candidates=8 seed=1
for name in treerm_k8_emblr3_8k treerm_k8_emblr3_s1; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
