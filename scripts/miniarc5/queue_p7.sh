#!/bin/bash
# Phase 7: follow-ups on the embedding-lr finding (puzzle_emb_lr=3e-2 gave the best standard-protocol run).
source "$(dirname "$0")/recipes.sh"
run_treerm treerm_emblr3_s1 "$DATA_STD" puzzle_emb_lr=3e-2 seed=1
run_treerm treerm_emblr10   "$DATA_STD" puzzle_emb_lr=1e-1
run_trm    trm_emblr3       "$DATA_STD" puzzle_emb_lr=3e-2
for name in treerm_emblr3_s1 treerm_emblr10 trm_emblr3; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
