#!/bin/bash
# Phase 9: the fair 8000-step pair for the embedding-lr recipe (TreeRM's 8k run is treerm_emblr3_8k).
source "$(dirname "$0")/recipes.sh"
run_trm trm_emblr3_8k "$DATA_STD" puzzle_emb_lr=3e-2 max_steps=8000
"$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/trm_emblr3_8k"
