#!/bin/bash
# Rerun of the K=8 variant (its first attempt was terminated by mistake while stopping the full-horizon run).
source "$(dirname "$0")/recipes.sh"
rm -rf "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_k8"
run_treerm treerm_k8 "$DATA_STD" arch.num_candidates=8
"$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/treerm_k8"
