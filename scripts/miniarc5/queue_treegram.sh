#!/bin/bash
# REPORTS_SCALING.md on MiniARC 5x5: TreeGRAM K=4 with the visit-weighted prior target (the queue_scaling.sh recipe).
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/recipes.sh"
run_treerm sc_tgv_k4 "$DATA_STD" arch=treegram arch.candidate_loss=wta puzzle_emb_lr=3e-2
