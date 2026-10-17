#!/bin/bash
# Phase 8: build on the embedding-lr finding (puzzle_emb_lr=3e-2): longer training, combine with K=8, and the held-out
# protocol with the faster embedding lr in pretraining and in TTT.
source "$(dirname "$0")/recipes.sh"
TASKS="task_ids_file=$DATA_HO/heldout_tasks.json"
run_treerm treerm_emblr3_8k   "$DATA_STD" puzzle_emb_lr=3e-2 max_steps=8000
run_treerm treerm_k8_emblr3   "$DATA_STD" puzzle_emb_lr=3e-2 arch.num_candidates=8
run_treerm treerm_ho_emblr3   "$DATA_HO"  puzzle_emb_lr=3e-2
ckpt=$(ckpt_of treerm_ho_emblr3); echo "== TTT on $ckpt"
bash "$HERE/ttt.sh" "$ckpt" full_emb3       $TASKS train_overrides.puzzle_emb_lr=3e-2
bash "$HERE/ttt.sh" "$ckpt" embed_emb3      $TASKS train_overrides.puzzle_emb_lr=3e-2 train_overrides.freeze_weights=true
bash "$HERE/ttt.sh" "$ckpt" embed_full_emb3 $TASKS train_overrides.puzzle_emb_lr=3e-2 train_overrides.freeze_alternation_epochs=10 train_overrides.lr_min_ratio=1.0
for name in treerm_emblr3_8k treerm_k8_emblr3 treerm_ho_emblr3; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
