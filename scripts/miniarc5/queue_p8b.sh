#!/bin/bash
# Phase 8b: the TTT variants on the held-out TreeRM trained with puzzle_emb_lr=3e-2, with the same embedding lr at test
# time (queue_p8's versions failed on the missing + prefix for a key the TTT config does not declare).
source "$(dirname "$0")/recipes.sh"
TASKS="task_ids_file=$DATA_HO/heldout_tasks.json"
ckpt=$(ckpt_of treerm_ho_emblr3); echo "== TTT on $ckpt"
bash "$HERE/ttt.sh" "$ckpt" full_e3       $TASKS +train_overrides.puzzle_emb_lr=3e-2
bash "$HERE/ttt.sh" "$ckpt" embed_e3      $TASKS +train_overrides.puzzle_emb_lr=3e-2 train_overrides.freeze_weights=true
bash "$HERE/ttt.sh" "$ckpt" embed_full_e3 $TASKS +train_overrides.puzzle_emb_lr=3e-2 train_overrides.freeze_alternation_epochs=10 train_overrides.lr_min_ratio=1.0
