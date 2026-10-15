#!/bin/bash
# Phase 12: the TreeRM analogue of Embed-TTT: backbone frozen, task embedding AND the search heads (candidate generator,
# scorer, value head) trained together, on the held-out model trained with puzzle_emb_lr=3e-2 (flat lr, emb lr 3e-2).
source "$(dirname "$0")/recipes.sh"
TASKS="task_ids_file=$DATA_HO/heldout_tasks.json"
BACKBONE_ONLY="train_overrides.freeze_param_substrings=[L_level,embed_tokens,lm_head,H_init,L_init]"
ckpt=$(ckpt_of treerm_ho_emblr3); echo "== TTT on $ckpt"
bash "$HERE/ttt.sh" "$ckpt" embed_heads_e3 $TASKS +train_overrides.puzzle_emb_lr=3e-2 train_overrides.lr_min_ratio=1.0 "$BACKBONE_ONLY"
