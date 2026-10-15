#!/bin/bash
# Phase 3b: the Embed-Full ordering with a flat lr (the cosine's decaying half otherwise starves the backbone phase),
# and Embed-TTT with more epochs, on the held-out checkpoints. usage: queue_p3b.sh [trm treerm varc]
source "$(dirname "$0")/recipes.sh"
TASKS="task_ids_file=$DATA_HO/heldout_tasks.json"
for arch in ${@:-trm treerm varc}; do
  ckpt=$(ckpt_of "${arch}_ho"); echo "== TTT 3b $arch: $ckpt"
  bash "$HERE/ttt.sh" "$ckpt" embed_full_flat $TASKS train_overrides.freeze_alternation_epochs=10 train_overrides.lr_min_ratio=1.0
  bash "$HERE/ttt.sh" "$ckpt" full_flat       $TASKS train_overrides.lr_min_ratio=1.0
  bash "$HERE/ttt.sh" "$ckpt" embed_e40       $TASKS train_overrides.freeze_weights=true epochs=40
done
