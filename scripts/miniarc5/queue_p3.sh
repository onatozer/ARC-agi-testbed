#!/bin/bash
# Phase 3: test-time training on the 30 held-out tasks, from the held-out checkpoints. usage: queue_p3.sh [trm varc treerm]
source "$(dirname "$0")/recipes.sh"
TASKS="task_ids_file=$DATA_HO/heldout_tasks.json"
HEADS="train_overrides.freeze_param_substrings=[candidate_,v_head]"
BACKBONE_ONLY="train_overrides.freeze_param_substrings=[L_level,embed_tokens,lm_head,H_init,L_init]"
for arch in ${@:-trm varc treerm}; do
  ckpt=$(ckpt_of "${arch}_ho"); echo "== TTT $arch: $ckpt"
  bash "$HERE/ttt.sh" "$ckpt" full       $TASKS
  bash "$HERE/ttt.sh" "$ckpt" embed      $TASKS train_overrides.freeze_weights=true
  bash "$HERE/ttt.sh" "$ckpt" embed_full $TASKS train_overrides.freeze_alternation_epochs=10
  if [ "$arch" = treerm ]; then
    bash "$HERE/ttt.sh" "$ckpt" full_heads_frozen       $TASKS "$HEADS"
    bash "$HERE/ttt.sh" "$ckpt" embed_full_heads_frozen $TASKS train_overrides.freeze_alternation_epochs=10 "$HEADS"
    bash "$HERE/ttt.sh" "$ckpt" embed_heads_only        $TASKS train_overrides.freeze_alternation_epochs=10 "$BACKBONE_ONLY"
  fi
  "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/${arch}_ho"
done
