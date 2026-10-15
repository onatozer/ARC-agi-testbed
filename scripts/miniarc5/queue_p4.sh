#!/bin/bash
# Phase 4: head freezing in regular training, as 1000-step continuations of the standard TreeRM checkpoint
# (freezing untrained heads from scratch would leave the search with a random prior and value).
source "$(dirname "$0")/recipes.sh"
ckpt=$(ckpt_of treerm_std); echo "== continuation from $ckpt"
CONT="load_checkpoint=$ckpt load_weights_only=true max_steps=1000 eval_interval_steps=1000 lr_warmup_steps=50"
run_treerm treerm_cont_ctrl      "$DATA_STD" $CONT
run_treerm treerm_cont_heads     "$DATA_STD" $CONT "freeze_param_substrings=[candidate_,v_head]"
run_treerm treerm_cont_alt       "$DATA_STD" $CONT freeze_alternation_epochs=25
run_treerm treerm_cont_alt_heads "$DATA_STD" $CONT freeze_alternation_epochs=25 "freeze_param_substrings=[candidate_,v_head]"
for name in treerm_cont_ctrl treerm_cont_heads treerm_cont_alt treerm_cont_alt_heads; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
