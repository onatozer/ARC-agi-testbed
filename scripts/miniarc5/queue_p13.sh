#!/bin/bash
# Phase 13: the heads during regular training, from scratch at the emb lr 3e-2 recipe (control: treerm_emblr3).
#  - heads/backbone alternation: 100-step blocks (25 epochs), even blocks train everything but the heads, odd blocks the
#    heads only (freeze_alternation_substrings)
#  - embeddings/weights alternation (the paper's schedule) at the same block length
#  - heads frozen for the second half: 2000 steps normal, then a 2000-step continuation with the heads frozen, and the
#    matching unfrozen continuation as its control (same lr warmup restart)
source "$(dirname "$0")/recipes.sh"
E3="puzzle_emb_lr=3e-2"
run_treerm treerm_e3_althd  "$DATA_STD" $E3 freeze_alternation_epochs=25 "freeze_alternation_substrings=[candidate_,v_head]"
run_treerm treerm_e3_altemb "$DATA_STD" $E3 freeze_alternation_epochs=25
run_treerm treerm_e3_2k     "$DATA_STD" $E3 max_steps=2000 eval_interval_steps=2000 checkpoint_interval_steps=2000
ckpt=$(ckpt_of treerm_e3_2k); echo "== continuation from $ckpt"
CONT="+load_checkpoint=$ckpt +load_weights_only=true max_steps=2000 eval_interval_steps=1000 lr_warmup_steps=50"
run_treerm treerm_e3_2k_heads "$DATA_STD" $E3 $CONT "freeze_param_substrings=[candidate_,v_head]"
run_treerm treerm_e3_2k_ctrl  "$DATA_STD" $E3 $CONT
for name in treerm_e3_althd treerm_e3_altemb treerm_e3_2k treerm_e3_2k_heads treerm_e3_2k_ctrl; do "$WT/venv/bin/python" "$HERE/summarize.py" "$HOME/ARC-agi-testbed/autoresearch_runs/miniarc5/$name"; done
